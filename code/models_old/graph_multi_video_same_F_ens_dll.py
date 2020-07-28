from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_flexible_temp import Graph_Layer
from graph_layer_flexible_temp import Graph_Layer_Wrapper
from normalize import Normalize
import random
import numpy as np

class Graph_Multi_Video(nn.Module):
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 feat_dim = None,
                 graph_size = None,
                 method = 'cos',
                 sparsify = [0.8],
                 non_lin = 'HT',
                 aft_nonlin = None,
                 sigmoid = False,
                 layer_bef = None,
                 graph_sum = False,
                 background = False,
                 just_graph = False
                 ):
        super(Graph_Multi_Video, self).__init__()
        
        self.num_classes = n_classes
        self.background = background
        
        if self.background:
            assert sigmoid
            n_classes+=1

        self.deno = deno
        self.graph_size = graph_size
        self.sparsify = sparsify
        self.graph_sum = graph_sum
        self.just_graph = just_graph

        if in_out is None:
            in_out = [2048,64]
        
        if feat_dim is None:
            feat_dim = [2048,64]

        num_layers = len(sparsify)
        
        print 'NUM LAYERS', num_layers, in_out
        
        self.bn =None
        # nn.BatchNorm1d(2048, affine = False)
        self.linear_layer = nn.Linear(feat_dim[0], feat_dim[1], bias = True)

        if layer_bef is None:
            self.layer_bef = None    
        else:
            self.layer_bef = []
            self.layer_bef.append(nn.Linear(layer_bef[0],layer_bef[1], bias = True))
            self.layer_bef.append(nn.ReLU())
            # self.layer_bef.append(Normalize())
            self.layer_bef = nn.Sequential(*self.layer_bef)

            
        
        
        self.graph_layers = nn.ModuleList()
        
        self.last_graphs = nn.ModuleList()
        
        for num_layer in range(num_layers): 
            if self.sparsify[num_layer]=='lin':
                lin_curr = []
                
                if non_lin=='HT':
                    lin_curr.append( nn.Hardtanh())
                elif non_lin.lower()=='rl':
                    lin_curr.append( nn.ReLU())
                elif non_lin is not None:
                    error_message = str('non_lin %s not recognized', non_lin)
                    raise ValueError(error_message)
                
                lin_curr.append(nn.Linear(in_out[0],in_out[1]))
                
                to_pend = aft_nonlin.split('_')
                for tp in to_pend:
                    if tp.lower()=='ht':
                        lin_curr.append(nn.Hardtanh())
                    elif tp.lower()=='rl':
                        lin_curr.append(nn.ReLU())
                    elif tp.lower()=='l2':
                        lin_curr.append(Normalize())
                    elif tp.lower()=='ln':
                        lin_curr.append(nn.LayerNorm(n_out))
                    elif tp.lower()=='bn':
                        lin_curr.append(nn.BatchNorm1d(n_out, affine = False, track_running_stats = False))
                    else:
                        error_message = str('non_lin %s not recognized', non_lin)
                        raise ValueError(error_message)
                lin_curr = nn.Sequential(*lin_curr)
                self.graph_layers.append(lin_curr)
            else:
                self.graph_layers.append(Graph_Layer_Wrapper(in_out[0],n_out = in_out[1], non_lin = non_lin, method = method, aft_nonlin = aft_nonlin))
            
            if self.just_graph:
                if sigmoid:
                    aft_nonlin_curr = 'sig'
                else:
                    aft_nonlin_curr = None
                last_graph = Graph_Layer_Wrapper(in_out[-1],n_classes, non_lin = non_lin, method = method, aft_nonlin = aft_nonlin_curr)
            else:
                last_graph = []
                last_graph.append(nn.Dropout(0.5))
                last_graph.append(nn.Linear(in_out[-1],n_classes))
                if sigmoid:
                    last_graph.append(nn.Sigmoid())
                last_graph = nn.Sequential(*last_graph)
            self.last_graphs.append(last_graph)

        self.num_branches = num_layers
        print 'self.num_branches', self.num_branches

    def process_background(self, out_pred):
        is_cuda = next(self.parameters()).is_cuda

        background = out_pred[:,-1].unsqueeze(1)
        out_pred = out_pred[:,:-1]
        
        # diff = out_pred - background
        
        # background = background.expand(background.size(0),out_pred.size(1))
        # background = 1-background
        # background = torch.min(background,out_pred)
        # background[diff>=0] = 0
        
        # mask = torch.ones(out_pred.size())
        # if is_cuda:
        #     mask = mask.cuda()
        # mask[diff<0] = 0
        # out_pred = out_pred*mask
        # out_pred = out_pred+background
        
        out_pred = out_pred - background

        return out_pred


    def forward(self, input, epoch_num = None, ret_bg =False, branch_to_test = -1):
        
        strip = False
        if type(input)!=type([]):
            input = [input]
            strip = True

        identity = False
        method = None
        # if not self.training:
        #     graph_size = 1
        #     # identity = True
        #     # method = 'cos'
        # el
        if self.graph_size is None:
            graph_size = len(input)
        elif self.graph_size=='rand':
            graph_size = random.randint(1,len(input))
        elif type(self.graph_size)==str and self.graph_size.startswith('randupto'):
            graph_size = int(self.graph_size.split('_')[1])
            graph_size = random.randint(1,min(graph_size, len(input)))
        else:
            graph_size = min(self.graph_size, len(input))

        # print graph_size
        # raw_input()
        # if self.graph_size is None:
        #     graph_size = len(input)
        # elif self.graph_size=='rand':
        #     import random
        #     graph_size = random.randint(1,len(input))
        # else:
        #     graph_size = min(self.graph_size, len(input))

        input_chunks = [input[i:i + graph_size] for i in xrange(0, len(input), graph_size)]

        is_cuda = next(self.parameters()).is_cuda
        # print 'Graph branch'
        
        pmf_all = [[] for i in range(self.num_branches)]
        x_all_all = [[] for i in range(self.num_branches)]
        graph_sums = []

        for input in input_chunks:
            input_sizes = [input_curr.size(0) for input_curr in input]
            input = torch.cat(input,0)

            if is_cuda:
                input = input.cuda()
            
            assert len(self.graph_layers)==(self.num_branches)
            
            if hasattr(self, 'layer_bef') and self.layer_bef is not None:
                input = self.layer_bef(input)

            feature_out = self.linear_layer(input)
            for col_num in range(len(self.graph_layers)):

                to_keep = self.sparsify[col_num]   
                if to_keep=='lin':
                    out_graph = self.graph_layers[col_num](input)
                else:             
                    out_graph = self.graph_layers[col_num](input, feature_out, to_keep = to_keep, graph_sum = self.graph_sum, identity = identity, method = method)
                    if self.graph_sum:
                        [out_graph, graph_sum] = out_graph       
                        graph_sums.append(graph_sum.unsqueeze(0))

                if hasattr(self, 'just_graph') and self.just_graph:
                    out_col = self.last_graphs[col_num](out_graph, feature_out,to_keep = to_keep, graph_sum = self.graph_sum, identity = identity, method = method)
                    if self.graph_sum:
                        [out_col, graph_sum] = out_col       
                        graph_sums.append(graph_sum.unsqueeze(0))                    
                else:
                    out_col = self.last_graphs[col_num](out_graph)
                if self.background:
                    out_col = self.process_background(out_col)
                x_all_all[col_num].append(out_col)


            for branch_num in range(len(x_all_all)):
                
                x = x_all_all[branch_num][-1]
                
                for idx_sample in range(len(input_sizes)):
                    if idx_sample==0:
                        start = 0
                    else:
                        start = sum(input_sizes[:idx_sample])

                    end = start+input_sizes[idx_sample]
                    x_curr = x[start:end,:]
                    pmf_all[branch_num] += [self.make_pmf(x_curr).unsqueeze(0)]
                
            
        if strip:
            for idx_pmf, pmf in enumerate(pmf_all):
                assert len(pmf)==1
                pmf_all[idx_pmf] = pmf[0].squeeze()

        for idx_x, x in enumerate(x_all_all):
            x_all_all[idx_x] = torch.cat(x,dim=0)
        
        if self.num_branches==1:
            x_all_all = x_all_all[0]
            pmf_all = pmf_all[0]

        if branch_to_test>-1:
            x_all_all = x_all_all[branch_to_test]
            pmf_all = pmf_all[branch_to_test]
        elif branch_to_test==-2:
            pmf_all = torch.cat([pmf.view(pmf.size(0),1) for pmf in pmf_all],dim = 1)
            pmf_all = torch.mean(pmf_all,dim = 1)

            x_all_all = torch.cat([torch.nn.functional.softmax(x_all,dim = 1).unsqueeze(2) for x_all in x_all_all],dim=2)
            # x_all_all = torch.cat([x_all.unsqueeze(2) for x_all in x_all_all],dim=2)
            x_all_all = torch.mean(x_all_all,dim=2)
        elif branch_to_test==-4 or branch_to_test==-3:
            pmf_all = torch.cat([pmf.view(pmf.size(0),1) for pmf in pmf_all],dim = 1)
            pmf_all = torch.mean(pmf_all,dim = 1)

            x_all_all = torch.cat([x_all.unsqueeze(2) for x_all in x_all_all],dim=2)
            x_all_all = torch.mean(x_all_all,dim=2)
        elif branch_to_test==-5:
            pmf_all = torch.cat([pmf.view(pmf.size(0),1) for pmf in pmf_all],dim = 1)
            pmf_all = torch.max(pmf_all,dim = 1)[0]

            x_all_all = torch.cat([x_all.unsqueeze(2) for x_all in x_all_all],dim=2)
            x_all_all = torch.max(x_all_all,dim=2)[0]

        if self.graph_sum:
            pmf_all = [pmf_all, torch.cat(graph_sums,dim = 0)]

        if ret_bg:
            return x_all_all, pmf_all, None
        else:
            return x_all_all, pmf_all
        

    def make_pmf(self,x):
        k = max(1,x.size(0)//self.deno)
        
        pmf,_ = torch.sort(x, dim=0, descending=True)
        pmf = pmf[:k,:]
        
        pmf = torch.sum(pmf[:k,:], dim = 0)/k
        return pmf

    def get_similarity(self,input,idx_graph_layer = 0,sparsify = False, nosum = False):

        # if sparsify is None:
        #     sparsify = self.sparsify

        is_cuda = next(self.parameters()).is_cuda

        input_sizes = [input_curr.size(0) for input_curr in input]
        input = torch.cat(input,0)


        if is_cuda:
            input = input.cuda()
        
        assert idx_graph_layer<len(self.graph_layers)
        
        if hasattr(self, 'layer_bef') and self.layer_bef is not None:
            input = self.layer_bef(input)

        feature_out = self.linear_layer(input)
        
        if sparsify:
            to_keep = self.sparsify[idx_graph_layer]                
        else:
            to_keep = None

        sim_mat = self.graph_layers[idx_graph_layer].get_affinity(feature_out, to_keep = to_keep,nosum = nosum)

        return sim_mat
    
    def printGraphGrad(self):
        grad_rel = self.graph_layers[0].graph_layer.weight.grad
        print torch.min(grad_rel).data.cpu().numpy(), torch.max(grad_rel).data.cpu().numpy()


class Network:
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 feat_dim = None,
                 graph_size = None,
                 method = 'cos',
                 sparsify = False,
                 non_lin = 'HT',
                 aft_nonlin = None,
                 sigmoid = False,
                 layer_bef = None,
                 graph_sum = False,
                 background = False,
                 just_graph = False
                 ):
        self.model = Graph_Multi_Video(n_classes, deno, in_out,feat_dim, graph_size, method, sparsify, non_lin, aft_nonlin,sigmoid, layer_bef, graph_sum, background, just_graph)
        print self.model
        raw_input()

    def get_lr_list(self, lr):
        
        
        lr_list = []

        modules = []
        if self.model.layer_bef is not None:
            modules.append(self.model.layer_bef)

        modules+=[self.model.linear_layer, self.model.graph_layers, self.model.last_graphs]

        for i,module in enumerate(modules):
            print i, lr[i]
            lr_list+= [{'params': [p for p in module.parameters() if p.requires_grad], 'lr': lr[i]}]

        # i = 0
        # if self.model.layer_bef is not None:
        #     print lr[i]
        #     lr_list+= [{'params': [p for p in self.model.layer_bef.parameters() if p.requires_grad], 'lr': lr[i]}]
        #     i+=1

        # print lr[i]
        # lr_list+= [{'params': [p for p in self.model.linear_layer.parameters() if p.requires_grad], 'lr': lr[i]}]
        # i+=1

        # print lr[i]
        # lr_list+= [{'params': [p for p in self.model.graph_layers.parameters() if p.requires_grad], 'lr': lr[i]}]        
        # i+=1

        # print lr[i]
        # lr_list+= [{'params': [p for p in self.model.last_graphs.parameters() if p.requires_grad], 'lr': lr[i]}]

        return lr_list

def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    net = Network(n_classes= 20, deno = 8)
    print net.model
    net.model = net.model.cuda()
    input = np.zeros((16,2048))
    input = torch.Tensor(input).cuda()
    input = Variable(input)
    output,pmf = net.model(input)
    # print output.shape


    print output.data.shape

if __name__=='__main__':
    main()