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
                 sameF = False
                 ):
        super(Graph_Multi_Video, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno
        self.graph_size = graph_size
        self.sparsify = sparsify
        self.graph_sum = graph_sum
        self.sameF = sameF

        if in_out is None:
            in_out = [2048,64]
        
        if feat_dim is None:
            feat_dim = [2048,64]

        num_layers = n_classes
        
        print 'NUM LAYERS', num_layers, in_out
        
        self.bn =None


        if layer_bef is None:
            self.layer_bef = None    
        else:
            self.layer_bef = []
            self.layer_bef.append(nn.Linear(layer_bef[0],layer_bef[1], bias = True))
            self.layer_bef.append(nn.ReLU())
            # self.layer_bef.append(Normalize())
            self.layer_bef = nn.Sequential(*self.layer_bef)

            
        
        if sameF:
            self.linear_layers = nn.Linear(feat_dim[0], feat_dim[1], bias = True)
        else:
            self.linear_layers = nn.ModuleList()

        self.graph_layers = nn.ModuleList()
        
        self.last_graphs = nn.ModuleList()
        
        for num_layer in range(num_layers): 
            if not sameF:
                self.linear_layers.append(nn.Linear(feat_dim[0], feat_dim[1], bias = True))
                    
            self.graph_layers.append(Graph_Layer_Wrapper(in_out[0],n_out = in_out[1], non_lin = non_lin, method = method, aft_nonlin = aft_nonlin))

            last_graph = []
            if in_out[-1]>1:
                last_graph.append(nn.Dropout(0.5))
                last_graph.append(nn.Linear(in_out[-1],1))
            if sigmoid:
                last_graph.append(nn.Sigmoid())
            last_graph = nn.Sequential(*last_graph)
            self.last_graphs.append(last_graph)

        self.num_branches = num_layers
        print 'self.num_branches', self.num_branches
        

    def forward(self, input, epoch_num = None, ret_bg =False, branch_to_test = -1):
        
        strip = False
        if type(input)!=type([]):
            input = [input]
            strip = True

        identity = False
        method = None
        if not self.training:
            graph_size = 1
        elif self.graph_size is None:
            graph_size = len(input)
        elif self.graph_size=='rand':
            graph_size = random.randint(1,len(input))
        elif type(self.graph_size)==str and self.graph_size.startswith('randupto'):
            graph_size = int(self.graph_size.split('_')[1])
            graph_size = random.randint(1,min(graph_size, len(input)))
        else:
            graph_size = min(self.graph_size, len(input))

        input_chunks = [input[i:i + graph_size] for i in xrange(0, len(input), graph_size)]

        is_cuda = next(self.parameters()).is_cuda
        
        pmf_all = []
        # [] for i in range(len(input_chunks))]
        x_all_all = []
        # [] for i in range(len(input_chunks))]
        graph_sums = []

        for idx_input, input in enumerate(input_chunks):
            input_sizes = [input_curr.size(0) for input_curr in input]
            input = torch.cat(input,0)

            if is_cuda:
                input = input.cuda()
            
            assert len(self.graph_layers)==(self.num_branches)
            
            if hasattr(self, 'layer_bef') and self.layer_bef is not None:
                input = self.layer_bef(input)

            if self.sameF:
                feature_out = self.linear_layers(input)

            out_curr_input = []
            for col_num in range(len(self.graph_layers)):

                if not self.sameF:
                    feature_out = self.linear_layers[col_num](input)                    
                to_keep = self.sparsify   
                if to_keep=='lin':
                    out_graph = self.graph_layers[col_num](input)
                else:             
                    out_graph = self.graph_layers[col_num](input, feature_out, to_keep = to_keep, graph_sum = self.graph_sum, identity = identity, method = method)
                    if self.graph_sum:
                        [out_graph, graph_sum] = out_graph       
                        graph_sums.append(graph_sum.unsqueeze(0))

                out_col = self.last_graphs[col_num](out_graph)
                out_curr_input.append(out_col)
            
            x = torch.cat(out_curr_input, dim = 1)
            x_all_all.append(x)

            for idx_sample in range(len(input_sizes)):
                if idx_sample==0:
                    start = 0
                else:
                    start = sum(input_sizes[:idx_sample])

                end = start+input_sizes[idx_sample]
                x_curr = x[start:end,:]
                pmf_all += [self.make_pmf(x_curr).unsqueeze(0)]
                
            
        if strip:
            assert len(pmf_all)==1
            pmf_all = pmf_all[0].squeeze()

        x_all_all = torch.cat(x_all_all,dim=0)
        

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
                 sameF = False
                 ):
        self.model = Graph_Multi_Video(n_classes, deno, in_out,feat_dim, graph_size, method, sparsify, non_lin, aft_nonlin,sigmoid, layer_bef, graph_sum, sameF)
        print self.model
        raw_input()

    def get_lr_list(self, lr):
        
        
        lr_list = []

        modules = []
        if self.model.layer_bef is not None:
            modules.append(self.model.layer_bef)

        modules+=[self.model.linear_layers, self.model.graph_layers, self.model.last_graphs]

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