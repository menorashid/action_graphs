from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_flexible import Graph_Layer
from graph_layer_flexible import Graph_Layer_Wrapper
from normalize import Normalize
import random
import numpy as np
import time

class Graph_Multi_Video(nn.Module):
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 feat_dim = None,
                 in_out_feat = None,
                 graph_size = None,
                 method = 'cos',
                 sparsify = 0.5,
                 non_lin = 'RL',
                 aft_nonlin = 'RL',
                 aft_nonlin_feat = 'RL',
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

        if in_out_feat is None:
            in_out_feat = [2048,1024]
        
        if in_out is None:
            in_out = [1024,512]
        
        if feat_dim is None:
            feat_dim = [1024,256]


        assert feat_dim[0]==in_out_feat[1]==in_out[0]

        # num_layers = 1
        
        # print 'NUM LAYERS', num_layers, in_out
        self.num_branches = 1
        print 'self.num_branches', self.num_branches

        self.bn =None
        # nn.BatchNorm1d(2048, affine = False)
        self.feature = []
        self.feature.append(nn.Linear(in_out_feat[0], in_out_feat[1], bias = True))
        to_pend = aft_nonlin_feat.split('_')
        for tp in to_pend:
            if tp.lower()=='ht':
                self.feature.append(nn.Hardtanh())
            elif tp.lower()=='rl':
                self.feature.append(nn.ReLU())
            elif tp.lower()=='l2':
                self.feature.append(Normalize())
            elif tp.lower()=='ln':
                self.feature.append(nn.LayerNorm(n_out))
            elif tp.lower()=='bn':
                self.feature.append(nn.BatchNorm1d(n_out, affine = False, track_running_stats = False))
            elif tp.lower()=='sig':
                self.feature.append(nn.Sigmoid())
            else:
                error_message = str('non_lin %s not recognized', non_lin)
                raise ValueError(error_message)

        self.feature = nn.Sequential(*self.feature)
        # self.feature_classifier = nn.Linear(in_out[-1],n_classes)


        self.linear_layer = nn.Linear(feat_dim[0], feat_dim[1], bias = True)
        
        self.graph_layer = Graph_Layer_Wrapper(in_out[0],n_out = in_out[1], non_lin = non_lin, method = method, aft_nonlin = aft_nonlin)
        
        last_graph = []
        last_graph.append(nn.Dropout(0.5))
        last_graph.append(nn.Linear(in_out[-1],n_classes))
        last_graph.append(nn.Softmax(dim=0))
        # if sigmoid:
        #     last_graph.append(nn.Sigmoid())
        self.last_graph = nn.Sequential(*last_graph)

        last_feat = []
        last_feat.append(nn.Dropout(0.5))
        last_feat.append(nn.Linear(in_out_feat[-1],n_classes))
        if sigmoid:
            last_feat.append(nn.Sigmoid())
        # last_feat.append(nn.Softmax(dim=0))
        self.last_feat = nn.Sequential(*last_feat)
        
        
    def forward(self, input, epoch_num = None, ret_bg =False, branch_to_test = -1):
        # print 'branch_to_test',branch_to_test

        strip = False
        if type(input)!=type([]):
            input = [input]
            strip = True

        identity = False
        method = None
        
        if self.graph_size is None:
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
        x_class = []
        x_det = []
        x_pred = []
        graph_sums = []

        # pmf_all = [[] for i in range(self.num_branches)]
        # x_all_all = [[] for i in range(self.num_branches)]

        for input in input_chunks:
            input_sizes = [input_curr.size(0) for input_curr in input]
            input = torch.cat(input,0)

            if is_cuda:
                input = input.cuda()
            
            inter_out = self.feature(input)

            # out_feat_branch = self.last_feat(input)


            feature_out = self.linear_layer(inter_out)

            to_keep = self.sparsify
            out_graph = self.graph_layer(inter_out, feature_out, to_keep = to_keep, graph_sum = self.graph_sum, identity = identity, method = method)
            
            
            if self.graph_sum:
                [out_graph, graph_sum] = out_graph       
                graph_sums.append(graph_sum.unsqueeze(0))
            
            for idx_sample in range(len(input_sizes)):
                if idx_sample==0:
                    start = 0
                else:
                    start = sum(input_sizes[:idx_sample])

                end = start+input_sizes[idx_sample]

                inter_out_curr = inter_out[start:end,:]
                out_graph_curr = out_graph[start:end,:]

                out_det = self.last_graph(out_graph_curr)
                out_class = self.last_feat(inter_out_curr)
                out_pred = out_class * out_det
                x_class.append(out_class)
                x_det.append(out_det)
                x_pred.append(out_pred)
                # print 'x_curr.size()',x_curr.size()
                pmf_all += [self.make_pmf(out_pred).unsqueeze(0)]
                
            
        if strip:
            # for idx_pmf, pmf in enumerate(pmf_all):
            #     assert len(pmf)==1
            #     pmf_all[idx_pmf] = pmf[0].squeeze()
            pmf_all = pmf_all[0].squeeze()
        
        x_class = torch.cat(x_class,dim=0)
        x_det = torch.cat(x_det,dim=0)
        x_pred = torch.cat(x_pred,dim=0)
        

        # for idx_x, x in enumerate(x_all_all):
        #     x_all_all[idx_x] = torch.cat(x,dim=0)
        
        # if branch_to_test>-1:
        #     x_all_all = x_all_all[branch_to_test]
        #     pmf_all = pmf_all[branch_to_test]

        # print len(graph_sums)
        # print len(pmf_all)
        if self.graph_sum:
            pmf_all = [pmf_all, torch.cat(graph_sums,dim = 0)]
        # print len(graph_sums)
        # print len(pmf_all)
        # print 'hello'
        if ret_bg:
            return x_det, pmf_all, None
        else:
            return x_det, pmf_all
        

    def make_pmf(self,x):
        
        if self.deno is None:
            pmf = torch.sum(x, dim = 0)
        else:    
            k = max(1,x.size(0)//self.deno)
            pmf,_ = torch.sort(x, dim=0, descending=True)
            pmf = pmf[:k,:]
            pmf = torch.sum(pmf[:k,:], dim = 0)

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

    def out_f(self, input):
        # is_cuda = next(self.parameters()).is_cuda
        # if is_cuda:
        #     input = input.cuda()
        
        identity = False
        method = None
        feature_out = self.linear_layer(input)
        to_keep = self.sparsify
        out_graph = self.graph_layer(input, feature_out, to_keep = to_keep, graph_sum = self.graph_sum, identity = identity, method = method)
        
        if self.graph_sum:
            [out_graph, graph_sum] = out_graph       
        
        return out_graph


class Network:
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 feat_dim = None,
                 in_out_feat = None,
                 graph_size = None,
                 method = 'cos',
                 sparsify = False,
                 non_lin = 'RL',
                 aft_nonlin = 'RL',
                 aft_nonlin_feat = 'RL',
                 sigmoid = False,
                 layer_bef = None,
                 graph_sum = False,
                 background = False,
                 just_graph = False
                 ):

        print 'network graph_size', graph_size
        self.model = Graph_Multi_Video(n_classes, deno, in_out,feat_dim,in_out_feat, graph_size, method, sparsify, non_lin, aft_nonlin, aft_nonlin_feat, sigmoid, layer_bef, graph_sum, background, just_graph)
        print self.model
        time.sleep(4)

    def get_lr_list(self, lr):
        
        
        lr_list = []

        modules = []
        modules+=[self.model.feature, self.model.linear_layer, self.model.graph_layer, self.model.last_graph, self.model.last_feat]

        for i,module in enumerate(modules):
            print i, lr[i]
            lr_list+= [{'params': [p for p in module.parameters() if p.requires_grad], 'lr': lr[i]}]

        return lr_list

def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    net = Network(n_classes= 20, deno = 8, graph_size = 2)
    print net.model
    net.model = net.model.cuda()
    input = np.zeros((16,2048))
    input = [torch.Tensor(input),torch.Tensor(input)]
    # .cuda()
    # input = Variable(input)
    output,pmf = net.model(input)
    print len(output),len(pmf)



    # print output.data.shape

if __name__=='__main__':
    main()