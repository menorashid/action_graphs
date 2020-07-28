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
                 graph_size = None,
                 method = 'cos',
                 sparsify = 0.5,
                 non_lin = 'RL',
                 aft_nonlin = 'RL',
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
            in_out = [2048,512]
        
        if feat_dim is None:
            feat_dim = [2048,256]

        num_layers = 1
        
        print 'NUM LAYERS', num_layers, in_out
        
        self.bn =None
        # nn.BatchNorm1d(2048, affine = False)
        self.linear_layer = nn.Linear(feat_dim[0], feat_dim[1], bias = True)

        # self.graph_layer = nn.ModuleList()
        
        # self.last_graph = nn.ModuleList()
        
        
        self.graph_layer = Graph_Layer_Wrapper(in_out[0],n_out = in_out[1], non_lin = non_lin, method = method, aft_nonlin = aft_nonlin)
        
        last_graph = []
        last_graph.append(nn.Linear(in_out[1],in_out[2], bias = True))
        last_graph.append(nn.ReLU())
        last_graph.append(nn.Dropout(0.5))
        last_graph.append(nn.Linear(in_out[-1],n_classes, bias = True))
        
        self.last_graph = nn.Sequential(*last_graph)
        
    def forward(self, input, epoch_num = None, ret_bg =False, branch_to_test = -1):
        
        strip = False
        if type(input)!=type([]):
            input = [input]
            strip = True

        identity = False
        method = None
        
        # print 'self.graph_size' , self.graph_size
        if not self.training:
            graph_size = 1
        else:
            if self.graph_size is None:
                graph_size = len(input)
            elif self.graph_size=='rand':
                graph_size = random.randint(1,len(input))
            elif type(self.graph_size)==str and self.graph_size.startswith('randupto'):
                graph_size = int(self.graph_size.split('_')[1])
                graph_size = random.randint(1,min(graph_size, len(input)))
            else:
                graph_size = min(self.graph_size, len(input))

        # print 'graph_size', graph_size, self.training
        
        input_chunks = [input[i:i + graph_size] for i in xrange(0, len(input), graph_size)]

        is_cuda = next(self.parameters()).is_cuda
        # print 'Graph branch'
        
        # pmf_all = [[] for i in range(self.num_branches)]
        # x_all_all = [[] for i in range(self.num_branches)]
        pmf_all = []
        x_all = []
        graph_sums = []

        for input in input_chunks:
            input_sizes = [input_curr.size(0) for input_curr in input]
            input = torch.cat(input,0)

            if is_cuda:
                input = input.cuda()
            
            # assert len(self.graph_layers)==(self.num_branches)
            
            # if hasattr(self, 'layer_bef') and self.layer_bef is not None:
            #     input = self.layer_bef(input)

            feature_out = self.linear_layer(input)
            # # for col_num in range(len(self.graph_layers)):
            # print 'feature_out.size()',feature_out.size()

            to_keep = self.sparsify
                # if to_keep=='lin':
            # out_graph = self.graph_layer(input)
            #     else:             
            out_graph = self.graph_layer(input, feature_out, to_keep = to_keep, graph_sum = self.graph_sum, identity = identity, method = method)
            
            
            if self.graph_sum:
                [out_graph, graph_sum] = out_graph       
                graph_sums.append(graph_sum.unsqueeze(0))

            # print 'out_graph.size()',out_graph.size()
        
            out_col = self.last_graph(out_graph)
            # print 'out_col.size()',out_col.size()
            x_all.append(out_col)

    
            x = x_all[-1]
            
            for idx_sample in range(len(input_sizes)):
                if idx_sample==0:
                    start = 0
                else:
                    start = sum(input_sizes[:idx_sample])

                end = start+input_sizes[idx_sample]
                x_curr = x[start:end,:]
                pmf_all += [self.make_pmf(x_curr).unsqueeze(0)]
                
            
        if strip:
            # for idx_pmf, pmf in enumerate(pmf_all):
            #     assert len(pmf)==1
            pmf_all = pmf_all[0].squeeze()
            # print torch.min(pmf_all), torch.max(pmf_all)


        # for idx_x, x in enumerate(x_all):
        x_all = torch.cat(x_all,dim=0)
        
        if self.graph_sum:
            pmf_all = [pmf_all, torch.cat(graph_sums,dim = 0)]

        if ret_bg:
            return x_all, pmf_all, None
        else:
            return x_all, pmf_all
        

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

    def out_f_f(self, input):
        # is_cuda = next(self.parameters()).is_cuda
        # if is_cuda:
        #     input = input.cuda()
        
        identity = False
        method = None
        feature_out = self.linear_layer(input)
        # to_keep = self.sparsify
        # out_graph = self.graph_layer(input, feature_out, to_keep = to_keep, graph_sum = self.graph_sum, identity = identity, method = method)
        
        # if self.graph_sum:
        #     [out_graph, graph_sum] = out_graph       
        
        return feature_out

class Network:
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 feat_dim = None,
                 graph_size = None,
                 method = 'cos',
                 sparsify = False,
                 non_lin = 'RL',
                 aft_nonlin = 'RL',
                 sigmoid = False,
                 layer_bef = None,
                 graph_sum = False,
                 background = False,
                 just_graph = False
                 ):

        print 'network graph_size', graph_size
        self.model = Graph_Multi_Video(n_classes, deno, in_out,feat_dim, graph_size, method, sparsify, non_lin, aft_nonlin,sigmoid, layer_bef, graph_sum, background, just_graph)
        print self.model
        time.sleep(4)
        # raw_input()

    def get_lr_list(self, lr):
        
        
        lr_list = []

        modules = []
        # if self.model.layer_bef is not None:
        #     modules.append(self.model.layer_bef)

        modules+=[self.model.linear_layer, self.model.graph_layer, self.model.last_graph]

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

    net = Network(n_classes= 20, deno = 8, graph_size = 2)
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