from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_flexible_temp import Graph_Layer
from graph_layer_flexible_temp import Graph_Layer_Wrapper
from normalize import Normalize

import numpy as np

class Graph_Multi_Video(nn.Module):
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 feat_dim = None,
                 graph_size = None,
                 method = 'cos',
                 sparsify = False,
                 non_lin = 'HT',
                 gk = 8,
                 aft_nonlin = None,
                 ):
        super(Graph_Multi_Video, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno
        self.graph_size = graph_size
        self.sparsify = sparsify
        self.gk = gk
        if in_out is None:
            in_out = [2048,64]
        if feat_dim is None:
            feat_dim = [2048,64]

        num_layers = len(in_out)-1
        
        print 'NUM LAYERS', num_layers, in_out
        
        self.linear_layer = nn.Linear(feat_dim[0], feat_dim[1], bias = True)

        
        self.graph_layers = nn.ModuleList()
        for num_layer in range(num_layers): 
            self.graph_layers.append(Graph_Layer_Wrapper(in_out[num_layer],n_out = in_out[num_layer+1], non_lin = non_lin, method = method, aft_nonlin = aft_nonlin))
        

        last_graph = []
        
        last_graph.append(nn.Dropout(0.5))
        last_graph.append(nn.Linear(in_out[-1],n_classes))
        last_graph = nn.Sequential(*last_graph)
        self.last_graph = last_graph
        


    def get_to_keep(self,input_sizes):
        
        # gk = 1
        k_all = [max(1,size_curr//self.gk) for idx_size_curr,size_curr in enumerate(input_sizes)]
        # print 'gtk',self.gk
        k_all = int(np.mean(k_all))
        
        return k_all
    
    def forward(self, input, epoch_num = None, ret_bg =False, branch_to_test = -1):
        
        strip = False
        if type(input)!=type([]):
            input = [input]
            strip = True

        if self.graph_size is None:
            graph_size = len(input)
        elif self.graph_size=='rand':
            import random
            graph_size = random.randint(1,len(input))
        else:
            graph_size = min(self.graph_size, len(input))

        input_chunks = [input[i:i + graph_size] for i in xrange(0, len(input), graph_size)]

        is_cuda = next(self.parameters()).is_cuda
        # print 'Graph branch'
        
        # pmf = 
        # [[] for i in range(self.num_branches)]
        # x_all_all = [[] for i in range(self.num_branches)]
        x_all = []
        pmf_all = []

        for input in input_chunks:
            input_sizes = [input_curr.size(0) for input_curr in input]
            input = torch.cat(input,0)

            if type(self.sparsify)==float:
                to_keep = self.sparsify
            elif self.sparsify==True:
                to_keep = (self.get_to_keep(input_sizes),input_sizes)
            else:
                to_keep = None

            if is_cuda:
                input = input.cuda()
            
            
            
            feature_out = self.linear_layer(input)

            input_graph = input
            for col_num in range(len(self.graph_layers)):
                input_graph = self.graph_layers[col_num](input_graph, feature_out, to_keep = to_keep)
                
                # x_all_all[col_num].append(out_col)
                

            x = self.last_graph(input_graph)
            x_all.append(x)
            
            for idx_sample in range(len(input_sizes)):
                if idx_sample==0:
                    start = 0
                else:
                    start = sum(input_sizes[:idx_sample])

                end = start+input_sizes[idx_sample]
                x_curr = x[start:end,:]
                pmf_all += [self.make_pmf(x_curr).unsqueeze(0)]
                
            
        if strip:
        #     for idx_pmf, pmf in enumerate(pmf_all):
            assert len(pmf_all)==1
            pmf_all = pmf_all[0].squeeze()

        # for idx_x, x in enumerate(x_all_all):
        #     x_all_all[idx_x] = torch.cat(x,dim=0)
        x_all = torch.cat(x_all,dim=0)

        # if branch_to_test>-1:
        #     x_all_all = x_all_all[branch_to_test]
        #     pmf_all = pmf_all[branch_to_test]

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

    def get_similarity(self,input,idx_graph_layer = 0,sparsify = None, nosum = False):

        if sparsify is None:
            sparsify = self.sparsify

        is_cuda = next(self.parameters()).is_cuda

        input_sizes = [input_curr.size(0) for input_curr in input]
        input = torch.cat(input,0)

        if type(sparsify)==float:
            to_keep = sparsify
        elif sparsify==True:
            to_keep = (self.get_to_keep(input_sizes),input_sizes)
        else:
            to_keep = None

        if is_cuda:
            input = input.cuda()

        
        assert idx_graph_layer<len(self.graph_layers)

        feature_out = self.linear_layer(input)
        sim_mat = self.graph_layers[idx_graph_layer].get_affinity(feature_out, to_keep = to_keep, nosum = nosum)
            
        
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
                 gk = 8,
                 aft_nonlin = None
                 ):
        self.model = Graph_Multi_Video(n_classes, deno, in_out,feat_dim, graph_size, method, sparsify, non_lin,gk, aft_nonlin)
        print self.model
        raw_input()

    def get_lr_list(self, lr):
        
        
        lr_list = []

        lr_list+= [{'params': [p for p in self.model.linear_layer.parameters() if p.requires_grad], 'lr': lr[0]}]

        # lr_list+= [{'params': [p for p in self.model.linear_layers_after.parameters() if p.requires_grad], 'lr': lr[0]}]
        
        lr_list+= [{'params': [p for p in self.model.graph_layers.parameters() if p.requires_grad], 'lr': lr[1]}]        
        # lr_list+= [{'params': [p for p in self.model.last_linear.parameters() if p.requires_grad], 'lr': lr[2]}]
        lr_list+= [{'params': [p for p in self.model.last_graph.parameters() if p.requires_grad], 'lr': lr[1]}]

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