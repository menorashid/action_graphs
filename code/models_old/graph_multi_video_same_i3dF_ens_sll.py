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
                 sparsify = [0.8],
                 non_lin = 'HT',
                 aft_nonlin = None,
                 ):
        super(Graph_Multi_Video, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno
        self.graph_size = graph_size
        self.sparsify = sparsify
        
        if in_out is None:
            in_out = [2048,64]
        
        if feat_dim is None:
            feat_dim = [2048,64]

        num_layers = len(sparsify)
        
        print 'NUM LAYERS', num_layers, in_out
        
        self.bn =None
        # nn.BatchNorm1d(2048, affine = False)


        # self.linear_layer = nn.Linear(feat_dim[0], feat_dim[1], bias = True)
        
        
        self.graph_layers = nn.ModuleList()
        
        # self.last_graph = nn.ModuleList()
        
        for num_layer in range(num_layers): 
            self.graph_layers.append(Graph_Layer_Wrapper(in_out[0],n_out = in_out[1], non_lin = non_lin, method = method, aft_nonlin = aft_nonlin))

        last_graph = []
        if len(in_out)==3:
            last_graph.append(nn.Linear(in_out[1]*num_layers,in_out[2]))
            last_graph.append(nn.Hardtanh())            
            last_graph.append(nn.Dropout(0.5))
            last_graph.append(nn.Linear(in_out[-1],n_classes))
        else:
            last_graph.append(nn.Dropout(0.5))
            last_graph.append(nn.Linear(in_out[1]*num_layers,n_classes))
        self.last_graph = nn.Sequential(*last_graph)
            # self.last_graphs.append(last_graph)

        self.num_branches = num_layers
        print 'self.num_branches', self.num_branches
        

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
        
        pmf_all = []
        x_all = []
        
        for input in input_chunks:
            input_sizes = [input_curr.size(0) for input_curr in input]
            input = torch.cat(input,0)

            if is_cuda:
                input = input.cuda()
            
            assert len(self.graph_layers)==(self.num_branches)
            
            temp = []
            # feature_out = self.linear_layer(input)
            
            for col_num in range(len(self.graph_layers)):

                to_keep = self.sparsify[col_num]
                
                out_graph = self.graph_layers[col_num](input, input, to_keep = to_keep)
                temp.append(out_graph)

                # raw_input()

            out_graph = torch.cat(temp, dim = 1)
            x = self.last_graph(out_graph)

            x_all.append(x)


            # for branch_num in range(len(x_all_all)):
                
            #     x = x_all_all[branch_num][-1]
                
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

        x_all = torch.cat(x_all,dim=0)
        

            

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

        if sparsify:
            to_keep = (self.get_to_keep(input_sizes),input_sizes)
        else:
            to_keep = None

        # print 'to_keep',to_keep

        if is_cuda:
            input = input.cuda()
        
        assert len(self.graph_layers)==(self.num_branches-1)
        assert idx_graph_layer<len(self.graph_layers)

        # if self.bn is not None:
        #     input = self.bn(input)

        input_graph = input
        for col_num in range(idx_graph_layer+1):
            graph_layer = self.graph_layers[col_num]
            linear_layer = self.linear_layers[col_num]
            linear_layer_after = self.linear_layers_after[col_num]
        
            feature_out = self.linear_layers[col_num](input_graph)
            out_col = self.linear_layers_after[col_num](feature_out)

            if self.attention:
                alpha = F.softmax(out_col,dim = 1)
                alpha,_ = torch.max(alpha,dim =1)
            else:
                alpha = None
            if col_num ==idx_graph_layer:
                sim_mat = self.graph_layers[col_num].get_affinity(feature_out, to_keep = to_keep, alpha = alpha, nosum = nosum)
            else:
                input_graph = self.graph_layers[col_num](input_graph, feature_out, to_keep = to_keep, alpha = alpha)
        
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
                 aft_nonlin = None
                 ):
        self.model = Graph_Multi_Video(n_classes, deno, in_out,feat_dim, graph_size, method, sparsify, non_lin, aft_nonlin)
        print self.model
        raw_input()

    def get_lr_list(self, lr):
        
        
        lr_list = []

        # lr_list+= [{'params': [p for p in self.model.linear_layer.parameters() if p.requires_grad], 'lr': lr[0]}]
        lr_list+= [{'params': [p for p in self.model.graph_layers.parameters() if p.requires_grad], 'lr': lr[0]}]        
        lr_list+= [{'params': [p for p in self.model.last_graph.parameters() if p.requires_grad], 'lr': lr[0]}]

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