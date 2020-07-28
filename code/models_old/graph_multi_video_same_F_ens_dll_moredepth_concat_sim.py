from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_flexible_concat_sim import Graph_Layer
from normalize import Normalize

import numpy as np

class Graph_Multi_Video(nn.Module):
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = [2048,64],
                 feat_dim = [2048,64],
                 graph_size = None,
                 method = 'cos',
                 num_branches = 1,
                 non_lin = 'HT',
                 non_lin_aft = 'RL',
                 aft_nonlin = None,
                 sigmoid = False,
                 layer_bef = None,
                 num_graphs = 1,
                 scaling_method = 'n'
                 ):
        super(Graph_Multi_Video, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno
        self.graph_size = graph_size
        self.num_branches = num_branches
        
        assert (len(in_out)-1)==num_graphs

        num_layers = self.num_branches
        
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

            graph_pipe = nn.ModuleList()
            for graph_num in range(num_graphs):
                graph_pipe.append(Graph_Layer(in_out[graph_num],feat_dim = feat_dim[1], n_out = in_out[graph_num+1], non_lin = non_lin,non_lin_aft = non_lin_aft, method = method, aft_nonlin = aft_nonlin, scaling_method = scaling_method))

            self.graph_layers.append(graph_pipe)
                
            last_graph = []
            last_graph.append(nn.Dropout(0.5))
            last_graph.append(nn.Linear(in_out[-1],n_classes))
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
        
        pmf_all = [[] for i in range(self.num_branches)]
        x_all_all = [[] for i in range(self.num_branches)]
        
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

                
                out_graph = input

                graph_pipe = self.graph_layers[col_num]

                for graph_num in range(len(graph_pipe)):
                    out_graph = graph_pipe[graph_num](out_graph, feature_out)

                out_col = self.last_graphs[col_num](out_graph)

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
            assert self.num_branches>1
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
        
        sim_mat = self.graph_layers[idx_graph_layer][0].get_affinity(feature_out, nosum = nosum)

        return sim_mat
    
    def printGraphGrad(self):
        grad_rel = self.graph_layers[0].graph_layer.weight.grad
        print torch.min(grad_rel).data.cpu().numpy(), torch.max(grad_rel).data.cpu().numpy()


class Network:
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = [2048,64],
                 feat_dim = [2048,64],
                 graph_size = None,
                 method = 'cos',
                 num_branches = 1,
                 non_lin = 'HT',
                 non_lin_aft = 'RL',
                 aft_nonlin = None,
                 sigmoid = False,
                 layer_bef = None,
                 num_graphs = 1,
                 scaling_method = 'sum'
                 ):

        self.model = Graph_Multi_Video(n_classes, deno, in_out, feat_dim, graph_size, method, num_branches, non_lin, non_lin_aft, aft_nonlin,sigmoid, layer_bef, num_graphs,scaling_method)
        print self.model
        raw_input()

    def get_lr_list(self, lr):
        
        
        lr_list = []
        i = 0
        if self.model.layer_bef is not None:
            print lr[i]
            lr_list+= [{'params': [p for p in self.model.layer_bef.parameters() if p.requires_grad], 'lr': lr[i]}]
            i+=1

        print lr[i]
        lr_list+= [{'params': [p for p in self.model.linear_layer.parameters() if p.requires_grad], 'lr': lr[i]}]
        i+=1

        print lr[i]
        lr_list+= [{'params': [p for p in self.model.graph_layers.parameters() if p.requires_grad], 'lr': lr[i]}]        
        lr_list+= [{'params': [p for p in self.model.last_graphs.parameters() if p.requires_grad], 'lr': lr[i]}]

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