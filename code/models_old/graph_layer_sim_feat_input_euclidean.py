from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class Graph_Layer(nn.Module):
    def __init__(self,in_size, n_out = None):
        super(Graph_Layer, self).__init__()
        
        self.in_size = in_size
        self.n_out = self.in_size if n_out is None else n_out

        self.weight = nn.Parameter(torch.randn(in_size,self.n_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.Softmax = nn.Softmax(dim = 1)
        self.pdist = nn.PairwiseDistance(p=2, eps=1e-06, keepdim=False)
        
    def forward(self, x, sim_feat):
        G = self.get_affinity(sim_feat)
        # if torch.min(G)<0:
        #     print 'NEG G'
        out = torch.mm(torch.mm(G,x),self.weight)
        
        return out

    def get_affinity(self,input):
        # norms = torch.norm(input, dim = 1, keepdim = True)
        # input = input/norms
        
        # G = torch.mm(input,torch.t(input))
        # G = self.pdist(input[idx,:].view(1,input.size(1)),)
        G = torch.zeros(input.size(0), input.size(0)).cuda()

        for vec_curr in range(input.size(0)):
            # for vec_b in range(vec_a+1,input.size(0)):
            #     G[vec_a,vec_b] = torch.pow(input[vec_a]-input[vec_b],2).sum()

            rep_input = input[vec_curr].view(1,input.size(1))
            rep_input = rep_input.repeat(input.size(0),1)
            G[vec_curr,:]= self.pdist(rep_input, input)
            del rep_input
            
        
        maxes,_ = torch.max(G, dim = 1, keepdim = True)
        G = G/maxes

        G = 1 - G
        # 'maxes.size()', print maxes.size()
        # print 'G.size()', G.size()
        # print 'torch.max(G), torch.min(G)', torch.max(G), torch.min(G)

        # raw_input()
        
        # self.Softmax(G)
        # print torch.max(G), torch.min(G)
        # raw_input()

        return G



class Graph_Layer_Wrapper(nn.Module):
    def __init__(self,in_size, n_out = None, non_lin = 'HT'):
        super(Graph_Layer_Wrapper, self).__init__()
        self.graph_layer = Graph_Layer(in_size, n_out)
        if non_lin=='HT':
            self.non_linearity = nn.Hardtanh()
        elif non_lin=='rl':
            self.non_linearity = nn.ReLU()
        else:
            error_message = str('non_lin %s not recognized', non_lin)
            raise ValueError(error_message)
    
    def forward(self, x, sim_feat):
        sim_feat = self.non_linearity(sim_feat)
        out = self.graph_layer(x, sim_feat)
        return out

    def get_affinity(self,input):
        return self.graph_layer.get_affinity(input)        