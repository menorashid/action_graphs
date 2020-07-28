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

        
    def forward(self, x, sim_feat, sim_feat1):
        G = self.get_affinity(sim_feat, sim_feat1)
        # if torch.min(G)<0:
        #     print 'NEG G'
        out = torch.mm(torch.mm(G,x),self.weight)
        
        return out

    def get_affinity(self,input, input1):
        norms = torch.norm(input, dim = 1, keepdim = True)
        input = input/norms
        
        norms = torch.norm(input1, dim = 1, keepdim = True)
        input1 = input1/norms

        G = torch.mm(input,torch.t(input1))

        # G = self.Softmax(G)

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
    
    def forward(self, x, sim_feat, sim_feat1):
        sim_feat = self.non_linearity(sim_feat)
        sim_feat = self.non_linearity(sim_feat1)
        out = self.graph_layer(x, sim_feat, sim_feat1)
        return out

    def get_affinity(self,input, input1):
        return self.graph_layer.get_affinity(input, input1)        