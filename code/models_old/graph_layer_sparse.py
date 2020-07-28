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

        
    def forward(self, x, sim_feat, k):
        G = self.get_affinity(sim_feat)
        temp = torch.mm(G,x)
        out = torch.mm(temp,self.weight)
        
        return out

    def get_affinity(self,input):
        input = F.normalize(input)
        # input = input[:5]

        G = torch.mm(input,torch.t(input)) 
        
        # set diag to zero
        eye_inv = (torch.eye(G.size(0)).cuda()+1) % 2
        G = G*eye_inv
        # print G

        # set diag to max of every row
        # G = G+torch.diagflat(torch.max(G,dim = 1)[0])
        # print G


        G = G/torch.sum(G,dim = 1, keepdim = True)
        # print G
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