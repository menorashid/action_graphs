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

        
    def forward(self, x, sim_feat):
        G = self.get_affinity(sim_feat)
        out = torch.mm(torch.mm(G,x),self.weight)
        
        return out

    def get_affinity(self,input):
        G = torch.mm(input,torch.t(input))
        G = self.Softmax(G)

        return G



