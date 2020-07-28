from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F

class Graph_Layer(nn.Module):
    def __init__(self,in_size, n_out = None):
        super(Graph_Layer, self).__init__()
        
        self.in_size = in_size
        # self.feature_size = feature_size
        self.n_out = self.in_size if n_out is None else n_out

        # self.transformers = nn.ModuleList([nn.Linear(in_size,feature_size, bias = False),nn.Linear(in_size,feature_size, bias = False)])
        
        self.weight = nn.Parameter(torch.randn(in_size,self.n_out))
        # nn.init.xavier_normal(self.weight.data)

        self.Softmax = nn.Softmax(dim = 1)

    def forward(self, x):

        G = self.get_affinity(x)
        out = torch.mm(torch.mm(G,x),self.weight)
        # print G.size(),out.size()
        # raw_input()

        # pmf = self.make_pmf(x)
        return out

    def get_affinity(self,input):

        # out = [layer_curr(input) for layer_curr in  self.transformers]
        G = torch.mm(input,torch.t(input))
        G = self.Softmax(G)

        return G



