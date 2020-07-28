from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class Graph_Layer(nn.Module):
    def __init__(self,in_size, n_out = None):
        super(Graph_Layer, self).__init__()
        
        self.in_size = in_size
        # self.feature_size = feature_size
        self.n_out = self.in_size if n_out is None else n_out

        # self.transformers = nn.ModuleList([nn.Linear(in_size,feature_size, bias = False),nn.Linear(in_size,feature_size, bias = False)])
        
        self.weight = nn.Parameter(torch.randn(in_size,self.n_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias = None
        # nn.Parameter(torch.zeros(1,in_size))
        # nn.init.xavier_normal(self.weight.data)

        self.Softmax = nn.Softmax(dim = 1)


        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)



    def forward(self, x):

        G = self.get_affinity(x)
        out = torch.mm(torch.mm(G,x),self.weight)
        # +self.bias
        # print G.size(),out.size()
        # print self.bias.size()
        # raw_input()

        # pmf = self.make_pmf(x)
        return out

    def get_affinity(self,input):

        # out = [layer_curr(input) for layer_curr in  self.transformers]
        # norms = torch.norm(input, dim = 1, keepdim = True)
        # input = input/norms
        is_cuda = next(self.parameters()).is_cuda

        input = F.normalize(input)
        
        G = torch.mm(input,torch.t(input)) 

        D = torch.diagflat(torch.rsqrt(torch.sum(G,dim = 1)))


        if is_cuda:
            D = D.cuda()

        G = torch.mm(torch.mm(D,G),D)

        
        # G = self.Softmax(G)
        return G



