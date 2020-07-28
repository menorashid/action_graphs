from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class Graph_Layer(nn.Module):
    def __init__(self,in_size, feature_size, n_out = None):
        super(Graph_Layer, self).__init__()
        
        self.in_size = in_size
        self.feature_size = feature_size
        self.n_out = self.in_size if n_out is None else n_out

        self.transformers = nn.Linear(in_size,feature_size, bias = False)
        # print 'no ortho'
        nn.init.orthogonal_(self.transformers.weight)
        # self.transformers.weight.requires_grad=False

        self.weight = nn.Parameter(torch.randn(in_size,self.n_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # nn.init.xavier_normal(self.weight.data)

        self.Softmax = nn.Softmax(dim = 1)

    def forward(self, x):

        G = self.get_affinity(x)
        # out = torch.mm(G,x)

        out = torch.mm(torch.mm(G,x),self.weight)

        return out

    def get_affinity(self,input):

        # out = [layer_curr(input) for layer_curr in  self.transformers]
        out = self.transformers(input)
        norms = torch.norm(out, dim = 1, keepdim = True)
        out = out/norms
        
        G = torch.mm(out,torch.t(out))
        # G = self.Softmax(G)

        return G



def main():
    gl = Graph_Layer(2048,512)
    print gl
    print gl.transformers.weight.data.size()
    print torch.mm(gl.transformers.weight, torch.t(gl.transformers.weight))

if __name__=='__main__':
    main()