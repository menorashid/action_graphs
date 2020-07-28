from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class Graph_Layer(nn.Module):
    def __init__(self,in_size, feature_size, n_out = None):
        super(Graph_Layer, self).__init__()
        
        self.in_size = in_size
        self.feature_size = feature_size
        self.n_out = self.in_size if n_out is None else n_out

        self.transformers = nn.ModuleList([nn.Linear(in_size,feature_size, bias = False),nn.Linear(in_size,feature_size, bias = False)])

        # self.transformers =nn.Linear(in_size,feature_size, bias = False)
        
        self.weight = nn.Parameter(torch.randn(in_size,self.n_out))

        self.weight_time = nn.Parameter(torch.randn(in_size,self.n_out))
        # nn.init.xavier_normal(self.weight.data)
        self.alpha = 0.5
        self.Softmax = nn.Softmax(dim = 1)

    def forward(self, x):

        G = self.get_affinity(x)
        G_time = self.get_time_affinity(x.size(0))

        out = torch.mm(torch.mm(G,x),self.weight)

        out_time = torch.mm(torch.mm(G_time,x),self.weight_time)

        out = self.alpha*out+(1-self.alpha)*out_time
        # print G.size(),out.size()
        # raw_input()

        # pmf = self.make_pmf(x)
        return out


    def get_time_affinity(self, num_feat):
        # num_feat = x.size(0)
        arr = np.array(range(num_feat))[np.newaxis,:]
        arr = np.triu(np.concatenate([np.roll(arr,num) for num in range(num_feat)],0))
        arr = num_feat - (np.flipud(np.fliplr(arr))+arr)
        arr = arr.astype(float)/np.sum(arr,axis = 1, keepdims = True)

        is_cuda = next(self.parameters()).is_cuda
        
        if is_cuda:
            G = torch.autograd.Variable(torch.FloatTensor(arr).cuda())
        else:
            G = torch.autograd.Variable(torch.FloatTensor(arr))

        return G


    def get_affinity(self,input):

        out = [layer_curr(input) for layer_curr in  self.transformers]
        G = torch.mm(out[0],torch.t(out[1]))

        # out = self.transformers(input)
        # G = torch.mm(out,torch.t(out))

        G = self.Softmax(G)

        return G



