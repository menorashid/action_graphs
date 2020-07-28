from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import time

class Graph_Layer(nn.Module):
    def __init__(self,in_size, n_out = None, method = 'cos', k = None):
        super(Graph_Layer, self).__init__()
        
        self.in_size = in_size
        self.n_out = self.in_size if n_out is None else n_out
        self.weight = nn.Parameter(torch.randn(in_size,self.n_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.method = method


        
    def forward(self, x, sim_feat, to_keep = None):
        G = self.get_affinity(sim_feat, to_keep)
        temp = torch.mm(G,x)
        out = torch.mm(temp,self.weight)
        
        return out

    def get_affinity(self,input, to_keep = None):

        # if to_keep is not None:
        #     input=input[:5]
        #     to_keep = 2

        if 'cos' in self.method:
            input = F.normalize(input)
        
        G = torch.mm(input,torch.t(input)) 
        
        if 'zero_self' in self.method:
            eye_inv = (torch.eye(G.size(0)).cuda()+1) % 2
            G = G*eye_inv
        
        
        


        if to_keep is not None:
            
            topk, indices = torch.topk(G, k=to_keep, dim =1)    
            G = G*0
            G = G.scatter(1, indices, topk)



        G = G/torch.sum(G,dim = 1, keepdim = True)
        
        # if to_keep is not None:        
        #     print G

        #     raw_input()        
        return G



class Graph_Layer_Wrapper(nn.Module):
    def __init__(self,in_size, n_out = None, non_lin = 'HT', method = 'cos'):
        super(Graph_Layer_Wrapper, self).__init__()
        self.graph_layer = Graph_Layer(in_size, n_out = n_out, method = method)
        if non_lin=='HT':
            self.non_linearity = nn.Hardtanh()
        elif non_lin.lower()=='rl':
            self.non_linearity = nn.ReLU()
        else:
            error_message = str('non_lin %s not recognized', non_lin)
            raise ValueError(error_message)
    
    def forward(self, x, sim_feat, to_keep = None):
        sim_feat = self.non_linearity(sim_feat)
        out = self.graph_layer(x, sim_feat, to_keep = to_keep)
        return out

    def get_affinity(self,input,to_keep = None):
        input = self.non_linearity(input)
        return self.graph_layer.get_affinity(input, to_keep = to_keep)        