from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import time
from normalize import Normalize
import numpy as np
from torch.autograd import Variable

class Graph_Layer(nn.Module):
    def __init__(self,in_size, feat_dim, n_out = None, method = 'cos', non_lin = 'HT', non_lin_aft = 'rl',scaling_method = 'sum',k = None,
        affinity_dict = None, aft_nonlin = None):
        super(Graph_Layer, self).__init__()
        
        self.in_size = in_size
        self.n_out = self.in_size if n_out is None else n_out
        self.weight = nn.Parameter(torch.randn(in_size,self.n_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        print self.weight.size()
        # self.feat_dim = feat_dim
        self.scaling_method = scaling_method
        
        edge_weighter = []

        if non_lin.lower()=='ht':
            edge_weighter.append( nn.Hardtanh() )
        elif non_lin.lower()=='rl':
            edge_weighter.append( nn.ReLU() )
        elif non_lin is not None:
            error_message = str('non_lin %s not recognized', non_lin)
            raise ValueError(error_message)
        
        edge_weighter.append(nn.Linear(feat_dim*2, 1, bias = False))

        if non_lin_aft is not None:
            non_lin_aft_all = non_lin_aft.split('_')
            for non_lin_aft in non_lin_aft_all:
                if non_lin_aft.lower()=='ht':
                    edge_weighter.append( nn.Hardtanh() )
                elif non_lin_aft.lower()=='rl':
                    edge_weighter.append( nn.ReLU() )
                elif non_lin_aft.lower()=='sig':
                    edge_weighter.append( nn.Sigmoid() )
                else:
                    error_message = str('non_lin_aft %s not recognized', non_lin_aft)
                    raise ValueError(error_message)
        
        self.edge_weighter = nn.Sequential(*edge_weighter)

        self.method = method

        self.aft = None
        if aft_nonlin is not None:
            self.aft = []

            to_pend = aft_nonlin.split('_')
            for tp in to_pend:
                if tp.lower()=='ht':
                    self.aft.append(nn.Hardtanh())
                elif tp.lower()=='rl':
                    self.aft.append(nn.ReLU())
                elif tp.lower()=='l2':
                    self.aft.append(Normalize())
                elif tp.lower()=='ln':
                    self.aft.append(nn.LayerNorm(n_out))
                elif tp.lower()=='bn':
                    self.aft.append(nn.BatchNorm1d(n_out, affine = False, track_running_stats = False))
                else:
                    error_message = str('non_lin %s not recognized', non_lin)
                    raise ValueError(error_message)
            self.aft = nn.Sequential(*self.aft)

        
    def forward(self, x, sim_feat, to_keep = None):

        G = self.get_affinity(sim_feat, to_keep = to_keep)
        temp = torch.mm(G,x)
        out = torch.mm(temp,self.weight)

        if hasattr(self,'aft') and self.aft is not None:
            out = self.aft(out)

        return out

    


    def get_affinity(self,input, to_keep = None, nosum = False):
        
        G = torch.autograd.Variable(torch.zeros(input.size(0),input.size(0))).cuda()
        # input = input.unsqueeze(2)
        # print 'input.size()',input.size()
        for b_num in range(input.size(0)):
            # print b_num,input.size()
            row = input[b_num].view(1,input.size(1))
            # print 'row.size()',row.size()
            row = row.expand(input.size(0)-b_num,input.size(1))
            # print 'row.size()',row.size()
            row = torch.cat([row,input[b_num:]],1)
            # print 'row.size()',row.size()
            edges_curr = self.edge_weighter(row).squeeze() # b,1
            # print 'edges_curr.size()',edges_curr.size()
            G[b_num,b_num:]=edges_curr
            
            if b_num<input.size(0)-1:
                # print 'hello'
                G[b_num+1:,b_num]=edges_curr[1:]


            # if b_num>0:
            #     print G[b_num-1:b_num+1,b_num-1:b_num+1]
            # else:
            #     print G[:2,:2]
            # raw_input()            
        #     G.append(edges_curr)

        # G = torch.cat(G,dim = 1)
        # .transpose()
        # print G.size()
        # raw_input()

        if not nosum:
            if self.scaling_method=='sum':
                sums = torch.sum(G,dim = 1, keepdim = True)
                sums[sums==0]=1
                G = G/sums
            elif self.scaling_method.lower()=='n':
                G = G/input.size(0)
            else:
                error_message = str('scaling_method %s not recognized', self.scaling_method)
                raise ValueError(error_message) 

       
        return G
      