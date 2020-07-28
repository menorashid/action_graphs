from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import time
from normalize import Normalize
import numpy as np
from torch.autograd import Variable


class Thresher(nn.Module):
    def __init__(self):
        super(Thresher,self).__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,G):
        G = G*F.sigmoid(self.weight)
        G = G+F.tanh(self.bias)
        G = F.relu(G)
        return G


class Graph_Layer_Cooc(nn.Module):
    def __init__(self,in_size, n_out = None):
        super(Graph_Layer_Cooc, self).__init__()
        
        self.in_size = in_size
        self.n_out = self.in_size if n_out is None else n_out
        self.weight = nn.Parameter(torch.randn(in_size,self.n_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        print self.weight.size()
        
        
    def forward(self, x, sim_feat, to_keep = None, alpha = None, graph_sum = False, identity = False, method = None):

        G = self.get_affinity(sim_feat, to_keep = to_keep, alpha = alpha, graph_sum = graph_sum, identity = identity, method = method)
        temp = torch.mm(G,x)
        
        # sums = torch.sum(temp,dim = 1, keepdim = True)
        # print temp.size(), sums.size()
        # sums[sums==0]=1
        # temp = temp/sums
        # print temp.size(), sums.size()
        # raw_input()

        out = torch.mm(temp,self.weight)

        return out

    def get_affinity(self,G, to_keep = None, alpha = None, nosum = False,graph_sum = False, identity = False, method = None):
        
        
        if to_keep is not None:
            if type(to_keep) == type(()):
                to_keep,input_sizes = to_keep
                topk = []
                indices = []
                
                for vid_idx in range(len(input_sizes)):
                    end_idx = sum(input_sizes[:vid_idx+1])
                    start_idx = sum(input_sizes[:vid_idx])
                    size_vid = input_sizes[vid_idx]
                    to_keep_curr = min(size_vid, to_keep)
                    
                    try:
                        # if alpha is not None:
                        #     _, indices_curr = torch.topk(torch.abs(alphamat[:,start_idx:end_idx]), to_keep_curr, dim = 1) 
                        # else:
                        _, indices_curr = torch.topk(torch.abs(G[:,start_idx:end_idx]), to_keep_curr, dim = 1) 
                    except:
                        print G.size()
                        print input_sizes
                        print start_idx, end_idx, size_vid, to_keep_curr, to_keep
                        raw_input()

                    indices_curr = indices_curr+start_idx   
                    
                    indices.append(indices_curr)
                    
                indices = torch.cat(indices,dim = 1)
                topk = torch.gather(G, 1, indices)
                

                G = G*0
                G = G.scatter(1, indices, topk)
            elif type(to_keep)==float:
                # print 'sparsifying!'
                # print G[0]
                if to_keep>0:
                    G[torch.abs(G)<to_keep] = 0
                else:
                    G[G<to_keep] = 0
            elif to_keep == 'mid':
                #zero diagonal
                eye_inv = (torch.eye(G.size(0)).cuda()+1) % 2
                # get max
                max_val = torch.max(G*eye_inv)
                thresh = 0.5*max_val
                # print max_val
                # print thresh
                # print torch.min(G[G>0])
                # print 0.5*max_val
                G[G<thresh] = 0
                # print torch.min(G[G>0])
                # raw_input()

                

                # print G[0]
                # raw_input()

        if not nosum:
            # print 'div summing right now'
            sums = torch.sum(G,dim = 1, keepdim = True)
            sums[sums==0]=1
            G = G/sums
            # G = G/G.size(1)
            # raw_input()

        return G




class Graph_Layer(nn.Module):
    def __init__(self,in_size, n_out = None, method = 'cos', k = None,
        affinity_dict = None):
        super(Graph_Layer, self).__init__()
        
        self.in_size = in_size
        self.n_out = self.in_size if n_out is None else n_out
        self.weight = nn.Parameter(torch.randn(in_size,self.n_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        print self.weight.size()
        self.method = method
        self.affinity_dict = affinity_dict
        if self.affinity_dict is not None:
            self.affinity_dict = Variable(torch.Tensor(np.load(self.affinity_dict))).cuda()
            # print torch.min(self.affinity_dict), torch.max(self.affinity_dict)

        
        if 'learn_thresh' in self.method:
            self.thresher = Thresher()
            
    def forward(self, x, sim_feat, to_keep = None, alpha = None, graph_sum = False, identity = False, method = None, nosum = False):

        G = self.get_affinity(sim_feat, to_keep = to_keep, alpha = alpha, graph_sum = graph_sum, identity = identity, method = method, nosum = nosum)
        if graph_sum:
            [G, gsum] = G
        temp = torch.mm(G,x)

        out = torch.mm(temp,self.weight)

        if graph_sum:
            return [out, gsum]
        else:
            return out

    def get_graph_stats(self, sim_feat):
        sim_feat = F.normalize(sim_feat)
            
        G = torch.mm(sim_feat,torch.t(sim_feat))
        G_up = G[torch.triu(torch.ones(G.size(0),G.size(0)),diagonal=1) == 1]
        # print G.size(), G_up.size()
        min_val = torch.min(G_up)
        max_val = torch.max(G_up)
        median = torch.median(G_up)
        # print min_val, max_val, median
        return min_val, max_val, median
            

    def get_affinity(self,input, to_keep = None, alpha = None, nosum = False,graph_sum = False, identity = False, method = None):
        
        # print 'to keep',to_keep,type(to_keep)

        # print input.size()
        if method is None:
            method = self.method

        # print method
        # raw_input()

        if identity:
            G = Variable(torch.eye(input.size(0)).cuda())
            gsum = torch.sum(G)
        elif 'affinity_dict' in self.method:
            assert self.affinity_dict is not None
            if next(self.parameters()).is_cuda and 'cuda' not in self.affinity_dict.type():
                self.affinity_dict = self.affinity_dict.cuda()
        
            aff_size = self.affinity_dict.size(0)
            in_size = input.size(0)
            
            # expand aff row wise
            index = input.view(in_size,1).expand(in_size,aff_size).long()
            G = torch.gather(self.affinity_dict,0,index)
            
            # expand aff col wise
            index = input.view(1,in_size).expand(in_size,in_size).long()
            G = torch.gather(G,1,index)
        else:
            if 'cos' in method:
                input = F.normalize(input)
            
            G = torch.mm(input,torch.t(input))
            
            
            if graph_sum==2:
                assert not nosum
                G_dt = torch.abs(G.clone())
                sums = torch.sum(G_dt, dim = 1, keepdim = True)
                G_dt = G_dt/sums
                k = torch.sum(G_dt>0,dim = 1, keepdim = True).type(torch.FloatTensor).cuda()
                G_dt[G_dt==0]=1
                G_dt = torch.sum(torch.log(G_dt),dim = 1, keepdim = True)
                gsum = torch.cat([G_dt, k], dim =1)
            else:
                gsum = torch.sum(torch.abs(G.clone()))/(G.size(0)*G.size(1))
            
            if 'exp' in method:
                G = torch.exp(G)

            if 'thresh_self' in method:
                assert type(to_keep)==float
                eye_curr = torch.eye(G.size(0)).cuda()
                eye_inv = (eye_curr+1) % 2
                eye_curr = to_keep*eye_curr
                
                G = (G*eye_inv)+eye_curr
            
            elif 'learn_thresh' in method:
                assert to_keep is None
                G = self.thresher(G)
              
            if 'zero_self' in method:
                eye_inv = (torch.eye(G.size(0)).cuda()+1) % 2
                G = G*eye_inv
            

            



        if alpha is not None:
            assert to_keep is None
            alpha = alpha.view(1,alpha.size(0))
            G = G*alpha
            
        if to_keep is not None:
            if type(to_keep) == type(()):
                to_keep,input_sizes = to_keep
                topk = []
                indices = []
                
                for vid_idx in range(len(input_sizes)):
                    end_idx = sum(input_sizes[:vid_idx+1])
                    start_idx = sum(input_sizes[:vid_idx])
                    size_vid = input_sizes[vid_idx]
                    to_keep_curr = min(size_vid, to_keep)
                    
                    try:
                        _, indices_curr = torch.topk(torch.abs(G[:,start_idx:end_idx]), to_keep_curr, dim = 1) 
                    except:
                        print G.size()
                        print input_sizes
                        print start_idx, end_idx, size_vid, to_keep_curr, to_keep
                        raw_input()

                    indices_curr = indices_curr+start_idx   
                    
                    indices.append(indices_curr)
                    
                indices = torch.cat(indices,dim = 1)
                topk = torch.gather(G, 1, indices)
                

                G = G*0
                G = G.scatter(1, indices, topk)
            elif type(to_keep)==float:
                if to_keep>0:
                    # print 'doing G shit here'
                    G[torch.abs(G)<to_keep] = 0
                    # G = torch.nn.functional.hardshrink(G.cpu(), lambd=0.5).cuda()
                else:
                    G[G<to_keep] = 0
            elif type(to_keep)==str:
                to_keep = to_keep.split('_')
                to_keep = float(to_keep[1])
                G_vals = torch.abs(G[torch.eye(G.size(0))==0])
                min_val = torch.min(G_vals)
                max_val = torch.max(G_vals)
                thresh = min_val+ (max_val - min_val)*to_keep
                # print 'thresh',thresh
                G[torch.abs(G)<thresh] = 0
                # eye_inv = (torch.eye(G.size(0)).cuda()+1) % 2

                
        
        # # print 'nosum',nosum
        # if type(nosum)==float:
        #     G_smax = torch.nn.functional.softmax(G, dim = 1)
        #     sums = torch.sum(G,dim = 1, keepdim = True)
        #     sums[sums==0]=1
        #     G_rg = G/sums
        #     # print nosum, (1-nosum)

        #     G = G_rg*nosum + (1-nosum)*G_smax
        # elif nosum==3:
        #     # print nosum
        #     G = torch.nn.functional.softmax(G, dim = 1)
        #     G = torch.nn.functional.softmax(G, dim = 0)
        # elif nosum==4:
        #     # print nosum
        #     sums = torch.sum(G,dim = 1, keepdim = True)
        #     sums[sums==0]=1
        #     G = G/sums
        #     sums = torch.sum(G,dim = 0, keepdim = True)
        #     sums[sums==0]=1
        #     G = G/sums
        # elif nosum==5:
        #     # print nosum
        #     sums = torch.sum(G,dim = 1, keepdim = True)
        #     sums[sums==0]=1
        #     G = G/sums
        #     G = torch.nn.functional.softmax(G, dim = 0)
            
        # # if nosum==2:
        # #     # sums = torch.sum(G,dim = 1, keepdim = True)
        # #     G = torch.nn.functional.softmax(G, dim = 1)
        # #     # G[sums==0]=1

        # #     G_sum =torch.sum(G, dim = 1, keepdim = True) 
        # el
            

        
            
            
        if not nosum:
            # print 'here'
            sums = torch.sum(G,dim = 1, keepdim = True)
            sums[sums==0]=1
            G = G/sums
        

        if graph_sum:
            return [G, gsum]
        else:
            return G



class Graph_Layer_Wrapper(nn.Module):
    def __init__(self,in_size, n_out = None, non_lin = 'HT', method = 'cos', aft_nonlin = None,
        affinity_dict = None, type_layer = 'regular'):

        super(Graph_Layer_Wrapper, self).__init__()
        
        n_out = in_size if n_out is None else n_out

        if type_layer=='regular':
            self.graph_layer = Graph_Layer(in_size, n_out = n_out, method = method,affinity_dict = affinity_dict)
        elif type_layer=='cooc':
            self.graph_layer = Graph_Layer_Cooc(in_size, n_out = n_out)

        

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
                elif tp.lower()=='sig':
                    self.aft.append(nn.Sigmoid())
                else:
                    error_message = str('non_lin %s not recognized', non_lin)
                    raise ValueError(error_message)
            self.aft = nn.Sequential(*self.aft)


        # self.do = nn.Dropout(0.5)
        if non_lin is None:
            self.non_linearity = None
        elif non_lin=='HT':
            self.non_linearity = nn.Hardtanh()
        elif non_lin.lower()=='rl':
            self.non_linearity = nn.ReLU()
        else:
            error_message = str('non_lin %s not recognized', non_lin)
            raise ValueError(error_message)
    
    def get_graph_stats(self,sim_feat):
        if self.non_linearity is not None:
            sim_feat = self.non_linearity(sim_feat)
        curr_graph_stats = self.graph_layer.get_graph_stats(sim_feat)
        return curr_graph_stats

    def forward(self, x, sim_feat, to_keep = None, alpha = None, graph_sum = False, identity = False, method= None,nosum=False):
        if self.non_linearity is not None:
            sim_feat = self.non_linearity(sim_feat)
        # sim_feat = self.do(sim_feat)
        out = self.graph_layer(x, sim_feat, to_keep = to_keep, alpha = alpha, graph_sum = graph_sum, identity = identity, method = method,nosum=nosum)
        
        if hasattr(self,'aft') and self.aft is not None:
            if graph_sum:
                # print self.aft
                # print len(out),out[0].size(),out[1].size()
                out[0] = self.aft(out[0])
            else:
                out = self.aft(out)

        return out

    def get_affinity(self,input,to_keep = None, alpha = None, nosum = False):
        if self.non_linearity is not None:
            input = self.non_linearity(input)
        # input = self.non_linearity(input)
        # sim_feat = self.do(sim_feat)
        return self.graph_layer.get_affinity(input, to_keep = to_keep, alpha = alpha, nosum = nosum)        