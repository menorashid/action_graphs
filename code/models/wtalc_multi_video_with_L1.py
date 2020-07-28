from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_flexible import Graph_Layer
from graph_layer_flexible import Graph_Layer_Wrapper
from normalize import Normalize
import random
import numpy as np
import time

class Graph_Multi_Video(nn.Module):
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 aft_nonlin = 'RL',
                 feat_ret = False
                 ):
        super(Graph_Multi_Video, self).__init__()
        
        self.num_classes = n_classes
        self.feat_ret = feat_ret
        self.deno = deno
        
        if in_out is None:
            in_out = [2048,512]
        
        self.linear_layer = [nn.Linear(in_out[0], in_out[1], bias = True)]
        if aft_nonlin is not None:
            
            to_pend = aft_nonlin.split('_')
            for tp in to_pend:
                if tp.lower()=='ht':
                    self.linear_layer.append(nn.Hardtanh())
                elif tp.lower()=='rl':
                    self.linear_layer.append(nn.ReLU())
                elif tp.lower()=='l2':
                    self.linear_layer.append(Normalize())
                elif tp.lower()=='ln':
                    self.linear_layer.append(nn.LayerNorm(n_out))
                elif tp.lower()=='bn':
                    self.linear_layer.append(nn.BatchNorm1d(n_out, affine = False, track_running_stats = False))
                elif tp.lower()=='sig':
                    self.linear_layer.append(nn.Sigmoid())
                else:
                    error_message = str('non_lin %s not recognized', non_lin)
                    raise ValueError(error_message)
            self.linear_layer = nn.Sequential(*self.linear_layer)

        
        last_graph = []
        last_graph.append(nn.Dropout(0.5))
        last_graph.append(nn.Linear(in_out[-1],n_classes, bias = True))
        
        self.last_graph = nn.Sequential(*last_graph)
        
    def forward(self, input_chunks, epoch_num = None, ret_bg =False, branch_to_test = -1):
        
        strip = False
        if type(input_chunks)!=type([]):
            input_chunks = [input_chunks]
            strip = True

        identity = False
        method = None
        
        # print 'self.graph_size' , self.graph_size
        graph_size = 1
        is_cuda = next(self.parameters()).is_cuda
        
        pmf_all = []
        x_all = []
        out_graph_all = []
        input_sizes_all = np.array([input_curr.size(0) for input_curr in input_chunks])
        
        for input in input_chunks:
            
            if is_cuda:
                input = input.cuda()
            
            
            feature_out = self.linear_layer(input)
            x_curr = self.last_graph(feature_out)
            x_all.append(x_curr)
            out_graph_all.append(feature_out)
            pmf_all += [self.make_pmf(x_curr).unsqueeze(0)]
                
            
        if strip:
            pmf_all = pmf_all[0].squeeze()
            x_all = x_all[0]
            

        if hasattr(self,'feat_ret') and self.feat_ret:
            pmf_all = [pmf_all, [None,out_graph_all,input_sizes_all]]
        
        if ret_bg:
            return x_all, pmf_all, None
        else:
            return x_all, pmf_all
        

    def make_pmf(self,x):
        k = max(1,x.size(0)//self.deno)
        
        pmf,_ = torch.sort(x, dim=0, descending=True)
        pmf = pmf[:k,:]
        
        pmf = torch.sum(pmf[:k,:], dim = 0)/k
        return pmf

    
class Network:
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 aft_nonlin = 'RL',
                 feat_ret = False
                 ):

        # print 'network graph_size', graph_size
        self.model = Graph_Multi_Video(n_classes, deno, in_out,aft_nonlin,feat_ret)
        print self.model
        time.sleep(4)
        # raw_input()

    def get_lr_list(self, lr):
        
        
        lr_list = []

        modules = []
        # if self.model.layer_bef is not None:
        #     modules.append(self.model.layer_bef)

        modules+=[self.model.linear_layer, self.model.last_graph]

        for i,module in enumerate(modules):
            print i, lr[i]
            lr_list+= [{'params': [p for p in module.parameters() if p.requires_grad], 'lr': lr[i]}]


        return lr_list

def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    net = Network(n_classes= 20, deno = 8, graph_size = 2)
    print net.model
    net.model = net.model.cuda()
    input = np.zeros((16,2048))
    input = torch.Tensor(input).cuda()
    input = Variable(input)
    output,pmf = net.model(input)
    # print output.shape


    print output.data.shape

if __name__=='__main__':
    main()