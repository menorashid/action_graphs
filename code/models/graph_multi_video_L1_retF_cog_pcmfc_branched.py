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
from graph_multi_video_with_L1_retF import Graph_Multi_Video as GAS
from graph_multi_video_cog_per_class_merged_fc import Graph_Multi_Video as GCO

class Graph_Multi_Video_Cog(nn.Module):
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 in_out_gco = None,
                 feat_dim = None,
                 graph_size = None,
                 method = 'cos',
                 sparsify = 0.5,
                 non_lin = None,
                 aft_nonlin = 'RL',
                 sigmoid = False,
                 layer_bef = None,
                 graph_sum = False,
                 background = False,
                 just_graph = False,
                 feat_ret = False,
                 smax = False
                 ):
        super(Graph_Multi_Video_Cog, self).__init__()

        self.deno = deno
        self.gas = GAS(n_classes,
                 deno,
                 in_out = in_out,
                 feat_dim = feat_dim,
                 graph_size = graph_size,
                 method = method,
                 sparsify = sparsify[0],
                 non_lin = non_lin,
                 aft_nonlin = aft_nonlin,
                 sigmoid = sigmoid,
                 layer_bef = layer_bef,
                 graph_sum = graph_sum,
                 background = background,
                 just_graph = just_graph,
                 feat_ret = feat_ret)

        self.gco = GCO(n_classes,
                 deno,
                 in_out = in_out_gco,
                 sparsify = sparsify[1],
                 aft_nonlin = aft_nonlin,
                 sigmoid = sigmoid,
                 feat_ret = False)
        if smax is 'smax':
            self.smax_layer = nn.Softmax(dim = 1)
        elif smax is 'sig':
            self.smax_layer = nn.Sigmoid()
        else:
            self.smax_layer = None

        
    def forward(self, input, epoch_num = None, ret_bg =False, branch_to_test = -1):
        
        strip = False
        if len(input)==2 and type(input[0])!=tuple:
            strip = True
            input = [input]


        out_gas, pmf_gas = self.gas([input_curr[0] for input_curr in input])
        out_gco, pmf_gco = self.gco(input)

        assert len(out_gas)==len(out_gco)
        assert len(pmf_gas)==2
        assert len(pmf_gas[1])==3

        pmf_all_all = [[],[],[]]
        x_all_all = [[],[],[]]

        for idx_input in range(len(out_gas)):
            # print out_gco[idx_input].size(), out_gas[idx_input].size()
            if self.smax is 'smax':
                out_combo = self.smax_layer(out_gco[idx_input])*self.smax_layer(out_gas[idx_input])
            elif self.smax is 'add':
                out_combo = out_gco[idx_input]+out_gas[idx_input]
            # elif self.smax is 'mul':

            else:
                out_combo = out_gco[idx_input]*out_gas[idx_input]

            x_all_all[0].append(out_gas[idx_input])
            x_all_all[1].append(out_gco[idx_input])
            x_all_all[2].append(out_combo)

            for idx_out_curr,out_curr in enumerate(x_all_all):
                pmf_curr = self.make_pmf(out_curr[-1])
                pmf_all_all[idx_out_curr].append(pmf_curr.unsqueeze(0))
            # pmf_all.append(pmf_curr.unsqueeze(0))


        

        # hack
        # x_all_all = [out_gas,out_gas,out_gas]
        # pmf_all_all = [pmf_gas[0],pmf_gas[0],pmf_gas[0]]

        if strip:
            for idx_curr in range(len(pmf_all_all)):
                pmf_all_all[idx_curr] = pmf_all_all[idx_curr][0].squeeze()
                x_all_all[idx_curr] = x_all_all[idx_curr][0]
        


        if branch_to_test>-1:
            x_all_all = x_all_all[branch_to_test]
            pmf_all_all = pmf_all_all[branch_to_test]

        pmf_all_all = [pmf_all_all, pmf_gas[1]]



        return x_all_all, pmf_all_all
        # x_all_all, pmf_all_all


        # print len(out_gas), len(out_gco)
        

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
                 in_out_gco = None,
                 feat_dim = None,
                 graph_size = None,
                 method = 'cos',
                 sparsify = False,
                 non_lin = None,
                 aft_nonlin = 'RL',
                 sigmoid = False,
                 layer_bef = None,
                 graph_sum = False,
                 background = False,
                 just_graph = False,
                 feat_ret = False,
                 smax = False
                 ):

        print 'network graph_size', graph_size
        self.model = Graph_Multi_Video_Cog(n_classes, deno, in_out,in_out_gco,feat_dim, graph_size, method, sparsify, non_lin, aft_nonlin,sigmoid, layer_bef, graph_sum, background, just_graph, feat_ret, smax)
        print self.model
        print self.model.gas.sparsify
        print self.model.gco.sparsify
        # time.sleep(4)
        # raw_input()

    def get_lr_list(self, lr):
        
        
        lr_list = []
        modules = []
        modules+=[self.model.gas.linear_layer, self.model.gas.graph_layer, self.model.gas.last_graph]
        modules +=[self.model.gco.graph_layers, self.model.gco.last_graph]
        
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