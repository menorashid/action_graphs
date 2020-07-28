from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_flexible import Graph_Layer
from graph_layer_flexible import Graph_Layer_Wrapper
from normalize import Normalize

import numpy as np

class Graph_Multi_Video(nn.Module):
    def __init__(self,
                 n_classes,
                 deno,
                 pretrained,
                 in_out = None,
                 graph_size = None,
                 method = 'cos',
                 num_switch = 1,
                 focus = 0,
                 sparsify = False,
                 non_lin = 'HT',
                 normalize = [True,True]
                 ):
        super(Graph_Multi_Video, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno
        self.graph_size = graph_size
        self.sparsify = sparsify

        if in_out is None:
            in_out = [2048,64,2048,64]

        num_layers = len(in_out)-3
        # non_lin = 'HT'

        print 'NUM LAYERS', num_layers, in_out

        
        self.linear_layer = nn.Linear(in_out[0], in_out[1], bias = False)
        # for param in self.linear_layer.parameters():
        #     param.requires_grad = False
        # non_lin = 'HT'

        if pretrained=='ucf':
            model_file = '../experiments/just_mill_flexible_deno_8_n_classes_20_layer_sizes_2048_64_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.001_0.001/model_99.pt'
        elif pretrained=='activitynet':
            model_file = '../experiments/just_mill_flexible_deno_8_n_classes_100_layer_sizes_2048_64_activitynet/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropy_50_step_50_0.1_0.001_0.001/model_49.pt'
        elif pretrained=='random':
            model_file = '../experiments/just_mill_flexible_deno_8_n_classes_20_layer_sizes_2048_64_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0_0.001/model_99.pt'
        elif pretrained=='default':
            model_file = None
        else:
            error_message = 'Similarity method %s not valid' % method
            raise ValueError(error_message)

        if model_file is not None:
            model_temp = torch.load(model_file)
            self.linear_layer.weight.data = model_temp.linear.weight.data
        else:
            print 'NO MODEL FILE AAAAAAAA'
        
        
        self.graph_layers = nn.ModuleList()
        for num_layer in range(num_layers): 
            self.graph_layers.append(Graph_Layer_Wrapper(in_out[num_layer+2],n_out = in_out[num_layer+3], non_lin = non_lin, method = method))
        
        last_linear = []
        if non_lin =='HT':
            last_linear.append(nn.Hardtanh())
        elif non_lin =='RL':
            last_linear.append(nn.ReLU())
        else:
            error_message = str('Non lin %s not valid', non_lin)
            raise ValueError(error_message)

        if normalize[0]:
            last_linear.append(Normalize())

        last_linear.append(nn.Dropout(0.5))
        last_linear.append(nn.Linear(in_out[1],n_classes))
        last_linear = nn.Sequential(*last_linear)
        self.last_linear = last_linear
        
        last_graph = []
        if non_lin =='HT':
            last_graph.append(nn.Hardtanh())
        elif non_lin =='RL':
            last_graph.append(nn.ReLU())
        else:
            error_message = str('Non lin %s not valid', non_lin)
            raise ValueError(error_message)

        if normalize[1]:
            last_graph.append(Normalize())
        
        # last_graph.append(nn.Hardtanh())
        # last_graph.append(Normalize())
        last_graph.append(nn.Dropout(0.5))
        last_graph.append(nn.Linear(in_out[-1],n_classes))
        last_graph = nn.Sequential(*last_graph)
        self.last_graph = last_graph

        if type(num_switch)==type(1):
            num_switch = [num_switch,num_switch]

        self.num_switch = num_switch
        self.epoch_counters = [0,0]
        self.focus = focus
        self.epoch_last = 0


    def get_to_keep(self,input_sizes):
        
        k_all = [max(1,size_curr)//self.deno for idx_size_curr,size_curr in enumerate(input_sizes)]
        k_all = int(np.mean(k_all))
        # print k_all
        # raw_input()
        # k_all = []
        # for idx_size_curr,size_curr in enumerate(input_sizes):
        #     deno_curr = max(1,size_curr)//self.deno
        #     k_all += [deno_curr]*size_curr 

        return k_all
        
        
        
    def forward_graph(self, input_chunks):

        pmf = []
        x_all = []

        is_cuda = next(self.parameters()).is_cuda

        for input in input_chunks:
            input_sizes = [input_curr.size(0) for input_curr in input]
            
            
            input = torch.cat(input,0)

            if self.sparsify:
                to_keep = self.get_to_keep(input_sizes)
            else:
                to_keep = None

            if is_cuda:
                input = input.cuda()
            
            features_out = self.linear_layer(input)
            
            input_graph = input
            for idx_graph_layer,graph_layer in enumerate(self.graph_layers):
                input_graph = graph_layer(input_graph, features_out,to_keep = to_keep)
            
            x = self.last_graph(input_graph)
            x_all.append(x)
            
            for idx_sample in range(len(input_sizes)):
                if idx_sample==0:
                    start = 0
                else:
                    start = sum(input_sizes[:idx_sample])
                end = start+input_sizes[idx_sample]
                x_curr = x[start:end,:]
                pmf += [self.make_pmf(x_curr).unsqueeze(0)]
        
        
        return x_all, pmf


    def forward_linear(self, input_chunks):
        
        pmf = []
        x_all = []
        
        is_cuda = next(self.parameters()).is_cuda

        for input in input_chunks:
            input_sizes = [input_curr.size(0) for input_curr in input]
            input = torch.cat(input,0)
            if is_cuda:
                input = input.cuda()

            features_out = self.linear_layer(input)

            x = self.last_linear(features_out)
            x_all.append(x)
            
            for idx_sample in range(len(input_sizes)):
                if idx_sample==0:
                    start = 0
                else:
                    start = sum(input_sizes[:idx_sample])

                end = start+input_sizes[idx_sample]
                x_curr = x[start:end,:]
                pmf += [self.make_pmf(x_curr).unsqueeze(0)]
        
        return x_all, pmf

    def forward(self, input, epoch_num = None, ret_bg =False, branch_to_test = -1):
        
        
        # if epoch_num>epoch_last:
        #     self.epoch_counters[focus]+=1

        # if self.epoch_last>0 and epoch_num>self.epoch_last:
        #     if epoch_num is not None and epoch_num%self.num_switch==0:
        #         self.focus = (self.focus+1)%2
        #         print 'FOCUS',epoch_num,self.focus, self.epoch_last
        
        if epoch_num is not None:
            if epoch_num>self.epoch_last:
                self.epoch_counters[self.focus]+=1

            if self.epoch_counters[self.focus]>0 and not self.epoch_counters[self.focus]%self.num_switch[self.focus]:
                self.epoch_counters[self.focus]=0
                self.focus = (self.focus+1)%2
                print 'FOCUS',epoch_num,self.focus, self.epoch_last,self.epoch_counters
                

            # if epoch_num is not None and epoch_num%self.num_switch==0:
            #     self.focus = (self.focus+1)%2
            #     print 'FOCUS',epoch_num,self.focus, self.epoch_last
                
                

        if branch_to_test>-1:
            self.focus = branch_to_test

        # print 'FOCUS',epoch_num,self.focus
        

        strip = False
        if type(input)!=type([]):
            input = [input]
            strip = True

        if self.graph_size is None:
            graph_size = len(input)
        else:
            graph_size = min(self.graph_size, len(input))

        input_chunks = [input[i:i + graph_size] for i in xrange(0, len(input), graph_size)]

        if not self.focus:
            x, pmf = self.forward_linear(input_chunks)
        else:
            x, pmf = self.forward_graph(input_chunks)

        if strip:
            assert len(pmf)==1
            pmf = pmf[0].squeeze()

        x = torch.cat(x,dim=0)
        
        if epoch_num is not None:
            self.epoch_last = epoch_num

        if ret_bg:
            return x, pmf, None
        else:
            return x, pmf 
        

    def make_pmf(self,x):
        k = max(1,x.size(0)//self.deno)
        
        pmf,_ = torch.sort(x, dim=0, descending=True)
        pmf = pmf[:k,:]
        
        pmf = torch.sum(pmf[:k,:], dim = 0)/k
        return pmf

    def get_similarity(self,input):
        is_cuda = next(self.parameters()).is_cuda
        
        input_sizes = [input_curr.size(0) for input_curr in input]
        input = torch.cat(input,0)

        if self.sparsify:
            to_keep = self.get_to_keep(input_sizes)
        else:
            to_keep = None
        print "TO KEEP",to_keep
        if is_cuda:
            input = input.cuda()
        
        feature_out = self.linear_layer(input)
        # feature_out = self.linear_layer(input)
        sim_mat = self.graph_layers[0].get_affinity(feature_out,to_keep = to_keep)
        return sim_mat
    
    def printGraphGrad(self):
        grad_rel = self.graph_layers[0].graph_layer.weight.grad
        print torch.min(grad_rel).data.cpu().numpy(), torch.max(grad_rel).data.cpu().numpy()


class Network:
    def __init__(self,
                 n_classes,
                 deno,
                 pretrained,
                 in_out = None,
                 graph_size = None,
                 method = 'cos',
                 num_switch = 1,
                 focus = 0,
                 sparsify = False,
                 non_lin = 'HT',
                 normalize = [True,True]
                 ):
        self.model = Graph_Multi_Video(n_classes, deno,pretrained, in_out, graph_size, method, num_switch, focus, sparsify, non_lin, normalize)
 
    def get_lr_list(self, lr):
        
        
        lr_list = []

        lr_list+= [{'params': [p for p in self.model.linear_layer.parameters() if p.requires_grad], 'lr': lr[0]}]
        lr_list+= [{'params': [p for p in self.model.graph_layers.parameters() if p.requires_grad], 'lr': lr[1]}]        
        lr_list+= [{'params': [p for p in self.model.last_linear.parameters() if p.requires_grad], 'lr': lr[2]}]
        lr_list+= [{'params': [p for p in self.model.last_graph.parameters() if p.requires_grad], 'lr': lr[2]}]

        return lr_list

def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    net = Network(n_classes= 20, deno = 8)
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