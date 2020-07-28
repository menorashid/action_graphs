from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_flexible import Graph_Layer
from graph_layer_flexible import Graph_Layer_Wrapper
from normalize import Normalize
import random
import numpy as np

class Graph_Multi_Video(nn.Module):
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 feat_dim = None,
                 graph_size = None,
                 method = 'cos',
                 sparsify = 0.5,
                 non_lin = 'RL',
                 aft_nonlin = 'RL',
                 sigmoid = False,
                 layer_bef = None,
                 graph_sum = False
                 ):
        super(Graph_Multi_Video, self).__init__()
        
        self.num_classes = n_classes
        # self.background = background
        
        # if self.background:
        #     assert sigmoid
        #     n_classes+=1

        self.deno = deno
        self.graph_size = graph_size
        self.sparsify = sparsify
        self.graph_sum = graph_sum
        # self.just_graph = just_graph

        if in_out is None:
            in_out = [2048,512]
        
        if feat_dim is None:
            feat_dim = [2048,256]

        num_layers = 1
        
        print 'NUM LAYERS', num_layers, in_out
        
        self.bn =None
        # nn.BatchNorm1d(2048, affine = False)
        self.linear_layer = nn.Linear(feat_dim[0], feat_dim[1], bias = True)

        # self.graph_layer = nn.ModuleList()
        
        # self.last_graph = nn.ModuleList()
        
        
        self.graph_layer = Graph_Layer_Wrapper(in_out[0],n_out = in_out[1], non_lin = non_lin, method = method, aft_nonlin = aft_nonlin)
        

        self.det_branch = []
        self.class_branch = []
        branches = [self.det_branch, self.class_branch]
        for branch in branches:
            # branch.append(nn.ReLU())
            branch.append(nn.Dropout(0.5))
            branch.append(nn.Linear(in_out[1],self.num_classes))            
        
        [self.det_branch, self.class_branch] = branches
        self.det_branch.append(nn.Hardtanh())
        self.det_branch_smax = nn.Softmax(dim=0)

        self.class_branch.append(nn.Softmax(dim=1))

        self.det_branch = nn.Sequential(*self.det_branch)
        self.class_branch = nn.Sequential(*self.class_branch)


        last_graph = []
        last_graph.append(nn.Dropout(0.5))
        last_graph.append(nn.Linear(in_out[-1],n_classes))
        
        self.last_graph = nn.Sequential(*last_graph)
        
    def forward(self, input, epoch_num = None, ret_bg =False, branch_to_test = -1):
        
        strip = False
        if type(input)!=type([]):
            input = [input]
            strip = True

        identity = False
        method = None
        
        # print 'self.graph_size' , self.graph_size
        if self.graph_size is None:
            graph_size = len(input)
        elif self.graph_size=='rand':
            graph_size = random.randint(1,len(input))
        elif type(self.graph_size)==str and self.graph_size.startswith('randupto'):
            graph_size = int(self.graph_size.split('_')[1])
            graph_size = random.randint(1,min(graph_size, len(input)))
        else:
            graph_size = min(self.graph_size, len(input))

        
        input_chunks = [input[i:i + graph_size] for i in xrange(0, len(input), graph_size)]

        is_cuda = next(self.parameters()).is_cuda
        # print 'Graph branch'
        
        # pmf_all = [[] for i in range(self.num_branches)]
        # x_all_all = [[] for i in range(self.num_branches)]
        pmf_all = []
        x_all = []
        graph_sums = []

        for input in input_chunks:
            input_sizes = [input_curr.size(0) for input_curr in input]
            input = torch.cat(input,0)

            if is_cuda:
                input = input.cuda()
            
            feature_out = self.linear_layer(input)
            
            to_keep = self.sparsify
            out_graph = self.graph_layer(input, feature_out, to_keep = to_keep, graph_sum = self.graph_sum, identity = identity, method = method)
            
            
            if self.graph_sum:
                [out_graph, graph_sum] = out_graph       
                graph_sums.append(graph_sum.unsqueeze(0))


            x_class = self.class_branch(out_graph) #n_instances x n_classes softmax along classes
            x_det = self.det_branch(out_graph) #n_instances x n_classes softmax along instances

            if len(input_sizes)>1:
                x_det_arr = []
                for idx_sample in range(len(input_sizes)):
                    if idx_sample==0:
                        start = 0
                    else:
                        start = sum(input_sizes[:idx_sample])
                    end = start+input_sizes[idx_sample]
                    x_det_arr.append(self.det_branch_smax(x_det[start:end,:])) 
                    # print x_det_arr[-1].size()
                x_det = torch.cat(x_det_arr, dim = 0)
                # print x_det.size()
                # raw_input()
            else:
                x_det = self.det_branch_smax(x_det)

            x_pred = x_class*x_det

            # pmf = self.make_pmf(x_pred)
                
            # print 'out_graph.size()',out_graph.size()
        
            # out_col = self.last_graph(out_graph)
            # print 'out_col.size()',out_col.size()
            # x_preds.append(x_pred)
            x_all.append(x_det)
    
            # x = x_all[-1]
            
            for idx_sample in range(len(input_sizes)):
                if idx_sample==0:
                    start = 0
                else:
                    start = sum(input_sizes[:idx_sample])

                end = start+input_sizes[idx_sample]
                x_curr = x_pred[start:end,:]
                pmf_all += [self.make_pmf(x_curr).unsqueeze(0)]
        
        # print len(x_all)
            
        if strip:
            # for idx_pmf, pmf in enumerate(pmf_all):
            #     assert len(pmf)==1
            pmf_all = pmf_all[0].squeeze()
            # print torch.min(pmf_all), torch.max(pmf_all)
            

        # for idx_x, x in enumerate(x_all):
        x_all = torch.cat(x_all,dim=0)
        
        if self.graph_sum:
            pmf_all = [pmf_all, torch.cat(graph_sums,dim = 0)]

        if ret_bg:
            return x_all, pmf_all, None
        else:
            return x_all, pmf_all
        

    def make_pmf(self,x):
        if self.deno is None:
            k = x.size(0)
            pmf = x
        else:
            k = max(1,x.size(0)//self.deno)
            pmf,_ = torch.sort(x, dim=0, descending=True)
            pmf = pmf[:k,:]
        
        # print torch.min(x), torch.max(x)
        pmf = torch.sum(pmf, dim = 0)
        # print torch.min(pmf), torch.max(pmf)
        # /k
        # print pmf.size()
        # raw_input()
        return pmf

    def printGraphGrad(self):
        grad_rel = self.graph_layers[0].graph_layer.weight.grad
        print torch.min(grad_rel).data.cpu().numpy(), torch.max(grad_rel).data.cpu().numpy()


class Network:
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 feat_dim = None,
                 graph_size = None,
                 method = 'cos',
                 sparsify = False,
                 non_lin = 'RL',
                 aft_nonlin = 'RL',
                 sigmoid = False,
                 layer_bef = None,
                 graph_sum = False
                 ):

        # print 'network graph_size', graph_size
        self.model = Graph_Multi_Video(n_classes, deno, in_out,feat_dim, graph_size, method, sparsify, non_lin, aft_nonlin,sigmoid, layer_bef, graph_sum)
        print self.model
        raw_input()

    def get_lr_list(self, lr):
        
        
        lr_list = []

        modules = []
        # if self.model.layer_bef is not None:
        #     modules.append(self.model.layer_bef)

        modules+=[self.model.linear_layer, self.model.graph_layer, self.model.class_branch, self.model.det_branch]

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