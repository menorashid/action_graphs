from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_flexible_temp import Graph_Layer
from graph_layer_flexible_temp import Graph_Layer_Wrapper
from normalize import Normalize

import numpy as np

class Graph_Multi_Video_PerfectG(nn.Module):
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 feat_dim = None,
                 graph_size = None,
                 method = 'cos',
                 sparsify = False,
                 non_lin = 'HT',
                 normalize = [True,True],
                 attention = False
                 ):
        super(Graph_Multi_Video_PerfectG, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno
        self.graph_size = graph_size
        self.sparsify = sparsify

        if in_out is None:
            in_out = [2048,64]
        if feat_dim is None:
            feat_dim = [2048,64]

        num_layers = len(in_out)-1
        
        print 'NUM LAYERS', num_layers, in_out

        self.linear_layers = nn.ModuleList()
        self.linear_layers_after = nn.ModuleList()
        for idx_layer_num,layer_num in enumerate(range(num_layers)):

            if non_lin =='HT':
                non_lin_curr = nn.Hardtanh()
            elif non_lin =='RL':
                non_lin_curr = nn.ReLU()
            else:
                error_message = str('Non lin %s not valid', non_lin)
                raise ValueError(error_message)

            idx_curr = idx_layer_num*2

            self.linear_layers.append(nn.Linear(feat_dim[idx_curr], feat_dim[idx_curr+1], bias = False))
            
            last_linear = []
            last_linear.append(non_lin_curr)
            if normalize[0]:
                last_linear.append(Normalize())
            last_linear.append(nn.Dropout(0.5))
            last_linear.append(nn.Linear(feat_dim[idx_curr+1],n_classes))
            last_linear = nn.Sequential(*last_linear)
            self.linear_layers_after.append(last_linear)


        
        self.graph_layers = nn.ModuleList()
        for num_layer in range(num_layers): 
            self.graph_layers.append(Graph_Layer_Wrapper(in_out[num_layer],n_out = in_out[num_layer+1], non_lin = non_lin, method = method))
        
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

        last_graph.append(nn.Dropout(0.5))
        last_graph.append(nn.Linear(in_out[-1],n_classes))
        last_graph = nn.Sequential(*last_graph)
        self.last_graph = last_graph

        self.num_branches = num_layers+1
        self.attention = attention
        print 'self.num_branches', self.num_branches
        


    def get_to_keep(self,input_sizes):
        
        k_all = [max(1,size_curr//self.deno) for idx_size_curr,size_curr in enumerate(input_sizes)]
        k_all = int(np.mean(k_all))
        
        return k_all
    
    def forward(self, input, epoch_num = None, ret_bg =False, branch_to_test = -1):
        
        
        gt_vec = input[1]
        input = input[0]
        assert len(input)==len(gt_vec)

        strip = False
        if type(input)!=type([]):
            input = [input]
            gt_vec = [gt_vec]
            strip = True

        if self.graph_size is None:
            graph_size = len(input)
        elif self.graph_size=='rand':
            import random
            graph_size = random.randint(1,len(input))
        else:
            graph_size = min(self.graph_size, len(input))

        input_chunks = [input[i:i + graph_size] for i in xrange(0, len(input), graph_size)]
        gt_vec_chunks = [gt_vec[i:i + graph_size] for i in xrange(0, len(input), graph_size)]

        is_cuda = next(self.parameters()).is_cuda
        # print 'Graph branch'
        
        pmf_all = [[] for i in range(self.num_branches)]
        x_all_all = [[] for i in range(self.num_branches)]
        
        for idx_input,input in enumerate(input_chunks):
            gt_vec = gt_vec_chunks[idx_input]
            input_sizes = [input_curr.size(0) for input_curr in input]
                    
            # print len(gt_vec)
            # for gt_vec_curr in gt_vec:
            #     print type(gt_vec_curr)
            #     print gt_vec_curr.size()
            gt_vec = torch.cat(gt_vec,0)
            # print gt_vec.size()
            gt_vec[gt_vec>0]=1
            # print torch.min(gt_vec), torch.max(gt_vec)

            # raw_input()

            input = torch.cat(input,0)

            if self.sparsify:
                to_keep = (self.get_to_keep(input_sizes),input_sizes)
            else:
                to_keep = None

            if is_cuda:
                input = input.cuda()
                gt_vec = gt_vec.cuda()
            
            assert len(self.graph_layers)==(self.num_branches-1)
            
            input_graph = input
            for col_num in range(len(self.graph_layers)):
                graph_layer = self.graph_layers[col_num]
                linear_layer = self.linear_layers[col_num]
                linear_layer_after = self.linear_layers_after[col_num]
            
                feature_out = self.linear_layers[col_num](input_graph)
                out_col = self.linear_layers_after[col_num](feature_out)

                if self.attention:
                    # alpha = F.softmax(out_col,dim = 1)
                    # alpha,_ = torch.max(alpha,dim =1)
                    # print alpha
                    # print alpha/torch.sum(alpha)
                    # raw_input()
                    alpha = gt_vec
                else:
                    alpha = None

                input_graph = self.graph_layers[col_num](input_graph, feature_out, to_keep = to_keep, alpha = alpha)
                
                x_all_all[col_num].append(out_col)
                

            x_all_all[len(self.graph_layers)].append(self.last_graph(input_graph))

            for branch_num in range(len(x_all_all)):
                
                x = x_all_all[branch_num][-1]
                
                for idx_sample in range(len(input_sizes)):
                    if idx_sample==0:
                        start = 0
                    else:
                        start = sum(input_sizes[:idx_sample])

                    end = start+input_sizes[idx_sample]
                    x_curr = x[start:end,:]
                    pmf_all[branch_num] += [self.make_pmf(x_curr).unsqueeze(0)]
                
            
        if strip:
            for idx_pmf, pmf in enumerate(pmf_all):
                assert len(pmf)==1
                pmf_all[idx_pmf] = pmf[0].squeeze()

        for idx_x, x in enumerate(x_all_all):
            x_all_all[idx_x] = torch.cat(x,dim=0)
        

        if branch_to_test>-1:
            x_all_all = x_all_all[branch_to_test]
            pmf_all = pmf_all[branch_to_test]

        if ret_bg:
            return x_all_all, pmf_all, None
        else:
            return x_all_all, pmf_all
        

    def make_pmf(self,x):
        k = max(1,x.size(0)//self.deno)
        
        pmf,_ = torch.sort(x, dim=0, descending=True)
        pmf = pmf[:k,:]
        
        pmf = torch.sum(pmf[:k,:], dim = 0)/k
        return pmf

    # def get_similarity(self,input,idx_graph_layer = 0):

    #     is_cuda = next(self.parameters()).is_cuda
    #     input_sizes = [input_curr.size(0) for input_curr in input]
    #     input = torch.cat(input,0)

    #     if self.sparsify:
    #         to_keep = (self.get_to_keep(input_sizes),input_sizes)
    #     else:
    #         to_keep = None

    #     print to_keep    
        
    #     if is_cuda:
    #         input = input.cuda()
        
    #     feature_out = self.linear_layers[idx_graph_layer][0](input)
    #     sim_mat = self.graph_layers[idx_graph_layer].get_affinity(feature_out,to_keep = to_keep)
        
    #     return sim_mat
    
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
                 non_lin = 'HT',
                 normalize = [True,True],
                 attention = False
                 ):
        self.model = Graph_Multi_Video_PerfectG(n_classes, deno, in_out,feat_dim, graph_size, method, sparsify, non_lin, normalize, attention)
        
    def get_lr_list(self, lr):
        
        
        lr_list = []

        lr_list+= [{'params': [p for p in self.model.linear_layers.parameters() if p.requires_grad], 'lr': lr[0]}]

        lr_list+= [{'params': [p for p in self.model.linear_layers_after.parameters() if p.requires_grad], 'lr': lr[0]}]
        
        lr_list+= [{'params': [p for p in self.model.graph_layers.parameters() if p.requires_grad], 'lr': lr[1]}]        
        # lr_list+= [{'params': [p for p in self.model.last_linear.parameters() if p.requires_grad], 'lr': lr[2]}]
        lr_list+= [{'params': [p for p in self.model.last_graph.parameters() if p.requires_grad], 'lr': lr[1]}]

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