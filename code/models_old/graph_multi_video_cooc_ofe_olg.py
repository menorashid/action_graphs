from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_flexible_temp import Graph_Layer
from graph_layer_flexible_temp import Graph_Layer_Wrapper
from normalize import Normalize
from globals import class_names
import numpy as np
import os

class Graph_Multi_Video_Cooc(nn.Module):
    def __init__(self,
                 n_classes,
                 deno,
                 in_out = None,
                 feat_dim = '100_all',
                 graph_size = None,
                 method = 'cos',
                 sparsify = False,
                 non_lin = 'HT',
                 gk = 8,
                 aft_nonlin = None,
                 post_pend = ''
                 ):
        super(Graph_Multi_Video_Cooc, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno
        self.graph_size = graph_size
        self.sparsify = sparsify
        self.gk = gk

        if in_out is None:
            in_out = [2048,64]
        
        feat_dim = feat_dim.split('_')
        dir_feat = os.path.join('../data/ucf101/kmeans/3_'+feat_dim[0],'npy_labeled')
        # if len(feat_dim)>1:
        #     post_pend = feat_dim[-1]
        # else:
        post_pend = ''
        # feat_dim = feat_dim[1:]
        feat_dicts = []
        for feat_curr in feat_dim:
            if 'iou'==feat_curr:
                dir_feat = os.path.join(dir_feat,'int_union')
            if 'all'==feat_curr:
                file_curr = os.path.join(dir_feat,'all_classes_mul.npy')
                print file_curr
                feat_dicts.append(file_curr)

        #     elif feat_curr=='ind':
        # print feat_dim, class_names
        for class_name in class_names:
            file_curr =os.path.join(dir_feat,class_name+post_pend+'.npy')
            print file_curr
            feat_dicts.append(file_curr)
            # else:
            #     # print os.path.join(dir_feat,feat_curr+'.npy')
            #     feat_dicts.append(os.path.join(dir_feat,feat_curr+'.npy'))


        # if feat_dim is None:
        #     feat_dim = [2048,64]

        num_layers = len(feat_dicts)
        # assert num_layers == n_classes
        # len(in_out)-1
        
        print 'NUM LAYERS', num_layers, in_out
        
        # self.linear_layer = nn.Linear(feat_dim[0], feat_dim[1], bias = True)

        
        self.graph_layers = nn.ModuleList()
        for num_layer in range(num_layers): 

            # branch_curr = []
            self.graph_layers.append(Graph_Layer_Wrapper(in_out[0],n_out = in_out[1], non_lin = non_lin, method = method, aft_nonlin = aft_nonlin, affinity_dict = feat_dicts[num_layer]))
            
            # self.graph_layers.append(branch_curr)
        
        self.num_branches = num_layers
        # self.last_graphs = nn.ModuleList()
        # for num_layer in range(num_layers): 
        last_graph = []
        last_graph.append(nn.Dropout(0.5))
        last_graph.append(nn.Linear(in_out[-1]*self.num_branches,self.num_classes))
        self.last_graph = nn.Sequential(*last_graph)
            # self.last_graphs.append(last_graph)
        


    def get_to_keep(self,input_sizes):
        
        # gk = 1
        k_all = [max(1,size_curr//self.gk) for idx_size_curr,size_curr in enumerate(input_sizes)]
        # print 'gtk',self.gk
        k_all = int(np.mean(k_all))
        
        return k_all
    
    def forward(self, input, epoch_num = None, ret_bg =False, branch_to_test = -1):
        
        # print len(input)
        gt_vec = input[1]
        input = input[0]
        # print len(gt_vec)
        # print len(input)

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
        
        pmf_all = []
        # [[] for i in range(self.num_branches)]
        # x_all_all = []
        # [[] for i in range(self.num_branches)]
        x_all = []
        # pmf_all = []

        for idx_input, input in enumerate(input_chunks):
            gt_vec = gt_vec_chunks[idx_input]
            input_sizes = [input_curr.size(0) for input_curr in input]
            input = torch.cat(input,0)
            gt_vec = torch.cat(gt_vec,0)
            # print 'gt_vec',gt_vec.size(),torch.min(gt_vec),torch.max(gt_vec)
            if type(self.sparsify)==float:
                to_keep = self.sparsify
            elif self.sparsify==True:
                to_keep = (self.get_to_keep(input_sizes),input_sizes)
            else:
                to_keep = None

            if is_cuda and 'cuda' not in input.type():
                input = input.cuda()
                gt_vec = gt_vec.cuda()
            
            
            
            # feature_out = self.linear_layer(input)

            # input_graph = input
            temp = []

            for col_num in range(len(self.graph_layers)):
                out_graph = self.graph_layers[col_num](input, gt_vec, to_keep = to_keep)
                temp.append(out_graph)

            # print len(temp),temp[0].size()
            
            out_graph = torch.cat(temp,dim = 1)
            # print out_graph.size()
            x = self.last_graph(out_graph)
            
            # print x.size()

            # raw_input()
            # x = self.last_graph(input_graph)
            x_all.append(x)
            # for branch_num in range(len(x_all_all)):
            # x = x_all[-1]

            for idx_sample in range(len(input_sizes)):
                if idx_sample==0:
                    start = 0
                else:
                    start = sum(input_sizes[:idx_sample])

                end = start+input_sizes[idx_sample]
                x_curr = x[start:end,:]
                pmf_all += [self.make_pmf(x_curr).unsqueeze(0)]
                
            
        if strip:
            # for idx_pmf, pmf in enumerate(pmf_all):
            assert len(pmf_all)==1
            pmf_all = pmf_all[0].squeeze()

        # for idx_x, x in enumerate(x_all_all):
        #     x_all_all[idx_x] = torch.cat(x,dim=0)
        x_all = torch.cat(x_all,dim=0)

        # if self.num_branches>1 and branch_to_test>-1:
        #     x_all_all = x_all_all[branch_to_test]
        #     pmf_all = pmf_all[branch_to_test]

        # if self.num_branches==1:
        #     x_all_all = x_all_all[branch_to_test]
        #     pmf_all = pmf_all[branch_to_test]

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

    def get_similarity(self,input,idx_graph_layer = 0,sparsify = None, nosum = False):

        gt_vec = input[1]
        input = input[0]
        
        if sparsify is None:
            sparsify = self.sparsify

        is_cuda = next(self.parameters()).is_cuda

        input_sizes = [input_curr.size(0) for input_curr in input]
        input = torch.cat(input,0)
        gt_vec = torch.cat(gt_vec,0)

        if type(sparsify)==float:
            to_keep = sparsify
        elif sparsify==True:
            to_keep = (self.get_to_keep(input_sizes),input_sizes)
        else:
            to_keep = None

        if is_cuda and 'cuda' not in input.type():
            input = input.cuda()
            gt_vec = gt_vec.cuda()

        
        assert idx_graph_layer<len(self.graph_layers)

        # feature_out = self.linear_layer(input)
        sim_mat = self.graph_layers[idx_graph_layer].get_affinity(gt_vec, to_keep = to_keep, nosum = nosum)
            
        
        return sim_mat
    
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
                 gk = 8,
                 aft_nonlin = None,
                 post_pend = ''
                 ):
        self.model = Graph_Multi_Video_Cooc(n_classes, deno, in_out,feat_dim, graph_size, method, sparsify, non_lin,gk, aft_nonlin,post_pend)
        # print self.model
        # print self.model.num_branches
        # raw_input()

    def get_lr_list(self, lr):
        
        
        lr_list = []

        # lr_list+= [{'params': [p for p in self.model.linear_layer.parameters() if p.requires_grad], 'lr': lr[0]}]

        # lr_list+= [{'params': [p for p in self.model.linear_layers_after.parameters() if p.requires_grad], 'lr': lr[0]}]
        
        lr_list+= [{'params': [p for p in self.model.graph_layers.parameters() if p.requires_grad], 'lr': lr[0]}]        
        # lr_list+= [{'params': [p for p in self.model.last_linear.parameters() if p.requires_grad], 'lr': lr[2]}]
        lr_list+= [{'params': [p for p in self.model.last_graph.parameters() if p.requires_grad], 'lr': lr[0]}]

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