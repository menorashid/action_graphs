from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
# from graph_layer_flexible_temp import Graph_Layer
# from graph_layer_flexible_temp import Graph_Layer_Wrapper
# from normalize import Normalize

import numpy as np

class Wsddn(nn.Module):
    def __init__(self,
                n_classes,
                deno = None,
                in_out = None,
                ret_fc = 0):
        super(Wsddn, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno
        self.ret_fc = ret_fc

        if in_out is None:
            in_out = [2048,512]
        
        self.linear_layer = []
        self.linear_layer.append(nn.Linear(in_out[0], in_out[1], bias = True))
        self.linear_layer.append(nn.ReLU())
        self.linear_layer.append(nn.Dropout(0.5))
        self.linear_layer = nn.Sequential(*self.linear_layer)
        
        self.det_branch = []
        self.class_branch = []
        # branches = [self.det_branch, self.class_branch]
        # for branch in branches:
        #     branch.append(nn.ReLU())
        #     branch.append(nn.Dropout(0.5))
            # branch.append(nn.Linear(in_out[1],self.num_classes))            
        
        # [self.det_branch, self.class_branch] = branches
        
        self.det_branch.append(nn.Linear(in_out[1],1))            
        self.class_branch.append(nn.Linear(in_out[1],self.num_classes))            

        # #                     nn.Linear(in_out[1],16),
        # #                     nn.ReLU(),
        # #                     nn.Linear(16,self.num_classes),
        # #                     nn.Softmax(dim=0)]

        # self.det_branch.append(nn.Hardtanh())
        self.det_branch.append(nn.Softmax(dim=0))
        
        self.class_branch.append(nn.Softmax(dim=1))

        self.det_branch = nn.Sequential(*self.det_branch)
        self.class_branch = nn.Sequential(*self.class_branch)

        # self.det_branch = nn.Sequential(*[nn.Linear(in_out[1],self.num_classes), nn.Softmax(dim=0)])
        # self.class_branch = nn.Sequential(*[nn.Linear(in_out[1],self.num_classes), nn.Softmax(dim=1)])

    def forward(self, input):
        # print self
        # raw_input()
        # print input.size()
        is_cuda = next(self.parameters()).is_cuda
        # print input.size()
        input = input[:,:1024]
        # print input.size()
        # raw_input()
        x = self.linear_layer(input)
        
        x_class = self.class_branch(x) #n_instances x n_classes softmax along classes
        x_det = self.det_branch(x) #n_instances x n_classes softmax along instances
        # print x_det.size()
        # print x_class.size()

        x_det = x_det.repeat(1,x_class.size(1))
        # print x_det.size()
        # raw_input()
        x_pred = x_class*x_det
        
        pmf = self.make_pmf(x_pred)
        
        # effective_window = 1
        if hasattr(self,'ret_fc') and self.ret_fc>0:
            effective_window = self.ret_fc
            max_pred, idx_max = torch.max(x_pred, dim = 0)
            
            lower_lim = torch.clamp(idx_max - effective_window, 1)
            upper_lim = torch.clamp(idx_max + effective_window, max = x_pred.size(0))+1

            diffs_vec = torch.zeros((self.num_classes,))
            if is_cuda:
                diffs_vec = diffs_vec.cuda()
            
            for class_num in range(self.num_classes):
                
                star_vec = x[idx_max[class_num]].unsqueeze(0)

                r_vecs = x[lower_lim[class_num]:upper_lim[class_num],:]
                diffs = star_vec - r_vecs
                
                diffs = torch.bmm(diffs.unsqueeze(1),diffs.unsqueeze(2))
                diffs_vec[class_num] = torch.sum(diffs)
                

            pmf = [pmf, [max_pred, diffs_vec]]

        return x_det, pmf

    def make_pmf(self, x):
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


class Network:
    def __init__(self, n_classes, deno = None, in_out = None, init = False, ret_fc = 0):
        model = Wsddn(n_classes, deno,in_out, ret_fc)

        self.model = model


    def get_lr_list(self, lr):
        modules = [self.model.linear_layer, self.model.class_branch, self.model.det_branch, ]
        lr_list = []
        # [p for p in self.model.last_graphs.parameters() if p.requires_grad]
        for idx_module, module in enumerate(modules):
            lr_list += [{'params': [p for p in module.parameters() if p.requires_grad], 'lr': lr[idx_module]}]
        
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
    print output.shape, pmf.shape

if __name__=='__main__':
    main()