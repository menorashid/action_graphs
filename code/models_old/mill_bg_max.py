from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F

class Just_Mill(nn.Module):
    def __init__(self, n_classes, deno):
        super(Just_Mill, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno


        self.features = []
        self.features.append(nn.Linear(2048,2048))
        self.features.append(nn.ReLU())
        self.features.append(nn.Dropout(0.5))
        self.features.append(nn.Linear(2048,n_classes+1))
        self.features.append(nn.Sigmoid())
        self.features = nn.Sequential(*self.features)
        # self.LogSoftmax = nn.LogSoftmax()

    def forward(self, input, ret_bg=False):
        x = self.features(input)
        
        bg = x[:,-1]
        x = x[:,:-1]

        bg = bg.view(bg.size(0),1).expand(-1,x.size(1))
        # x = x-bg
        # print torch.min(x), torch.max(x)
        # # gt_classes[5]=9
        # if gt_classes is not None:
        #     idx_gt = torch.nonzero(gt_classes)[0]
        #     bg = bg.view(bg.size(0),1).expand(-1,idx_gt.size(0))
        #     # print 'bg.size()',bg.size()
            
            
        #     gt_conf = x[:,idx_gt]
        #     # print 'gt_conf.size(), bg.size()',gt_conf.size(), bg.size()
        #     index_move = bg>gt_conf
        #     # print 'index_move.size()',index_move.size()

        #     for idx_idx_gt_curr, idx_gt_curr in enumerate(idx_gt):
        #         idx_check = torch.nonzero(index_move[:,idx_idx_gt_curr])
        #         # print 'idx_check',idx_check
        #         # print 'x[idx_check,idx_gt_curr]',x[idx_check,idx_gt_curr]
        #         x[:,idx_gt_curr].masked_scatter_(index_move[:,idx_idx_gt_curr],bg[index_move[:,idx_idx_gt_curr],idx_idx_gt_curr])
        #         # print 'bg[idx_check,idx_idx_gt_curr]',bg[idx_check,idx_idx_gt_curr]
        #         # print 'x[idx_check,idx_gt_curr]',x[idx_check,idx_gt_curr]
        #         # raw_input()
                
        #     # raw_input()
        # # x = F.softmax(x, dim = 1)
        pmf = self.make_pmf(x-bg)
        # raw_input()
        if not ret_bg:
            return x, pmf
        else:
            return x, pmf, bg

    def make_pmf(self,x):
        k = max(1,x.size(0)//self.deno)
        # print 'k',k

        pmf,_ = torch.sort(x, dim=0, descending=True)
        # print pmf.size()
        pmf = pmf[:k,:]
        # print pmf.size()
        pmf = torch.sum(pmf[:k,:], dim = 0)/k
        # print torch.min(pmf), torch.max(pmf)
        # print pmf.size()
        # pmf = pmf
        # print pmf.size()
        return pmf

    

class Network:
    def __init__(self, n_classes, deno, init = False):
        model = Just_Mill(n_classes, deno)

        if init:
            for idx_m,m in enumerate(model.features):
                if isinstance(m,nn.Linear):
                    # print m,1
                    nn.init.xavier_normal(m.weight.data)
                    nn.init.constant(m.bias.data,0.)
                
        self.model = model


    def get_lr_list(self, lr):
        lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]
        return lr_list

def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    net = Network(n_classes= 20, deno = 8)
    print net.model
    net.model = net.model.cuda()
    input = np.zeros((32,2048))
    input = torch.Tensor(input).cuda()
    input = Variable(input)
    output, pmf = net.model(input)

    print output.data.shape

if __name__=='__main__':
    main()

