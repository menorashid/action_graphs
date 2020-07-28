
import torch.nn as nn
import torch
import torch.nn.functional as F

class Mill_Centerloss(nn.Module):
    def __init__(self, n_classes, deno):
        super(Mill_Centerloss, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno

        self.features_first = []
        self.features_first.append(nn.Linear(2048,2048))
        self.features_first.append(nn.ReLU())
        self.features_first = nn.Sequential(*self.features_first)

        self.features = []
        self.features.append(nn.Dropout(0.5))
        self.features.append(nn.Linear(2048,n_classes+1))
        self.features.append(nn.Sigmoid())
        self.features = nn.Sequential(*self.features)
        self.centers = None
        # self.LogSoftmax = nn.LogSoftmax()

    def forward(self, input, labels):
        preds = []
        class_pred = []
        features = []
        labels_all = []
        is_cuda = next(self.parameters()).is_cuda
        for idx_sample, sample in enumerate(input):
            if is_cuda:
                sample = sample.cuda()
            pmf, class_pred_curr, features_curr = self.forward_single(sample)
            class_pred.append(class_pred_curr)
            features.append(features_curr)
            labels_all.append(labels[idx_sample,:].view(1,labels.size(1)).repeat(features_curr.size(0),1))
            preds.append(pmf.unsqueeze(0))
        
        preds = torch.cat(preds, 0)        
        features = torch.cat(features, 0)
        class_pred = torch.cat(class_pred, 0)
        labels_all = torch.cat(labels_all, 0)

        return preds, [labels_all, features, class_pred]


    def forward_single(self, input, ret_bg=False):
        
        features = self.features_first(input)
        
        x = self.features(features)
        
        pmf = self.make_pmf(x)
        class_pred = F.softmax(x, dim = 1)
        return pmf, class_pred, features
        
    def forward_single_test(self, input, ret_bg=False):
        features = self.features_first(input)
        x = self.features(features)
        pmf = self.make_pmf(x)

        bg = x[:,-1]
        x = x[:,:-1]
        # bg = bg.view(bg.size(0),1).expand(-1,x.size(1))
        # x = x-bg


        
        return x, pmf
    


    def make_pmf(self,x):

        bg = x[:,-1]
        x = x[:,:-1]
        bg = bg.view(bg.size(0),1).expand(-1,x.size(1))
        x = x-bg

        k = max(1,x.size(0)//self.deno)

        pmf,_ = torch.sort(x, dim=0, descending=True)
        pmf = pmf[:k,:]
        pmf = torch.sum(pmf[:k,:], dim = 0)/k
        return pmf

    

class Network:
    def __init__(self, n_classes, deno, init = False):
        model = Mill_Centerloss(n_classes, deno)

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

