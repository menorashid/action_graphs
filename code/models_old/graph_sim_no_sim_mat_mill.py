from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_no_sim_mat import Graph_Layer

class Graph_Sim_Mill(nn.Module):
    def __init__(self, n_classes, deno):
        super(Graph_Sim_Mill, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno


        self.features = []
        self.features.append(nn.Linear(2048,2048))
        self.features.append(nn.ReLU())
        self.features.append(nn.Dropout(0.5))
        
        self.features.append(Graph_Layer(2048))
        self.features.append(nn.ReLU())
        # self.features.append(torch.nn.LayerNorm(2048, eps=1e-05, elementwise_affine=False))
        self.features.append(nn.Dropout(0.5))

        # self.features.append(Graph_Layer(512,4,2048))
        # self.features.append(nn.ReLU())
        # self.features.append(nn.Dropout(0.5))

        # self.features.append(Graph_Layer(2048,32, n_out = n_classes))
        # self.features.append(nn.ReLU())
        # self.features.append(nn.Dropout(0.5))
        
        self.features.append(nn.Linear(2048,n_classes))
        # self.features.append(nn.Linear(2048,n_classes))
        self.features = nn.Sequential(*self.features)
        



        # self.graph_layer = 
        # self.LogSoftmax = nn.LogSoftmax()

    def forward(self, input):
        x = self.features(input)
        # print x.size()
        # x = self.graph_layer(x)

        # return x

        pmf = self.make_pmf(x)
        return x, pmf

    def make_pmf(self,x):
        k = max(1,x.size(0)//self.deno)
        # print 'k',k

        pmf,_ = torch.sort(x, dim=0, descending=True)
        # print pmf.size()
        pmf = pmf[:k,:]
        # print pmf.size()
        pmf = torch.sum(pmf[:k,:], dim = 0)
        # print pmf.size()
        # pmf = pmf
        # print pmf.size()
        return pmf

    

class Network:
    def __init__(self, n_classes, deno, init = False):
        model = Graph_Sim_Mill(n_classes, deno)

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
    input = np.zeros((16,2048))
    input = torch.Tensor(input).cuda()
    input = Variable(input)
    output,pmf = net.model(input)
    # print output.shape


    print output.data.shape

if __name__=='__main__':
    main()

