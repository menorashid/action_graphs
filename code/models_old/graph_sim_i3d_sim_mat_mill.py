from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_i3d_sim_mat import Graph_Layer

class Graph_Sim_Mill(nn.Module):
    def __init__(self, n_classes, deno):
        super(Graph_Sim_Mill, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno


        # self.features = []
        # self.features.append(nn.Linear(2048,2048))
        # self.features.append(nn.ReLU())
        # self.features.append(nn.Dropout(0.5))
        
        # self.graph_layer = []
        print 'NO GRAPH'
        self.graph_layer = Graph_Layer(2048)
        
        # self.features.append(Graph_Layer(512,4,2048))
        # self.features.append(nn.ReLU())
        # self.features.append(nn.Dropout(0.5))

        # self.features.append(Graph_Layer(2048,32, n_out = n_classes))
        # self.features.append(nn.ReLU())
        # self.features.append(nn.Dropout(0.5))
        self.final_layer = []
        # self.final_layer.append(torch.nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False))
        self.final_layer.append(nn.Hardtanh())
        self.final_layer.append(nn.Dropout(0.5))
        self.final_layer.append(nn.Linear(2048,n_classes)) 
        # self.final_layer.append(nn.Sigmoid())
        # self.features.append(nn.Linear(2048,n_classes))
        # self.features = nn.Sequential(*self.features)
        # self.graph_layer = nn.Sequential(*self.graph_layer)
        self.final_layer = nn.Sequential(*self.final_layer)
        



        # self.graph_layer = 
        # self.LogSoftmax = nn.LogSoftmax()

    def forward(self, input, ret_bg = False):
        # print input.size()
        # x = self.features(input)
        # print x.size()
        x = self.graph_layer(input)
        x = self.final_layer(x)
        # print x.size()
        # x = self.graph_layer(x)

        # return x
        pmf = self.make_pmf(x)
        if ret_bg:
            return x, pmf, None
        else:
            return x, pmf

    def make_pmf(self,x):
        k = max(1,x.size(0)//self.deno)
        # print 'k',k

        pmf,_ = torch.sort(x, dim=0, descending=True)
        # print pmf.size()
        pmf = pmf[:k,:]
        # print pmf.size()
        pmf = torch.sum(pmf[:k,:], dim = 0)/k
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
        lr_list= [{'params': self.model.graph_layer.parameters(), 'lr': lr[0]},
        {'params': self.model.final_layer.parameters(), 'lr': lr[0]}]
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

