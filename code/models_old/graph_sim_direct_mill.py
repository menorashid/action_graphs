from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_sim_feat_input import Graph_Layer

class Graph_Sim_Mill(nn.Module):
    def __init__(self, n_classes, deno):
        super(Graph_Sim_Mill, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno

        self.features = nn.ModuleList()

        feature = [nn.Linear(2048,1024)]
        feature.append(nn.ReLU())
        feature = nn.Sequential(*feature)

        graph_layer = Graph_Layer(2048,1024)

        last_layer = []
        last_layer.append(nn.ReLU())
        last_layer.append(nn.Dropout(0.5))
        last_layer.append(nn.Linear(2048,n_classes))
        last_layer = nn.Sequential(*last_layer)
        
        self.features.append(feature)
        self.features.append(graph_layer)
        self.features.append(last_layer)

        
    def forward(self, input, ret_bg =False):
        feature_out = self.features[0](input)
        graph_out = self.features[1](input, feature_out)
        cat_out = torch.cat([feature_out, graph_out],dim = 1)
        x = self.features[2](cat_out)
        pmf = self.make_pmf(x)
        
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
        feature_out = self.features[0](input)
        sim_mat = self.features[1].get_affinity( feature_out)
        return sim_mat
    

class Network:
    def __init__(self, n_classes, deno):
        self.model = Graph_Sim_Mill(n_classes, deno)
 
    def get_lr_list(self, lr):
        lr_list= [{'params': [p for p in self.model.features.parameters() if p.requires_grad], 'lr': lr[0]}]
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

