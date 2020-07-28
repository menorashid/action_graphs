from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_sim_feat_input_cosine import Graph_Layer
from graph_layer_sim_feat_input_cosine import Graph_Layer_Wrapper


class Graph_Sim_Mill(nn.Module):
    def __init__(self, n_classes, deno):
        super(Graph_Sim_Mill, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno

        # num_layers = 2
        # in_out = [2048,512,1024]
        # print 'NUM LAYERS', num_layers, in_out

        num_layers = 1
        in_out = [2048,1024]
        print 'NUM LAYERS', num_layers, in_out


        self.linear_layer = nn.Linear(2048, 1024, bias = False)
        nn.init.orthogonal_(self.linear_layer.weight)

        lin_size = 1024
    
        self.graph_layers = nn.ModuleList()
        for num_layer in range(num_layers): 
            self.graph_layers.append(Graph_Layer_Wrapper(in_out[num_layer],in_out[num_layer+1]))
            

        self.last_layer = []
        self.last_layer.append(nn.Hardtanh())
        self.last_layer.append(nn.Dropout(0.5))
        self.last_layer.append(nn.Linear(2048,n_classes))
        self.last_layer = nn.Sequential(*self.last_layer)
        

        
    def forward(self, input, ret_bg =False):

        features_out = [self.linear_layer(input)]

        input_graph = input
        for idx_graph_layer,graph_layer in enumerate(self.graph_layers):
            input_graph = graph_layer(input_graph, features_out[0])

        features_out.append(input_graph)

        cat_out = torch.cat(features_out,dim = 1)
        x = self.last_layer(cat_out)
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
        feature_out = self.linear_layers[0](input)
        sim_mat = self.graph_layers[0].get_affinity(feature_out)
        return sim_mat
    

class Network:
    def __init__(self, n_classes, deno):
        self.model = Graph_Sim_Mill(n_classes, deno)
 
    def get_lr_list(self, lr):
        
        modules = [self.model.linear_layer, self.model.graph_layers, self.model.last_layer]
        lr_list = []
        for lr_curr, module in zip(lr,modules):
            lr_list+= [{'params': [p for p in module.parameters() if p.requires_grad], 'lr': lr_curr}]
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