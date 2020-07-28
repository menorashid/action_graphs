from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
# from graph_layer_sim_feat_dot_softmax import Graph_Layer
from graph_layer_perfectG import Graph_Layer
# from graph_layer_sim_feat_input_cosine import Graph_Layer_Wrapper
# from normalize import Normalize

class Graph_PerfectG(nn.Module):
    def __init__(self, n_classes, deno, in_out = None):
        super(Graph_PerfectG, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno

        # num_layers = 2
        # in_out = [2048,512,1024]
        # print 'NUM LAYERS', num_layers, in_out

        if in_out is None:
            in_out = [2048,2048]
        # in_out = [2048,512,2048]
        num_layers = len(in_out)-1
        non_lin = 'rl'

        print 'NUM LAYERS', num_layers, in_out

        self.graph_layers = nn.ModuleList()
        for num_layer in range(num_layers): 
            self.graph_layers.append(Graph_Layer(in_out[num_layer],in_out[num_layer+1]))
        
        last_layer = []
        
        last_layer.append(nn.ReLU())
        # last_layer.append(Normalize())
        last_layer.append(nn.Dropout(0.5))
        last_layer.append(nn.Linear(in_out[-1],n_classes))
        last_layer = nn.Sequential(*last_layer)
        self.last_layer = last_layer

        

        
    def forward(self, input, ret_bg =False, branch_to_test = -1):

        # features_out = self.linear_layer(input)

        input_graph = input[0]
        gt_vec = input[1]
        # print input_graph.size()
        # print gt_vec.size()
        for idx_graph_layer,graph_layer in enumerate(self.graph_layers):
            input_graph = graph_layer(input_graph, gt_vec)

        # feature_curr = self.non_lin(input_graph)
        # # if idx==0:
        # feature_curr = F.normalize(feature_curr)

        x = self.last_layer(input_graph)
        # x = input_graph
        pmf = self.make_pmf(x)
        
        # x = features_out
        # pmf = self.make_pmf(x)

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
        # feature_out = self.linear_layer(input)
        sim_mat = self.graph_layers[0].get_affinity(input)
        return sim_mat
    
    def printGraphGrad(self):
        grad_rel = self.graph_layers[0].weight.grad
        print torch.min(grad_rel).data.cpu().numpy(), torch.max(grad_rel).data.cpu().numpy()


class Network:
    def __init__(self, n_classes, deno, in_out = None):
        self.model = Graph_PerfectG(n_classes, deno, in_out)
 
    def get_lr_list(self, lr):
        
        
        lr_list = []
        # for p in self.model.linear_layer.parameters() :
        #     print p.size()
        #     print p.requires_grad

        # lr_list+= [{'params': [p for p in self.model.linear_layer.parameters() if p.requires_grad], 'lr': lr[0]}]
        # print 'len lr list', len(lr_list),lr_list
        # raw_input()
        lr_list+= [{'params': [p for p in self.model.graph_layers.parameters() if p.requires_grad], 'lr': lr[1]}]        
        lr_list+= [{'params': [p for p in self.model.last_layer.parameters() if p.requires_grad], 'lr': lr[2]}]


        # for lr_curr, module in zip(lr,modules):
        #     print lr_curr
        #     lr_list+= [{'params': [p for p in module.parameters() if p.requires_grad], 'lr': lr_curr}]
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