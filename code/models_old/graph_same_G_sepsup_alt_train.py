from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_sim_feat_input_cosine import Graph_Layer
from graph_layer_sim_feat_input_cosine import Graph_Layer_Wrapper

# from graph_layer_sim_feat_input_euclidean import Graph_Layer
# from graph_layer_sim_feat_input_euclidean import Graph_Layer_Wrapper


class Graph_Sim_Mill(nn.Module):
    def __init__(self, n_classes, deno, num_switch = 1):
        super(Graph_Sim_Mill, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno

        # num_layers = 2
        # in_out = [2048,512,1024]
        # print 'NUM LAYERS', num_layers, in_out

        num_layers = 1
        in_out = [2048,1024]
        non_lin = 'HT'

        print 'NUM LAYERS', num_layers, in_out


        self.linear_layer = nn.Linear(2048, 1024, bias = False)
        # nn.init.orthogonal_(self.linear_layer.weight)

        lin_size = 1024
    
        self.graph_layers = nn.ModuleList()
        for num_layer in range(num_layers): 
            self.graph_layers.append(Graph_Layer_Wrapper(in_out[num_layer],in_out[num_layer+1], non_lin))

        if non_lin=='rl':
            self.non_lin = nn.ReLU()
        else:
            self.non_lin = nn.Hardtanh()        
        
        self.last_linear = []
        self.last_linear.append(nn.Dropout(0.5))
        self.last_linear.append(nn.Linear(1024,n_classes))
        self.last_linear = nn.Sequential(*self.last_linear)

        self.last_graph = []
        self.last_graph.append(nn.Dropout(0.5))
        self.last_graph.append(nn.Linear(1024,n_classes))
        self.last_graph = nn.Sequential(*self.last_graph)
        
        self.num_switch = num_switch
        self.focus = 0
        self.epoch_last = 0
        

        
    def forward(self, input, epoch_num=None, ret_bg =False, branch_to_test = -1):


        # if epoch_num is not None:
        #     print epoch_num, self.num_switch, epoch_num%self.num_switch, self.focus 

        if self.epoch_last>0 and epoch_num>self.epoch_last:
            if epoch_num is not None and epoch_num%self.num_switch==0:
                self.focus = (self.focus+1)%2

        if branch_to_test>-1:

            self.focus = branch_to_test

        print 'FOCUS',self.focus
        
        linear_out = self.linear_layer(input)
        if not self.focus:
            feature_curr = self.non_lin(linear_out)
            feature_curr = F.normalize(feature_curr)
            x = self.last_linear(feature_curr)
            pmf = self.make_pmf(x)
        else:
            input_graph = input
            for idx_graph_layer,graph_layer in enumerate(self.graph_layers):
                input_graph = graph_layer(input_graph, linear_out)
            feature_curr = self.non_lin(input_graph)
            feature_curr = F.normalize(feature_curr)
            x = self.last_graph(feature_curr)
            pmf = self.make_pmf(x)

        # features_out = [self.linear_layer(input)]

        # input_graph = input
        # for idx_graph_layer,graph_layer in enumerate(self.graph_layers):
        #     input_graph = graph_layer(input_graph, features_out[0])

        # features_out.append(input_graph)

        # pmfs = []
        # xs=[]
        # for idx, feature_curr in enumerate(features_out):
        #     # cat_out = torch.cat(features_out,dim = 1)
        #     feature_curr = self.non_lin(feature_curr)
        #     # if idx==0:
        #     feature_curr = F.normalize(feature_curr)

        #     x = self.last_layers[idx](feature_curr)
        #     pmf = self.make_pmf(x)
        #     pmfs.append(pmf)
        #     xs.append(x)
        
        # if branch_to_test>-1:
        #     x = xs[branch_to_test]
        #     pmfs = pmfs[branch_to_test]
        self.epoch_last = epoch_num

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
        feature_out = self.linear_layer(input)
        sim_mat = self.graph_layers[0].get_affinity(feature_out)
        return sim_mat
    
    def printGraphGrad(self):
        grad_rel = self.graph_layers[0].graph_layer.weight.grad
        if grad_rel is not None:
            print torch.min(grad_rel).data.cpu().numpy(), torch.max(grad_rel).data.cpu().numpy()


class Network:
    def __init__(self, n_classes, deno, num_switch):
        self.model = Graph_Sim_Mill(n_classes, deno, num_switch)
 
    def get_lr_list(self, lr):
        
        
        lr_list = []

        lr_list+= [{'params': [p for p in self.model.linear_layer.parameters() if p.requires_grad], 'lr': lr[0]}]
        lr_list+= [{'params': [p for p in self.model.graph_layers.parameters() if p.requires_grad], 'lr': lr[1]}]        
        lr_list+= [{'params': [p for p in self.model.last_linear.parameters() if p.requires_grad], 'lr': lr[2]}]
        lr_list+= [{'params': [p for p in self.model.last_graph.parameters() if p.requires_grad], 'lr': lr[3]}]


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