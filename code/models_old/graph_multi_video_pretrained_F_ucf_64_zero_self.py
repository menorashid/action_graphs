from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from graph_layer_input_f_cos_row_div_max_self_con import Graph_Layer
from graph_layer_input_f_cos_row_div_max_self_con import Graph_Layer_Wrapper
from normalize import Normalize

class Graph_Multi_Video(nn.Module):
    def __init__(self, n_classes, deno, in_out = None,graph_size = None):
        super(Graph_Multi_Video, self).__init__()
        
        self.num_classes = n_classes
        self.deno = deno
        self.graph_size = graph_size

        if in_out is None:
            in_out = [2048,64]

        num_layers = len(in_out)-1
        non_lin = 'HT'

        print 'NUM LAYERS', num_layers, in_out

        
        self.linear_layer = nn.Linear(2048, 64, bias = False)
        # for param in self.linear_layer.parameters():
        #     param.requires_grad = False

        model_file = '../experiments/just_mill_ht_unit_norm_no_bias_fix_64_ucf/all_classes_False_just_primary_False_deno_8_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.001/model_99.pt'
        non_lin = 'HT'


        # model_file = '../experiments/just_mill_relu_unit_norm_no_bias_ucf/all_classes_False_just_primary_False_deno_8_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.0001_128/model_99.pt'

        model_temp = torch.load(model_file)
        self.linear_layer.weight.data = model_temp.linear.weight.data
        
        
        self.graph_layers = nn.ModuleList()
        for num_layer in range(num_layers): 
            self.graph_layers.append(Graph_Layer_Wrapper(in_out[num_layer],in_out[num_layer+1], non_lin))
        
        
        
        last_layer = []
        
        last_layer.append(nn.Hardtanh())
        last_layer.append(Normalize())
        last_layer.append(nn.Dropout(0.5))
        last_layer.append(nn.Linear(in_out[-1],n_classes))
        last_layer = nn.Sequential(*last_layer)
        self.last_layer = last_layer

        

        
    def forward(self, input, ret_bg =False, branch_to_test = -1):

        strip = False
        if type(input)!=type([]):
            input = [input]
            strip = True

        is_cuda = next(self.parameters()).is_cuda

        if self.graph_size is None:
            graph_size = len(input)
        else:
            graph_size = min(self.graph_size, len(input))

        pmf = []
        x_all = []
        
        input_chunks = [input[i:i + graph_size] for i in xrange(0, len(input), graph_size)]

        for input in input_chunks:
            input_sizes = [input_curr.size(0) for input_curr in input]
            input = torch.cat(input,0)
            if is_cuda:
                input = input.cuda()

            

            features_out = self.linear_layer(input)

            input_graph = input
            for idx_graph_layer,graph_layer in enumerate(self.graph_layers):
                input_graph = graph_layer(input_graph, features_out)

            # feature_curr = self.non_lin(input_graph)
            # # if idx==0:
            # feature_curr = F.normalize(feature_curr)

            x = self.last_layer(input_graph)
            x_all.append(x)
            
            for idx_sample in range(len(input_sizes)):
                if idx_sample==0:
                    start = 0
                else:
                    start = sum(input_sizes[:idx_sample])

                end = start+input_sizes[idx_sample]
                x_curr = x[start:end,:]
                pmf += [self.make_pmf(x_curr).unsqueeze(0)]
        
        x_all = torch.cat(x_all,dim=0)
        x =x_all
        
        if strip:
            assert len(pmf)==1
            pmf = pmf[0].squeeze()
            # x = x_all[0].squeeze()
        


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
        print torch.min(grad_rel).data.cpu().numpy(), torch.max(grad_rel).data.cpu().numpy()


class Network:
    def __init__(self, n_classes, deno, in_out = None,graph_size = None):
        self.model = Graph_Multi_Video(n_classes, deno, in_out, graph_size)
 
    def get_lr_list(self, lr):
        
        
        lr_list = []
        # for p in self.model.linear_layer.parameters() :
        #     print p.size()
        #     print p.requires_grad

        lr_list+= [{'params': [p for p in self.model.linear_layer.parameters() if p.requires_grad], 'lr': lr[0]}]
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