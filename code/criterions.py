import os
import torch
import numpy as np
# from torchvision import transforms
from helpers import util, visualize
import torch.nn as nn
from wtalc_criterions import MyLoss_triple,MyLoss_triple_old,MyLoss_triple_noExclusive

class MultiCrossEntropy(nn.Module):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = None):
        super(MultiCrossEntropy, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim = 1)
        # print self.class_weights
        # raw_input()
        if class_weights is None:
            self.class_weights = None
        else: 
            self.class_weights = nn.Parameter(torch.Tensor(class_weights[np.newaxis,:]), requires_grad = False)

    def forward(self, gt, pred):
        pred = self.LogSoftmax(pred)

        if self.class_weights is not None:
            assert self.class_weights.size(1)==pred.size(1)
            loss = self.class_weights*-1*gt*pred
            loss = torch.sum(loss, dim = 1)
        else:
            loss = -1*gt* pred
            loss = torch.sum(loss, dim = 1)
            loss = torch.mean(loss)
        return loss

class MultiCrossEntropy_noSoftmax(nn.Module):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = None):
        super(MultiCrossEntropy_noSoftmax, self).__init__()
        # self.Log = nn.Log(dim = 1)
        if class_weights is None:
            self.class_weights = None
        else: 
            self.class_weights = nn.Parameter(torch.Tensor(class_weights[np.newaxis,:]), requires_grad = False)

    def forward(self, gt, pred):
        pred = torch.log(pred)

        if self.class_weights is not None:
            assert self.class_weights.size(1)==pred.size(1)
            loss = self.class_weights*-1*gt*pred
        else:
            loss = -1*gt* pred

        loss = torch.sum(loss, dim = 1)
        loss = torch.mean(loss)
        return loss


class Wsddn_Loss(nn.Module):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = None):
        super(Wsddn_Loss, self).__init__()
        
        if class_weights is None:
            self.class_weights = None
        else: 
            self.class_weights = nn.Parameter(torch.Tensor(class_weights[np.newaxis,:]), requires_grad = False)

    def forward(self, gt, pred):
        gt[gt>0]=1.
        gt[gt<=0]=-1.

        in_log_val = torch.clamp(gt*(pred - 0.5)+0.5,1e-10,1)
        loss = -1*torch.log(in_log_val)
        
        if self.class_weights is not None:
            assert self.class_weights.size(1)==pred.size(1)
            loss = self.class_weights*loss

        loss = torch.sum(loss, dim = 1)
        loss = torch.mean(loss)
        if loss.eq(float('-inf')).any() or loss.eq(float('inf')).any():
            print torch.min(pred), torch.max(pred)
            print torch.min(in_log_val), torch.max(in_log_val)
            print torch.log(torch.min(in_log_val)),torch.log(torch.max(in_log_val))

            raw_input()


        return loss


class Wsddn_Loss_WithL1(Wsddn_Loss):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = None, window_size = 3):
        num_branches = max(num_branches,1)
        super(Wsddn_Loss_WithL1, self).__init__(class_weights=class_weights, loss_weights = loss_weights, num_branches = num_branches)
        self.loss_weights = loss_weights
        # self.att_weight = loss_weights[-1]
        
    def forward(self, gt, preds, att):
        # print gt.size()
        # print preds.size()
        # print att.size()
        # raw_input()
        loss_regular = super(Wsddn_Loss_WithL1,self).forward(gt, preds)
        # max_preds = preds[

        max_preds = torch.cat([max_pred_curr.unsqueeze(0) for max_pred_curr,_ in att],0)
        dots = torch.cat([dot_curr.unsqueeze(0) for _,dot_curr in att],0)

        max_preds = 0.5*(max_preds**2)

        loss_spatial = torch.sum(max_preds*dots)
        # print loss_spatial
        loss_spatial = loss_spatial/(max_preds.size(0)*max_preds.size(1))
        # print loss_spatial

        # print max_preds.size()
        
        # print att[0][0]
        # print max_preds[0]

        
        # print dots.size()
        # print att[0][1]
        # print dots[0]


        # for fc, max_idx in att:
        #     print 'in loss'
        #     print fc.size()
        #     print max_idx.size()
        #     print max_idx
        

        


        # l1 = torch.mean(torch.abs(att))
        # l1 = self.att_weight*l1
        # loss_all = l1+loss_regular
        # print loss_spatial
        # print loss_regular
        
        loss_all = self.loss_weights[0]*loss_regular + self.loss_weights[1]*loss_spatial
        
        return loss_all

class MultiCrossEntropyMultiBranch(nn.Module):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2):
        super(MultiCrossEntropyMultiBranch, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim = 1)
        self.num_branches = num_branches
        
        # print class_weights

        # if class_weights is not None and len(class_weights)==100:
        #     class_weights = class_weights*5
        #     print 'scaling class_weights'
        #     # class_weights = None
        # print class_weights        
        # raw_input()
        if class_weights is None:
            self.class_weights = None
        else: 
            self.class_weights = nn.Parameter(torch.Tensor(class_weights[np.newaxis,:]), requires_grad = False)

        if loss_weights is None:
            self.loss_weights = [1 for i in range(self.num_branches)]
        else:
            self.loss_weights = loss_weights

    def forward(self, gt, preds, collate = True):

        # print gt.size()
        # print gt[:10]
        # print preds[0].size()

        # raw_input()
        # gt[gt>0]=1
        if collate:
            loss_all = 0
        else:
            loss_all = []

        assert len(preds) == self.num_branches
        for idx_pred, pred in enumerate(preds):
            pred = self.LogSoftmax(pred)
            # torch.log(pred)
            # 
            # print pred.size()
            if self.class_weights is not None:
                assert self.class_weights.size(1)==pred.size(1)
                loss = self.class_weights*-1*gt*pred
                loss = torch.sum(loss, dim = 1)
                loss = torch.mean(loss)
                # sum(loss)
                    # , dim = 1)
            else:
                loss = -1*gt* pred
                loss = torch.sum(loss, dim = 1)
            # print 'meaning'
                loss = torch.mean(loss)
            # print 'meaning'


            if collate:
                loss_all += loss*self.loss_weights[idx_pred]
            else:
                # print 'loss',loss
                loss_all.append(loss)

        return loss_all


class BinaryCrossEntropyMultiBranch(nn.Module):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2):
        super(BinaryCrossEntropyMultiBranch, self).__init__()

        if class_weights is not None:
            [pos_weight, class_weights] = class_weights
            n_classes = class_weights.size
            class_weights = None
            self.BCE = nn.BCEWithLogitsLoss(pos_weight = pos_weight*torch.ones([n_classes]))
        else:
            # pos_weight = 5
            # n_classes = 100
            print 'hello margin'
            self.BCE = nn.MultiLabelSoftMarginLoss()
            # nn.BCEWithLogitsLoss(reduction = 'none')
            # pos_weight = pos_weight*torch.ones([100]))
        # print class_weights.size
        # print pos_weight, n_classes
        

        if class_weights is None:
            # self.BCE = nn.BCEWithLogitsLoss()
            self.class_weights = None
        else: 
            self.class_weights = torch.Tensor(class_weights[np.newaxis,:])

        self.num_branches = num_branches

        if loss_weights is None:
            self.loss_weights = [1 for i in range(self.num_branches)]
        else:
            self.loss_weights = loss_weights

    def forward(self, gt, preds, collate = True):

        # print gt.size()
        # print gt[:10]
        # # print preds[0].size()
        gt[gt>0]=1
        # print self.class_weights
        if self.class_weights is not None:
            weights = self.class_weights
            weights = weights.repeat(gt.size(0),1).cuda()
            # print weights.size()
            # print weights[:10]

            self.BCE.weight = weights
            # raw_input()

        # print self.BCE.weight.size()

        # print gt[:10]
        # raw_input()

        if collate:
            loss_all = 0
        else:
            loss_all = []

        assert len(preds) == self.num_branches
        for idx_pred, pred in enumerate(preds):
            
            loss = self.BCE(pred, gt)
            # print loss.size()
            
            # loss = torch.sum(loss, dim = 1)
            # /gt.size(1)
            # print loss.size()
            
            # loss = torch.mean(loss)
            # print loss.size()
            # print gt.size()
            # print pred.size()
            # raw_input()

            if collate:
                loss_all += loss*self.loss_weights[idx_pred]
            else:
                # print 'loss',loss
                loss_all.append(loss)

        return loss_all

class MultiCrossEntropyMultiBranchWithL1(MultiCrossEntropyMultiBranch):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2, att_weight = 0.5, num_similar = 0):
        num_branches = max(num_branches,1)
        self.att_weight = loss_weights[-1]

        super(MultiCrossEntropyMultiBranchWithL1, self).__init__(class_weights=class_weights, loss_weights = loss_weights[:-1], num_branches = num_branches)
        # self.loss_weights_all = loss_weights
        
    def forward(self, gt, preds, att, collate = True):
        if self.num_branches ==1:
            preds = [preds]

        loss_regular = super(MultiCrossEntropyMultiBranchWithL1,self).forward(gt, preds, collate = collate)
        # print 'min_val',torch.min(torch.abs(att))
        

        l1 = torch.mean(torch.abs(att))
        # print l1
        # print 'att',att
        # print 'l1',l1
        if collate:
            l1 = self.att_weight*l1
            loss_all = l1+loss_regular
        else:
            loss_regular.append(l1)
            loss_all = loss_regular

        return loss_all

class MultiCrossEntropyMultiBranchWithDT(MultiCrossEntropyMultiBranch):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2, att_weight = 0.5, num_similar = 0, alpha = 0.5):
        num_branches = max(num_branches,1)
        self.att_weight = loss_weights[-1]

        super(MultiCrossEntropyMultiBranchWithDT, self).__init__(class_weights=class_weights, loss_weights = loss_weights[:-1], num_branches = num_branches)
        self.alpha = alpha
        # self.loss_weights_all = loss_weights
        
    def forward(self, gt, preds, att, collate = True):
        if self.num_branches ==1:
            preds = [preds]

        loss_regular = super(MultiCrossEntropyMultiBranchWithDT,self).forward(gt, preds, collate = collate)
        # print 'min_val',torch.min(torch.abs(att))
        
        # print att.size()
        # print att[:10]
        k = att[:,1]
        att = att[:,0]

        alpha_curr = self.alpha/k
        # print alpha_curr
        lbeta = k*torch.mvlgamma(alpha_curr,1)-torch.mvlgamma(k*alpha_curr,1)
        # print lbeta[:10]
        # raw_input()
        l1 = torch.mean((1-alpha_curr)*att+lbeta)
        
        if collate:
            l1 = self.att_weight*l1
            loss_all = l1+loss_regular
        else:
            loss_regular.append(l1)
            loss_all = loss_regular

        return loss_all

class MultiCrossEntropyMultiBranchWithDTL1_CASL(MultiCrossEntropyMultiBranchWithDT):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2, att_weight = 0.5, num_similar = 0, alpha = 0.5):
        # self.att_weight =None
        
        super(MultiCrossEntropyMultiBranchWithDTL1_CASL,self).__init__(class_weights=class_weights, loss_weights = loss_weights[:-1], num_branches = num_branches, alpha = alpha)
        
        self.num_similar = num_similar
        self.casl_weight = loss_weights[-1]
        self.loss_weights_all = loss_weights
        self.loss_strs = ['CrossEnt'+str(idx) for idx in range(len(self.loss_weights_all)-2)]+['L1','CASL']

        
    def forward(self, gt, preds, att, out, collate = True):
        graph_sum = att[0]
        feature = att[1]
        input_sizes = att[2]
        
        loss_everything_else = super(MultiCrossEntropyMultiBranchWithDTL1_CASL,self).forward(gt, preds, graph_sum, collate = collate)

        if self.num_branches>1:
            out = out[0]
        
        if self.casl_weight==0:
            casl_loss = 0.
        else:
            casl_loss = MyLoss_triple_noExclusive(feature, out, input_sizes, gt, type_loss = 'casl', debug = False, num_similar = self.num_similar)
        
        if collate:
            loss = loss_everything_else + self.casl_weight*casl_loss    
        else:
            loss_everything_else.append(casl_loss)
            loss = 0
            for idx_loss, loss_curr in enumerate(loss_everything_else):
                loss += self.loss_weights_all[idx_loss]*loss_curr
            loss_everything_else = [self.loss_weights_all[idx_loss]*loss_curr for idx_loss, loss_curr in enumerate(loss_everything_else)] 
            loss = [loss,loss_everything_else]
        
        return loss


class BinaryCrossEntropyMultiBranchWithL1(BinaryCrossEntropyMultiBranch):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2, att_weight = 0.5, num_similar = 0):
        num_branches = max(num_branches,1)
        self.att_weight = loss_weights[-1]

        super(BinaryCrossEntropyMultiBranchWithL1, self).__init__(class_weights=class_weights, loss_weights = loss_weights[:-1], num_branches = num_branches)
        # self.loss_weights_all = loss_weights
        
    def forward(self, gt, preds, att, collate = True):
        if self.num_branches ==1:
            preds = [preds]

        loss_regular = super(BinaryCrossEntropyMultiBranchWithL1,self).forward(gt, preds, collate = collate)
        # print 'min_val',torch.min(torch.abs(att))
        

        l1 = torch.mean(torch.abs(att))
        # print l1
        # print 'att',att
        # print 'l1',l1
        if collate:
            l1 = self.att_weight*l1
            loss_all = l1+loss_regular
        else:
            loss_regular.append(l1)
            loss_all = loss_regular

        return loss_all

class MultiCrossEntropyMultiBranchWithL1_withplot(MultiCrossEntropyMultiBranch):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2, att_weight = 0.5, num_similar = 0):
        num_branches = max(num_branches,1)
        self.att_weight = loss_weights[-1]
        self.loss_weights_all = loss_weights
        # print self.loss_weights_all
        # raw_input()
        super(MultiCrossEntropyMultiBranchWithL1_withplot, self).__init__(class_weights=class_weights, loss_weights = loss_weights[:-1], num_branches = num_branches)
        
        self.loss_strs = ['CrossEnt'+str(idx) for idx in range(len(self.loss_weights_all)-1)]+['L1']
        # self.loss_weights_all = loss_weights
        
    def forward(self, gt, preds, att, collate = True):
        if self.num_branches ==1:
            preds = [preds]

        loss_regular = super(MultiCrossEntropyMultiBranchWithL1_withplot,self).forward(gt, preds, collate = collate)
        # print 'min_val',torch.min(torch.abs(att))
        

        l1 = torch.mean(torch.abs(att))

        
        # print l1
        # print 'att',att
        # print 'l1',l1
        if collate:
            l1 = self.att_weight*l1
            loss_all = l1+loss_regular
        else:
            loss_regular.append( l1)
            loss = 0
            for idx_loss, loss_curr in enumerate(loss_regular):
                # print 'here',idx_loss, self.loss_weights_all[idx_loss],loss_curr
                # print loss_curr, self.loss_weights_all[idx_loss]*loss_curr
                loss += self.loss_weights_all[idx_loss]*loss_curr
            loss_all = [loss,loss_regular]
            # print loss_all,loss_regular
            # loss_all = loss_regular

        return loss_all


class MultiCrossEntropyMultiBranchWithL1_CASL(MultiCrossEntropyMultiBranchWithL1):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2, att_weight = 0.5, num_similar = 0):
        # self.att_weight =None
        
        super(MultiCrossEntropyMultiBranchWithL1_CASL,self).__init__(class_weights=class_weights, loss_weights = loss_weights[:-1], num_branches = num_branches)
        # print 'att_super',super(MultiCrossEntropyMultiBranchWithL1_CASL,self).att_weight
        self.num_similar = num_similar
        self.casl_weight = loss_weights[-1]
        self.loss_weights_all = loss_weights
        self.loss_strs = ['CrossEnt'+str(idx) for idx in range(len(self.loss_weights_all)-2)]+['L1','CASL']

        
    def forward(self, gt, preds, att, out, collate = True):
        graph_sum = att[0]
        feature = att[1]
        input_sizes = att[2]
        # print graph_sum
        # print input_sizes

        # for att_curr in att[1:]:
            # print len(att_curr), att_curr.size()
        # print len(att_curr), att_curr[1].size()
        # raw_input()

        loss_everything_else = super(MultiCrossEntropyMultiBranchWithL1_CASL,self).forward(gt, preds, graph_sum, collate = collate)

        if self.num_branches>1:
            # print len(out)
            # print len(out[0])
            # .size()
            out = out[0]
            # raw_input()

        if self.casl_weight==0:
            casl_loss = 0.
        else:
            casl_loss = MyLoss_triple_noExclusive(feature, out, input_sizes, gt, type_loss = 'casl', debug = False, num_similar = self.num_similar)
        # print 'casl_loss',casl_loss
        
        if collate:
            loss = loss_everything_else + self.casl_weight*casl_loss    
        else:
            loss_everything_else.append(casl_loss)
            loss = 0
            for idx_loss, loss_curr in enumerate(loss_everything_else):
                # print 'here',idx_loss, self.loss_weights_all[idx_loss],loss_curr
                # print loss_curr, self.loss_weights_all[idx_loss]*loss_curr
                loss += self.loss_weights_all[idx_loss]*loss_curr
            loss = [loss,loss_everything_else]
        # print loss_everything_else
        return loss

class BinaryCrossEntropyMultiBranchWithL1_CASL(BinaryCrossEntropyMultiBranchWithL1):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2, att_weight = 0.5, num_similar = 0):
        # self.att_weight =None
        
        super(BinaryCrossEntropyMultiBranchWithL1_CASL,self).__init__(class_weights=class_weights, loss_weights = loss_weights[:-1], num_branches = num_branches)
        # print 'att_super',super(BinaryCrossEntropyMultiBranchWithL1_CASL,self).att_weight
        self.num_similar = num_similar
        self.casl_weight = loss_weights[-1]
        self.loss_weights_all = loss_weights
        self.loss_strs = ['CrossEnt'+str(idx) for idx in range(len(self.loss_weights_all)-2)]+['L1','CASL']

        
    def forward(self, gt, preds, att, out, collate = True):
        graph_sum = att[0]
        feature = att[1]
        input_sizes = att[2]
        # print graph_sum
        # print input_sizes

        # for att_curr in att[1:]:
            # print len(att_curr), att_curr.size()
        # print len(att_curr), att_curr[1].size()
        # raw_input()

        loss_everything_else = super(BinaryCrossEntropyMultiBranchWithL1_CASL,self).forward(gt, preds, graph_sum, collate = collate)

        if self.num_branches>1:
            # print len(out)
            # print len(out[0])
            # .size()
            out = out[0]
            # raw_input()

        if self.casl_weight==0:
            casl_loss = 0.
        else:
            casl_loss = MyLoss_triple_noExclusive(feature, out, input_sizes, gt, type_loss = 'casl', debug = False, num_similar = self.num_similar)
        # print 'casl_loss',casl_loss
        
        if collate:
            loss = loss_everything_else + self.casl_weight*casl_loss    
        else:
            loss_everything_else.append(casl_loss)
            loss = 0
            for idx_loss, loss_curr in enumerate(loss_everything_else):
                # print 'here',idx_loss, self.loss_weights_all[idx_loss],loss_curr
                # print loss_curr, self.loss_weights_all[idx_loss]*loss_curr
                # print loss_curr, self.loss_weights_all[idx_loss]*loss_curr
                loss += self.loss_weights_all[idx_loss]*loss_curr
            loss = [loss,loss_everything_else]
        # print loss_everything_else
        return loss


class MultiCrossEntropyMultiBranchFakeL1_CASL(MultiCrossEntropyMultiBranch):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2, att_weight = 0.5, num_similar = 0):
        # self.att_weight =None
        num_branches = max(num_branches,1)
        
        super(MultiCrossEntropyMultiBranchFakeL1_CASL,self).__init__(class_weights=class_weights, loss_weights = loss_weights[:-1], num_branches = num_branches)
        # print 'att_super',super(MultiCrossEntropyMultiBranchWithL1_CASL,self).att_weight
        self.num_similar = num_similar
        self.casl_weight = loss_weights[-1]
        self.loss_weights_all = loss_weights
        self.loss_strs = ['CrossEnt'+str(idx) for idx in range(len(self.loss_weights_all)-2)]+['CASL']

        
    def forward(self, gt, preds, att, out, collate = True):
        # graph_sum = att[0]
        if self.num_branches ==1:
            preds = [preds]

        feature = att[1]
        input_sizes = att[2]
        loss_everything_else = super(MultiCrossEntropyMultiBranchFakeL1_CASL,self).forward(gt, preds, collate = collate)

        casl_loss = MyLoss_triple_noExclusive(feature, out, input_sizes, gt, type_loss = 'casl', debug = False, num_similar = self.num_similar)
        # casl_loss = MyLoss_triple(feature, out, input_sizes, gt, type_loss = 'casl', debug = False, num_similar = self.num_similar)
        
        if collate:
            loss = loss_everything_else + self.casl_weight*casl_loss    
        else:
            loss_everything_else.append(casl_loss)
            loss = 0
            for idx_loss, loss_curr in enumerate(loss_everything_else):
                loss += self.loss_weights_all[idx_loss]*loss_curr
            loss = [loss,loss_everything_else]

        return loss



class MultiCrossEntropyMultiBranchWithL1_CASL_pushH(MultiCrossEntropyMultiBranchWithL1):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2, att_weight = 0.5):
        # self.att_weight =None
        
        super(MultiCrossEntropyMultiBranchWithL1_CASL_pushH,self).__init__(class_weights=class_weights, loss_weights = loss_weights[:-1], num_branches = num_branches)
        # print 'att_super',super(MultiCrossEntropyMultiBranchWithL1_CASL_pushH,self).att_weight
        
        self.casl_weight = loss_weights[-1]
        self.loss_weights_all = loss_weights
        self.loss_strs = ['CrossEnt'+str(idx) for idx in range(len(self.loss_weights_all)-2)]+['L1','PushH']

        
    def forward(self, gt, preds, att, out, collate = True):
        graph_sum = att[0]
        feature = att[1]
        input_sizes = att[2]
        loss_everything_else = super(MultiCrossEntropyMultiBranchWithL1_CASL_pushH,self).forward(gt, preds, graph_sum, collate = collate)

        
        casl_loss = MyLoss_triple(feature, out, input_sizes, gt, type_loss = 'pushH', debug = False)
        
        if collate:
            loss = loss_everything_else + self.casl_weight*casl_loss    
        else:
            loss_everything_else.append(casl_loss)
            loss = 0
            for idx_loss, loss_curr in enumerate(loss_everything_else):
                loss += self.loss_weights_all[idx_loss]*loss_curr
            loss = [loss,loss_everything_else]

        return loss


class MultiCrossEntropyMultiBranchWithL1_CASL_pushDiff(MultiCrossEntropyMultiBranchWithL1):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2, att_weight = 0.5):
        # self.att_weight =None
        
        super(MultiCrossEntropyMultiBranchWithL1_CASL_pushDiff,self).__init__(class_weights=class_weights, loss_weights = loss_weights[:-1], num_branches = num_branches)
        # print 'att_super',super(MultiCrossEntropyMultiBranchWithL1_CASL_pushDiff,self).att_weight
        
        self.casl_weight = loss_weights[-1]
        self.loss_weights_all = loss_weights
        self.loss_strs = ['CrossEnt'+str(idx) for idx in range(len(self.loss_weights_all)-2)]+['L1','PushH']

        
    def forward(self, gt, preds, att, out, collate = True):
        graph_sum = att[0]
        feature = att[1]
        input_sizes = att[2]
        loss_everything_else = super(MultiCrossEntropyMultiBranchWithL1_CASL_pushDiff,self).forward(gt, preds, graph_sum, collate = collate)

        
        casl_loss = MyLoss_triple_old(feature, out, input_sizes, gt, type_loss = 'pushDiff', debug = False)
        
        if collate:
            loss = loss_everything_else + self.casl_weight*casl_loss    
        else:
            loss_everything_else.append(casl_loss)
            loss = 0
            for idx_loss, loss_curr in enumerate(loss_everything_else):
                loss += self.loss_weights_all[idx_loss]*loss_curr
            loss = [loss,loss_everything_else]

        return loss


class MCE_CenterLoss_Combo(nn.Module):
    def __init__(self, n_classes, feat_dim, bg, lambda_param, alpha_param, class_weights = None):
        super(MCE_CenterLoss_Combo, self).__init__()  
        self.lambda_param = lambda_param
        self.alpha_param = alpha_param
        self.mce = MultiCrossEntropy(class_weights)
        self.cl = CenterLoss(n_classes, feat_dim, bg)
        self.optimizer_centloss = torch.optim.SGD(self.cl.parameters(), lr=alpha_param)

    def forward(self, gt, pred):
        [gt, gt_all, features, class_pred] = gt
        loss_mce = self.mce(gt, pred)
        # loss_cl = self.cl(gt_all, features, class_pred)
        loss_total = loss_mce
         # + self.lambda_param*loss_cl
        # self.optimizer_centloss.zero_grad()
        return loss_total

    # def backward(self):
    #     pass
        # multiple (1./alpha) in order to remove the effect of alpha on updating centers
        # for param in self.cl.parameters():
        #     param.grad.data *= (1./self.lambda_param)
        # self.optimizer_centloss.step()





class CenterLoss(nn.Module):
    def __init__(self, n_classes, feat_dim, bg):
        super(CenterLoss, self).__init__()
        
        self.bg = bg

        if self.bg:
            n_classes+=1

        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.n_classes, self.feat_dim).cuda())
        # [None]*self.n_classes

    # def make_clusters(self,features, gt):
    #     # find features of each gt class (both for multilabel)
    #     for class_num in range(self.n_classes):
    #         if self.centers[class_num] is not None:
    #             continue
    #         rel_features = features[gt[:,class_num]>0,:]
    #         if rel_features.size(0)==0:
    #             continue
    #         rel_features = torch.mean(rel_features,dim =0)
    #         self.centers[class_num] = rel_features

    # def update_clusters(self, features, gt):
    #     for class_num in range(self.n_classes):
            
    #         rel_features = features[gt[:,class_num]>0,:]
    #         if rel_features.size(0)==0:
    #             continue
            
    #         rel_features = torch.mean(rel_features,dim =0)
    #         self.centers[class_num] = rel_features        


    def forward(self, gt, features, class_pred = None):
        
        num_instances = features.size(0)

        is_cuda = features.is_cuda
        
        if self.bg:
            assert class_pred is not None
            assert class_pred.size(1)==self.n_classes
            assert class_pred.size(1) - gt.size(1)==1

            zeros = torch.zeros(gt.size(0),1)
            if is_cuda:
                zeros = zeros.cuda()
            
            gt = torch.cat([gt,zeros],dim = 1)
            bin_bg = torch.argmax(class_pred, dim = 1)==(class_pred.size(1)-1)
            gt[bin_bg,:-1] = 0
            gt[bin_bg,-1] = 1

        # self.make_clusters(features,gt)
        
        loss_total = 0
        for class_num in range(self.n_classes):

            rel_features = features[gt[:,class_num]>0,:]
            if rel_features.size(0)==0:
                continue
            
            center_rel = self.centers[class_num]
            center_rel = center_rel.view(1,center_rel.size(0)).expand(rel_features.size(0),-1)
            distance = torch.sum(torch.sum(torch.pow(rel_features - center_rel,2),1),0)
            loss_total += distance

        loss_total = loss_total/num_instances        
        
        return loss_total    



class MultiCrossEntropyMultiBranchWithSigmoid(nn.Module):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2):
        super(MultiCrossEntropyMultiBranch, self).__init__()
        self.Sigmoid = nn.Sigmoid()
        self.LogSoftmax = nn.LogSoftmax(dim = 1)
        self.num_branches = num_branches
        
        # print class_weights

        # if class_weights is not None and len(class_weights)==100:
        #     class_weights = class_weights*5
        #     print 'scaling class_weights'
        #     # class_weights = None
        print class_weights        
        raw_input()
        if class_weights is None:
            self.class_weights = None
        else: 
            self.class_weights = nn.Parameter(torch.Tensor(class_weights[np.newaxis,:]), requires_grad = False)

        if loss_weights is None:
            self.loss_weights = [1 for i in range(self.num_branches)]
        else:
            self.loss_weights = loss_weights

    def forward(self, gt, preds, collate = True):

        # print gt.size()
        # print gt[:10]
        # print preds[0].size()

        # raw_input()
        # gt[gt>0]=1
        if collate:
            loss_all = 0
        else:
            loss_all = []

        assert len(preds) == self.num_branches
        for idx_pred, pred in enumerate(preds):
            pred = self.Sigmoid(pred)
            pred = self.LogSoftmax(pred)
            # torch.log(pred)
            # 
            # print pred.size()
            if self.class_weights is not None:
                assert self.class_weights.size(1)==pred.size(1)
                loss = self.class_weights*-1*gt*pred
                loss = torch.sum(loss, dim = 1)
                loss = torch.mean(loss)
                # sum(loss)
                    # , dim = 1)
            else:
                loss = -1*gt* pred
                loss = torch.sum(loss, dim = 1)
            # print 'meaning'
                loss = torch.mean(loss)
            # print 'meaning'


            if collate:
                loss_all += loss*self.loss_weights[idx_pred]
            else:
                # print 'loss',loss
                loss_all.append(loss)

        return loss_all


class MultiCrossEntropyMultiBranchWithSigmoidWithL1(MultiCrossEntropyMultiBranch):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2, att_weight = 0.5, num_similar = 0):
        num_branches = max(num_branches,1)
        self.att_weight = loss_weights[-1]

        super(MultiCrossEntropyMultiBranchWithSigmoidWithL1, self).__init__(class_weights=class_weights, loss_weights = loss_weights[:-1], num_branches = num_branches)
        # self.loss_weights_all = loss_weights
        
    def forward(self, gt, preds, att, collate = True):
        if self.num_branches ==1:
            preds = [preds]

        loss_regular = super(MultiCrossEntropyMultiBranchWithSigmoidWithL1,self).forward(gt, preds, collate = collate)
        # print 'min_val',torch.min(torch.abs(att))
        

        l1 = torch.mean(torch.abs(att))
        # print l1
        # print 'att',att
        # print 'l1',l1
        if collate:
            l1 = self.att_weight*l1
            loss_all = l1+loss_regular
        else:
            loss_regular.append(l1)
            loss_all = loss_regular

        return loss_all



class MultiCrossEntropyMultiBranchWithSigmoidWithL1_CASL(MultiCrossEntropyMultiBranchWithSigmoidWithL1):
    def __init__(self,class_weights=None, loss_weights = None, num_branches = 2, att_weight = 0.5, num_similar = 0):
        # self.att_weight =None
        
        super(MultiCrossEntropyMultiBranchWithSigmoidWithL1_CASL,self).__init__(class_weights=class_weights, loss_weights = loss_weights[:-1], num_branches = num_branches)
        # print 'att_super',super(MultiCrossEntropyMultiBranchWithSigmoidWithL1_CASL,self).att_weight
        self.num_similar = num_similar
        self.casl_weight = loss_weights[-1]
        self.loss_weights_all = loss_weights
        self.loss_strs = ['CrossEnt'+str(idx) for idx in range(len(self.loss_weights_all)-2)]+['L1','CASL']

        
    def forward(self, gt, preds, att, out, collate = True):
        graph_sum = att[0]
        feature = att[1]
        input_sizes = att[2]
        # print graph_sum
        # print input_sizes

        # for att_curr in att[1:]:
            # print len(att_curr), att_curr.size()
        # print len(att_curr), att_curr[1].size()
        # raw_input()

        loss_everything_else = super(MultiCrossEntropyMultiBranchWithSigmoidWithL1_CASL,self).forward(gt, preds, graph_sum, collate = collate)

        if self.num_branches>1:
            # print len(out)
            # print len(out[0])
            # .size()
            out = out[0]
            # raw_input()

        if self.casl_weight==0:
            casl_loss = 0.
        else:
            casl_loss = MyLoss_triple_noExclusive(feature, out, input_sizes, gt, type_loss = 'casl', debug = False, num_similar = self.num_similar)
        # print 'casl_loss',casl_loss
        
        if collate:
            loss = loss_everything_else + self.casl_weight*casl_loss    
        else:
            loss_everything_else.append(casl_loss)
            loss = 0
            for idx_loss, loss_curr in enumerate(loss_everything_else):
                # print 'here',idx_loss, self.loss_weights_all[idx_loss],loss_curr
                # print loss_curr, self.loss_weights_all[idx_loss]*loss_curr
                loss += self.loss_weights_all[idx_loss]*loss_curr
            loss = [loss,loss_everything_else]
        # print loss_everything_else
        return loss

        