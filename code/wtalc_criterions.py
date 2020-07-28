from __future__ import print_function
import os
import torch
import numpy as np
import torch.nn.functional as F
# from torchvision import transforms
# from helpers import util, visualize
# import torch.nn as nn
import itertools

# def MyLoss_triple_old(x, element_logits, seq_len, labels, type_loss = 'original', debug = False):
#     ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature), 
#         element_logits should be torch tensor of dimension (n_similar, n_element, n_class) 
#         seq_len should be numpy array of dimension (B,)
#         labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

#     batch_size = labels.size(0)
#     sim_loss = 0.
#     n_tmp = 0.
#     labels[labels>0]=1
#     # labels = torch.FloatTensor(labels)
#     for triple in itertools.combinations(range(batch_size),3):
#         triple = list(triple)
#         joint_label = torch.sum(labels[triple,:],dim = 0)
        
#         if torch.max(joint_label)!=2 or torch.sum(joint_label==1)==0:
#             # print ('zero continue')
#             continue

#         for same_idx in itertools.combinations(triple,2):
#             diff = [idx for idx in triple if idx not in same_idx]
#             assert len(diff)==1
            
#             ssd = list(same_idx)+ diff
            
#             # check same is same
#             same_sum = torch.sum(labels[same_idx,:], dim = 0)
#             if torch.max(same_sum)!=2 or torch.sum(same_sum==1)!=0:
#                 continue

#             #check diff is diff
#             if torch.max(joint_label[labels[ssd[-1],:]>0])!=1:
#                 print ('second continue')
#                 continue
            
#             #check seq len is atleast 2
#             if np.sum(seq_len[ssd]<2)>0:
#                 print ('third continue')
#                 continue

#             labels_curr = [labels[idx,:] for idx in ssd]
#             # if debug:
#             #     print ('labels_curr',labels_curr)
#             #     print ('ssd',ssd)
#             #     raw_input()

#             # alpha = []
#             # x_curr = []
#             # for idx in ssd:
#             #     start_curr = np.sum(seq_len[:idx])
#             #     end_curr = np.sum(seq_len[:idx+1])
#             #     alpha.append(element_logits[start_curr:end_curr,:])
#             #     x_curr.append(x[start_curr:end_curr,:])
                
#             # alpha = [element_logits[idx][:seq_len[idx]] for idx in ssd]
#             # x_curr = [x[idx][:seq_len[idx]] for idx in ssd]
#             # print (len(x), x[0].size())

#             alpha = [element_logits[idx] for idx in ssd]
#             x_curr = [x[idx] for idx in ssd]
#             if debug:
#                 for idx_x, x_pr in enumerate(x_curr):
#                     print (labels[ssd])
#                     print (x_pr.size())
#                     print (alpha[idx_x].size())
#                 raw_input()

#             n_tmp +=1

#             if type_loss =='pushDiff':
#                 sim_loss += single_pushDiff_loss(alpha, labels_curr, x_curr, debug = debug)
#                 break
#             # elif type_loss =='casl_us':
#             # sim_loss += single_casl_loss(alpha, labels_curr, x_curr, debug = debug)
#             # break
#             # elif type_loss =='new_loss':
#             #     sim_loss += single_new_loss(alpha, labels_curr, x_curr, debug = debug)
#             else:
#                 raise ValueError('Loss type '+type_loss+' not valid')
#             #     sim_loss += single_triplet_loss(alpha, labels_curr, x_curr, debug = debug)
        
#         # input()
#     # print (n_tmp, sim_loss)
#     sim_loss = sim_loss/max(n_tmp,1)
#     # print (sim_loss)
#     # input()
#     return sim_loss




def MyLoss_triple_old(x, element_logits, seq_len, labels, type_loss = 'original', debug = False):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature), 
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class) 
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    batch_size = labels.size(0)
    sim_loss = 0.
    n_tmp = 0.
    labels[labels>0]=1
    # labels = torch.FloatTensor(labels)
    for same_idx in itertools.combinations(range(batch_size),2):
        same_idx = list(same_idx)
        joint_label = torch.sum(labels[same_idx,:],dim = 0)
        
        if torch.max(joint_label)==2 and torch.sum(joint_label==1)==0:

            for idx_diff in range(batch_size):
                if idx_diff in same_idx:
                    continue
                ssd = same_idx+[idx_diff]
                joint_label = torch.sum(labels[ssd,:],dim = 0)
            
                if torch.max(joint_label[labels[ssd[-1],:]>0])!=1:
                    # print ('second continue')
                    # print (labels[ssd])
                    # raw_input()
                    continue

                #check seq len is atleast 2
                if np.sum(seq_len[ssd]<2)>0:
                    print ('third continue')
                    continue

                labels_curr = [labels[idx,:] for idx in ssd]
                
                alpha = [element_logits[idx] for idx in ssd]
                x_curr = [x[idx] for idx in ssd]
                if debug:
                    print (ssd)
                    print (labels[ssd])
                    for idx_x, x_pr in enumerate(x_curr):
                        print (x_pr.size())
                        print (alpha[idx_x].size())
                    raw_input()

                n_tmp +=1

                if type_loss =='pushDiff':
                    sim_loss += single_pushDiff_loss(alpha, labels_curr, x_curr, debug = debug)
                else:
                    raise ValueError('Loss type '+type_loss+' not valid')
            
    sim_loss = sim_loss/max(n_tmp,1)
    return sim_loss




def single_pushDiff_loss(alpha, labels, x, delta = 0.5, margin = 0.5, debug = False):
    atns = [F.softmax(alpha_curr, dim = 0) for alpha_curr in alpha]
    Hfs = [torch.mm(torch.transpose(x_curr, 1, 0), atns[idx_x_curr])[:,labels[idx_x_curr]>0] for idx_x_curr, x_curr in enumerate(x)]
    # Lfs = [torch.mm(torch.transpose(x_curr, 1, 0), (1-atns[idx_x_curr]))[:,labels[idx_x_curr]>0] for idx_x_curr, x_curr in enumerate(x)]
    
    h_norms = [torch.norm(vec_curr,2,dim = 0, keepdim = True) for vec_curr in Hfs]
    # l_norms = [torch.norm(vec_curr,2,dim = 0, keepdim = True) for vec_curr in Lfs]

    h1h1 = torch.diag(cos_sim(Hfs[:2])).unsqueeze(1)
    h1h2a = cos_sim([Hfs[0], Hfs[2]])
    h1h2b = cos_sim([Hfs[1], Hfs[2]])

    # h1l1ab = torch.diag(cos_sim([Hfs[0], Lfs[1]])).unsqueeze(1)
    # h1l1ba = torch.diag(cos_sim([Hfs[1], Lfs[0]])).unsqueeze(1)

    first_terms = [ h1h1, h1h1]
    second_terms = [h1h2a, h1h2b]

    first_term_strs =  ['h1h1','h1h1']
    second_term_strs =  ['h1h2a', 'h1h2b']
    assert len(first_terms)==len(second_terms)

    loss = 0
    for idx_term, first_term in enumerate(first_terms):
        # print (h_norms)
        # print (l_norms)
        second_term = second_terms[idx_term]
        # print (first_term.size(), second_term.size())
        

        relud = torch.nn.functional.relu(first_term - second_term +margin)
        # print ('relud', relud.size())

        loss+= 1./len(first_terms)*torch.mean(relud)

        if debug:
            print (first_term_strs[idx_term],first_term)
            print (second_term_strs[idx_term],second_term)
            print (first_term - second_term +margin)
            print (loss, relud, torch.mean(relud))
            raw_input()
    return loss



def MyLoss_triple(x, element_logits, seq_len, labels, type_loss = 'original', debug = False, num_similar = 0):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature), 
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class) 
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    
    batch_size = labels.size(0)
    sim_loss = 0.
    n_tmp = 0.
    labels[labels>0]=1
    total_pairs = 0
    # print (labels.cpu().data)
    # labels = torch.FloatTensor(labels)

    if num_similar==0:
        iterator = itertools.combinations(range(batch_size),2)
    else:
        iterator = [(idx, idx+1) for idx in range(0,min(2*num_similar,batch_size),2)] 

    print (num_similar,iterator,batch_size)
    # debug = True
    # raw_input()

    for triple in iterator:
        total_pairs +=1
        triple = list(triple)
        
        print (labels[triple,:])
        
        joint_label = torch.sum(labels[triple,:],dim = 0)

        if torch.max(joint_label)!=2:
            # print ('first_continue')
            continue

        ssd = list(triple)
        same_idx = ssd
        
        # check same is same
        same_sum = torch.sum(labels[same_idx,:], dim = 0)
        print (torch.max(same_sum),torch.sum(same_sum==1))
        if torch.max(same_sum)!=2 or torch.sum(same_sum==1)!=0:
            # print ('second_continue')
            continue

        #check seq len is atleast 2
        if np.sum(seq_len[ssd]<2)>0:
            print ('third continue')
            continue

        # print (same_idx)
        labels_curr = [labels[idx,:] for idx in ssd]
        alpha = [element_logits[idx] for idx in ssd]
        x_curr = [x[idx] for idx in ssd]
        if debug:
            for idx_x, x_pr in enumerate(x_curr):
                print (x_pr.size())
                print (alpha[idx_x].size())
            raw_input()

        n_tmp +=1

        if type_loss =='pushH':
            sim_loss += single_pushH_loss(alpha, labels_curr, x_curr, debug = debug)
        elif type_loss =='casl':
            sim_loss += single_casl_loss(alpha, labels_curr, x_curr, debug = debug)
    

    # print (n_tmp
    sim_loss = sim_loss/max(n_tmp,1)
    print ('n_tmp',n_tmp, 'total_pairs', total_pairs)
    raw_input()
    return sim_loss


def MyLoss_triple_noExclusive(x, element_logits, seq_len, labels, type_loss = 'original', debug = False, num_similar = 0):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature), 
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class) 
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    
    batch_size = labels.size(0)
    sim_loss = 0.
    n_tmp = 0.
    labels = labels.clone()
    labels[labels>0]=1
    total_pairs = 0
    # print (labels.cpu().data)
    # labels = torch.FloatTensor(labels)

    if num_similar==0:
        iterator = itertools.combinations(range(batch_size),2)
    else:
        iterator = [(idx, idx+1) for idx in range(0,min(2*num_similar,batch_size-1),2)] 

    # print (num_similar,iterator,batch_size)
    # debug = True
    # raw_input()

    for triple in iterator:
        # if triple[1]>=batch_size:
        #     continue

        total_pairs +=1
        triple = list(triple)
        
        # print (labels[triple,:])
        
        joint_label = torch.sum(labels[triple,:],dim = 0)
        # print (torch.sum(joint_label==2))

        if torch.max(joint_label)!=2:
            # print ('first_continue')
            continue

        # if torch.sum(joint_label==2)>2:
        #     # print ('second_continue')
        #     continue

        ssd = list(triple)
        
        #check seq len is atleast 2
        if np.sum(seq_len[ssd]<2)>0:
            print ('third continue')
            continue

        labels_curr = [labels[idx,:].clone() for idx in ssd]
        # print (labels_curr)
        # print (joint_label)
        for label_curr in labels_curr:
            label_curr[joint_label<2]=0
            label_curr[joint_label>2]=1
        # print (labels_curr)
        # print ('...')
        # print (labels[idx,:])
        
        # raw_input()

        alpha = [element_logits[idx] for idx in ssd]
        x_curr = [x[idx] for idx in ssd]
        if debug:
            for idx_x, x_pr in enumerate(x_curr):
                print (x_pr.size())
                print (alpha[idx_x].size())
            raw_input()

        n_tmp +=1

        if type_loss =='pushH':
            sim_loss += single_pushH_loss(alpha, labels_curr, x_curr, debug = debug)
        elif type_loss =='casl':
            sim_loss += single_casl_loss(alpha, labels_curr, x_curr, debug = debug)
    

    # print (n_tmp
    sim_loss = sim_loss/max(n_tmp,1)
    # print ('no exclusive n_tmp',n_tmp, 'total_pairs', total_pairs)
    # raw_input()
    return sim_loss


def cos_sim(vecs):

    vecs = [vec_curr/torch.norm(vec_curr,2,dim = 0, keepdim = True) for vec_curr in vecs]
    d1 = 1 - torch.mm(torch.transpose(vecs[0],0,1),vecs[1])
    return d1


def single_pushH_loss(alpha, labels, x, delta = 0.5, margin = 0.5, debug = False):
    atns = [F.softmax(alpha_curr, dim = 0) for alpha_curr in alpha]
    Hfs = [torch.mm(torch.transpose(x_curr, 1, 0), atns[idx_x_curr])[:,labels[idx_x_curr]>0] for idx_x_curr, x_curr in enumerate(x)]
    # Lfs = [torch.mm(torch.transpose(x_curr, 1, 0), (1-atns[idx_x_curr]))[:,labels[idx_x_curr]>0] for idx_x_curr, x_curr in enumerate(x)]
    
    h_norms = [torch.norm(vec_curr,2,dim = 0, keepdim = True) for vec_curr in Hfs]
    # l_norms = [torch.norm(vec_curr,2,dim = 0, keepdim = True) for vec_curr in Lfs]

    h1h1 = torch.diag(cos_sim(Hfs[:2])).unsqueeze(1)

    # h1l1ab = torch.diag(cos_sim([Hfs[0], Lfs[1]])).unsqueeze(1)
    # h1l1ba = torch.diag(cos_sim([Hfs[1], Lfs[0]])).unsqueeze(1)

    first_terms = [ h1h1]
    # , h1h1]
    # second_terms = [h1l1ab, h1l1ba]

    first_term_strs =  ['h1h1']
    # second_term_strs =  ['h1l1ab', 'h1l1ba']
    # assert len(first_terms)==len(second_terms)

    loss = 0
    for idx_term, first_term in enumerate(first_terms):
        # print (h_norms)
        # print (l_norms)
        # second_term = second_terms[idx_term]
        # print (first_term.size(), second_term.size())
        

        relud = torch.nn.functional.relu(1 - first_term)
        # print ('relud', relud.size())

        loss+= 1./len(first_terms)*torch.mean(relud)

        if debug:
            print (first_term_strs[idx_term],first_term)
            # print (second_term_strs[idx_term],second_term)
            print (1 - first_term)
            print (loss, relud, torch.mean(relud))
            raw_input()
    return loss

def single_casl_loss(alpha, labels, x, delta = 0.5, margin = 0.5, debug = False):

    atns = [F.softmax(alpha_curr, dim = 0) for alpha_curr in alpha]
    Hfs = [torch.mm(torch.transpose(x_curr, 1, 0), atns[idx_x_curr])[:,labels[idx_x_curr]>0] for idx_x_curr, x_curr in enumerate(x)]
    Lfs = [torch.mm(torch.transpose(x_curr, 1, 0), (1-atns[idx_x_curr]))[:,labels[idx_x_curr]>0] for idx_x_curr, x_curr in enumerate(x)]
    
    h_norms = [torch.norm(vec_curr,2,dim = 0, keepdim = True) for vec_curr in Hfs]
    l_norms = [torch.norm(vec_curr,2,dim = 0, keepdim = True) for vec_curr in Lfs]

    h1h1 = torch.diag(cos_sim(Hfs[:2])).unsqueeze(1)

    h1l1ab = torch.diag(cos_sim([Hfs[0], Lfs[1]])).unsqueeze(1)
    h1l1ba = torch.diag(cos_sim([Hfs[1], Lfs[0]])).unsqueeze(1)

    first_terms = [ h1h1, h1h1]
    second_terms = [h1l1ab, h1l1ba]

    first_term_strs =  ['h1h1', 'h1h1']
    second_term_strs =  ['h1l1ab', 'h1l1ba']
    assert len(first_terms)==len(second_terms)

    loss = 0
    for idx_term, first_term in enumerate(first_terms):
        # print (h_norms)
        # print (l_norms)
        second_term = second_terms[idx_term]
        # print (first_term.size(), second_term.size())
        

        relud = torch.nn.functional.relu(first_term - second_term + margin)
        # print ('relud', relud.size())

        loss+= 1./len(first_terms)*torch.mean(relud)

        if debug:
            atns_rel = [atns[idx][:,labels[idx]>0] for idx in range(len(x))]
            for atn_rel_curr in atns_rel:
                print ('atn_rel_curr.size()',atn_rel_curr.size())
                print ('torch.min(atn_rel_curr),torch.max(atn_rel_curr),torch.sum(atn_rel_curr)',torch.min(atn_rel_curr),torch.max(atn_rel_curr),torch.sum(atn_rel_curr))
            print (first_term_strs[idx_term],first_term)
            print (second_term_strs[idx_term],second_term)
            print ('first_term - second_term',first_term - second_term)
            print ('loss, first_term-second_term+margin, relud, torch.mean(relud)',loss, first_term-second_term+margin, relud, torch.mean(relud))
            raw_input()
    return loss