from helpers import util, visualize
import random
import globals as globs
import torch.utils
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import models
import matplotlib.pyplot as plt
import time
import os
import itertools
import glob
import sklearn.metrics
import analysis.evaluate_thumos as et
import debugging_graph as dg
import wtalc
from data_processors import preprocess_charades as pc
import figuring_out_anet as foa
import threshes_temp
import scipy.special
class Exp_Lr_Scheduler:
    def __init__(self, optimizer,step_curr, init_lr, decay_rate, decay_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.step_curr = step_curr
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        

    def step(self):
        self.step_curr += 1
        # print 'STEPPING',len(self.optimizer.param_groups)
        for idx_param_group,param_group in enumerate(self.optimizer.param_groups): 
            # print 'outside',idx_param_group,self.init_lr[idx_param_group],param_group['lr']
            if self.init_lr[idx_param_group]!=0:
                new_lr = self.init_lr[idx_param_group] * self.decay_rate **(self.step_curr/float(self.decay_steps))
                new_lr = max(new_lr ,self.min_lr)
                # print idx_param_group,param_group['lr'], new_lr
                param_group['lr'] = new_lr
            # print param_group['lr']

def test_model_core(model, test_dataloader, criterion, log_arr, multibranch = 1):
    model.eval()

    if multibranch>1:
        preds = [[] for i in range(multibranch)]
    else:
        preds = []
    
    labels_all = []
    loss_iter_total = 0.
    model_name = model.__class__.__name__.lower()
    criterion_str = criterion.__class__.__name__.lower()
    
    for num_iter_test,batch in enumerate(test_dataloader):
        samples = batch['features']
        labels = batch['label'].cuda()
        
        # print model.__class__.__name__.lower()
        # print 'centerloss' in model.__class__.__name__.lower()
        # raw_input()

        if 'centerloss' in model_name:
            preds_mini,extra = model.forward(samples, labels)
            preds.append(preds_mini)
            labels = [labels]+extra
        else:
            if multibranch>1:
                preds_mini = [[] for i in range(multibranch)]
            else:
                preds_mini = []
                att = []

            for idx_sample, sample in enumerate(samples):
                if 'multi_video' in model_name:
                    # print model_name
                    if 'perfectg' in model_name or 'cooc' in model_name:
                        # print 'hello'
                        out,preds_mini = model.forward([samples,batch['gt_vec']])
                        # print len(preds_mini)
                    else:
                        out,preds_mini = model.forward(samples)
                    
                    if 'l1' in criterion_str:
                        [preds_mini, att] = preds_mini

                    if multibranch>1:

                        for idx_preds_curr,preds_curr in enumerate(preds_mini):
                            preds[idx_preds_curr]+=[torch.nn.functional.softmax(pmf).data.cpu().numpy() for pmf in preds_curr]    
                    else:
                        preds+=[torch.nn.functional.softmax(pmf_mini).data.cpu().numpy() for pmf_mini in preds_mini]
                    
                    break
                elif 'perfectg' in model_name:
                    out,pmf = model.forward([sample.cuda(),batch['gt_vec'][idx_sample].cuda()])
                else:
                    out,pmf = model.forward(sample.cuda())

                if 'l1' in criterion_str:
                    [pmf, att_curr] = pmf
                    att.append(att_curr)

                if multibranch>1:
                    preds.append(torch.nn.functional.softmax(pmf[0].unsqueeze(0)).data.cpu().numpy())
                        
                    for idx in range(len(pmf)):
                        # print idx, pmf[idx].size(), len(preds_mini[idx])
                        preds_mini[idx].append(pmf[idx].unsqueeze(0))
                else:
                    preds.append(torch.nn.functional.softmax(pmf.unsqueeze(0)).data.cpu().numpy())
                    preds_mini.append(pmf.unsqueeze(0))


            if multibranch>1:
                # print 'hello;',len(preds_mini[0]), len(preds_mini[1])
                preds_mini = [torch.cat(preds_curr,0) for preds_curr in preds_mini]
                # print preds_mini[0].size(), preds_mini[1].size()

            else:
                preds_mini = torch.cat(preds_mini,0)
        
        if 'casl' in criterion_str:
            loss = criterion(labels, preds_mini, att, out)
        elif 'l1' in criterion_str:
            loss = criterion(labels, preds_mini, att)
        else:
            loss = criterion(labels, preds_mini)
        

        labels_all.append(labels.data.cpu().numpy())
        loss_iter = loss.item()
        loss_iter_total+=loss_iter    
        str_display = 'val iter: %d, val loss: %.4f' %(num_iter_test,loss_iter)
        log_arr.append(str_display)
        print str_display
        
    
    labels_all = np.concatenate(labels_all,axis = 0)
    
    # if 'centerloss' not in criterion.__class__.__name__.lower():    
    #     loss = criterion(labels_all, preds)
    #     loss_iter = loss.item()

    #     str_display = 'val total loss: %.4f' %(loss_iter)
    #     log_arr.append(str_display)
    #     print str_display
    

    # labels_all = labels_all.data.cpu().numpy()
    loss_iter = loss_iter_total/len(test_dataloader)
    labels_all[labels_all>0]=1
    assert len(np.unique(labels_all)==2)
    # print labels_all.shape, np.min(labels_all), np.max(labels_all)
    # print preds.shape, np.min(preds), np.max(preds)
    if multibranch==1:
        preds_all = [preds]
    else:
        preds_all = preds

    for preds in preds_all:
        preds = np.concatenate(preds,axis = 0)
        accuracy = sklearn.metrics.average_precision_score(labels_all, preds)
        
        # print accuracy.shape
        # print accuracy
        
        str_display = 'val accuracy: %.4f' %(accuracy)
        log_arr.append(str_display)
        print str_display
        
    del preds_all
    torch.cuda.empty_cache()

    model.train(True)

    return accuracy, loss_iter

def merge_detections(bin_keep, det_conf, det_time_intervals, merge_with = 'max',dummy_ret = False):
    bin_keep = bin_keep.astype(int)
    bin_keep_rot = np.roll(bin_keep, 1)
    bin_keep_rot[0] = 0
    diff = bin_keep - bin_keep_rot
    # diff[-3]=1
    idx_start_all = list(np.where(diff==1)[0])
    idx_end_all = list(np.where(diff==-1)[0])
    if len(idx_start_all)>len(idx_end_all):
        assert len(idx_start_all)-1==len(idx_end_all)
        idx_end_all.append(bin_keep.shape[0])
    
    assert len(idx_start_all)==len(idx_end_all)
    num_det = len(idx_start_all)
    
    det_conf_new = np.zeros((num_det,))
    det_time_intervals_new = np.zeros((num_det,2))
    dummy = np.array(det_conf)
    # np.ones(det_conf.shape)*-1
    # 
    # .shape)

    for idx_curr in range(num_det):
        idx_start = idx_start_all[idx_curr]
        idx_end = idx_end_all[idx_curr]

        det_conf_rel = det_conf[idx_start:idx_end]
        if merge_with=='max':
            det_conf_new[idx_curr]=np.max(det_conf_rel)
            dummy[idx_start:idx_end] = np.max(det_conf_rel)
        elif merge_with=='min':
            det_conf_new[idx_curr]=np.min(det_conf_rel)
        elif merge_with=='mean':
            det_conf_new[idx_curr]=np.mean(det_conf_rel)
        else:
            error_message = str('merge_with %s not recognized', merge_with)
            raise ValueError(error_message)


        # print det_time_intervals.shape, idx_start
        det_time_intervals_new[idx_curr,0]=det_time_intervals[idx_start,0]
        # print idx_end, det_time_intervals.shape, idx_curr, num_det
        det_time_intervals_new[idx_curr,1]=det_time_intervals[idx_end,0] if idx_end<det_time_intervals.shape[0] else det_time_intervals[idx_end-1,1]

        # print bin_keep[idx_start:idx_end]
        # print diff[idx_start:idx_end]
        assert np.all(bin_keep[idx_start:idx_end]==1)

    # print det_conf.shape
    # print det_time_intervals.shape
    # print det_conf_new.shape
    # print det_time_intervals_new.shape

    # raw_input()
    # return det_conf_new, det_time_intervals_new, 
    if dummy_ret:
        return dummy, det_time_intervals_new, 
    else:
        return det_conf_new, det_time_intervals_new, 

def visualize_dets(model, test_dataloader, dir_viz, first_thresh , second_thresh, bin_trim = None,  det_class = -1, branch_to_test =-1, criterion_str= None, dataset = 'ucf'):

    if dataset=='ucf_untf':
        fps_stuff = 0.1
    else:
        fps_stuff = 16./25.

    print fps_stuff

    # fps_stuff = 1./10.

    # 10./30.
    model.eval()
    model_name = model.__class__.__name__.lower()
    
    preds = []
    labels_all = []

    det_vid_names_ac = [os.path.split(line.split(' ')[0])[1][:-4] for line in test_dataloader.dataset.files]
    
    outs = []
    min_all = None
    max_all = None

    det_vid_names = []
    det_events_class = []

    det_time_intervals_all = []
    det_conf_all = []

    det_vid_names_merged = []
    det_events_class_merged = []
    
    det_time_intervals_merged_all = []
    det_conf_merged_all = []
    
    
    out_shapes = {}
    idx_test = 0

    threshes_all = []
    plot_dict = {}
    for num_iter_test,batch in enumerate(test_dataloader):
        samples = batch['features']
        labels = batch['label']

        preds_mini = []
        for idx_sample, sample in enumerate(samples):
            # print idx_test
            if branch_to_test>-1:
                out, pmf, bg = model.forward(sample.cuda(), ret_bg = True, branch_to_test = branch_to_test)
            elif 'perfectg' in model_name:
                out, pmf, bg = model.forward([sample.cuda(),batch['gt_vec'][idx_sample].cuda()], ret_bg = True)
                # out, pmf, bg = model.forward(sample.cuda(), ret_bg = True)
            else:
                out, pmf = model.forward(sample.cuda())
                # , ret_bg = True)
            # out = out-bg

            if 'l1' in criterion_str:
                [pmf, att] = pmf


            if bin_trim is not None:
                out = out[:,np.where(bin_trim)[0]]
                pmf = pmf[np.where(bin_trim)[0]]

            if branch_to_test==-1:
                out = torch.nn.functional.softmax(out,dim = 1)
                print 'smaxing'
            # else:
            #     print 'not smaxing'

            start_seq = np.array(range(0,out.shape[0]))*fps_stuff
            end_seq = np.array(range(1,out.shape[0]+1))*fps_stuff
            det_time_intervals_meta = np.concatenate([start_seq[:,np.newaxis],end_seq[:,np.newaxis]],axis=1)
            

            pmf = pmf.data.cpu().numpy()
            out = out.data.cpu().numpy()
            # print det_class
            # raw_input()
            if det_class==-1:
                class_idx = np.where(labels[idx_sample].numpy())[0]
                # [0][0]
                if len(class_idx)>2:
                    # print class_idx
                    class_idx = class_idx[2]
                else:
                    class_idx = class_idx[0]
                # else:
                #     continue

                class_idx_gt = class_idx
            elif det_class ==-2:
                bg = bg.data.cpu().numpy()
                bg = bg[:,:1]
                out = np.concatenate([out,bg],axis = 1)
                class_idx = out.shape[1] - 1
                class_idx_gt = np.where(labels[idx_sample].numpy())[0][0]
            else:
                class_idx = det_class
                class_idx_gt = class_idx

            if det_class>=-1:
                if first_thresh==-1:
                    bin_not_keep = labels[idx_sample].data.cpu().numpy()==0
                else:
                    bin_not_keep = pmf<first_thresh
                # print pmf
                # print bin_not_keep
                # print class_idx
                # raw_input()
                # for class_idx in range(pmf.size):

                # if bin_not_keep[class_idx]:
                #     idx_test +=1
                #     print 'PROBLEM'
                #     continue
            class_idx_all = np.where(labels[idx_sample].numpy())[0]
            
            # if len(class_idx_all)>1:
            #     print idx_test, idx_sample, labels[idx_sample],  det_vid_names_ac[idx_test]
            
            # if '1468' not in det_vid_names_ac[idx_test]:
            #     idx_test+=1
            #     continue

            for class_idx in class_idx_all:
                
                # print class_idx, class_idx_all
                class_idx_gt = class_idx
                det_conf = out[:,class_idx]
                if second_thresh<=0:
                    thresh = second_thresh
                else:
                    thresh = np.max(det_conf)-(np.max(det_conf)-np.min(det_conf))*second_thresh
                bin_second_thresh = det_conf>thresh

                # print bin_second_thresh



                det_conf_merged, det_time_intervals_merged = merge_detections(bin_second_thresh, det_conf, det_time_intervals_meta)
                # if '0000324' in det_vid_names_ac[idx_test]:
                #     # print det_conf
                #     print det_conf_merged
                #     print det_time_intervals_merged
                #     print det_vid_names_ac[idx_test]

                # raw_input()


                det_time_intervals = det_time_intervals_meta

                det_vid_names.extend([det_vid_names_ac[idx_test]]*det_conf.shape[0])
                det_events_class.extend([class_idx_gt]*det_conf.shape[0])
                # out_shapes.extend([out.shape[0]]*det_conf.shape[0])
                out_shapes[det_vid_names_ac[idx_test]] = out.shape[0]
                
                det_conf_all.append(det_conf)
                det_time_intervals_all.append(det_time_intervals)

                det_conf_merged_all.append(det_conf_merged)
                det_time_intervals_merged_all.append(det_time_intervals_merged)
                det_vid_names_merged.extend([det_vid_names_ac[idx_test]]*det_conf_merged.shape[0])
                det_events_class_merged.extend([class_idx_gt]*det_conf_merged.shape[0])
                # out_shapes_merged.extend([out.shape[0]]*det_conf_merged.shape[0])

            idx_test +=1
            # break
    

    det_conf_all = np.concatenate(det_conf_all,axis =0)
    det_time_intervals_all = np.concatenate(det_time_intervals_all,axis = 0)

    det_conf_merged_all = np.concatenate(det_conf_merged_all,axis =0)
    det_time_intervals_merged_all = np.concatenate(det_time_intervals_merged_all,axis = 0)    

    det_events_class_all = np.array(det_events_class)
    
    det_events_class_merged = np.array(det_events_class_merged)
    det_vid_names_merged = np.array(det_vid_names_merged)

    # out_shapes = np.array(out_shapes)
    # print np.min(det_conf_all), np.max(det_conf_all)
    # raw_input()
    plot_dict = {}
    plot_dict['Det'] = [det_conf_all,det_time_intervals_all, det_events_class_all, np.array(det_vid_names)]
    plot_dict['Merged'] = [det_conf_merged_all,det_time_intervals_merged_all,det_events_class_merged, det_vid_names_merged]

    # et.viz_overlap(dir_viz, det_vid_names, det_conf_all, det_time_intervals_all, det_events_class_all,out_shapes)
    et.viz_overlap_multi(dir_viz,  plot_dict, out_shapes, fps_stuff, dataset = dataset)

    # np.savez('../scratch/debug_det_graph.npz', det_vid_names = det_vid_names, det_conf = det_conf, det_time_intervals = det_time_intervals, det_events_class = det_events_class)

def test_model_overlap(model, test_dataloader, criterion, log_arr,first_thresh , second_thresh , bin_trim = None , multibranch =1, branch_to_test = -1,dataset = 'ucf', save_outfs = None, test_method = 'original', fps_stuff = 16./25., matlab_arr = None, soft_out = True):

    # model, 
    # print 'test_dataloader ',test_dataloader 
    # print 'criterion ',criterion 
    # print 'log_arr',log_arr
    # print 'first_thresh  ',first_thresh  
    # print 'second_thresh  ',second_thresh  
    # print 'bin_trim',bin_trim
    # print 'multibranch',multibranch
    # print 'branch_to_test',branch_to_test
    # print 'dataset',dataset
    # print 'save_outfs',save_outfs
    # raw_input()
    # print 'test_method',test_method
    # print 'fps_stuff',fps_stuff


    # if dataset=='ucf':
    #     fps_stuff = 16./25.
    if dataset=='ucf_untf':
        fps_stuff = 0.1
    else:
        fps_stuff = 16./25.

    # print fps_stuff
    # 10./25.

    # print 'SECOND THRESH', second_thresh
    # raw_input()
    # out_dir = '../scratch/graph_2_nononlin_b'
    # util.mkdir(out_dir)
    # print out_dir
    # raw_input()

    model.eval()
    limit_bef =  test_dataloader.dataset.feature_limit
    test_dataloader.dataset.select_front = True
    # test_dataloader.dataset.feature_limit = None

    is_cuda = next(model.parameters()).is_cuda
    if not is_cuda:
        model = model.cuda()

    model_name = model.__class__.__name__.lower()
    criterion_str = criterion.__class__.__name__.lower()

    preds = []
    labels_all = []

    det_vid_names_ac = [os.path.split(line.split(' ')[0])[1][:-4] for line in test_dataloader.dataset.files]
    
    outs = []
    min_all = None
    max_all = None

    det_events_class = []
    det_time_intervals_all = []
    det_conf_all = []
    det_vid_names = []
    idx_test = 0
    predictions = []
    pmfs = []
    labels_all = []
    out_fs = []
    threshes_all = []
    for num_iter_test,batch in enumerate(test_dataloader):
        samples = batch['features']
        labels = batch['label'].cuda()

        preds_mini = []
        for idx_sample, sample in enumerate(samples):

            if 'centerloss' in model_name:
                out, pmf = model.forward_single_test(sample.cuda())
            else:    
                
                if multibranch>1:
                    if 'perfectg' in model_name or 'cooc' in model_name:
                        out, pmf = model.forward([sample.cuda(),batch['gt_vec'][idx_sample].cuda()], branch_to_test = branch_to_test)
                    elif 'cog' in model_name:
                        out, pmf = model.forward(sample, branch_to_test = branch_to_test)
                    else:
                        out, pmf = model.forward(sample.cuda(), branch_to_test = branch_to_test)
                    
                else:
                    if 'perfectg' in model_name or 'cooc' in model_name:
                        out,pmf = model.forward([sample.cuda(),batch['gt_vec'][idx_sample].cuda()])
                    else:  
                        # if type(sample)==type([]):
                        # if save_outfs:
                        #     model.feat_ret = True

                        out, pmf = model.forward(sample)

                        # print out.size(), sample.size(), save_outfs
                        # raw_input()

                        # else:  
                        #     out, pmf = model.forward(sample.cuda())
                        # print pmf
                        # print model
                        if save_outfs:
                            outf = out.data.cpu().numpy()
                            # print np.min(outf), np.max(outf)
                            vid_name = det_vid_names_ac[idx_test]
                            # out_file_f = os.path.join(save_outfs,vid_name+'.npy')
                            # np.save(out_file_f,outf)

                            # print pmf[0].data.cpu().numpy()
                            # print labels[idx_test]
                            # raw_input()
                            # bin_keep = pmf[0].data.cpu().numpy()>first_thresh
                            # bin_keep = bin_keep[np.newaxis,:]
                            # outf  = outf*bin_keep

                            start_seq = np.array(range(0,outf.shape[0]))*fps_stuff
                            end_seq = np.array(range(1,outf.shape[0]+1))*fps_stuff
                            dtm = np.concatenate([start_seq[:,np.newaxis],end_seq[:,np.newaxis]],axis=1)

                            bin_second_thresh = outf>second_thresh
                            dca = []
                            for class_idx in range(outf.shape[1]):
                                det_conf, _ = merge_detections(bin_second_thresh[:,class_idx], outf[:,class_idx], dtm,dummy_ret = True)
                                # det_conf = det_conf+1 
                                dca.append(det_conf[:,np.newaxis])
                            dca = np.concatenate(dca, axis = 1)
                            
                            
                            # print idx_test, len(det_vid_names_ac), vid_name, outf.shape
                            out_file_f = os.path.join(save_outfs,vid_name+'.npy')
                            np.save(out_file_f,dca)


                        #     # print outf.shape
                        #     # print pmf[0].shape
                        #     out_dir_pmf = os.path.join(os.path.split(save_outfs)[0],'outpmf')
                        #     # print out_dir_pmf
                        #     util.mkdir(out_dir_pmf)
                        #     out_file_pmf = os.path.join(out_dir_pmf,vid_name+'.npy')
                        #     np.save(out_file_pmf, pmf[0].data.cpu().numpy())

                        #     # raw_input()
                            idx_test+=1
                            continue

                        # if save_outfs:
                        #     outf = pmf[1][1]
                        #     # .cpu().numpy()
                        #     # print len(outf)
                        #     # print outf[0].shape
                        #     outf = outf[0].data.cpu().numpy()
                        #     # raw_input()
                        #     # .size()
                        #     pmf = pmf[:2]
                        #     # model.out_f(sample.cuda()).data.cpu().numpy()
                        #     vid_name = det_vid_names_ac[idx_test]
                        #     out_file_f = os.path.join(save_outfs,vid_name+'.npy')
                            
                        #     np.save(out_file_f,outf)
                        # elif test_method=='best_worst_dot':
                        #     out_fs.append(model.out_f(sample.cuda()).data.cpu().numpy())

                if test_method=='best_worst_dot':
                    out_fs.append(model.out_f_f(sample.cuda()).data.cpu().numpy())

            if 'l1' in criterion_str:
                [pmf, att] = pmf

            # print out.size(),torch.min(out), torch.max(out)
            # print pmf.size(),torch.min(pmf), torch.max(pmf)

            # raw_input()

            if bin_trim is not None:
                # print out.size()
                out = out[:,np.where(bin_trim)[0]]
                pmf = pmf[np.where(bin_trim)[0]]

            # print det_vid_names_ac[idx_test]
            # print out.size()
            # raw_input()

            if (second_thresh>=0 and soft_out and branch_to_test!=-2 and branch_to_test!=-4 and branch_to_test!=-5) or (type(second_thresh)==str and second_thresh.startswith('smax')):
                print 'smaxing'
                out_temp = out.data.cpu().numpy()
                # idx_interesting = np.where(np.sum(out_temp,axis = 1)==0)[0]
                # print idx_interesting
                # print out_temp[idx_interesting,:]
                # print torch.sum(torch.sum(out,dim =1 )==0)

                out = torch.nn.functional.softmax(out,dim = 1)
                # out[idx_interesting,:] = 0.
                out_temp = out.data.cpu().numpy()
                # print np.sum(np.isnan(out_temp)), out_temp.size
                
                # print out_temp[idx_interesting,:]

                # raw_input()

            start_seq = np.array(range(0,out.shape[0]))*fps_stuff
            # print start_seq
            end_seq = np.array(range(1,out.shape[0]+1))*fps_stuff
            # raw_input()
            det_time_intervals_meta = np.concatenate([start_seq[:,np.newaxis],end_seq[:,np.newaxis]],axis=1)
            
            # print pmf.size()
            # print pmf
            # print labels[idx_sample]
            # raw_input()
            if first_thresh>=0:
                pmf = torch.nn.functional.softmax(pmf)
                pmf = pmf.data.cpu().numpy()
            else:
                # print 'not smacing'
                pmf = pmf.data.cpu().numpy()
                # first_thresh = -1*first_thresh
            # print pmf.shape
            # print pmf

            # pmf = softmax(pmf)
            # print pmf.shape
            # raw_input()

            out = out.data.cpu().numpy()
            if test_method is not 'original':
                det_vid_names.append(det_vid_names_ac[idx_test])
                predictions.append(out)
                pmfs.append(pmf)
                labels_all.append(labels[idx_sample].data.cpu().numpy())
            else:
                if first_thresh==-1:
                    bin_not_keep = labels[idx_sample].data.cpu().numpy()==0
                else:
                    # print first_thresh, pmf
                    bin_not_keep = pmf<-1*first_thresh

                # print bin_not_keep
                # raw_input()
                # print 'in overlap',det_vid_names_ac[idx_test]
                for class_idx in range(pmf.size):
                    if bin_not_keep[class_idx]:
                        continue

                    det_conf = out[:,class_idx]
                    if type(second_thresh)==str:
                        if second_thresh=='otsu':
                            thresh = foa.get_otsu_thresh(det_conf, 10)
                            # print thresh, np.min(det_conf), np.max(det_conf)
                        # elif second_thresh=='min_max_0.5_per_class':
                        elif second_thresh=='otsu_per_class_pmfthresh_justpos_0':
                            threshes = threshes_temp.min_max_per_class_anet
                            thresh = threshes[class_idx]
                            thresh_comp = np.max(det_conf)-(np.max(det_conf)-np.min(det_conf))*0.5
                            # print class_idx, thresh, thresh_comp

                        elif second_thresh=='otsu_per_class':
                            threshes = [ -5.318356728553772 , -4.973605537414551 , -5.033505606651307 , -4.882340455055237 , -5.094739747047424 , -4.715932703018188 , -5.515528774261474 , -3.4971811771392822 , -4.985876345634461 , -4.548909533023835 , -5.307925057411193 , -5.285209989547729 , -5.345892333984375 , -4.907059001922608 , -5.419295883178711 , -4.888721156120301 , -5.3396998882293705 , -5.426804924011231 , -5.097824430465698 , -5.448168039321899]
                            thresh = threshes[class_idx]
                        elif second_thresh == 'otsu_per_class_gt':
                            # threshes = [ 1.6653436422348018 , 1.6350783109664917 , -3.241373896598816 , 0.5096382975578302 , 0.8811956644058228 , -0.4336070775985714 , -0.9082340478897097 , -4.3277980327606205 , -4.335655236244202 , -3.006533443927765 , -4.732422733306885 , 1.7015709877014151 , -5.088234949111939 , 1.6076517105102544 , 1.8010571002960205 , -4.549804472923279 , -4.643194222450257 , 1.4283066987991329 , 1.7541022300720206 , -5.030483508110047  ]
                            threshes = [ -11.493487238883972 , -11.715554904937743 , -11.567303442955016 , -0.29134426116943235 , -9.127959728240967 , -11.774006605148315 , -10.586105346679688 , -10.005760955810548 , -4.0798876762390135 , -6.865327882766723 , -6.464476490020751 , -10.147429347038269 , -10.552372121810912 , -10.41338300704956 , -10.107991337776184 , -8.000696849823 , -3.2308075666427607 , -8.325735449790955 , 0.0018072128295898438 , -10.581804275512695 , -9.085333108901978 , -4.380960249900818 , -7.76091194152832 , -7.8100281953811646 , -8.015686750411987 , -10.228572845458984 , -8.89403510093689 , -10.035913586616516 , -6.485160398483275 , 1.5468014478683454 , -10.466089797019958 , -10.352537393569946 , -11.080861330032349 , -4.079550766944886 , -9.870902061462402 , -11.705468130111694 , -11.277372598648071 , -11.920127391815186 , -8.292747259140015 , -8.435733556747437 , -3.6880436658859246 , -9.962094902992249 , -10.136343955993652 , -10.096235990524292 , 1.8210916519165021 , -9.617901945114134 , -0.6553510189056384 , -8.892017841339111 , -9.910322070121765 , -5.507466173171997 , -9.935795688629149 , -0.40978360176086426 , -8.750633478164673 , -10.75163221359253 , -11.562832188606261 , -10.59936091899872 , -12.900600695610045 , -8.079093027114867 , -11.099324131011961 , -2.6412230968475336 , -11.352792143821716 , 1.7469805479049691 , -0.3136753559112533 , -12.025232648849489 , -9.967067241668701 , -11.40596890449524 , -12.218655109405518 , -11.722469282150268 , -8.90257215499878 , -8.502108573913574 , -4.826782011985779 , -12.014808773994446 , -10.319413661956787 , -11.360230445861816 , -9.46954345703125 , -8.555493474006653 , -12.345401000976562 , -10.13498330116272 , -6.36683886051178 , -10.883997225761412 , -7.914173603057861 , -12.55906960964203 , -10.209265351295471 , -8.52416455745697 , -9.731002688407898 , -11.799030280113222 , -10.429827213287354 , -11.352923154830933 , -9.692192912101746 , -4.177016687393189 , -10.813226962089537 , -10.608894610404967 , -11.23424062728882 , -9.35195436477661 , -11.124334073066713 , -10.96947717666626 , -10.344857096672058 , -0.38236346244812136 , -11.854193639755248 , -4.933973860740662  ]
                            thresh = threshes[class_idx]
                            # print thresh, class_idx
                        elif second_thresh =='max_-4':
                            thresh = np.max(det_conf)-(np.max(det_conf)-np.min(det_conf))*0.5
                            thresh = min(-4,thresh)
                        elif second_thresh.startswith('smax'):
                            thresh = np.max(det_conf)-(np.max(det_conf)-np.min(det_conf))*0.5
                    elif second_thresh<=-1:
                        thresh = second_thresh
                    elif second_thresh<=0:
                        thresh = second_thresh
                    else:
                        # print 'else'
                        thresh = np.max(det_conf)-(np.max(det_conf)-np.min(det_conf))*second_thresh
                        print thresh
                    bin_second_thresh = det_conf>thresh
         

                    det_conf, det_time_intervals = merge_detections(bin_second_thresh, det_conf, det_time_intervals_meta)
                    
                    det_vid_names.extend([det_vid_names_ac[idx_test]]*det_conf.shape[0])
                    det_events_class.extend([class_idx]*det_conf.shape[0])
                    # print 'hello im here'
                    det_conf_all.append(det_conf)
                    det_time_intervals_all.append(det_time_intervals)
                
            idx_test +=1

    
    if test_method=='wtalc':
        aps, dmap, iou, class_names = wtalc.getDetectionMAP(predictions, pmfs, det_vid_names, first_thresh, second_thresh)
        print len(aps), len(aps[0])

        for idx_curr, arr_curr in enumerate(aps):
            aps[idx_curr].append(dmap[idx_curr])
        # aps = [arr_curr.append(dmap[idx_curr]) ]
        aps = np.array(aps).T
        aps[:-1,:] = aps[:-1,:]*100
        
        # for ap_curr in aps:
        #     print len(ap_curr)

        class_names.append('Average')
        et.print_overlap(aps, class_names, iou, log_arr)
    elif test_method.startswith('top'):
        num_top = int(test_method.split('_')[1])
        # print len(predictions), type(predictions), type(predictions[0]), predictions[0].shape
        # print len(det_vid_names)
        # predictions = np.array(predictions)
        # det_vid_names = np.array(det_vid_names)
        # np.savez('../scratch/temp.npz',predictions = predictions, det_vid_names= det_vid_names)
        # raw_input()
        if dataset=='ucf':
            class_list = globs.class_names

        precisions, ap, class_names = wtalc.getTopKPrecRecall(num_top, predictions, det_vid_names, class_list)
        iou = [1., 1.]
        for idx,p in enumerate(precisions):
            p.append(ap[idx]) 
        aps = np.array(precisions).T
        aps[:-1,:] = aps[:-1,:]*100
        class_names.append('Average')
        et.print_overlap(aps, class_names, iou, log_arr)

        # iou = [1.]
        # precisions.append(ap)
        # aps = np.array(precisions)[np.newaxis,:]
        # aps[:-1,:] = aps[:-1,:]*100
        # class_names.append('Average')
        # et.print_overlap(aps, class_names, iou, log_arr)

        # for c, p, r in zip(precision, recall, class_names):
        #     str_print = c+'\t%.2f\t%.2f' % (precision, recall)
        #     print str_print
        # raw_input()
    elif test_method =='best_worst_dot':
        print len(predictions), type(predictions), type(predictions[0]), predictions[0].shape
        print len(det_vid_names)
        print len(out_fs), type(out_fs[0]), out_fs[0].shape
        predictions = np.array(predictions)
        det_vid_names = np.array(det_vid_names)
        out_fs = np.array(out_fs)
        out_file_feats = '../scratch/graph_l1_supervise_W_outF.npz'
        # graph_l1_graphOut_noZero.npz'
        # check_best_worst.npz'
        # graph_nosparse_feats.npz'
        np.savez_compressed(out_file_feats ,predictions = predictions, det_vid_names= det_vid_names, out_fs = out_fs)
        print 'saved', out_file_feats
        raw_input()
    else:
        # print 'hello'
        if save_outfs:
            print 'creating charades det file'
            pc.create_charades_det_file(save_outfs)
            return None
        # pmfs = np.array(pmfs)
        # labels_all = np.array(labels_all)
        # labels_all[labels_all>0] = 1
        # labels_all = np.array(labels_all, dtype = np.int)

        # print pmfs.shape, labels_all.shape, type(labels_all)
        # print pmfs[0,:10]
        # print labels_all[0]
        # print sklearn.metrics.average_precision_score(labels_all, pmfs)
        # print len(pmfs), len(labels_all)
        # # .shape, np.min(labels), np.max(labels)
        # raw_input()

        # threshes_all = np.concatenate(threshes_all,0)
        det_conf = np.concatenate(det_conf_all,axis =0)
        # print np.min(det_conf), np.max(det_conf)
        # bin_keep = det_conf>0.25
        # print det_conf.shape, np.sum(bin_keep), bin_keep.shape

        det_time_intervals = np.concatenate(det_time_intervals_all,axis = 0)
        det_events_class = np.array(det_events_class)
        
        # det_time_intervals = det_time_intervals[bin_keep]
        # det_events_class = det_events_class[bin_keep]
        # det_conf = det_conf[bin_keep]
        # det_vid_names =np.array(det_vid_names)[bin_keep]
        # det_vid_names = list(det_vid_names)
        # print globs.class_names

        # class_idx = [7,9,12,21,22,23,24,26,31,33,36,40,45,51,68,79,85,92,93,97]
        # if matlab_arr is not None:
        #     for idx_det in range(len(det_vid_names)):
        #         vid_curr = det_vid_names[idx_det]
        #         start_time = det_time_intervals[idx_det,0]
        #         end_time = det_time_intervals[idx_det,1]
        #         conf = det_conf[idx_det]
        #         class_curr = det_events_class[idx_det]
        #         class_idx_curr = class_idx[class_curr]
        #         str_curr = []
        #         str_curr.append(vid_curr+'.mpeg')
        #         str_curr.append('%.2f'%start_time)
        #         str_curr.append('%.2f'%end_time)
        #         str_curr.append('%d'%class_idx_curr)
        #         str_curr.append('%.10f'%conf)
        #         str_curr = ' '.join(str_curr)
        #         matlab_arr.append(str_curr)


        aps = et.test_overlap(det_vid_names, det_conf, det_time_intervals,det_events_class,log_arr = log_arr, dataset = dataset)
    
    model.train()
    
    # test_dataloader.dataset.feature_limit = limit_bef
    
    return aps


def test_model_overlap_pairs(model, test_dataloader, criterion, log_arr,first_thresh , second_thresh , bin_trim = None , multibranch =1, branch_to_test = -1,dataset = 'ucf', save_outfs = None):
    fps_stuff = 10./30.
    # print 'FIRST THRESH', first_thresh
    # print 'SECOND THRESH', second_thresh
    # raw_input()
    # out_dir = '../scratch/graph_2_nononlin_b'
    # util.mkdir(out_dir)
    # print out_dir
    # raw_input()

    model.eval()
    model = model.cuda()
    model_name = model.__class__.__name__.lower()
    criterion_str = criterion.__class__.__name__.lower()

    preds = []
    labels_all = []

    det_vid_names_ac = [os.path.split(line.split(' ')[0])[1][:-4] for line in test_dataloader.dataset.files]
    det_vid_names_ac = det_vid_names_ac[::20]

    outs = []
    min_all = None
    max_all = None

    det_events_class = []
    det_time_intervals_all = []
    det_conf_all = []
    det_vid_names = []
    idx_test = 0

    threshes_all = []
    for num_iter_test,batch in enumerate(test_dataloader):
        print num_iter_test
        print idx_test

        samples = batch['features']
        labels = batch['label'].cuda()
        
        preds_mini = []
        
        assert len(samples)==20
        chunk_size = 2
        
        sample_chunks = [samples[i:i + chunk_size] for i in xrange(0, len(samples), chunk_size)]
        
        if 'perfectg' in model_name or 'cooc' in model_name:
            gt_vecs = batch['gt_vec']
            gt_vecs_chunks = [gt_vecs[i:i + chunk_size] for i in xrange(0, len(gt_vecs), chunk_size)]

        # out_chunk = []
        # pmf_chunk = []
        for idx_sample, sample in enumerate(sample_chunks):
            sample_size = sample[0].size(0)
            

            if multibranch>1:
                if 'perfectg' in model_name or 'cooc' in model_name:
                    out, pmf = model.forward([sample,gt_vecs_chunks[idx_sample]], branch_to_test = branch_to_test)
                else:
                    out, pmf = model.forward(sample, branch_to_test = branch_to_test)
            else:
                if 'perfectg' in model_name or 'cooc' in model_name:
                    out,pmf = model.forward([sample,gt_vecs_chunks[idx_sample]])
                else:    
                    out, pmf = model.forward(sample)


            if 'l1' in criterion_str:
                [pmf, att] = pmf

            out = out[:sample_size]
            pmf = pmf[0].squeeze()
            

            # print out.size(),torch.min(out), torch.max(out)
            # print pmf.size(),torch.min(pmf), torch.max(pmf)

            # raw_input()

            if bin_trim is not None:
                # print out.size()
                out = out[:,np.where(bin_trim)[0]]
                pmf = pmf[np.where(bin_trim)[0]]


                # print out.size()
                # raw_input()
            if second_thresh>=0 and branch_to_test!=-2 and branch_to_test!=-4 and branch_to_test!=-5:
                out = torch.nn.functional.softmax(out,dim = 1)

            start_seq = np.array(range(0,out.shape[0]))*fps_stuff
            end_seq = np.array(range(1,out.shape[0]+1))*fps_stuff
            det_time_intervals_meta = np.concatenate([start_seq[:,np.newaxis],end_seq[:,np.newaxis]],axis=1)
            

            pmf = pmf.data.cpu().numpy()
            out = out.data.cpu().numpy()

            if first_thresh==-1:
                bin_not_keep = labels[idx_sample].data.cpu().numpy()==0
            else:
                bin_not_keep = pmf<first_thresh
            
            # print np.min(pmf), np.max(pmf)
            # print bin_not_keep
            # print labels[idx_sample]
            # print pmf
            # print first_thresh
            # raw_input()


            for class_idx in range(pmf.size):
                if bin_not_keep[class_idx]:
                    continue

                det_conf = out[:,class_idx]
                if second_thresh<0:
                    thresh = 0
                else:
                    thresh = np.max(det_conf)-(np.max(det_conf)-np.min(det_conf))*second_thresh
                bin_second_thresh = det_conf>thresh
                # print thresh, np.sum(bin_second_thresh)


                det_conf, det_time_intervals = merge_detections(bin_second_thresh, det_conf, det_time_intervals_meta)
                # print class_idx, labels[idx_sample][class_idx],np.min(det_conf), np.max(det_conf),len(det_time_intervals)

                # det_time_intervals = det_time_intervals_meta
                
                det_vid_names.extend([det_vid_names_ac[idx_test]]*det_conf.shape[0])
                det_events_class.extend([class_idx]*det_conf.shape[0])
                det_conf_all.append(det_conf)
                det_time_intervals_all.append(det_time_intervals)
            
            # raw_input()

        idx_test +=1

    # threshes_all = np.concatenate(threshes_all,0)
    det_conf = np.concatenate(det_conf_all,axis =0)
    det_time_intervals = np.concatenate(det_time_intervals_all,axis = 0)
    det_events_class = np.array(det_events_class)
    # class_keep = np.argmax(det_conf , axis = 1)

    # np.savez('../scratch/debug_det_graph.npz', det_vid_names = det_vid_names, det_conf = det_conf, det_time_intervals = det_time_intervals, det_events_class = det_events_class)
    # raw_input()

    aps = et.test_overlap(det_vid_names, det_conf, det_time_intervals,det_events_class,log_arr = log_arr, dataset = dataset)
    
    return aps



def test_model(out_dir_train,
                model_num, 
                test_data, 
                batch_size_val = None,
                criterion = nn.CrossEntropyLoss(),
                gpu_id = 0,
                num_workers = 0,
                post_pend = '',
                trim_preds = None,
                first_thresh = 0,
                second_thresh = 0.5, 
                visualize = False,
                det_class = -1, 
                multibranch = 1,
                branch_to_test = -1,
                dataset = 'ucf',
                save_outfs = None,
                test_pair = False,
                test_method = 'original'):
    out_dir_results = os.path.join(out_dir_train,'results_model_'+str(model_num)+post_pend+'_'+str(first_thresh)+'_'+str(second_thresh))
    criterion_str = criterion.__class__.__name__.lower()

    if branch_to_test!=-1:
        out_dir_results = out_dir_results +'_'+str(branch_to_test)

    if test_pair:
        out_dir_results = out_dir_results +'_test_pair'        

    print out_dir_results
    util.mkdir(out_dir_results)

    if save_outfs:
        save_outfs = os.path.join(out_dir_results,'outf')
        util.mkdir(save_outfs)
    
    model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    if branch_to_test>-1:
        append_name = '_'+str(branch_to_test)
    else:
        append_name = ''    

    log_file = os.path.join(out_dir_results,'log'+append_name+'.txt')
    matlab_file = os.path.join(out_dir_results,'dets_for_matlab'+append_name+'.txt')
    out_file = os.path.join(out_dir_results,'aps'+append_name+'.npy')
    log_arr=[]
    matlab_arr = []

    model = torch.load(model_file)
    # print model
    # print model.graph_layers[0].graph_layer
    # return
    if multibranch==1 and branch_to_test>-1:
        model.focus = branch_to_test
    


    if batch_size_val is None:
        batch_size_val = len(test_data)

    # print 'batch_size_val', batch_size_val
    # raw_input()
    
    test_dataloader = torch.utils.data.DataLoader(test_data, 
                        batch_size = batch_size_val,
                        collate_fn = test_data.collate_fn,
                        shuffle = False, 
                        num_workers = num_workers)
    
    torch.cuda.device(gpu_id)
    
    model = model.cuda()
    
    criterion = criterion.cuda()

    if trim_preds is not None:
        bin_trim = np.in1d(np.array(trim_preds[0]),np.array(trim_preds[1]))
        new_trim = np.array(trim_preds[0])[bin_trim]
        old_trim = np.array(trim_preds[1])
        assert np.all(new_trim==old_trim)
    else:
        bin_trim = None
    
    if visualize:
        dir_viz = os.path.join(out_dir_results, '_'.join([str(val) for val in ['viz',det_class, first_thresh, second_thresh, branch_to_test]]))
        util.mkdir(dir_viz)
        if multibranch>1:
            branch_to_test_pass = branch_to_test
        elif branch_to_test<0:
            branch_to_test_pass = branch_to_test
        else:
            branch_to_test_pass = -1
        # print branch_to_test_pass
        # raw_input()
        visualize_dets(model, test_dataloader,  dir_viz,first_thresh = first_thresh, second_thresh = second_thresh,bin_trim = bin_trim,det_class = det_class, branch_to_test = branch_to_test_pass, criterion_str = criterion_str,dataset = dataset)
    elif test_pair:
        aps = test_model_overlap_pairs(model, test_dataloader, criterion, log_arr ,first_thresh = first_thresh, second_thresh = second_thresh, bin_trim = bin_trim, multibranch = multibranch, branch_to_test = branch_to_test,dataset = dataset, save_outfs = save_outfs)
        np.save(out_file, aps)
        util.writeFile(log_file, log_arr)
    else:
        aps = test_model_overlap(model, test_dataloader, criterion, log_arr ,first_thresh = first_thresh, second_thresh = second_thresh, bin_trim = bin_trim, multibranch = multibranch, branch_to_test = branch_to_test,dataset = dataset, save_outfs = save_outfs, test_method = test_method, matlab_arr = matlab_arr)
        np.save(out_file, aps)
        util.writeFile(log_file, log_arr)
        util.writeFile(matlab_file, matlab_arr)


def visualize_sim_mat(out_dir_train,
                model_num, 
                test_data, 
                batch_size_val = None,
                gpu_id = 0,
                num_workers = 0,
                post_pend = '', first_thresh = 0, second_thresh = 0.5, dataset = 'ucf'):
    
    out_dir_results = os.path.join(out_dir_train,'results_model_'+str(model_num)+post_pend+'_'+str(first_thresh)+'_'+str(second_thresh))
    util.mkdir(out_dir_results)
    model_file = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    

    model = torch.load(model_file)

    if batch_size_val is None:
        batch_size_val = len(test_data)

    
    test_dataloader = torch.utils.data.DataLoader(test_data, 
                        batch_size = batch_size_val,
                        collate_fn = test_data.collate_fn,
                        shuffle = False, 
                        num_workers = num_workers)
    
    torch.cuda.device(gpu_id)
    
    model = model.cuda()
    
    dir_viz = os.path.join(out_dir_results, '_'.join([str(val) for val in ['viz_sim_mat']]))
    util.mkdir(dir_viz)


    visualize_sim_mat_inner(model, test_dataloader,  dir_viz, dataset = dataset)
     
            
def visualize_sim_mat_inner(model, test_dataloader, dir_viz, dataset = 'ucf'):

    model.eval()
    model_name = model.__class__.__name__.lower()
    preds = []
    labels_all = []

    det_vid_names_ac = [os.path.split(line.split(' ')[0])[1][:-4] for line in test_dataloader.dataset.files]
    
    outs = []
    min_all = None
    max_all = None

    det_events_class = []
    det_time_intervals_all = []
    det_conf_all = []
    det_vid_names = []
    out_shapes = []
    idx_test = 0

    threshes_all = []
    for num_iter_test,batch in enumerate(test_dataloader):
        samples = batch['features']
        labels = batch['label']

        preds_mini = []
        for idx_sample, sample in enumerate(samples[:100]):
            print idx_sample
            vid_name = det_vid_names_ac[idx_test]
            out_shape_curr = sample.size(0)

            class_idx = np.where(labels[idx_sample].numpy())[0][0]

            if 'perfectg' in model_name:
                sim_mat = model.get_similarity(batch['gt_vec'][idx_sample].cuda())
            else:
                sim_mat = model.get_similarity(sample.cuda())
            # print sim_mat.size()
            sim_mat = sim_mat.data.cpu().numpy()
            dg.save_sim_viz(vid_name, out_shape_curr, sim_mat, class_idx, dir_viz, dataset = dataset)

            idx_test +=1
    
    dg.make_htmls(dir_viz)        


def train_model(out_dir_train,
                train_data,
                test_data,
                batch_size = None,
                batch_size_val = None,
                num_epochs = 100,
                save_after = 20,
                disp_after = 1,
                plot_after = 10,
                test_after = 1,
                lr = 0.0001,
                dec_after = 100, 
                model_name = 'alexnet',
                criterion = nn.CrossEntropyLoss(),
                gpu_id = 0,
                num_workers = 0,
                model_file = None,
                epoch_start = 0,
                network_params = None,
                weight_decay = 0, 
                multibranch = 1):

    util.mkdir(out_dir_train)
    log_file = os.path.join(out_dir_train,'log.txt')
    plot_file = os.path.join(out_dir_train,'loss.jpg')
    
    log_file_writer = open(log_file,'wb')

    plot_file = os.path.join(out_dir_train,'loss.jpg')
    log_arr = []
    plot_arr = [[],[]]
    plot_val_arr =  [[],[]]
    plot_val_acc_arr = [[],[]]
    plot_strs_posts = ['Loss']
    plot_acc_file = os.path.join(out_dir_train,'val_accu.jpg')

    network = models.get(model_name,network_params)

    if model_file is not None:
        network.model = torch.load(model_file)

    model = network.model
    
    if batch_size is None:
        batch_size = len(train_data)

    if batch_size_val is None:
        batch_size_val = len(test_data)

    train_dataloader = torch.utils.data.DataLoader(train_data, 
                        batch_size = batch_size,
                        collate_fn = train_data.collate_fn,
                        shuffle = True, 
                        num_workers = num_workers)
    
    test_dataloader = torch.utils.data.DataLoader(test_data, 
                        batch_size = batch_size_val,
                        collate_fn = test_data.collate_fn,
                        shuffle = False, 
                        num_workers = num_workers)
    
    torch.cuda.device(gpu_id)
    
    model = model.cuda()
    model.train(True)
    model_str = str(model)

    log_file_writer.write(model_str+'\n')
    print model_str
    # out_file = os.path.join(out_dir_train,'model_-1.pt')
    # print 'saving',out_file
    # torch.save(model,out_file)    
    # return
    criterion_str = criterion.__class__.__name__.lower()

    optimizer = torch.optim.Adam(network.get_lr_list(lr),weight_decay=weight_decay)

    if dec_after is not None:
        print dec_after
        if dec_after[0] is 'step':
            print dec_after
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=dec_after[1], gamma=dec_after[2])
        elif dec_after[0] is 'exp':
            print 'EXPING',dec_after
            exp_lr_scheduler = Exp_Lr_Scheduler(optimizer,epoch_start*len(train_dataloader),[lr_curr for lr_curr in lr if lr_curr!=0],dec_after[1],dec_after[2],dec_after[3])
    
    criterion = criterion.cuda()

    for num_epoch in range(epoch_start,num_epochs):

        plot_arr_epoch = []
        for num_iter_train,batch in enumerate(train_dataloader):

            samples = batch['features']
            labels = batch['label'].cuda()

            if 'centerloss' in model_name:
                preds,extra = model.forward(samples, labels)
                labels = [labels]+extra

            else:
                preds = []
                if multibranch>1:
                    preds = [[] for i in range(multibranch)]
                elif 'l1' in criterion_str:
                    preds = [[],[]]
                
                for idx_sample, sample in enumerate(samples):

                    if 'alt_train' in model_name and 'multi_video' in model_name:
                        out,preds = model.forward(samples, epoch_num = num_epoch)
                        break
                    elif ('cooc' in model_name or 'perfectG' in model_name) and 'multi_video' in model_name:
                        out,preds = model.forward([samples,batch['gt_vec']])
                        break
                    elif 'alt_train' in model_name:
                        out,pmf = model.forward(sample.cuda(), epoch_num=num_epoch)
                    elif 'perfectG' in model_name:
                        out,pmf = model.forward([sample.cuda(),batch['gt_vec'][idx_sample].cuda()])
                    elif 'multi_video' in model_name:
                        out,preds = model.forward(samples)
                        break
                    else:    
                        out,pmf = model.forward(sample.cuda())

                    if multibranch>1:
                        for idx in range(len(pmf)):
                            preds[idx].append(pmf[idx].unsqueeze(0))
                    elif 'l1' in criterion_str:
                        preds[0].append(pmf[0].unsqueeze(0))
                        preds[1].append(pmf[1])
                    else:
                        preds.append(pmf.unsqueeze(0))
                

                if 'l1' in criterion_str:
                    [preds, att] = preds


                if multibranch>1:
                    preds = [torch.cat(preds_curr,0) for preds_curr in preds]        
                else:
                    preds = torch.cat(preds,0)        

            if 'l1' in criterion_str:
                loss = criterion(labels, preds,att)
            else:
                loss = criterion(labels, preds)
            loss_iter = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # model.printGraphGrad()
            # grad_rel = model.graph_layers[0].graph_layer.weight.grad
            # print torch.min(grad_rel).data.cpu().numpy(), torch.max(grad_rel).data.cpu().numpy()

            # print criterion.__class__.__name__.lower()
            # if 'centerloss' in criterion.__class__.__name__.lower():
            #     criterion.backward()
            
            num_iter = num_epoch*len(train_dataloader)+num_iter_train
            
            plot_arr_epoch.append(loss_iter)
            str_display = 'lr: %.6f, iter: %d, loss: %.4f' %(optimizer.param_groups[-1]['lr'],num_iter,loss_iter)
            log_arr.append(str_display)
            print str_display

        plot_arr[0].append(num_epoch)
        plot_arr[1].append(np.mean(plot_arr_epoch))

        if num_epoch % plot_after== 0 and num_iter>0:
            
            for string in log_arr:
                log_file_writer.write(string+'\n')
            
            log_arr = []

            if len(plot_val_arr[0])==0:
                visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
            else:
                
                lengend_strs = [pre_str+plot_str_posts for pre_str in ['Train ','Val '] for plot_str_posts in plot_strs_posts]

                # print len(plot_arr),len(plot_val_arr)
                plot_vals = [(arr[0],arr[1]) for arr in [plot_arr]+[plot_val_arr]]
                # print plot_vals
                # print lengend_strs
                visualize.plotSimple(plot_vals,out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=lengend_strs)

                visualize.plotSimple([(plot_val_acc_arr[0],plot_val_acc_arr[1])],out_file = plot_acc_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Accuracy',legend_entries=['Val'])
                


        if (num_epoch+1) % test_after == 0 or num_epoch==0:

            accuracy, loss_iter = test_model_core(model, test_dataloader, criterion, log_arr, multibranch  = multibranch)
            
            # num_iter = num_epoch*len(train_dataloader)+len(train_dataloader)
            
            plot_val_arr[0].append(num_epoch); plot_val_arr[1].append(loss_iter)
            plot_val_acc_arr[0].append(num_epoch); plot_val_acc_arr[1].append(accuracy)
           

        if (num_epoch+1) % save_after == 0 or num_epoch==0:
            out_file = os.path.join(out_dir_train,'model_'+str(num_epoch)+'.pt')
            print 'saving',out_file
            torch.save(model,out_file)

        if dec_after is not None and dec_after[0]=='reduce':
            # exp_lr_scheduler
            if accuracy>=best_val:
                best_val = accuracy
                out_file_best = os.path.join(out_dir_train,'model_bestVal.pt')
                print 'saving',out_file_best
                torch.save(model,out_file_best)            
            exp_lr_scheduler.step(loss_iter)

        elif dec_after is not None and dec_after[0]!='exp':
            exp_lr_scheduler.step()
    
    out_file = os.path.join(out_dir_train,'model_'+str(num_epoch)+'.pt')
    print 'saving',out_file
    torch.save(model,out_file)
    
    for string in log_arr:
        log_file_writer.write(string+'\n')
                
    if len(plot_val_arr[0])==0:
        visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
    else:
        
        lengend_strs = [pre_str+plot_str_posts for pre_str in ['Train ','Val '] for plot_str_posts in plot_strs_posts]

        # print len(plot_arr),len(plot_val_arr)
        plot_vals = [(arr[0],arr[1]) for arr in [plot_arr]+[plot_val_arr]]
        # print plot_vals
        # print lengend_strs
        visualize.plotSimple(plot_vals,out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=lengend_strs)

        visualize.plotSimple([(plot_val_acc_arr[0],plot_val_acc_arr[1])],out_file = plot_acc_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Accuracy',legend_entries=['Val'])

    log_file_writer.close()









    