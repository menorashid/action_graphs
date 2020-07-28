import sys
sys.path.append('./')
import os
import numpy as np
import scipy.io
import glob
from helpers import util, visualize
import globals as globals

def interval_single_overlap_val_seconds(i1, i2, norm_type = 0):
    i1 = [np.min(i1), np.max(i1)]
    i2 = [np.min(i2), np.max(i2)]

    ov = 0
    if norm_type<0:
        ua = 1
    elif norm_type==1:
        ua = i1[1] - i1[0]
    elif norm_type==2:
        ua = i2[1] - i2[0]
    else:
        bu = [min(i1[0], i2[0]), max(i1[1], i2[1])]
        ua = bu[1] - bu[0]

    bi = [max(i1[0], i2[0]), min(i1[1], i2[1])]
    iw = bi[1] - bi[0]

    if iw>0:
        if norm_type<0:
            ov = iw
        else:
            ov = iw/float(ua)

    return ov

def interval_overlap_val_seconds(i1, i2, norm_type=0):
    ov = np.zeros((i1.shape[0],i2.shape[0]))

    for i in range(i1.shape[0]):
        for j in range(i2.shape[0]):
            ov[i,j] = interval_single_overlap_val_seconds(i1[i,:], i2[j,:], norm_type)
            
    return ov


def pr_ap(rec, prec):
    ap = 0
    recall_points = np.arange(0,1.1,0.1)
    # print recall_points
    for t in recall_points:
        p = prec[rec>=t]
        if p.size ==0:
            p=0
        else:
            p= np.max(p)
        
        ap = ap + p/float(recall_points.size)
        # print t,p,ap

    return ap





def test_ov():
    loaded = scipy.io.loadmat('../TH14evalkit/i1_i2.mat')
    i1 = loaded['i1']
    i2 = loaded['i2']
    ov_org = loaded['ov']
    print i1.shape,i2.shape,ov_org.shape
    ov = interval_overlap_val_seconds(i1, i2,2)
    print ov.shape
    print ov
    print ov_org
    print np.abs(ov_org-ov)

def test_pr_ap():
    loaded = scipy.io.loadmat('../TH14evalkit/rec_prec.mat')
    rec = loaded['rec']
    prec = loaded['prec']
    ap_org = loaded['ap'][0][0]
    print rec.shape, prec.shape, ap_org.shape
    ap = pr_ap(rec, prec)
    print ap
    print ap_org
    assert np.all(ap_org-ap)    

def event_det_pr(det_vid_names, det_time_intervals, det_class_names, det_conf, gt_vid_names, gt_time_intervals, gt_class_names, class_name, overlap_thresh):

    video_names = np.unique(det_vid_names+gt_vid_names)
    num_pos = gt_class_names.count(class_name)
    # print np.unique(gt_class_names).shape
    # print class_name,num_pos
    assert num_pos>0

    gt_class_names = np.array(gt_class_names)
    gt_vid_names = np.array(gt_vid_names)
    det_vid_names = np.array(det_vid_names)
    det_class_names = np.array(det_class_names)

    ind_gt_class = gt_class_names==class_name
    ind_amb_class = gt_class_names=='Ambiguous'
    ind_det_class = det_class_names==class_name
    # print det_class_names[:10]

    tp_conf = []
    fp_conf = []
    for idx_video_name, video_name in enumerate(video_names):
        # print video_name
        gt = np.logical_and(gt_vid_names==video_name, ind_gt_class)
        amb = np.logical_and(gt_vid_names==video_name, ind_amb_class)
        det = np.logical_and(det_vid_names==video_name, ind_det_class)
        
        # print 'np.sum(det_vid_names==video_name)', np.sum(det_vid_names==video_name)
        # print 'np.sum(ind_det_class)', np.sum(ind_det_class)
        # det = det_vid_names==video_name

        if np.sum(det)>0:
            
            ind_free = np.ones((np.sum(det),))
            ind_amb = np.zeros((np.sum(det),))
            

            det_conf_curr = det_conf[det]
            idx_sort = np.argsort(det_conf_curr)[::-1]
            idx_sort = np.where(det)[0][idx_sort]

            det_conf_curr = det_conf[idx_sort]
            
            det_vid_names_curr = det_vid_names[idx_sort]
            assert  np.unique(det_vid_names_curr).size==1 and det_vid_names_curr[0]== video_name

            if np.sum(gt)>0:
                
                # print det_time_intervals[idx_sort,:]
                # print gt_time_intervals[gt,:]
                ov = interval_overlap_val_seconds(gt_time_intervals[gt,:], det_time_intervals[idx_sort,:])
                
                for k in range(ov.shape[0]):    
                    ind = np.where(ind_free>0)[0]
                    if ind.size==0:
                        continue

                    im = np.argmax(ov[k,ind])
                    vm = ov[k,ind][im]
                    if vm>overlap_thresh:
                        ind_free[ind[im]]=0

            if np.sum(amb)>0:
                ov_amb = interval_overlap_val_seconds(gt_time_intervals[amb,:], det_time_intervals[idx_sort,:])
                ind_amb = np.sum(ov_amb,0)

            tp_conf.extend(list(det_conf_curr[np.where(ind_free==0)[0]]))
            fp_conf.extend(list(det_conf_curr[np.where(np.logical_and(ind_free==1, ind_amb==0))[0]]))
    # raw_input()

    # print tp_conf, fp_conf
    conf = np.array([tp_conf + fp_conf,list(np.ones((len(tp_conf),))) + list(2*np.ones((len(fp_conf),)))])
    idx_sort = np.argsort(conf[0,:])[::-1]
    tp = np.cumsum(conf[1,idx_sort]==1)
    fp = np.cumsum(conf[1,idx_sort]==2)
    rec = tp/float(num_pos)
    prec = tp/(fp+tp).astype(float)
    ap = pr_ap(rec, prec)
    return rec, prec, ap
        
def load_activitynet_gt(train,select=False):
    dir_gt = '../data/activitynet/gt_npys'
    if select:
        if train:
            anno_file = os.path.join(dir_gt, 'train_select.npz')
        else:
            print 'VAL SELECT'
            anno_file = os.path.join(dir_gt, 'val_select.npz')
    else:
        if train:
            anno_file = os.path.join(dir_gt, 'train.npz')
        else:
            anno_file = os.path.join(dir_gt, 'val_pruned.npz')


    data = np.load(anno_file)
    gt_class_names = data['gt_class_names']
    gt_vid_names = data['gt_vid_names']
    gt_time_intervals = data['gt_time_intervals']
    gt_class_names = [str(val) for val in gt_class_names]
    gt_vid_names = [str(val) for val in gt_vid_names]

    # print type(gt_class_names),type(gt_class_names[0]),len(gt_class_names),gt_class_names[0]
    # print type(gt_vid_names),type(gt_vid_names[0]),len(gt_vid_names),gt_vid_names[0]
    # print gt_time_intervals.shape,gt_time_intervals[0]
    
    return gt_vid_names, gt_class_names, gt_time_intervals

def load_ucf_gt(train):
    class_name = 'BaseballPitch'
    if train:
        mat_file = os.path.join('../TH14evalkit',class_name+'.mat')
    else:
        mat_file = os.path.join('../TH14evalkit','mat_files', class_name+'_test.mat')

    loaded = scipy.io.loadmat(mat_file)
    
    gt_vid_names = loaded['gtvideonames'][0]
    gt_class_names = loaded['gt_events_class'][0]
    gt_time_intervals = loaded['gt_time_intervals'][0]

    arr_meta = [gt_vid_names, gt_class_names]

    arr_out = []
    for arr_curr in arr_meta:
        arr_curr = [str(a[0]) for a in arr_curr]
        arr_out.append(arr_curr)

    [gt_vid_names, gt_class_names] = arr_out
    gt_time_intervals = np.array([a[0] for a in gt_time_intervals])
    return gt_vid_names, gt_class_names, gt_time_intervals

def load_multithumos_gt(train):
    anno_dir = '../data/multithumos'
    if train:
        mat_file = os.path.join(anno_dir, 'train.npz')
    else:
        mat_file = os.path.join(anno_dir, 'test.npz')

    loaded = np.load(mat_file)
    
    gt_vid_names = list(loaded['gt_vid_names'])
    gt_class_names = list(loaded['gt_class_names'])
    gt_time_intervals = loaded['gt_time_intervals']

    arr_meta = [gt_vid_names, gt_class_names]

    # # arr_out = []
    # # for arr_curr in arr_meta:
    # #     arr_curr = [str(a[0]) for a in arr_curr]
    # #     arr_out.append(arr_curr)

    # # [gt_vid_names, gt_class_names] = arr_out
    # gt_time_intervals = np.array([a[0] for a in gt_time_intervals])
    return gt_vid_names, gt_class_names, gt_time_intervals

def load_charades_gt(train):
    anno_dir = '../data/charades'
    if train:
        mat_file = os.path.join(anno_dir, 'annos_train.npz')
    else:
        mat_file = os.path.join(anno_dir, 'annos_test.npz')

    loaded = np.load(mat_file)
    # print loaded.keys()
    # raw_input()
    gt_vid_names = list(loaded['gt_vid_names'])
    gt_class_names = list(loaded['gt_class_names'])
    gt_time_intervals = loaded['gt_time_intervals']

    return gt_vid_names, gt_class_names, gt_time_intervals

def print_overlap(aps,class_names,overlap_thresh_all, log_arr):
    
    # aps = np.array(aps)
    # print aps.shape
    assert aps.shape[0]==len(class_names)
    assert aps.shape[1]==len(overlap_thresh_all)
    
    
    str_print = '\t'.join(['Overlap\t']+['%.1f' % ov for ov in overlap_thresh_all])
    print str_print
    log_arr.append(str_print)

    for idx_class_name, class_name in enumerate(class_names):
        str_print = [class_name+'\t' if len(class_name)<8 else class_name]+['%.2f' % ap_curr for ap_curr in aps[idx_class_name,:]]
        str_print = '\t'.join(str_print)
    
        print str_print
        log_arr.append(str_print)
    
    # aps[-1,:]= np.mean(aps[:len(class_names),:],0)
    # class_names_curr = class_names[:]
    # class_names_curr.append('Average')
    
    # print aps
    # str_print = '\t'.join(['%.1f' % ov for ov in overlap_thresh_all])
    # log_arr.append(str_print)
    # # print log_arr[-1]
    # for idx_class_name, class_name in enumerate(class_names):
    #     # if len(class_name)<8 and dataset=='ucf':
    #     #     class_name +='\t'
    #     str_print = [class_name]+['%.2f' % ap_curr for ap_curr in aps[idx_class_name,:]]
    #     str_print = '\t'.join(str_print)
    #     log_arr.append(str_print)
    
    # print log_arr[-1]

    return aps


def test_overlap(det_vid_names_all, det_conf_all, det_time_intervals_all, second_thresh, train=False, log_arr = [], dataset = 'ucf'):
    
    print dataset

    if dataset =='ucf' or dataset.startswith('ucf_cooc'):
        # class_names = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']
        # class_names.sort()
        class_names = globals.class_names_ucf
        gt_vid_names, gt_class_names, gt_time_intervals = load_ucf_gt(train)
        aps = np.zeros((len(class_names)+1,5))
        overlap_thresh_all = np.arange(0.1,0.6,0.1)
        fps_stuff = 16./25.
    elif dataset =='multithumos':
        class_names = globals.class_names_multithumos
        gt_vid_names, gt_class_names, gt_time_intervals = load_multithumos_gt(train)
        
        overlap_thresh_all = np.arange(0.1,0.2,0.1)
        aps = np.zeros((len(class_names)+1,1))
        fps_stuff = 16./25.
    elif dataset.startswith('charades'):
        class_names = globals.class_names_charades
        gt_vid_names, gt_class_names, gt_time_intervals = load_charades_gt(train)
        overlap_thresh_all = np.arange(0.1,0.2,0.1)
        aps = np.zeros((len(class_names)+1,1))
        fps_stuff = 16./25.
    elif dataset =='activitynet':
        class_names = globals.class_names_activitynet
        gt_vid_names, gt_class_names, gt_time_intervals = load_activitynet_gt(train)
        overlap_thresh_all = np.array([0.5,0.7,0.9])
        # overlap_thresh_all = np.arange(0.1,0.6,0.1)
        aps = np.zeros((len(class_names)+1,overlap_thresh_all.size))
    elif dataset =='activitynet_select':
        class_names = globals.class_names_activitynet_select
        gt_vid_names, gt_class_names, gt_time_intervals = load_activitynet_gt(train,True)
        overlap_thresh_all = np.array([0.5,0.7,0.9])
        aps = np.zeros((len(class_names)+1,overlap_thresh_all.size))
    elif dataset=='ucf_untf':
        class_names = globals.class_names_ucf
        gt_vid_names, gt_class_names, gt_time_intervals = load_ucf_gt(train)
        aps = np.zeros((len(class_names)+1,5))
        overlap_thresh_all = np.arange(0.1,0.6,0.1)
        fps_stuff = 10./30.
    else:
        raise ValueError('Problem. '+dataset+' not valid')

    # print 'fps', fps_stuff
    # raw_input()
    
    str_print = '\t'.join(['Overlap\t']+['%.1f' % ov for ov in overlap_thresh_all])
    print str_print
    for idx_class_name, class_name in enumerate(class_names):
        # if idx_class_name<6:
        #     continue

        

        bin_keep = second_thresh == idx_class_name
        det_conf = det_conf_all[bin_keep]
        det_time_intervals = det_time_intervals_all[bin_keep]
        det_vid_names = list(np.array(det_vid_names_all)[bin_keep])
        det_class_names = [class_name]*det_conf.shape[0]

        
        # det_conf = det_conf_all[:,idx_class_name]
        # bin_keep = det_conf>=second_thresh[:,idx_class_name]
        # det_time_intervals = det_time_intervals_all[bin_keep,:]
        # det_vid_names = list(np.array(det_vid_names_all)[bin_keep])
        # det_conf = det_conf[bin_keep]
        # det_class_names = [class_name]*det_conf.shape[0]
        

        # bin_keep = class_keep == idx_class_name
        # det_time_intervals = det_time_intervals_all[bin_keep,:]
        # det_vid_names = list(np.array(det_vid_names_all)[bin_keep])
        # det_conf = det_conf_all[bin_keep]
        # det_class_names = [class_name]*det_conf.shape[0]
        
        for idx_overlap_thresh, overlap_thresh in enumerate(overlap_thresh_all):
            rec, prec, ap = event_det_pr(det_vid_names, det_time_intervals, det_class_names, det_conf, gt_vid_names, gt_time_intervals, gt_class_names, class_name, overlap_thresh)
            aps[idx_class_name, idx_overlap_thresh] = ap*100
        
        str_print = [class_name+'\t' if len(class_name)<8 else class_name]+['%.2f' % ap_curr for ap_curr in aps[idx_class_name,:]]
        str_print = '\t'.join(str_print)
        print str_print
    
    aps[-1,:]= np.mean(aps[:len(class_names),:],0)
    # class_names_curr = class_names[:]
    # class_names_curr.append('Average')
    
    # print aps
    str_print = '\t'.join(['%.1f' % ov for ov in overlap_thresh_all])
    log_arr.append(str_print)
    # print log_arr[-1]
    for idx_class_name, class_name in enumerate(class_names+['Average']):
        if len(class_name)<8 and dataset=='ucf':
            class_name +='\t'
        str_print = [class_name]+['%.2f' % ap_curr for ap_curr in aps[idx_class_name,:]]
        str_print = '\t'.join(str_print)
        log_arr.append(str_print)
    
    print log_arr[-1]

    return aps



def test_event_det_pr():
    # loaded = scipy.io.loadmat('../TH14evalkit/gt_det_stuff.mat')
    class_names = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']
    
    gt_ov = util.readLinesFromFile('analysis/test.txt')
    gt_ov = [float(val) for val in gt_ov]
    diffs = []
    idx_ov = 0

    for class_name in class_names:
        print class_name
        mat_file = os.path.join('../TH14evalkit',class_name+'.mat')
        loaded = scipy.io.loadmat(mat_file)
        gt_vid_names = loaded['gtvideonames'][0]
        gt_time_intervals = loaded['gt_time_intervals'][0]
        gt_class_names = loaded['gt_events_class'][0]
        
        det_vid_names = loaded['detvideonames'][0]
        det_class_names = loaded['det_events_class'][0]
        det_time_intervals = loaded['det_time_intervals'][0]
        det_conf = loaded['det_conf'][0]

        # class_name = 'BaseballPitch'
        # overlap_thresh = 0.1

        arr_meta = [gt_vid_names,det_vid_names,gt_class_names,det_class_names]

        arr_out = []
        for arr_curr in arr_meta:
            arr_curr = [str(a[0]) for a in arr_curr]
            arr_out.append(arr_curr)
        
        [gt_vid_names,det_vid_names,gt_class_names,det_class_names] = arr_out

        gt_time_intervals = np.array([a[0] for a in gt_time_intervals])
        det_time_intervals = np.array([a[0] for a in det_time_intervals])
        det_conf = np.array([a[0][0] for a in det_conf])

        for overlap_thresh in np.arange(0.1,0.6,0.1):
            rec, prec, ap = event_det_pr(det_vid_names, det_time_intervals, det_class_names, det_conf, gt_vid_names, gt_time_intervals, gt_class_names, class_name, overlap_thresh)
            print ap, overlap_thresh
            diffs.append(ap - gt_ov[idx_ov])
            idx_ov+=1
    
    diffs = np.abs(np.array(diffs))
    print np.mean(diffs), np.min(diffs), np.max(diffs)


def viz_overlap(out_dir_meta, det_vid_names, det_conf_all, det_time_intervals_all, det_events_class_all ,out_shapes):
    
    class_names = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']
    class_names.sort()

    aps = np.zeros((len(class_names)+1,5))
    overlap_thresh_all = np.arange(0.1,0.6,0.1)
    
    for idx_class_name, class_name in enumerate(class_names):
        # if idx_class_name<6:
        #     continue
        out_dir = os.path.join(out_dir_meta,class_name)
        util.mkdir(out_dir)

        mat_file = os.path.join('../TH14evalkit','mat_files', class_name+'_test.mat')

        loaded = scipy.io.loadmat(mat_file)
        
        gt_vid_names_all = loaded['gtvideonames'][0]
        gt_class_names = loaded['gt_events_class'][0]

        gt_time_intervals = loaded['gt_time_intervals'][0]
        
        arr_meta = [gt_vid_names_all, gt_class_names]
        arr_out = []
        for arr_curr in arr_meta:
            arr_curr = [str(a[0]) for a in arr_curr]
            arr_out.append(arr_curr)

        [gt_vid_names_all, gt_class_names] = arr_out
        gt_time_intervals_all = np.array([a[0] for a in gt_time_intervals])

        gt_vid_names = list( np.unique(np.array(gt_vid_names_all)[np.array(gt_class_names)==class_name]))
        det_vid_names = np.array(det_vid_names)
        # print len(det_vid_names)
        # print np.unique(det_vid_names).shape
        # print gt_vid_name
        # print len(gt_vid_names)

        for gt_vid_name in gt_vid_names:
            bin_keep = det_vid_names == gt_vid_name
            # if np.sum(bin_keep):
            #    print idx_class_name, np.sum(bin_keep) 
            # print gt_vid_name
            # print np.sum(det_events_class_all==idx_class_name)
            # raw_input()
            bin_keep = np.logical_and(bin_keep, det_events_class_all==idx_class_name)
            if np.sum(bin_keep)==0:
                print 'Continuing'
                continue
            gt_time_intervals = gt_time_intervals_all[np.array(gt_vid_names_all)==gt_vid_name]
            
            det_conf = det_conf_all[bin_keep]
            det_time_intervals = det_time_intervals_all [bin_keep,:]
            
            out_shape_curr = out_shapes[bin_keep]
            assert len(np.unique(out_shape_curr))==1
            out_shape_curr = np.unique(out_shape_curr)[0]
            # print out_shape_curr

            det_time_intervals_merged = det_time_intervals
            
            det_times = det_time_intervals[:,0]

            det_times = np.array(range(0,out_shape_curr+1))*16./25.
            
            
            gt_vals = np.zeros(det_times.shape)
            # print gt_time_intervals.shape
            # print det_times.shape

            for gt_time_curr in gt_time_intervals:
                idx_start = np.argmin(np.abs(det_times-gt_time_curr[0]))
                idx_end = np.argmin(np.abs(det_times-gt_time_curr[1]))
                gt_vals[idx_start:idx_end] = np.max(det_conf)

            det_vals = np.zeros(det_times.shape)
            for idx_det_time_curr, det_time_curr in enumerate(det_time_intervals_merged):
                idx_start = np.argmin(np.abs(det_times-det_time_curr[0]))
                idx_end = np.argmin(np.abs(det_times-det_time_curr[1]))
                det_vals[idx_start:idx_end] = det_conf[idx_det_time_curr]
            
            out_file_curr = os.path.join(out_dir,gt_vid_name+'.jpg')

            visualize.plotSimple([(det_times,det_vals),(det_times,gt_vals)],out_file = out_file_curr,title = 'det conf over time',xlabel = 'time',ylabel = 'det conf',legend_entries=['Det','GT'])
        
        visualize.writeHTMLForFolder(out_dir)


def viz_overlap_multi(out_dir_meta, det_conf_all_dict, out_shapes, fps_stuff, title= None, dataset = 'ucf'):
    # print 'HELLO'
    # fps_stuff = 1./10.
    # activitynet = False
    print dataset
    raw_input()
    if dataset =='activitynet':
        class_names = globals.class_names_activitynet
        gt_vid_names_all, gt_class_names, gt_time_intervals_all = load_activitynet_gt(False)
    elif dataset =='activitynet_select':
        class_names = globals.class_names_activitynet_select
        gt_vid_names_all, gt_class_names, gt_time_intervals_all = load_activitynet_gt(False,True)
    elif dataset=='multithumos':
        class_names = globals.class_names_multithumos
        gt_vid_names_all, gt_class_names, gt_time_intervals_all = load_multithumos_gt(False)
    else:
        class_names = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']
        gt_vid_names_all, gt_class_names, gt_time_intervals_all = load_ucf_gt(False)
        class_names.sort()

    aps = np.zeros((len(class_names)+1,5))
    overlap_thresh_all = np.arange(0.1,0.6,0.1)
    
    for idx_class_name, class_name in enumerate(class_names):
        # if idx_class_name<6:
        #     continue
        out_dir = os.path.join(out_dir_meta,class_name)
        util.mkdir(out_dir)

        # if not dataset.startswith('activitynet'):
        #     mat_file = os.path.join('../TH14evalkit','mat_files', class_name+'_test.mat')

        #     loaded = scipy.io.loadmat(mat_file)
            
        #     gt_vid_names_all = loaded['gtvideonames'][0]
        #     gt_class_names = loaded['gt_events_class'][0]

        #     gt_time_intervals = loaded['gt_time_intervals'][0]
            
        #     arr_meta = [gt_vid_names_all, gt_class_names]
        #     arr_out = []
        #     for arr_curr in arr_meta:
        #         arr_curr = [str(a[0]) for a in arr_curr]
        #         arr_out.append(arr_curr)

        #     [gt_vid_names_all, gt_class_names] = arr_out
        #     gt_time_intervals_all = np.array([a[0] for a in gt_time_intervals])


        gt_vid_names = list( np.unique(np.array(gt_vid_names_all)[np.array(gt_class_names)==class_name]))
       

        for gt_vid_name in gt_vid_names:

            gt_time_intervals = gt_time_intervals_all[np.logical_and(np.array(gt_vid_names_all)==gt_vid_name ,np.array(gt_class_names)==class_name)]
            if gt_vid_name not in list(out_shapes.keys()):
                continue

            out_shape_curr = out_shapes[gt_vid_name]
            det_times = np.array(range(0,out_shape_curr+1))*fps_stuff
            gt_vals = np.zeros(det_times.shape)

            plot_arr = []
            legend_entries = []

            max_det_conf = None
            for k in det_conf_all_dict.keys():
                

                [det_conf_curr,det_time_intervals_all, det_events_class_all, det_vid_names] = det_conf_all_dict[k]
                
                bin_keep = det_vid_names == gt_vid_name
                
                # print det_vid_names[0], gt_vid_names, np.sum(bin_keep)

                bin_keep = np.logical_and(bin_keep, det_events_class_all==idx_class_name)
                if np.sum(bin_keep)==0:
                    # print 'Continuing'
                    continue

                # print 'not Continuing'
                # det_conf_curr = det_conf_all_dict[k][0]
                det_time_intervals_merged = det_time_intervals_all[bin_keep,:]
                det_conf_curr = det_conf_curr[bin_keep]
                # print k, det_time_intervals_merged

                if max_det_conf is None:
                    max_det_conf = np.max(det_conf_curr)
                else:
                    max_det_conf = max(max_det_conf, np.max(det_conf_curr))

                det_vals = np.zeros(det_times.shape)
                for idx_det_time_curr, det_time_curr in enumerate(det_time_intervals_merged):
                    idx_start = np.argmin(np.abs(det_times-det_time_curr[0]))
                    idx_end = np.argmin(np.abs(det_times-det_time_curr[1]))
                    det_vals[idx_start:idx_end] = det_conf_curr[idx_det_time_curr]

                legend_entries.append(k)
                plot_arr.append((det_times,det_vals))
                
            for gt_time_curr in gt_time_intervals:
                idx_start = np.argmin(np.abs(det_times-gt_time_curr[0]))
                idx_end = np.argmin(np.abs(det_times-gt_time_curr[1]))
                gt_vals[idx_start:idx_end] = max_det_conf

            plot_arr.append((det_times,gt_vals))
            legend_entries.append('GT')

            out_file_curr = os.path.join(out_dir,gt_vid_name+'.jpg')

            if title is None:
                title = 'det conf over time'
            # print plot_arr
            out_file_first = out_file_curr[:out_file_curr.rindex('.')]
            # plot_arr_for_save = {}
            for idx_arr in range(len(plot_arr)):
                # plot_arr_for_save[legend_entries[idx_arr]]=np.array(list(plot_arr[idx_arr]))
                arr_curr = np.array(list(plot_arr[idx_arr])) 
                np.save(out_file_first+'_'+legend_entries[idx_arr]+'.npy',arr_curr)


                # print legend_entries[idx_arr],plot_arr_for_save[legend_entries[idx_arr]].shape



            
            visualize.plotSimple(plot_arr,out_file = out_file_curr,title = title,xlabel = 'Time',ylabel = 'Detection Confidence'
            ,legend_entries=legend_entries)
            
        
        visualize.writeHTMLForFolder(out_dir)




def testing_gt_mat():
    class_names = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']
    class_names.sort()
    train = False

    for idx_class_name, class_name in enumerate(class_names):
        
        if train:
            mat_file = os.path.join('../TH14evalkit',class_name+'.mat')
        else:
            mat_file = os.path.join('../TH14evalkit','mat_files', class_name+'_test.mat')
        print 'TRAIN',train,mat_file

        loaded = scipy.io.loadmat(mat_file)
        
        gt_vid_names = loaded['gtvideonames'][0]
        gt_class_names = loaded['gt_events_class'][0]
        gt_time_intervals = loaded['gt_time_intervals'][0]
        
        arr_meta = [gt_vid_names, gt_class_names]

        arr_out = []
        for arr_curr in arr_meta:
            arr_curr = [str(a[0]) for a in arr_curr]
            arr_out.append(arr_curr)

        [gt_vid_names, gt_class_names] = arr_out
        
        gt_vid_names = np.array(gt_vid_names)
        gt_class_names = np.array(gt_class_names)
        gt_time_intervals = np.array([a[0] for a in gt_time_intervals])
        if idx_class_name>0:
            print np.all(gt_vid_names_old==gt_vid_names)
            print np.all(gt_class_names_old==gt_class_names)
            print np.all(gt_time_intervals_old==gt_time_intervals)

        gt_vid_names_old = gt_vid_names
        gt_time_intervals_old = gt_time_intervals
        gt_class_names_old = gt_class_names





def test_overlap_dummy_baselines(dataset = 'activitynet'):
    
    print dataset
    train = True
    if dataset =='ucf' or dataset.startswith('ucf_cooc'):
        class_names = globals.class_names_ucf
        gt_vid_names, gt_class_names, gt_time_intervals = load_ucf_gt(train)
        aps = np.zeros((len(class_names)+1,5))
        overlap_thresh_all = np.arange(0.1,0.6,0.1)
        fps_stuff = 16./25.
    elif dataset =='activitynet':
        class_names = globals.class_names_activitynet
        gt_vid_names, gt_class_names, gt_time_intervals = load_activitynet_gt(train)
        overlap_thresh_all = np.array([0.5,0.7,0.9])
        aps = np.zeros((len(class_names)+1,overlap_thresh_all.size))
    elif dataset=='ucf_untf':
        class_names = globals.class_names_ucf
        gt_vid_names, gt_class_names, gt_time_intervals = load_ucf_gt(train)
        aps = np.zeros((len(class_names)+1,5))
        overlap_thresh_all = np.arange(0.1,0.6,0.1)
        fps_stuff = 10./30.
    else:
        raise ValueError('Problem. '+dataset+' not valid')



    print len(gt_vid_names), type(gt_vid_names), type(gt_class_names)

    print len(set(gt_vid_names))
    mat_load = np.load('../data/activitynet/ids_durations.npz')
    durations = mat_load['durations']
    ids = mat_load['ids']
    print gt_vid_names[0]
    print ids.shape, ids[0]

    gt_vid_names_t = np.array(gt_vid_names)
    gt_class_names_t = np.array(gt_class_names)
    bin_keep = np.isin(ids,gt_vid_names_t)
    durations_keep = durations[bin_keep]
    ids_keep = ids[bin_keep]


    second_thresh = []
    det_conf_all = []
    det_time_intervals_all = []
    det_vid_names_all = []

    for idx_id_curr,id_curr in enumerate(ids_keep):
        duration_rel = durations_keep[idx_id_curr]
        idx_rel = np.where(gt_vid_names_t == id_curr)[0]

        gt_labels_curr = np.unique(gt_class_names_t[idx_rel])
      
        for label_curr in gt_labels_curr:
       
            idx_label = class_names.index(label_curr)
            second_thresh.append(idx_label)
            det_conf_all.append(1.)

            det_time_intervals_all.append([0,duration_rel])
            det_vid_names_all.append(id_curr)


    det_time_intervals_all = np.array(det_time_intervals_all)
    det_vid_names_all = np.array(det_vid_names_all)
    det_conf_all = np.array(det_conf_all)
    second_thresh = np.array(second_thresh)

    log_arr = []

    for idx_class_name, class_name in enumerate(class_names):

        bin_keep = second_thresh == idx_class_name
        det_conf = det_conf_all[bin_keep]
        det_time_intervals = det_time_intervals_all[bin_keep]
        det_vid_names = list(np.array(det_vid_names_all)[bin_keep])
        det_class_names = [class_name]*det_conf.shape[0]

        for idx_overlap_thresh, overlap_thresh in enumerate(overlap_thresh_all):
            rec, prec, ap = event_det_pr(det_vid_names, det_time_intervals, det_class_names, det_conf, gt_vid_names, gt_time_intervals, gt_class_names, class_name, overlap_thresh)
            aps[idx_class_name, idx_overlap_thresh] = ap*100
        
        str_print = [class_name+'\t' if len(class_name)<8 else class_name]+['%.2f' % ap_curr for ap_curr in aps[idx_class_name,:]]
        str_print = '\t'.join(str_print)
        print str_print
    
    aps[-1,:]= np.mean(aps[:len(class_names),:],0)
    
    str_print = '\t'.join(['%.1f' % ov for ov in overlap_thresh_all])
    log_arr.append(str_print)
    for idx_class_name, class_name in enumerate(class_names+['Average']):
        if len(class_name)<8 and dataset=='ucf':
            class_name +='\t'
        str_print = [class_name]+['%.2f' % ap_curr for ap_curr in aps[idx_class_name,:]]
        str_print = '\t'.join(str_print)
        log_arr.append(str_print)
    
    print log_arr[-1]


def main():
    print 'hello'

    test_overlap_dummy_baselines()
    





if __name__=='__main__':
    main()


