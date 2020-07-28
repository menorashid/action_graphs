import sys
sys.path.append('./')
import os
import numpy as np
import scipy.io
import glob
from helpers import util, visualize

def merge_detections(bin_keep, det_conf, det_time_intervals):
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

    for idx_curr in range(num_det):
        idx_start = idx_start_all[idx_curr]
        idx_end = idx_end_all[idx_curr]

        det_conf_rel = det_conf[idx_start:idx_end]
        det_conf_new[idx_curr]=np.mean(det_conf_rel)

        # print det_time_intervals.shape, idx_start
        det_time_intervals_new[idx_curr,0]=det_time_intervals[idx_start,0]
        print idx_end, det_time_intervals.shape, idx_curr, num_det
        det_time_intervals_new[idx_curr,1]=det_time_intervals[idx_end,0] if idx_end<det_time_intervals.shape[0] else det_time_intervals[idx_end-1,1]

        # print bin_keep[idx_start:idx_end]
        # print diff[idx_start:idx_end]
        assert np.all(bin_keep[idx_start:idx_end]==1)

    # print det_conf.shape
    # print det_time_intervals.shape
    # print det_conf_new.shape
    # print det_time_intervals_new.shape

    # raw_input()
    return det_conf_new, det_time_intervals_new

    # print idx_start.shape
    # print idx_end.shape


    # print bin_keep[20:40]
    # print bin_keep_rot[20:40]
    # raw_input()

def script_debug_old_testing():
    print 'hello'

    train = False

    det_file = '../scratch/debug_det.npz'
    out_dir_meta = '../scratch/seeing_dets'
    util.mkdir(out_dir_meta)

    det_data = np.load(det_file)

    det_vid_names = det_data['det_vid_names']
    det_conf_all = det_data['det_conf']
    det_time_intervals_all = det_data['det_time_intervals']
    print det_vid_names.shape, det_vid_names[0], det_vid_names[-1]
    print det_conf_all.shape, np.min(det_conf_all), np.max(det_conf_all)
    print det_time_intervals_all.shape, np.min(det_time_intervals_all), np.max(det_time_intervals_all)


    class_names = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']
    class_names.sort()

    aps = np.zeros((len(class_names)+1,5))
    overlap_thresh_all = np.arange(0.1,0.6,0.1)
    
    for idx_class_name, class_name in enumerate(class_names):
        # if idx_class_name<6:
        #     continue
        out_dir = os.path.join(out_dir_meta,class_name)
        util.mkdir(out_dir)

        if train:
            mat_file = os.path.join('../TH14evalkit',class_name+'.mat')
        else:
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
        print class_name, len(gt_vid_names)
        
        for gt_vid_name in gt_vid_names:
            print gt_vid_name
            bin_keep = det_vid_names == gt_vid_name
            gt_time_intervals = gt_time_intervals_all[np.array(gt_vid_names_all)==gt_vid_name]
            

            # print det_vid_names[bin_keep]
            print det_conf_all.shape, idx_class_name, bin_keep.shape
            det_conf = det_conf_all[bin_keep,idx_class_name]
            det_time_intervals = det_time_intervals_all [bin_keep,:]

            thresh = np.max(det_conf)-(np.max(det_conf)-np.min(det_conf))*0.5
            bin_second_thresh = det_conf>thresh

            det_conf, det_time_intervals_merged = merge_detections(bin_second_thresh, det_conf, det_time_intervals)


            # det_conf[det_conf<thresh]=0

            
            det_times = det_time_intervals[:,0]
            
            gt_vals = np.zeros(det_times.shape)
            for gt_time_curr in gt_time_intervals:
                idx_start = np.argmin(np.abs(det_times-gt_time_curr[0]))
                idx_end = np.argmin(np.abs(det_times-gt_time_curr[1]))
                gt_vals[idx_start:idx_end] = np.max(det_conf)

            det_vals = np.zeros(det_times.shape)
            for idx_det_time_curr, det_time_curr in enumerate(det_time_intervals_merged):
                idx_start = np.argmin(np.abs(det_times-det_time_curr[0]))
                idx_end = np.argmin(np.abs(det_times-det_time_curr[1]))
                det_vals[idx_start:idx_end] = det_conf[idx_det_time_curr]
            


            out_file_curr = os.path.join(out_dir,'dets_'+gt_vid_name+'_merged.jpg')

            
            visualize.plotSimple([(det_times,det_vals),(det_times,gt_vals)],out_file = out_file_curr,title = 'det conf over time',xlabel = 'time',ylabel = 'det conf',legend_entries=['Det','GT'])

            # print out_file_curr
            # raw_input()

        visualize.writeHTMLForFolder(out_dir)
        
        # det_conf = det_conf_all[:,idx_class_name]
        # bin_keep = det_conf>=second_thresh[:,idx_class_name]
        
        # # print np.sum(bin_keep), det_conf.shape
        # # raw_input()
        # det_time_intervals = det_time_intervals_all[bin_keep,:]
        # det_vid_names = list(np.array(det_vid_names_all)[bin_keep])
        # # det_vid_names_all
        # det_conf = det_conf[bin_keep]
        # det_class_names = [class_name]*det_conf.shape[0]
        



def main():
    print 'hello'

    train = False

    det_file = '../scratch/debug_det_graph.npz'
    out_dir_meta = '../scratch/seeing_dets_graph'
    util.mkdir(out_dir_meta)

    det_data = np.load(det_file)

    det_vid_names = det_data['det_vid_names']
    det_conf_all = det_data['det_conf']
    det_time_intervals_all = det_data['det_time_intervals']
    det_events_class_all = det_data['det_events_class']
    print det_vid_names.shape, det_vid_names[0], det_vid_names[-1]
    print det_conf_all.shape, np.min(det_conf_all), np.max(det_conf_all)
    print det_time_intervals_all.shape, np.min(det_time_intervals_all), np.max(det_time_intervals_all)
    print det_events_class_all.shape, np.min(det_events_class_all), np.max(det_events_class_all)


    class_names = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']
    class_names.sort()

    aps = np.zeros((len(class_names)+1,5))
    overlap_thresh_all = np.arange(0.1,0.6,0.1)
    
    for idx_class_name, class_name in enumerate(class_names):
        # if idx_class_name<6:
        #     continue
        out_dir = os.path.join(out_dir_meta,class_name)
        util.mkdir(out_dir)

        if train:
            mat_file = os.path.join('../TH14evalkit',class_name+'.mat')
        else:
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
        print class_name, len(gt_vid_names)
        
        for gt_vid_name in gt_vid_names:
            print gt_vid_name
            bin_keep = det_vid_names == gt_vid_name
            bin_keep = np.logical_and(bin_keep, det_events_class_all==idx_class_name)
            gt_time_intervals = gt_time_intervals_all[np.array(gt_vid_names_all)==gt_vid_name]
            

            # print det_vid_names[bin_keep]
            print det_conf_all.shape, idx_class_name, bin_keep.shape
            det_conf = det_conf_all[bin_keep]
            det_time_intervals = det_time_intervals_all [bin_keep,:]

            # thresh = np.max(det_conf)-(np.max(det_conf)-np.min(det_conf))*0.5
            # bin_second_thresh = det_conf>thresh

            # det_conf, det_time_intervals_merged = merge_detections(bin_second_thresh, det_conf, det_time_intervals)


            # det_conf[det_conf<thresh]=0
            det_time_intervals_merged = det_time_intervals
            
            det_times = det_time_intervals[:,0]
            
            gt_vals = np.zeros(det_times.shape)
            for gt_time_curr in gt_time_intervals:
                idx_start = np.argmin(np.abs(det_times-gt_time_curr[0]))
                idx_end = np.argmin(np.abs(det_times-gt_time_curr[1]))
                gt_vals[idx_start:idx_end] = np.max(det_conf)

            det_vals = np.zeros(det_times.shape)
            for idx_det_time_curr, det_time_curr in enumerate(det_time_intervals_merged):
                idx_start = np.argmin(np.abs(det_times-det_time_curr[0]))
                idx_end = np.argmin(np.abs(det_times-det_time_curr[1]))
                det_vals[idx_start:idx_end] = det_conf[idx_det_time_curr]
            


            out_file_curr = os.path.join(out_dir,'dets_'+gt_vid_name+'_merged.jpg')

            
            visualize.plotSimple([(det_times,det_vals),(det_times,gt_vals)],out_file = out_file_curr,title = 'det conf over time',xlabel = 'time',ylabel = 'det conf',legend_entries=['Det','GT'])

        
        visualize.writeHTMLForFolder(out_dir)
        
        



if __name__=='__main__':
    main()
