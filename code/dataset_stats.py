import numpy as np
import os
import glob
from helpers import visualize,util
from analysis import evaluate_thumos as et
import globals as globals

def duration_stats():
    anno_file = '../data/activitynet/train_test_files/val.txt'
    dataset = 'anet'

    anno_file = '../data/ucf101/train_test_files/test.txt'
    dataset = 'ucf'
    anno_file = '../data/charades/train_test_files/i3d_charades_both_test.txt'
    dataset = 'charades'
    fps_stuff = 16./25.

    
    out_file = anno_file[:-4]+'_lengths.txt'
    
    # lines = util.readLinesFromFile(anno_file)
    # lengths = []
    # npy_files = [line.split(' ')[0] for line in lines]
    # lengths = [np.load(npy_file).shape[0] for npy_file in npy_files]
    # lines_to_write = [npy_file+' '+str(lengths[idx_file]) for idx_file, npy_file in enumerate(npy_files)]
    # util.writeFile(out_file, lines_to_write)
    # print out_file
    
    lines = util.readLinesFromFile(out_file)
    lines_split = [line.split() for line in lines]
    vid_files = [os.path.split(line[0])[1][:-4] for line in lines_split]
    durations = [int(line[1])*fps_stuff for line in lines_split]
    
    if dataset =='anet':
        gt_vid_names, gt_class_names, gt_time_intervals = et.load_activitynet_gt(False)
        class_names = globals.class_names_activitynet
    elif dataset == 'ucf':
        gt_vid_names, gt_class_names, gt_time_intervals = et.load_ucf_gt(False)
        class_names = globals.class_names
    elif dataset == 'charades':
        class_names = globals.class_names_charades
        gt_vid_names, gt_class_names, gt_time_intervals = et.load_charades_gt(False)

    gt_vid_names = np.array(gt_vid_names)
    print gt_vid_names[0]
    print gt_time_intervals.shape, gt_vid_names.shape
    duration_arr = np.zeros(gt_vid_names.shape)
    for gt_vid_name in np.unique(gt_vid_names):
        if gt_vid_name not in vid_files:
            print 'continuing'
            continue
        idx = vid_files.index(gt_vid_name)
        duration = durations[idx]
        duration_arr[gt_vid_names==gt_vid_name] = duration

    fg_duration = gt_time_intervals[:,1]-gt_time_intervals[:,0]
    print fg_duration.shape
    percent_total = fg_duration/duration_arr
    lims = [0.,1/8.,1/4.,1/2.,1]
    print dataset
    num_vids_total = np.unique(gt_vid_names).size
    for idx_lim, lim in enumerate(lims[:-1]):
        high_lim = lims[idx_lim+1]
        bin_curr = np.logical_and(lim<percent_total,percent_total<=high_lim)
        num_vids = np.unique(gt_vid_names[bin_curr]).size
        
        # count = np.sum(bin_curr)
        
        # print percent_total[bin_curr][:10]
        # print 

        print high_lim, num_vids/float(num_vids_total)
        

def main():

    # anno_file = '../data/charades/train_test_files/i3d_charades_both_test.txt'
    anno_file = '../data/ucf101/train_test_files/test.txt'
    anno = util.readLinesFromFile(anno_file)
    # anno_file = '../data/charades/train_test_files/i3d_charades_both_train_wmiss.txt'
    anno_file = '../data/ucf101/train_test_files/train.txt'
    anno+= util.readLinesFromFile(anno_file)

    print len(anno)
    anno = [[int(val) for val in anno_curr.split(' ')[1:]] for anno_curr in anno]
    anno = np.array(anno)
    print anno.shape
    print np.min(anno), np.max(anno)
    print np.sum(anno)/float(anno.shape[0])
   
        
    


if __name__=='__main__':
    main()