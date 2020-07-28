import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import scipy.misc
import numpy as np
import random
import scipy.io


def view_old_mat():
    print 'hello'
    class_name = 'BaseballPitch'
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

    for arr in [gt_vid_names, gt_class_names, gt_time_intervals]:
        print type(arr), len(arr)

    print gt_time_intervals.shape

    # return gt_vid_names, gt_class_names, gt_time_intervals

def save_all_anno():
    meta_dir = '../data/multithumos'
    anno_dir = os.path.join(meta_dir,'annotations')

    out_file = os.path.join(meta_dir,'annos_all.npz')

    anno_files = glob.glob(os.path.join(anno_dir,'*.txt'))
    anno_files.sort()

    gt_class_names = []
    gt_time_intervals = []
    gt_vid_names = []

    for anno_file in anno_files:
        lines = util.readLinesFromFile(anno_file)
        gt_class_name = os.path.split(anno_file)[1]
        gt_class_name = gt_class_name[:gt_class_name.rindex('.')]
        for line in lines:
            line = line.split(' ')
            [vid_name, start_time, end_time] = line
            gt_time_intervals.append([float(start_time), float(end_time)])
            gt_vid_names.append(vid_name)
            gt_class_names.append(gt_class_name)


    gt_vid_names = np.array(gt_vid_names)
    gt_time_intervals = np.array(gt_time_intervals)
    gt_class_names = np.array(gt_class_names)

    print gt_time_intervals.shape
    print gt_class_names.shape
    print gt_vid_names.shape


    np.savez(out_file, gt_time_intervals = gt_time_intervals ,gt_class_names = gt_class_names ,gt_vid_names = gt_vid_names)

def save_test_train_files():
    meta_dir = '../data/multithumos'
    out_dir = os.path.join(meta_dir,'train_test_files')
    util.mkdir(out_dir)

    out_train_file = os.path.join(out_dir,'train.txt')
    out_test_file = os.path.join(out_dir, 'test.txt')
    dir_train_features = '../data/i3d_features/Thumos14-I3D-JOINTFeatures_val'
    dir_test_features = '../data/i3d_features/Thumos14-I3D-JOINTFeatures_test'

    anno_file = os.path.join(meta_dir,'annos_all.npz')

    loaded = np.load(anno_file)

    gt_time_intervals = loaded['gt_time_intervals']
    gt_class_names = loaded['gt_class_names']
    gt_vid_names = loaded['gt_vid_names']

    class_names = list(np.unique(gt_class_names))
    class_names.sort()
    print class_names

    num_classes = len(class_names)
    print num_classes

    out_files = [out_train_file, out_test_file]
    feature_dirs = [dir_train_features, dir_test_features]

    str_match = '_validation_'
    idx_files_keep = [1 if str_match in vid_name else 0 for vid_name in gt_vid_names ]

    idx_files_keep = np.array(idx_files_keep)

    bin_keeps = [idx_files_keep==1, idx_files_keep==0]

    bin_classes = []

    for out_file, bin_keep, dir_features in zip(out_files, bin_keeps, feature_dirs):
        lines_to_write = []

        vid_names = gt_vid_names[bin_keep]
        for vid_name in np.unique(vid_names):

            feature_file = os.path.join(dir_features, vid_name+'.npy')
            if not os.path.exists(feature_file):
                print 'not exists',feature_file
                continue


            rel_classes = np.unique(gt_class_names[gt_vid_names==vid_name])
            bin_curr = np.zeros(num_classes).astype(int)
            for rel_class in rel_classes:
                bin_curr[class_names.index(rel_class)] = 1

            line_curr = ' '.join([str(val) for val in [feature_file]+list(bin_curr)])
            lines_to_write.append(line_curr)
            # print lines_to_write

        print out_file, len(lines_to_write)

        util.writeFile(out_file, lines_to_write)
        

def split_annos():
    meta_dir = '../data/multithumos'
    anno_file = os.path.join(meta_dir,'annos_all.npz')

    out_train_file = os.path.join(meta_dir,'train.npz')
    out_test_file = os.path.join(meta_dir, 'test.npz')

    loaded = np.load(anno_file)

    gt_time_intervals = loaded['gt_time_intervals']
    gt_class_names = loaded['gt_class_names']
    gt_vid_names = loaded['gt_vid_names']

    out_files = [out_train_file, out_test_file]
    
    str_match = '_validation_'
    idx_files_keep = [1 if str_match in vid_name else 0 for vid_name in gt_vid_names ]
    idx_files_keep = np.array(idx_files_keep)

    
    
    bin_keeps = [idx_files_keep==1, idx_files_keep==0]

    for out_file, bin_keep in zip(out_files, bin_keeps):
        time_keep = gt_time_intervals[bin_keep,:]
        class_names_keep = gt_class_names[bin_keep]
        vid_names_keep = gt_vid_names[bin_keep]
        print out_file, time_keep.shape, class_names_keep.shape, vid_names_keep.shape
        np.savez(out_file, gt_time_intervals = time_keep, gt_class_names = class_names_keep,gt_vid_names = vid_names_keep)



def main():

    # save_test_train_files()
    # split_annos()
    meta_dir = '../data/multithumos'
    
    anno_file = os.path.join(meta_dir,'annos_all.npz')

    loaded = np.load(anno_file)

    # gt_time_intervals = loaded['gt_time_intervals']
    gt_class_names = loaded['gt_class_names']
    # gt_vid_names = loaded['gt_vid_names']

    class_names = list(np.unique(gt_class_names))
    class_names.sort()
    print class_names

    



    # for anno_file in anno_files:
    #     print anno_file

    # print len(anno_files)








if __name__=='__main__':
    main()
