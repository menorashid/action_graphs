import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import numpy as np
import scipy.io

def ck():
    data_dir = '../data/ck_96/train_test_files'
    facs_anno_dir = '../data/ck_original/FACS'

    all_files = []
    fold_num = 0
    train_file = os.path.join(data_dir,'train_'+str(fold_num)+'.txt')
    test_file = os.path.join(data_dir,'test_'+str(fold_num)+'.txt')
    all_files = all_files+util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)
    assert len(all_files)==len(set(all_files))

    existence = []
    for file_curr in all_files:
        file_curr_split = file_curr.split(' ')
        anno = file_curr_split[1]
        im_name = os.path.split(file_curr_split[0])[1]
        im_name_split = im_name[:im_name.rindex('.')].split('_')
        facs_file = os.path.join(facs_anno_dir,im_name_split[0],im_name_split[1],'_'.join(im_name_split)+'_facs.txt')
        print facs_file, os.path.exists(facs_file),anno
        existence.append(os.path.exists(facs_file))

    print len(existence)
    print sum(existence)


def i3d():
    data_dir = '../data/i3d_features'
    
    for x in  (glob.glob(os.path.join(data_dir,'*'))):
        print x
        a = np.load(x)
        print a.shape
        print a[0].shape

def checking_val_order():
    # val_meta_path  = '../data/ucf101/val_data/validation_set_meta/validation_set_meta/validation_set.mat'
    # key = 'validation_videos'
    # to_check = glob.glob(os.path.join('../data/ucf101/val_data','anno_rough','*.npy'))

    val_meta_path = '../data/ucf101/test_data/test_set_meta.mat'
    key = 'test_videos'
    to_check = glob.glob(os.path.join('../data/ucf101/test_data','anno_rough','*.npy'))

    to_check = [os.path.split(file_curr)[1].replace('.npy','') for file_curr in to_check]
    print len(to_check)

    val = scipy.io.loadmat(val_meta_path, struct_as_record = False)[key][0]
    print len(val)
    idx_old = 0

    actions = []
    for val_curr in val:
        
        # for k in val_curr.__dict__.keys():
        #     print k
        if val_curr.video_name[0].replace('.mp4','') not in to_check:
            continue
        if val_curr.primary_action[0]=='Haircut':
            print val_curr.primary_action,val_curr.video_name
        # print val_curr.video_name[0], val_curr.primary_action[0]
        idx_new = int(val_curr.video_name[0].split('_')[-1])
        actions.append(val_curr.primary_action[0])

    print len(actions)
    actions = list(set(actions))
    actions.sort()
    print len(actions)
    for action in actions:
        print action



        # diff = idx_new -idx_old
        # print idx_new, idx_old, diff
        # assert diff==1
        # # print val_curr.video_name[0]
        # idx_old = idx_new
        # raw_input()


def main():
    # i3d()
    checking_val_order()
    
        

        

if __name__=='__main__':
    main()