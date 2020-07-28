import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import numpy as np
import subprocess
import scipy.io
import cv2
import matplotlib.pyplot as plt
import skimage.transform
import multiprocessing
import random
# import preprocess_bp4d

def script_make_im_gray():
    dir_meta = '../data/disfa'
    out_dir_im = os.path.join(dir_meta, 'Videos_LeftCamera_frames_200')
    # ../data/disfa/preprocess_im_200_color_align
    # out_dir_files = os.path.join(dir_meta, 'train_test_files_110_color_align')
    # out_dir_files_new = os.path.join(dir_meta, 'train_test_files_110_gray_align')
    out_dir_im_new = os.path.join(dir_meta, 'preprocess_im_110_gray_align')
    util.mkdir(out_dir_im_new)

    num_folds = 3
    im_size = [110,110]
    # [96,96]
    # all_im = []
    # for fold_curr in range(num_folds):
    #     train_file = os.path.join(out_dir_files,'train_'+str(fold_curr)+'.txt')
    #     test_file = os.path.join(out_dir_files,'test_'+str(fold_curr)+'.txt')
    #     all_data = util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)
    #     all_im = all_im + [line_curr.split(' ')[0] for line_curr in all_data]

    all_im = glob.glob(os.path.join(out_dir_im,'*','*.jpg'))
    print len(all_im ),len(set(all_im))
    
    all_im = list(set(all_im))
    args = []
    for idx_file_curr,file_curr in enumerate(all_im):
        out_file_curr = file_curr.replace(out_dir_im,out_dir_im_new)
        dir_curr = os.path.split(out_file_curr)[0]
        util.makedirs(dir_curr)
        # print out_file_curr
        # print dir_curr
        if not os.path.exists(out_file_curr):
            args.append((file_curr,out_file_curr,im_size,idx_file_curr))

    print len(args)
    # for arg in args:
    #     print arg
    #     preprocess_bp4d.save_color_as_gray(arg)
    #     raw_input()


    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(preprocess_bp4d.save_color_as_gray,args)
    

def script_change_train_test():
    dir_meta = '../data/disfa'
    
    out_dir_files = os.path.join(dir_meta,'train_test_10_6_method')
    out_dir_files_new = os.path.join(dir_meta, 'train_test_10_6_method_110_gray_align')
    util.mkdir(out_dir_files_new)

    out_dir_im = os.path.join(dir_meta, 'Videos_LeftCamera_frames_200')
    out_dir_im_new = os.path.join(dir_meta, 'preprocess_im_110_gray_align')

    num_folds = 10

    for fold_curr in range(num_folds):
        for file_pre in ['train','test']:
            file_curr = os.path.join(out_dir_files,file_pre+'_'+str(fold_curr)+'.txt')
            lines = util.readLinesFromFile(file_curr)

            out_file = os.path.join(out_dir_files_new,file_pre+'_'+str(fold_curr)+'.txt')
            
            out_lines = []
            for line_curr in lines:
                out_line = line_curr.replace(out_dir_im,out_dir_im_new)
                im_curr = line_curr.split(' ')[0]
                assert os.path.exists(im_curr)
                out_lines.append(out_line)

            # print out_lines[0]
            print len(out_lines)
            print out_file

            # raw_input()
            util.writeFile(out_file,out_lines)


def script_save_mean_std_files():

    dir_meta = '../data/disfa'
    # out_dir_files_new = os.path.join(dir_meta, 'train_test_10_6_method_110_gray_align')
    # num_folds = 10
    # jump = 1

    out_dir_files_new = os.path.join(dir_meta, 'train_test_8_au_all_method_110_gray_align')
    num_folds = 3
    jump = 10

    for fold_curr in range(num_folds):
        train_file = os.path.join(out_dir_files_new,'train_'+str(fold_curr)+'.txt')
        lines = util.readLinesFromFile(train_file)
        ims_rel = [line_curr.split(' ')[0] for line_curr in lines]
        ims_rel = ims_rel[::jump]
        print len(ims_rel)
        out_file_mean = os.path.join(out_dir_files_new,'train_'+str(fold_curr)+'_mean.png')
        out_file_std = os.path.join(out_dir_files_new,'train_'+str(fold_curr)+'_std.png')

        print out_file_mean
        print out_file_std
        print train_file
        preprocess_bp4d.save_mean_std_im(ims_rel,out_file_mean,out_file_std)

    




def get_frame_au_anno(au_files):
    num_frames = len(util.readLinesFromFile(au_files[0]))
    anno = [[val] for val in range(num_frames)]
    for au_file in au_files:
        au_curr = os.path.split(au_file)[1]
        au_curr = int(au_curr[:-4].split('_')[-1][2:])
        
        lines = util.readLinesFromFile(au_file)
        for line in lines:
            frame, intensity = line.split(',')

            intensity = int(intensity)
            if intensity>0:
                anno[int(frame)-1].append(au_curr)
                anno[int(frame)-1].append(intensity)

    anno = [anno_curr for anno_curr in anno if len(anno_curr)>1]
    return anno

def get_emotion_count():
    anno_video_all = []
    meta_anno_dir = '../data/disfa/ActionUnit_Labels'
    video_anno_dirs = glob.glob(os.path.join(meta_anno_dir,'*'))
    for video_anno_dir in video_anno_dirs:
        au_files = glob.glob(os.path.join(video_anno_dir,'*.txt'))
        au_files.sort()
        anno_video = get_frame_au_anno(au_files)
        anno_video_all.extend(anno_video)

    # print len(anno_video_all)
    # return

    emotion_lists = [[6,12],[1,4,15],[1,2,5,26],[1,2,4,5,20,26],[9,15]]
    emotion_lists = emotion_lists+ [[1,15],[4,15],[1,2,5],[1,2,26],[1,2,4,20],[1,2,4,5,20],[1,2,5,20]]
    # for emotion_list in emotion_lists:
    emotion_lists = [[6,12],
                    [4,5,7,22,23,24],
                    [4,5,22,23,24],
                    [4,7,22,23,24],
                    [1,4,15],
                    [1,4,15,17],
                    [9,25,26],
                    [10,25,26],
                    [9,10,25,26],
                    [1,2,4,5,7,20,25,26],
                    [1,2,4,5,7,20,26],
                    [1,2,5,25,26],
                    [1,2,5,26]]

    
    anno_combination_list = []

    for anno_curr in anno_video_all:
        aus = anno_curr[1::2]
        aus.sort()
        aus = [str(au_curr) for au_curr in aus]
        anno_combination_list.append(' '.join(aus))
    
    emotion_lists = [' '.join([str(val) for val in emotion_list]) for emotion_list in emotion_lists]
    for anno_combo in emotion_lists:
    # set(anno_combination_list):
        print anno_combo,anno_combination_list.count(anno_combo)

def save_frames(out_dir,video_file,out_size=None):
    video_name = os.path.basename(video_file)
    video_name = video_name[:video_name.rindex('.')]

    out_dir = os.path.join(out_dir,video_name)
    util.mkdir(out_dir)
    frame_name = os.path.join(out_dir,video_name+'_%05d.jpg')

    command = []
    command.extend(['ffmpeg','-i'])
    command.append(video_file)
    command.append(frame_name)
    if out_size is not None:
        command.extend(['-s',str(out_size[0])+'x'+str(out_size[1])])
    command.append('-hide_banner')
    command = ' '.join(command)
    print command
    subprocess.call(command, shell=True)

def script_save_frames():
    data_dir = '../data/disfa/Videos_LeftCamera'
    out_dir = data_dir+'_frames'
    util.mkdir(out_dir)
    # video_files = glob.glob(os.path.join(data_dir,'*.avi'))
    # print video_files
    # for video_file in video_files:
    video_file = '../data/disfa/Videos_LeftCamera/LeftVideoSN013_comp.avi'
    save_frames(out_dir, video_file)


def script_view_bad_kp():
    out_dir = '../scratch/disfa/kp_check'
    util.makedirs(out_dir)
    out_file = os.path.join(out_dir,'SN001_0000_lm.jpg')
    mat_dir = '../data/disfa/Landmark_Points' 
    # SN001 'frame_lm'
    frame_dir ='../data/disfa/Videos_LeftCamera_frames'
    
    problem_dict = {'SN030':[ 939, 962, 1406, 1422, 2100, 2132, 2893, 2955],
                    'SN029':[ 4090, 4543],
                    'SN028':[ 1875, 1885, 4571, 4690],
                    'SN027':[ 3461, 3494, 4738, 4785],
                    'SN025':[ 4596, 4662, 4816, 4835],
                    'SN023':[ 1021, 1049, 3378, 3557, 3584, 3668, 4547, 4621, 4741, 4772, 4825, 4845],
                    'SN021':[ 574, 616, 985, 1164, 1190, 1205, 1305, 1338, 1665, 1710, 1862, 2477, 2554, 4657, 4710, 4722],
                    'SN011':[ 4529, 4533, 4830, 4845,  ],
                    'SN009':[ 1736, 1808, 1851, 1885],
                    'SN006':[ 1349, 1405],
                    'SN004':[ 4541, 4555],
                    'SN002':[ 800, 826],
                    'SN001':[ 398, 420, 3190, 3243]}

    for video_name in problem_dict.keys():
        range_starts = problem_dict[video_name][::2]
        range_ends = problem_dict[video_name][1::2]
                
        out_dir_curr = os.path.join(out_dir,video_name)
        util.mkdir(out_dir_curr)
        print video_name
        for idx_range in range(len(range_starts)):
            print range_starts[idx_range],range_ends[idx_range]

            for anno_num in range(range_starts[idx_range]-1,range_ends[idx_range]):


    # for anno_num in range(397,421)+range(3189,3244):
                str_num_mat = '0'*(4-len(str(anno_num)))+str(anno_num)
                str_num_im = '0'*(5-len(str(anno_num+1)))+str(anno_num+1)
                
                
                mat_file = os.path.join(mat_dir,video_name,'tmp_frame_lm',video_name+'_'+str_num_mat+'_lm.mat')
                if not os.path.exists(mat_file):
                    mat_file = os.path.join(mat_dir,video_name,'tmp_frame_lm','l0'+str_num_mat+'_lm.mat')

                im_file = os.path.join(frame_dir,'LeftVideo'+video_name+'_comp','LeftVideo'+video_name+'_comp_'+str_num_im+'.jpg')

                out_file = os.path.join(out_dir_curr,video_name+'_'+str_num_mat+'_provided.jpg')
                if os.path.exists(out_file):
                    continue
                
                im = scipy.misc.imread(im_file)
                pts = scipy.io.loadmat(mat_file)
                pts = pts['pts']
                for pt_curr in pts:
                    pt_curr = (int(pt_curr[0]),int(pt_curr[1]))
                    cv2.circle(im, pt_curr, 5, (255,0,0),-1)


                scipy.misc.imsave(out_file,im)  

        visualize.writeHTMLForFolder(out_dir_curr)

def script_save_avg_kp():
    out_dir = '../data/disfa'
    all_kp_files = glob.glob(os.path.join('../data/disfa/Landmark_Points','*','*','*.mat'))
    all_kp = None
    # all_kp_files = all_kp_files[::100]
    for kp_file in all_kp_files:
        kp = scipy.io.loadmat(kp_file)['pts']
        kp = kp-np.min(kp,0)
        if all_kp is None:
            all_kp = kp
        else:
            all_kp = all_kp+kp

    avg_kp = all_kp/len(all_kp_files)
    
    out_file = os.path.join(out_dir,'avg_kp.npy')
    np.save(out_file,avg_kp)

    plt.figure()
    plt.plot(avg_kp[:,0],np.max(avg_kp[:,1])-avg_kp[:,1],'*b')
    plt.savefig(os.path.join(out_dir,'avg_kp.jpg'))
    plt.close()

def script_save_avg_kp():
    out_dir = '../data/disfa'
    avg_kp = np.load(os.path.join(out_dir,'avg_kp.npy'))
    avg_kp = avg_kp - np.min(avg_kp,0)
    avg_kp = avg_kp /np.max(avg_kp,0)
    ratios = [75,20,5]
    assert sum(ratios)==100
    center = ratios[0]*200/100
    top = ratios[1]*200/100
    side = (100-ratios[0])/2. * 200/100
    avg_kp = avg_kp*center
    avg_kp[:,1]= avg_kp[:,1]+top
    avg_kp[:,0]= avg_kp[:,0]+side
    print side
    out_name = 'avg_kp_200_'+'_'.join([str(val) for val in ratios])
    out_file = os.path.join(out_dir,out_name+'.npy')
    np.save(out_file,avg_kp)
    
    plt.figure()
    plt.plot(avg_kp[:,0],avg_kp[:,1],'*b')
    plt.savefig(os.path.join('../scratch/disfa/kp_check', out_name+'.jpg'))
    plt.close()


def save_registered_face((avg_pts_file,mat_file,im_file,out_file,idx)):
    if not idx%100:
        print idx

    avg_pts = np.load(avg_pts_file)

    pts = scipy.io.loadmat(mat_file)
    pts = pts['pts']

    im = scipy.misc.imread(im_file)
    
    tform = skimage.transform.estimate_transform('similarity', pts, avg_pts)
    im_new = skimage.transform.warp(im, tform.inverse, output_shape=(200,200), order=1, mode='edge')

    # # print im.shape
    # im_new = im_new*255
    # for pt_curr in avg_pts:
    #   pt_curr = (int(pt_curr[0]),int(pt_curr[1]))
    #   print pt_curr
    #   cv2.circle(im_new, pt_curr, 3, (255,0,0),-1)

    # out_file = '../scratch/disfa/warp.jpg'
    scipy.misc.imsave(out_file,im_new)


def script_save_registered_faces():
    mat_dir_meta = '../data/disfa/Landmark_Points' 
    frame_dir_meta ='../data/disfa/Videos_LeftCamera_frames'
    out_dir_meta = '../data/disfa/Videos_LeftCamera_frames_200'
    util.mkdir(out_dir_meta)

    avg_pts_file = '../data/disfa/avg_kp_200_75_20_5.npy'
    video_names = [dir_curr for dir_curr in os.listdir(mat_dir_meta) if os.path.isdir(os.path.join(mat_dir_meta,dir_curr))]

    args = []
    
    for video_name in video_names:
        mat_dir = os.path.join(mat_dir_meta,video_name,'tmp_frame_lm')
        mat_files = glob.glob(os.path.join(mat_dir,'*.mat'))
        im_files = glob.glob(os.path.join(frame_dir_meta,'LeftVideo'+video_name+'_comp','*.jpg'))
        im_files.sort()
        mat_files.sort()
        if len(mat_files)>len(im_files):
            mat_files = mat_files[1:]
        assert len(mat_files)==len(im_files)
        

        for idx_mat_file,(mat_file,im_file) in enumerate(zip(mat_files,im_files)):
            mat_num = int(mat_file[-11:-7])
            im_num = int(im_file[-8:-4])
            if im_num!=mat_num:
                assert im_num-mat_num==1

            out_file = im_file.replace(frame_dir_meta,out_dir_meta)
            if not os.path.exists(out_file):
                util.makedirs(os.path.dirname(out_file))
                args.append((avg_pts_file,mat_file,im_file,out_file,len(args)))
    
    print len(args)

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(save_registered_face,args)


def checking_warp():
    out_dir = '../data/disfa'
    # avg_pts = np.load(os.path.join(out_dir,'avg_kp_200_80.npy'))
    avg_pts = np.load(os.path.join(out_dir,'avg_kp_200_75_20_5.npy'))
    mat_file = '../data/disfa/Landmark_Points/SN001/tmp_frame_lm/SN001_0000_lm.mat'

    pts = scipy.io.loadmat(mat_file)
    pts = pts['pts']
    # print np.min(pts,0),np.max(pts,0)
    

    # # im_file = '../data/disfa/Video_RightCamera_frames/RightVideoSN001_Comp/RightVideoSN001_Comp_00001.jpg'
    im_file = '../data/disfa/Videos_LeftCamera_frames/LeftVideoSN001_comp/LeftVideoSN001_comp_00001.jpg'
    im = scipy.misc.imread(im_file)
    import skimage.transform
    tform = skimage.transform.estimate_transform('similarity', pts, avg_pts)

    im_new = skimage.transform.warp(im, tform.inverse, output_shape=(200,200), order=1, mode='edge')

    # print im.shape
    im_new = im_new*255
    for pt_curr in avg_pts:
        pt_curr = (int(pt_curr[0]),int(pt_curr[1]))
        print pt_curr
        cv2.circle(im_new, pt_curr, 3, (255,0,0),-1)

    out_file = '../scratch/disfa/warp.jpg'
    scipy.misc.imsave(out_file,im_new)

def fill_in_volume(file_nums, dir_anno_meta, sub, list_aus):
    volume = np.zeros((len(file_nums),len(list_aus)))
    for idx_au_curr,au_curr in enumerate(list_aus):
        file_curr = os.path.join(dir_anno_meta,sub,sub+'_au'+str(au_curr)+'.txt')
        annos = util.readLinesFromFile(file_curr)
        annos = [[int(val) for val in line_curr.split(',')] for line_curr in annos]
        annos = np.array(annos)
        assert len(file_nums)==annos.shape[0]
        volume[:,idx_au_curr]=annos[:,1]
    return volume

def save_anno_volume():
    dir_meta = '../data/disfa'
    dir_im_meta = os.path.join(dir_meta,'Videos_LeftCamera_frames_200')
    dir_anno_meta = os.path.join(dir_meta,'ActionUnit_Labels')
    out_dir_volume = os.path.join(dir_meta,'anno_volume')
    util.mkdir(out_dir_volume)

    list_subs = glob.glob(os.path.join(dir_anno_meta,'*'))
    list_subs = [os.path.split(sub)[1] for sub in list_subs]
    # list_aus = glob.glob(os.path.join(list_subs[0],'*.txt'))
    list_aus = [1,12,15,17,2,20,25,26,4,5,6,9]
    list_aus.sort()
    print list_aus

    pre_im_str = 'LeftVideo'
    post_im_str = '_comp'

    for sub in list_subs:
        im_dir = os.path.join(dir_im_meta,pre_im_str+sub+post_im_str)
        out_file_volume = os.path.join(out_dir_volume, sub+'.npy')
        list_files = glob.glob(os.path.join(im_dir,'*.jpg'))
        list_files.sort()
        
        file_nums = [int(file_curr[file_curr.rindex('_')+1:file_curr.rindex('.')]) for file_curr in list_files]

        volume_anno = fill_in_volume(file_nums,dir_anno_meta,sub,list_aus)

        print volume_anno.shape
        print out_file_volume
        np.save(out_file_volume,volume_anno)

def make_folds():
    dir_meta = '../data/disfa'
    dir_im_meta = os.path.join(dir_meta,'Videos_LeftCamera_frames_200')
    dir_anno_meta = os.path.join(dir_meta,'ActionUnit_Labels')
    # num_folds = 10
    # out_dir_folds = os.path.join(dir_meta,'folds_'+str(num_folds))

    num_folds = 3
    out_dir_folds = os.path.join(dir_meta,'folds_'+str(num_folds))
    
    util.mkdir(out_dir_folds)

    list_subs = glob.glob(os.path.join(dir_anno_meta,'*'))
    list_subs = [os.path.split(sub)[1] for sub in list_subs]
    
    list_subs.sort()
    
    all_folds = []
    for idx_fold in range(num_folds):
        all_folds.append(list_subs[idx_fold::num_folds])
    
    for idx_fold in range(num_folds):
        train_file = os.path.join(out_dir_folds,'train_'+str(idx_fold)+'.txt')
        test_file = os.path.join(out_dir_folds,'test_'+str(idx_fold)+'.txt')
        train_fold = []
        for i in range(len(all_folds)):
            if i!=idx_fold:
                train_fold = train_fold+all_folds[i]
        test_fold = all_folds[idx_fold]
        print idx_fold,len(train_fold),len(test_fold)

        assert len(set(train_fold+test_fold))==len(train_fold+test_fold)==27
        util.writeFile(train_file,train_fold)
        util.writeFile(test_file,test_fold)

def make_folds_val():
    dir_meta = '../data/disfa'
    dir_im_meta = os.path.join(dir_meta,'Videos_LeftCamera_frames_200')
    dir_anno_meta = os.path.join(dir_meta,'ActionUnit_Labels')
    out_dir_folds = os.path.join(dir_meta,'folds_val')
    util.mkdir(out_dir_folds)

    list_subs = glob.glob(os.path.join(dir_anno_meta,'*'))
    list_subs = [os.path.split(sub)[1] for sub in list_subs]
    
    list_subs.sort()
    num_folds = 9
    all_folds = []
    for idx_fold in range(num_folds):
        all_folds.append(list_subs[idx_fold::num_folds])
    
    for idx_fold in range(num_folds):
        train_file = os.path.join(out_dir_folds,'train_'+str(idx_fold)+'.txt')
        test_file = os.path.join(out_dir_folds,'test_'+str(idx_fold)+'.txt')
        val_file = os.path.join(out_dir_folds,'val_'+str(idx_fold)+'.txt')

        train_fold = []
        val_fold_num = (idx_fold+1)%num_folds
        print idx_fold,val_fold_num

        for i in range(len(all_folds)):
            if i!=idx_fold and i!=val_fold_num:
                train_fold = train_fold+all_folds[i]
        test_fold = all_folds[idx_fold]
        val_fold = all_folds[val_fold_num]
        
        print val_fold
        print test_fold
        print train_fold

        assert len(set(train_fold+test_fold))==len(train_fold+test_fold)
        assert len(set(train_fold+val_fold))==len(train_fold+val_fold)

        util.writeFile(train_file,train_fold)
        util.writeFile(test_file,test_fold)
        util.writeFile(val_file,val_fold)



def make_disfa_800_1600_anno():

    dir_meta = '../data/disfa'
    dir_im_meta = os.path.join(dir_meta,'Videos_LeftCamera_frames_200')
    pre_im_str = 'LeftVideo'
    post_im_str = '_comp'

    dir_anno_meta = os.path.join(dir_meta,'ActionUnit_Labels')
    out_dir_volume = os.path.join(dir_meta,'anno_volume')
    out_dir_annos = os.path.join(dir_meta,'sub_annos_800_1600_method')
    util.mkdir(out_dir_annos)

    list_subs = glob.glob(os.path.join(dir_anno_meta,'*'))
    list_subs = [os.path.split(sub)[1] for sub in list_subs]    

    aus_keep = [1,2,4,6,9,12,25,26]
    aus_keep.sort()

    list_aus = [1,12,15,17,2,20,25,26,4,5,6,9]
    list_aus.sort()
    idx_keep = [1 if au in aus_keep else 0 for au in list_aus ]
    
    total_reasonable = 0


    for sub in list_subs:
        im_dir = os.path.join(dir_im_meta,pre_im_str+sub+post_im_str)
        im_files = glob.glob(os.path.join(im_dir,'*.jpg'))
        im_files.sort()


        volume_file = os.path.join(out_dir_volume,sub+'.npy')
        volume = np.load(volume_file)
        

        
        assert volume.shape[0]==len(im_files)

        volume = volume[:,np.array(idx_keep)>0]
        print np.unique(volume)
        
        volume_pos = np.array(volume)
        volume_pos[volume_pos<3] = 0
        volume_pos[volume_pos>=3] = 1
        bin_pos = np.sum(volume_pos,1)>0

        print 'num_pos',np.sum(bin_pos) 
        
        volume_neg = np.array(volume)
        
        volume_neg[volume_neg==0]=-1
        volume_neg[volume_neg>0]=0
        volume_neg[volume_neg<0]=1

        bin_neg = np.sum(volume_neg,1)==volume_neg.shape[1]
        # bin_neg = volume_neg<1
        # bin_neg = np.sum(volume_neg,1)>0

        print 'num_neg', np.sum(bin_neg)

        rel_pos = volume[bin_pos,:]
        print rel_pos.shape
        print rel_pos[0]
        print volume_pos[bin_pos,:][0]


        rel_neg = volume[bin_neg,:]
        print rel_neg.shape
        print rel_neg[0]
        print volume_neg[bin_neg,:][0]
        
        raw_input()


        volume_sum = np.sum(volume,1)
        to_keep= volume_sum>6
        print to_keep.shape, np.sum(to_keep)

        

        total_reasonable+=np.sum(to_keep)

        im_files = np.array(im_files)
        im_files_to_keep = im_files[to_keep]
        volume_to_keep = volume[to_keep,:]

        out_file_sub = os.path.join(out_dir_annos,sub+'.txt')
        lines = []
        for idx in range(im_files_to_keep.shape[0]):
            im_file = im_files_to_keep[idx]
            volume_file = volume_to_keep[idx,:] 
            anno = [im_file]+[str(int(val)) for val in volume_file]
            anno = ' '.join(anno)
            lines.append(anno)
        print len(lines),out_file_sub
        util.writeFile(out_file_sub,lines)

        

    print total_reasonable



def make_disfa_8au_anno_all():

    dir_meta = '../data/disfa'
    dir_im_meta = os.path.join(dir_meta,'Videos_LeftCamera_frames_200')
    pre_im_str = 'LeftVideo'
    post_im_str = '_comp'

    dir_anno_meta = os.path.join(dir_meta,'ActionUnit_Labels')
    out_dir_volume = os.path.join(dir_meta,'anno_volume')
    out_dir_annos = os.path.join(dir_meta,'sub_annos_8_au_all_method')
    util.mkdir(out_dir_annos)

    list_subs = glob.glob(os.path.join(dir_anno_meta,'*'))
    list_subs = [os.path.split(sub)[1] for sub in list_subs]    

    aus_keep = [1,2,4,6,9,12,25,26]
    aus_keep.sort()

    list_aus = [1,12,15,17,2,20,25,26,4,5,6,9]
    list_aus.sort()
    idx_keep = [1 if au in aus_keep else 0 for au in list_aus ]
    
    total_reasonable = 0


    for sub in list_subs:
        im_dir = os.path.join(dir_im_meta,pre_im_str+sub+post_im_str)
        im_files = glob.glob(os.path.join(im_dir,'*.jpg'))
        im_files.sort()


        volume_file = os.path.join(out_dir_volume,sub+'.npy')
        volume = np.load(volume_file)

        
        assert volume.shape[0]==len(im_files)

        volume_to_keep = volume[:,np.array(idx_keep)>0]
        print volume_to_keep.shape,np.sum(np.sum(volume_to_keep,1)>0)
        raw_input()
        # volume_sum = np.sum(volume,1)
        # to_keep= volume_sum>6
        # print to_keep.shape, np.sum(to_keep)

        

        total_reasonable+=volume_to_keep.shape[0]

        # np.sum(to_keep)

        im_files_to_keep = np.array(im_files)
        # im_files_to_keep = im_files[to_keep]
        # volume_to_keep = volume[to_keep,:]

        out_file_sub = os.path.join(out_dir_annos,sub+'.txt')
        lines = []
        for idx in range(im_files_to_keep.shape[0]):
            im_file = im_files_to_keep[idx]
            volume_file = volume_to_keep[idx,:] 
            anno = [im_file]+[str(int(val)) for val in volume_file]
            anno = ' '.join(anno)
            lines.append(anno)
        print len(lines),out_file_sub
        util.writeFile(out_file_sub,lines)

        

    print total_reasonable
        




def make_disfa_10_16_anno():

    dir_meta = '../data/disfa'
    dir_im_meta = os.path.join(dir_meta,'Videos_LeftCamera_frames_200')
    pre_im_str = 'LeftVideo'
    post_im_str = '_comp'

    dir_anno_meta = os.path.join(dir_meta,'ActionUnit_Labels')
    out_dir_volume = os.path.join(dir_meta,'anno_volume')
    out_dir_annos = os.path.join(dir_meta,'sub_annos_10_6_method')
    util.mkdir(out_dir_annos)

    list_subs = glob.glob(os.path.join(dir_anno_meta,'*'))
    list_subs = [os.path.split(sub)[1] for sub in list_subs]    

    aus_keep = [1,2,4,6,9,12,15,17,25,26]
    aus_keep.sort()

    list_aus = [1,12,15,17,2,20,25,26,4,5,6,9]
    list_aus.sort()
    idx_keep = [1 if au in aus_keep else 0 for au in list_aus ]
    
    total_reasonable = 0


    for sub in list_subs:
        im_dir = os.path.join(dir_im_meta,pre_im_str+sub+post_im_str)
        im_files = glob.glob(os.path.join(im_dir,'*.jpg'))
        im_files.sort()


        volume_file = os.path.join(out_dir_volume,sub+'.npy')
        volume = np.load(volume_file)
        

        
        assert volume.shape[0]==len(im_files)

        volume = volume[:,np.array(idx_keep)>0]
        
        volume_sum = np.sum(volume,1)
        to_keep= volume_sum>6
        print to_keep.shape, np.sum(to_keep)

        

        total_reasonable+=np.sum(to_keep)

        im_files = np.array(im_files)
        im_files_to_keep = im_files[to_keep]
        volume_to_keep = volume[to_keep,:]

        out_file_sub = os.path.join(out_dir_annos,sub+'.txt')
        lines = []
        for idx in range(im_files_to_keep.shape[0]):
            im_file = im_files_to_keep[idx]
            volume_file = volume_to_keep[idx,:] 
            anno = [im_file]+[str(int(val)) for val in volume_file]
            anno = ' '.join(anno)
            lines.append(anno)
        print len(lines),out_file_sub
        util.writeFile(out_file_sub,lines)

        

    print total_reasonable


def save_train_test_files():
    dir_meta = '../data/disfa'
    dir_im_meta = os.path.join(dir_meta,'Videos_LeftCamera_frames_200')
    pre_im_str = 'LeftVideo'
    post_im_str = '_comp'

    dir_anno_meta = os.path.join(dir_meta,'ActionUnit_Labels')
    out_dir_volume = os.path.join(dir_meta,'anno_volume')

    # out_dir_annos = os.path.join(dir_meta,'sub_annos_10_6_method')
    # out_dir_train_test = os.path.join(dir_meta,'train_test_10_6_method')
    # num_folds = 10
    # out_dir_folds =  os.path.join(dir_meta,'folds_10')

    out_dir_annos = os.path.join(dir_meta,'sub_annos_8_au_all_method')
    # out_dir_train_test = os.path.join(dir_meta,'train_test_8_au_all_method_110_gray_align')
    # num_folds = 3
    # out_dir_folds =  os.path.join(dir_meta,'folds_3')
    # new_dir_im_meta = os.path.join(dir_meta,'preprocess_im_110_gray_align')

    out_dir_train_test = os.path.join(dir_meta,'train_test_8_au_all_method_256_color_align')
    num_folds = 3
    out_dir_folds =  os.path.join(dir_meta,'folds_3')
    new_dir_im_meta = os.path.join(dir_meta,'preprocess_im_256_color_align')
    
    util.mkdir(out_dir_train_test)

    
    for fold_curr in range(num_folds):
        fold_curr = str(fold_curr)
        print fold_curr
        train_subs = util.readLinesFromFile(os.path.join(out_dir_folds,'train_'+fold_curr+'.txt'))
        test_subs = util.readLinesFromFile(os.path.join(out_dir_folds,'test_'+fold_curr+'.txt'))
        assert len(train_subs)+len(test_subs)==27

        train_sub_files = [os.path.join(out_dir_annos,sub_curr+'.txt') for sub_curr in train_subs]
        test_sub_files = [os.path.join(out_dir_annos,sub_curr+'.txt') for sub_curr in test_subs]
        
        out_file_train = os.path.join(out_dir_train_test,'train_'+str(fold_curr)+'.txt')
        train_anno = []
        for train_sub_file in train_sub_files:
            train_anno = train_anno+util.readLinesFromFile(train_sub_file)
        print len(train_anno)

        train_anno = [line.replace(dir_im_meta,new_dir_im_meta) for line in train_anno]
        for line_curr in train_anno:
            im_curr = line_curr.split(' ')[0]
            assert os.path.exists(im_curr)

        



        print train_anno[0]
        # raw_input()
        random.shuffle(train_anno)
        util.writeFile(out_file_train,train_anno)

        out_file_test = os.path.join(out_dir_train_test,'test_'+str(fold_curr)+'.txt')
        test_anno = []
        for test_sub_file in test_sub_files:
            test_anno = test_anno+util.readLinesFromFile(test_sub_file)
        print len(test_anno)

        test_anno = [line.replace(dir_im_meta,new_dir_im_meta) for line in test_anno]
        for line_curr in test_anno:
            im_curr = line_curr.split(' ')[0]
            assert os.path.exists(im_curr)

        random.shuffle(test_anno)
        util.writeFile(out_file_test,test_anno)

    # mean_std = np.array([[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]])
    # print mean_std.shape
    # print mean_std
    # np.save(os.path.join(out_dir_train_test,'mean_std.npy'),mean_std)


def save_resize_im((in_file,out_file,im_size,idx_file_curr)):
    if idx_file_curr%1000 ==0:
        print idx_file_curr

    img = cv2.imread(in_file);
    # gray  =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if im_size is not None:
    img_new = cv2.resize(img,tuple(im_size));
    cv2.imwrite(out_file,img_new)


def script_save_256_im():
    dir_meta = '../data/disfa'
    out_dir_train_test = os.path.join(dir_meta,'train_test_8_au_all_method_110_gray_align')

    im_size = [256,256]
    out_dir_im = os.path.join(dir_meta,'preprocess_im_256_color_align')

    num_folds = 3
    im_files_all = []
    for fold_curr in range(num_folds):
        for file_pre in ['train','test']:
            file_curr = os.path.join(out_dir_train_test,file_pre+'_'+str(fold_curr)+'.txt')
            print file_curr,len(util.readLinesFromFile(file_curr))

            im_files = [line_curr.split(' ')[0] for line_curr in util.readLinesFromFile(file_curr)]
            im_files_all.extend(im_files)

    print len(im_files_all),len(list(set(im_files_all)))
    im_files_all = list(set(im_files_all))
    str_replace_in_files = [os.path.join(dir_meta,'preprocess_im_110_gray_align'),os.path.join(dir_meta,'Videos_LeftCamera_frames_200')]
    im_files_all = [file_curr.replace(str_replace_in_files[0],str_replace_in_files[1]) for file_curr in im_files_all]

    str_replace_out_files = [str_replace_in_files[1],out_dir_im]
    args =[]
    for idx_file_curr, im_file_curr in enumerate(im_files_all):
        out_file_curr = im_file_curr.replace(str_replace_out_files[0],str_replace_out_files[1])
        out_dir_curr = os.path.split(out_file_curr)[0]
        util.makedirs(out_dir_curr)
        if os.path.exists(out_file_curr):
            continue
        args.append((im_file_curr,out_file_curr,im_size,idx_file_curr))

    print len(im_files_all)
    print len(args)
    # for arg in args:
    #     print arg
    #     raw_input()
    #     save_resize_im(arg)
    #     break

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(save_resize_im, args)






def main():
    save_train_test_files()
    # script_save_256_im()


    # make_disfa_8au_anno_all()
    # make_folds()

    # make_disfa_800_1600_anno()

    # script_save_mean_std_files()
    # script_change_train_test()
    # save_train_test_files()
    # script_make_im_gray()
    
    








    












        

    
if __name__=='__main__':
    main()