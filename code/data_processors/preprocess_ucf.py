# import cv2
import sys
sys.path.append('./')
import os
from helpers import util, visualize
import numpy as np
import glob
import scipy.misc
import scipy.stats
import multiprocessing
import subprocess
import scipy.io
import random
from globals import class_names

dir_meta = '../data/ucf101'
dir_meta_features = '../data/i3d_features'
def compare_rgb(old_rgb,new_rgb_folder):
    old_rgb = np.load(old_rgb)
    new_rgb = make_rgb_np(new_rgb_folder,224)
    assert new_rgb.shape[1]==old_rgb.shape[1]

    # print len(new_rgb_paths)
    print old_rgb.shape

    
    print old_rgb.shape, new_rgb.shape
    print np.min(old_rgb), np.max(old_rgb),np.mean(old_rgb)
    print np.min(new_rgb), np.max(new_rgb),np.mean(new_rgb)

    diff_mat = np.abs(new_rgb-old_rgb)
    diff_mat = np.around(diff_mat,3)
    uni_val, counts = np.unique(diff_mat,return_counts = True)
    mode = uni_val[np.argmax(counts)]
    print counts[np.argmax(counts)]

    print np.min(diff_mat), np.max(diff_mat), np.mean(diff_mat), mode


def compare_flo(old_flo, new_flo_folders):
    old_flo = np.load(old_flo)
    new_flo = make_flo_np(new_flo_folders[0],new_flo_folders[1], 224)

    assert new_flo.shape[1]==old_flo.shape[1]
    print old_flo.shape, new_flo.shape
    print np.min(old_flo), np.max(old_flo),np.mean(old_flo)
    print np.min(new_flo), np.max(new_flo),np.mean(new_flo)

    diff_mat = np.abs(new_flo-old_flo)
    diff_mat = np.around(diff_mat,5)
    uni_val, counts = np.unique(diff_mat,return_counts = True)
    mode = uni_val[np.argmax(counts)]
    print counts[np.argmax(counts)]

    print np.min(diff_mat), np.max(diff_mat), np.mean(diff_mat), mode

def make_flo_np(dir_flo_u,dir_flo_v,crop_size):
    new_flo_paths = glob.glob(os.path.join(dir_flo_u,'*.jpg'))
    new_flo_paths.sort()
    
    new_flo = np.zeros((1,len(new_flo_paths),crop_size,crop_size,2))
    for idx_im_curr, flo_u in enumerate(new_flo_paths):
        # print idx_im_curr
        flo_v = os.path.join(dir_flo_v,os.path.split(flo_u)[1])
        assert os.path.exists(flo_v)
        new_flo[0,idx_im_curr]= preprocess_flo(flo_u,flo_v)
    return new_flo


def make_rgb_np(new_rgb_folder,crop_size):
    new_rgb_paths = glob.glob(os.path.join(new_rgb_folder,'*.jpg'))
    new_rgb_paths.sort()
    # new_rgb_paths = new_rgb_paths
    # [:-1]
    new_rgb = np.zeros((1,len(new_rgb_paths),crop_size,crop_size,3))
    for idx_im_curr, im_curr in enumerate(new_rgb_paths):
        new_rgb[0,idx_im_curr]= preprocess_rgb(im_curr)

    return new_rgb


def preprocess_flo(flo_u,flo_v,crop_size = 224):
    u_im = scipy.misc.imread(flo_u).astype(float)[:,:,np.newaxis]
    v_im = scipy.misc.imread(flo_v).astype(float)[:,:,np.newaxis]
    flo = np.concatenate([u_im,v_im],2)
    
    start_row = (flo.shape[0]-crop_size)//2 
    start_col = (flo.shape[1]-crop_size)//2 

    flo_crop = flo[start_row:start_row+crop_size,start_col:start_col+crop_size]
    flo_crop = flo_crop/255. *2 -1
    return flo_crop

def preprocess_rgb(im_path,crop_size = 224):
    im = scipy.misc.imread(im_path).astype(float)
    
    start_row = (im.shape[0]-crop_size)//2 
    start_col = (im.shape[1]-crop_size)//2 

    im_crop = im[start_row:start_row+crop_size,start_col:start_col+crop_size]

    im_crop = im_crop/255 *2 -1
    return im_crop

def save_numpy((video, dir_rgb, dir_flos, crop_size, out_file, idx_video)):
    if idx_video%10==0:
        print idx_video

    try:
        rgb_dir = os.path.join(dir_rgb,video)
        u_dir = os.path.join(dir_flos[0], video)
        v_dir = os.path.join(dir_flos[1], video)
        assert os.path.exists(u_dir)
        assert os.path.exists(v_dir)
        assert os.path.exists(rgb_dir)

        rgb_np = make_rgb_np(rgb_dir, crop_size)
        flo_np = make_flo_np(u_dir, v_dir, crop_size)
        # print rgb_np.shape, flo_np.shape

        assert rgb_np.shape[1]>=flo_np.shape[1]

        if rgb_np.shape[1]> flo_np.shape[1]:
            rgb_np = rgb_np[:,:flo_np.shape[1],:,:]

        assert np.all(rgb_np.shape[:-1]==flo_np.shape[:-1])

        # raw_input()
        # print out_file
        np.savez(out_file, rgb= rgb_np, flo=flo_np)
    except:
        print 'ERROR', out_file, dir_rgb

def get_numpys(video, dir_rgb, dir_flos, crop_size):
    rgb_dir = os.path.join(dir_rgb,video)
    u_dir = os.path.join(dir_flos[0], video)
    v_dir = os.path.join(dir_flos[1], video)
    assert os.path.exists(u_dir)
    assert os.path.exists(v_dir)
    assert os.path.exists(rgb_dir)

    rgb_np = make_rgb_np(rgb_dir, crop_size)
    flo_np = make_flo_np(u_dir, v_dir, crop_size)
    # print rgb_np.shape, flo_np.shape

    assert rgb_np.shape[1]>=flo_np.shape[1]

    if rgb_np.shape[1]> flo_np.shape[1]:
        rgb_np = rgb_np[:,:flo_np.shape[1],:,:]

    assert np.all(rgb_np.shape[:-1]==flo_np.shape[:-1])
    return rgb_np, flo_np



def script_save_numpys():
    
    out_dir = os.path.join(dir_meta,'npys')
    util.mkdir(out_dir)

    dir_rgb = os.path.join(dir_meta, 'rgb_ziss/jpegs_256')

    dir_flos = os.path.join(dir_meta,'flow_ziss/tvl1_flow')
    dir_flos = [os.path.join(dir_flos,'u'),os.path.join(dir_flos,'v')]

    videos = [os.path.split(dir_curr)[1] for dir_curr in glob.glob(os.path.join(dir_rgb,'*')) if os.path.isdir(dir_curr)]
    print len(videos)

    args = []
    for idx_video, video in enumerate(videos):
        out_file = os.path.join(out_dir,video+'.npz')
        if os.path.exists(out_file):
            continue

        arg_curr = (video, dir_rgb, dir_flos, 224, out_file, idx_video)
        args.append(arg_curr)

    print len(args)

    # for arg_curr in args:
    #   save_numpy(arg_curr)
    #   break

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(save_numpy,args)


def redo_numpys():
    out_dir = '../data/ucf101/npys'
    all_numpys = glob.glob(os.path.join(out_dir,'*.npz'))[:10]
    problem_numpys = []
    problem_count = 0
    
    for idx_np_curr, np_curr in enumerate(all_numpys):
        
        # if idx_np_curr%100==0:
        print idx_np_curr

        # try:
        # print 'no problem'
        data = np.load(np_curr)
            # np.savez_compressed(np_curr, rgb= data['rgb'], flo=data['flo'])
        # except:
            # print 'problem'
            # problem_count+=1
            # problem_numpys.append(np_curr)


    print len(all_numpys), problem_count
    out_file = 'problem_npys.txt'
    util.writeFile(out_file, problem_npys)


def extract_frames((input_path,out_dir,fps,out_res,idx)):
    # command = 'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of default=nw=1:nk=1 '
    # command = command+input_path
    # ret_vals = os.popen(command).readlines()
    
    # ret_vals = [int(val.strip('\n')) for val in ret_vals]
    
    # if ret_vals[0]>ret_vals[1]:
    #   pass
    # else:
    #   print 'width bigger!!!'
    print idx
    util.mkdir(out_dir)
    command = ['ffmpeg']

    command += ['-i',input_path]

    command += ['-r' , str(fps)]
    command += ['-vf' ,'scale='+ '-1:'+str(out_res)]
    command += [os.path.join(out_dir,'frame%06d.jpg')]
    command += ['-hide_banner']
    command += ['> /dev/null 2> err.txt']
    command = ' '.join(command)

    # print command
    subprocess.call(command,shell=True)


def checking_frame_extraction(in_dir=None,out_dir=None, small_dim = 256, fps = 10):
    
    if in_dir is None:
        in_dir = os.path.join(dir_meta,'val_data','validation')

    if out_dir is None:
        out_dir = os.path.join(dir_meta,'val_data','rgb_'+str(fps)+'_fps_'+str(small_dim))

    util.mkdir(out_dir)

    in_files = glob.glob(os.path.join(in_dir,'*.mp4'))
    in_files.sort()

    print len(in_files)
    
    out_file_problem = os.path.join(out_dir,'problem_files.txt')

    problem = []
    for idx_in_file, in_file in enumerate(in_files):
        print idx_in_file
        out_dir_curr = os.path.split(in_file)[1]
        out_dir_curr = out_dir_curr[:out_dir_curr.rindex('.')]
        out_dir_curr = os.path.join(out_dir,out_dir_curr)

        command = ['ffmpeg', '-i']
        command += [in_file, '2>&1']
        command += ['|','grep "Duration"']
        command += ['|', 'cut -d', "' '",'-f 4']
        command += ['|', 'sed s/,//']
        # command += ['|', 'sed','s@\..*@@g' ]
        # command += ['|', 'awk',"'{", 'split($1, A, ":");', 'split(A[3], B, ".");', 'print 3600*A[1] + 60*A[2] + B[1]', "}'"]
        command = ' '.join(command)
        # print command

        secs = os.popen(command).readlines()
        try:
            secs = secs[0].strip('\n')
            secs = secs.split(':')
            assert len(secs)==3
            secs = secs[:-1] + secs[-1].split('.')[:1]
            secs = int(secs[0])*3600+int(secs[1])*60+int(secs[2])
            # print secs
            num_frames = secs*fps

            num_frames_ac = len(glob.glob(os.path.join(out_dir_curr,'*.jpg')))
            if (num_frames_ac-num_frames)<0:
                print 'PROBLEM',num_frames_ac,num_frames
                problem.append(' '.join([in_file,str(num_frames_ac),str(num_frames)]))
        except:
            print 'PROBLEM SERIOUS',in_file
            problem.append(in_file)

    util.writeFile(out_file_problem,problem)

        # break


def get_video_duration(in_file):
    print in_file
    command = ['ffmpeg', '-i']
    command += [in_file, '2>&1']
    command += ['|','grep "Duration"']
    command += ['|', 'cut -d', "' '",'-f 4']
    command += ['|', 'sed s/,//']
    command = ' '.join(command)
    
    secs = os.popen(command).readlines()
    secs = secs[0].strip('\n')
    secs = secs.split(':')
    assert len(secs)==3
    secs = secs[:-1] + secs[-1].split('.')[:1]
    secs = int(secs[0])*3600+int(secs[1])*60+int(secs[2])
    return secs


def script_extract_frames(in_dir=None,out_dir=None, small_dim = 256, fps = 10):
    
    if in_dir is None:
        in_dir = os.path.join(dir_meta,'val_data','validation')

    if out_dir is None:
        out_dir = os.path.join(dir_meta,'val_data','rgb_'+str(fps)+'_fps_'+str(small_dim))

    util.mkdir(out_dir)

    in_files = glob.glob(os.path.join(in_dir,'*.mp4'))
    in_files.sort()

    print len(in_files)

    args = []
    for idx_in_file, in_file in enumerate(in_files):
        out_dir_curr = os.path.split(in_file)[1]
        out_dir_curr = out_dir_curr[:out_dir_curr.rindex('.')]
        out_dir_curr = os.path.join(out_dir,out_dir_curr)
        
        if os.path.exists(os.path.join(out_dir_curr,'frame000001.jpg')):
            continue

        args.append((in_file, out_dir_curr, fps, small_dim, idx_in_file))

    print len(args)
    # print args[0]
    # extract_frames(args[0])
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(extract_frames, args)
    # print 'done'

def split_and_save_i3d():
    np_file ='../data/i3d_features/Thumos14-I3D-JOINTFeatures.npy'
    
    
    videos_path = '../data/ucf101/val_data/validation'
    out_dir = '../data/i3d_features/Thumos14-I3D-JOINTFeatures_val'
    test=False
    util.mkdir(out_dir)

    videos_path = '../data/ucf101/test_data/TH14_test_set_mp4'
    out_dir = '../data/i3d_features/Thumos14-I3D-JOINTFeatures_test'
    test=True
    util.mkdir(out_dir)

    val_videos = glob.glob(os.path.join(videos_path, '*.mp4'))
    val_videos.sort()

    print len(val_videos)
    print val_videos[0]
    print val_videos[-1]

    features = np.load(np_file)

    if test:
        features = features[-len(val_videos):]
    else:
        features = features[:len(val_videos)]

    print len(features),len(val_videos)

    for features_idx, features_curr in enumerate(features[:len(val_videos)]):
        # print features_idx, val_videos[features_idx]
        # if features_idx<1009:
        #     continue
        out_file = os.path.join(out_dir,os.path.split(val_videos[features_idx])[1].replace('.mp4','.npy'))
        # print out_file
        # print features_curr.shape
        print features_idx,out_file,features_curr.shape
        # raw_input()
        np.save(out_file,features_curr)



    

def script_get_video_durations():
    dir_data = os.path.join(dir_meta,'test_data','TH14_test_set_mp4')
    out_file_durations = os.path.join(dir_meta,'test_data','durations.txt')

    video_list = glob.glob(os.path.join(dir_data,'*.mp4'))
    video_list.sort()
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    durations = pool.map(get_video_duration,video_list)
    lines = [video_list[idx_curr]+' '+str(duration_curr) for idx_curr,duration_curr in enumerate(durations)]
    util.writeFile(out_file_durations,lines)

def check_feature_length(dir_features=None, file_duration_file=None, fps = 25):
    if dir_features is None:
        dir_features = os.path.join('../data','i3d_features', 'Thumos14-I3D-JOINTFeatures_val')
    
    if file_duration_file is None:
        file_duration_file = os.path.join(dir_meta,'val_data','durations.txt')

    problem_files = []
    out_file_problems = os.path.join(os.path.split(file_duration_file)[0], 'problem_lengths.txt')

    lines = util.readLinesFromFile(file_duration_file)
    
    for idx_line_curr, line_curr in enumerate(lines):
        print idx_line_curr
        line_curr = line_curr.split(' ')
        video_curr = line_curr[0]
        duration = int(line_curr[1])
        
        feature_file = os.path.join(dir_features,os.path.split(video_curr)[1].replace('.mp4','.npy'))
        

        feature_curr = np.load(feature_file)
        num_features = feature_curr.shape[0]
        pred_num_features = (duration*fps)//16
        if not np.abs(pred_num_features - num_features)<=2:
            to_print = ' '.join(line_curr+[str(num_features), str(pred_num_features)])
            print to_print
            problem_files.append(to_print)

        if '262' in video_curr:
            print duration, fps, pred_num_features
            print num_features
            raw_input()
            
        
    util.writeFile(out_file_problems, problem_files)

    # lines = [line_curr.split(' ') for line_curr in lines]
    # videos = [line_curr[0] for line_curr in lines]

def make_rough_anno_mat(test = False):
    dir_meta_curr = os.path.join(dir_meta, 'val_data')
    
    if test:
        dir_meta_curr = os.path.join(dir_meta, 'test_data')

    anno_dir = os.path.join(dir_meta_curr, 'annotation')

    out_dir_anno_rough = os.path.join(dir_meta_curr, 'anno_rough')
    util.mkdir(out_dir_anno_rough)

    files_anno = glob.glob(os.path.join(anno_dir,'*.txt'))
    files_anno = [file_curr for file_curr in files_anno if not os.path.split(file_curr)[1].startswith('Ambiguous')]
    classes = [os.path.split(file_curr[:file_curr.rindex('_')])[1] for file_curr in files_anno]
    classes.sort()
    assert len(classes)==20

    dict_annos = {}

    for file_curr in files_anno:
        annos = util.readLinesFromFile(file_curr)
        class_curr = os.path.split(file_curr[:file_curr.rindex('_')])[1]
        anno_num = classes.index(class_curr)

        for anno_curr in annos:
            anno_curr = anno_curr.split()
            key = anno_curr[0]
            start = float(anno_curr[1])
            end = float(anno_curr[2])
            if key not in dict_annos:
                dict_annos[key] = [[] for idx in range(20)]
            dict_annos[key][anno_num].append([start, end])

    for key in dict_annos.keys():
        annos = dict_annos[key]
        annos = [np.array(anno_curr) for anno_curr in annos]
        assert len(annos)==20

        out_file = os.path.join(out_dir_anno_rough, key+'.npy')
        # print out_file
        np.save(out_file, annos)

        # annos = np.load(out_file)
        # print annos.shape
        # for anno_curr in annos:
        #     print anno_curr.shape

        # break
            
def make_real_anno_mat(test = False, fps = 25):
    dir_meta_curr = os.path.join(dir_meta, 'val_data')
    feature_dir = os.path.join(dir_meta_features,'Thumos14-I3D-JOINTFeatures_val')
    
    if test:
        dir_meta_curr = os.path.join(dir_meta, 'test_data')
        feature_dir = os.path.join(dir_meta_features,'Thumos14-I3D-JOINTFeatures_test')

    anno_dir = os.path.join(dir_meta_curr, 'annotation')

    dir_anno_rough = os.path.join(dir_meta_curr, 'anno_rough')
    dir_anno_mat = os.path.join(dir_meta_curr, 'anno_mat_temporal')

    util.mkdir(dir_anno_mat)

    video_list = glob.glob(os.path.join(dir_anno_rough, '*.npy'))
    for video_file in video_list:
        video_name = os.path.split(video_file)[1]
        feature_file = os.path.join(feature_dir,video_name)
        out_file = os.path.join(dir_anno_mat, video_name)
        # print feature_file, out_file

        features = np.load(feature_file)
        num_features = features.shape[0]
        annos = np.load(video_file)
        anno_mat = np.zeros((num_features,20))

        for idx_anno_curr, anno_curr in enumerate(annos):
            # print idx_anno_curr, anno_curr.shape
            if anno_curr.size==0:
                continue
            
            # print anno_curr.shape, np.min(anno_curr), np.max(anno_curr)
            anno_curr = np.round(anno_curr).astype(int)
            # print anno_curr.shape, np.min(anno_curr), np.max(anno_curr)
            anno_curr = anno_curr * fps
            # print anno_curr.shape, np.min(anno_curr), np.max(anno_curr)
            anno_curr = anno_curr//16
            # assert np.min(anno_curr)>=0
            # print anno_curr.shape, np.min(anno_curr), np.max(anno_curr)
            # print num_features
            
            for [start,end] in anno_curr:
                # assert np.max(anno_curr) <=num_features 
                if end>num_features:
                    end = num_features
                if start>num_features:
                    print 'PROBLEM', feature_file, idx_anno_curr, num_features, start, end
                    continue
                # print start,end
                anno_mat[start:end,idx_anno_curr]=1

        np.save(out_file, anno_mat)


def verify_class_labels(test= False):
    dir_meta_curr = os.path.join(dir_meta, 'val_data')
    val_meta_path  = '../data/ucf101/val_data/validation_set_meta/validation_set_meta/validation_set.mat'
    key = 'validation_videos'
    
    if test:
        dir_meta_curr = os.path.join(dir_meta, 'test_data')
        val_meta_path = '../data/ucf101/test_data/test_set_meta.mat'
        key = 'test_videos'


    anno_dir = os.path.join(dir_meta_curr, 'annotation')
    rough_dir = os.path.join(dir_meta_curr, 'anno_rough')

    files_anno = glob.glob(os.path.join(anno_dir,'*.txt'))
    files_anno = [file_curr for file_curr in files_anno if not os.path.split(file_curr)[1].startswith('Ambiguous')]
    classes = [os.path.split(file_curr[:file_curr.rindex('_')])[1] for file_curr in files_anno]
    classes.sort()
    assert len(classes)==20
    print classes

    to_check = glob.glob(os.path.join(rough_dir,'*.npy'))

    to_check = [os.path.split(file_curr)[1].replace('.npy','') for file_curr in to_check]

    val = scipy.io.loadmat(val_meta_path, struct_as_record = False)[key][0]
    print len(val)
    idx_old = 0

    actions = []
    anno_mat_1 = []
    anno_mat_2 = []



    for val_curr in val:
        
        

        if val_curr.video_name[0].replace('.mp4','') not in to_check:
            continue
    
        rel_anno = os.path.join(rough_dir, val_curr.video_name[0]+'.npy')
        rel_anno = np.load(rel_anno)
        sizes = [0 if mat.size==0 else 1 for mat in rel_anno ]

        class_1 = list(val_curr.primary_action)
        if len(val_curr.secondary_actions)>0:
            class_1 +=list(val_curr.secondary_actions[0][0])
        
        anno_1 = [1 if classes[idx] in class_1 else 0 for idx in range(20)]
        
        class_2 = [classes[idx_curr] for idx_curr in np.where(sizes)[0]]
        # for idx_curr in np.where(sizes)[0]:
        #     print val_curr.video_name, class_1, classes[idx_curr], class_1==classes[idx_curr], class_1 in classes, len(np.where(sizes)[0])

        anno_mat_1.append(anno_1)
        anno_mat_2.append(sizes)

        good_one  = np.all(anno_1==sizes)
        if not good_one or sum(anno_1)==0:
            print val_curr.video_name[0], class_1, class_2, sum(anno_1)



def write_train_test_files(test = False, just_primary = False):
    out_dir_files = os.path.join(dir_meta,'train_test_files')
    util.mkdir(out_dir_files)


    dir_meta_curr = os.path.join(dir_meta, 'val_data')
    val_meta_path  = '../data/ucf101/val_data/validation_set_meta/validation_set_meta/validation_set.mat'
    key = 'validation_videos'
    feature_dir = os.path.join(dir_meta_features,'Thumos14-I3D-JOINTFeatures_val')
    out_file = os.path.join(out_dir_files,'train.txt')

    if test:
        dir_meta_curr = os.path.join(dir_meta, 'test_data')
        val_meta_path = '../data/ucf101/test_data/test_set_meta.mat'
        key = 'test_videos'
        feature_dir = os.path.join(dir_meta_features,'Thumos14-I3D-JOINTFeatures_test')
        out_file = os.path.join(out_dir_files,'test.txt')

    anno_dir = os.path.join(dir_meta_curr, 'annotation')
    rough_dir = os.path.join(dir_meta_curr, 'anno_rough')

    if just_primary:
        out_file = out_file[:out_file.rindex('.')]+'_just_primary.txt'

    files_anno = os.path.join(anno_dir,'*.txt') if not test else os.path.join(anno_dir,'*_test.txt')
    files_anno = glob.glob(files_anno)
    print len(files_anno)
    for file_curr in files_anno:
        print file_curr 

    assert len(files_anno)==21
    files_anno = [file_curr for file_curr in files_anno if not os.path.split(file_curr)[1].startswith('Ambiguous')]
    classes = [os.path.split(file_curr[:file_curr.rindex('_')])[1] for file_curr in files_anno]
    classes.sort()
    assert len(classes)==20

    to_check = glob.glob(os.path.join(rough_dir,'*.npy'))
    to_check = [os.path.split(file_curr)[1].replace('.npy','') for file_curr in to_check]
    val = scipy.io.loadmat(val_meta_path, struct_as_record = False)[key][0]
    print len(val)
    idx_old = 0

    out_lines = []


    for val_curr in val:

        if val_curr.video_name[0].replace('.mp4','') not in to_check:
            continue
    
        class_1 = list(val_curr.primary_action)
        if len(val_curr.secondary_actions)>0 and not just_primary:
            class_1 +=list(val_curr.secondary_actions[0][0])
        
        anno_1 = [1 if classes[idx] in class_1 else 0 for idx in range(20)]
        if sum(anno_1)>0:
            file_feat = os.path.join(feature_dir,val_curr.video_name[0]+'.npy')
            assert os.path.exists(file_feat)
            out_line = ' '.join([str(val) for val in [file_feat]+anno_1])
            out_lines.append(out_line)

    print out_file, len(out_lines)
    util.writeFile(out_file,out_lines)

def write_train_test_files_all(test = False, just_primary = False):
    out_dir_files = os.path.join(dir_meta,'train_test_files')
    util.mkdir(out_dir_files)


    dir_meta_curr = os.path.join(dir_meta, 'val_data')
    val_meta_path  = '../data/ucf101/val_data/validation_set_meta/validation_set_meta/validation_set.mat'
    key = 'validation_videos'
    feature_dir = os.path.join(dir_meta_features,'Thumos14-I3D-JOINTFeatures_val')
    out_file = os.path.join(out_dir_files,'train_all.txt')

    if test:
        dir_meta_curr = os.path.join(dir_meta, 'test_data')
        val_meta_path = '../data/ucf101/test_data/test_set_meta.mat'
        key = 'test_videos'
        feature_dir = os.path.join(dir_meta_features,'Thumos14-I3D-JOINTFeatures_test')
        out_file = os.path.join(out_dir_files,'test_all.txt')

    if just_primary:
        out_file = out_file[:out_file.rindex('.')]+'_just_primary.txt'

    anno_dir = os.path.join(dir_meta_curr, 'annotation')
    rough_dir = os.path.join(dir_meta_curr, 'anno_rough')


    classes_file = os.path.join(dir_meta,'train_data','Class Index.txt')
    lines = util.readLinesFromFile(classes_file)
    classes = [line_curr.split(' ')[1].strip('\r') for line_curr in lines]
    
    print classes
    assert len(classes)==101

    # to_check = glob.glob(os.path.join(rough_dir,'*.npy'))
    # to_check = [os.path.split(file_curr)[1].replace('.npy','') for file_curr in to_check]
    val = scipy.io.loadmat(val_meta_path, struct_as_record = False)[key][0]
    print len(val)
    # idx_old = 0

    out_lines = []


    for val_curr in val:

    #     if val_curr.video_name[0].replace('.mp4','') not in to_check:
    #         continue
    
        class_1 = list(val_curr.primary_action)
        if len(val_curr.secondary_actions)>0 and not just_primary:
            class_1 +=list(val_curr.secondary_actions[0][0])
        
        anno_1 = [1 if classes[idx] in class_1 else 0 for idx in range(len(classes))]
        if sum(anno_1)>0:
            file_feat = os.path.join(feature_dir,val_curr.video_name[0]+'.npy')
            assert os.path.exists(file_feat)
            out_line = ' '.join([str(val) for val in [file_feat]+anno_1])
            out_lines.append(out_line)

    print out_file, len(out_lines)
    print  out_lines[0] 
    assert not os.path.exists(out_file)
    util.writeFile(out_file,out_lines)

def write_classes_list_files():
    classes_file = os.path.join(dir_meta,'train_data','Class Index.txt')
    lines = util.readLinesFromFile(classes_file)
    classes = [line_curr.split(' ')[1].strip('\r') for line_curr in lines]
    
    out_file = os.path.join(dir_meta,'train_test_files','classes_all_list.txt')
    util.writeFile(out_file, classes)

    classes= ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']
    
    out_file = os.path.join(dir_meta,'train_test_files','classes_rel_list.txt')
    util.writeFile(out_file, classes)


def save_test_pair_file():
    dir_train_test_files = '../data/ucf101/train_test_files'
    num_to_pick = 20    
    train_file = os.path.join(dir_train_test_files,'train_ultra_correct.txt')
    test_file = os.path.join(dir_train_test_files,'test_ultra_correct.txt')

    train_data = util.readLinesFromFile(train_file)
    test_data = util.readLinesFromFile(test_file)

    test_new_data = []
    for line_curr in test_data:
        # rand_idx = random.randint(0,len(train_data)-1)
        num_rand = random.sample(xrange(len(train_data)), num_to_pick)
        for rand_idx in num_rand:
        
        # for line_train in train_data:
            test_new_data.append(line_curr)
            test_new_data.append(train_data[rand_idx])

    print len(test_new_data)
    print len(test_data)
    print len(train_data)
    print len(test_data)*len(train_data)*2
    out_file = os.path.join(dir_train_test_files,'test_pair_rand'+str(num_to_pick)+'_ultra_correct.txt')
    print out_file
    # raw_input()
    util.writeFile(out_file, test_new_data)


def write_cooc_train_test_files(num_neighbors = 100):
    dir_train_test = '../data/ucf101/train_test_files'
    in_files = ['train.txt','test.txt']
    out_post = '_cooc_'+str(num_neighbors).replace('/','_')
    cooc_dir = '../data/ucf101/i3d_dists_just_train/arr_coocs_'+str(num_neighbors)

    for in_file in in_files:
        just_name = in_file[:in_file.rindex('.')]
        out_file = os.path.join(dir_train_test, just_name+out_post+'.txt')
        in_file = os.path.join(dir_train_test, in_file)
        # print in_file, out_file
        lines = util.readLinesFromFile(in_file)
        new_lines = []
        for line_curr in lines:
            line_split = line_curr.split(' ')
            just_vid_name = os.path.split(line_split[0])[1]
            just_vid_name = just_vid_name[:just_vid_name.rindex('.')]
            cooc_file = os.path.join(cooc_dir, just_vid_name+'.npy')
            assert os.path.exists(cooc_file)
            line_new = [line_split[0],cooc_file]+line_split[1:]
            line_new = ' '.join(line_new)
            new_lines.append(line_new)
            # print line_new
            # raw_input()
        print out_file, len(new_lines), new_lines[0]
        raw_input()
        util.writeFile(out_file, new_lines)
        

def write_cooc_per_class_train_test_files():
    dir_train_test = '../data/ucf101/train_test_files'
    in_files = ['train.txt','test.txt']
    out_post = '_cooc_per_class'
    cooc_dir = '../data/ucf101/i3d_dists_just_train/arr_coocs_per_class'

    for in_file in in_files:
        just_name = in_file[:in_file.rindex('.')]
        out_file = os.path.join(dir_train_test, just_name+out_post+'.txt')
        in_file = os.path.join(dir_train_test, in_file)
        # print in_file, out_file
        lines = util.readLinesFromFile(in_file)
        new_lines = []
        for line_curr in lines:
            line_split = line_curr.split(' ')
            just_vid_name = os.path.split(line_split[0])[1]
            just_vid_name = just_vid_name[:just_vid_name.rindex('.')]
            line_new = [line_split[0]]
            # ,cooc_file]+line_split[1:]
            for class_name in class_names:
                cooc_file = os.path.join(cooc_dir,class_name, just_vid_name+'.npz')
                assert os.path.exists(cooc_file)
                line_new.append(cooc_file)

            line_new +=line_split[1:]
            
            line_new = ' '.join(line_new)
            new_lines.append(line_new)
            # print line_new
            # raw_input()
        print out_file, len(new_lines), new_lines[0]
        raw_input()
        util.writeFile(out_file, new_lines)
        

def merge_coocs_per_class():
    dir_train_test = '../data/ucf101/train_test_files'
    in_files = ['train_cooc_per_class.txt','test_cooc_per_class.txt']
    cooc_dir = '../data/ucf101/i3d_dists_just_train/arr_coocs_per_class'
    out_dir = os.path.join(cooc_dir,'merged')
    util.mkdir(out_dir)

    for in_file in in_files:
        file_curr = os.path.join(dir_train_test,in_file)
        lines = util.readLinesFromFile(file_curr)
        for line in lines:
            line_split = line.split(' ')
            vid_name = os.path.split(line_split[0])[1]
            vid_name = vid_name[:vid_name.rindex('.')]
            out_file = os.path.join(out_dir,vid_name+'.npy')
            if os.path.exists(out_file):
                continue
            print vid_name
            assert len(line_split)==41
            npz_files = line_split[1:21]
            npzs = [np.load(npz_file_curr)['arr_0'][np.newaxis,:,:] for npz_file_curr in npz_files]
            npzs = np.concatenate(npzs, axis = 0)
            # print npzs.shape

            
            # print out_file
            # raw_input()
            # np.savez_compressed(out_file, npzs)
            np.save(out_file, npzs)
        #     break
        # break



def main():
    # file_npz = '../data/ucf101/i3d_dists_just_train/arr_coocs_per_class/merged/video_validation_0000051.npz'
    # file_npy = file_npz.replace('.npz','.npy')
    # import time
    # t = time.time()
    # arr = np.load(file_npz)['arr_0']
    # print time.time()-t

    # t = time.time()
    # arr = np.load(file_npy)
    # print time.time()-t

    # merge_coocs_per_class()
    write_cooc_train_test_files(num_neighbors = 'per_class/merged')
    # write_cooc_per_class_train_test_files()

    # save_test_pair_file()

    # just_primary = True
    # write_train_test_files_all(False, just_primary = just_primary)
    # write_train_test_files_all(True, just_primary = just_primary)
    # write_train_test_files(False, just_primary = just_primary)
    # write_train_test_files(True, just_primary = just_primary)


    # write_train_test_files()
    # write_train_test_files(True)

    # verify_class_labels(False)
    # make_real_anno_mat(True)
    # check_feature_length(dir_features=os.path.join('../data','i3d_features', 'Thumos14-I3D-JOINTFeatures_test'), file_duration_file=os.path.join(dir_meta, 'test_data','durations.txt'))
    # check_feature_length()

    # script_get_video_durations()
    # write_train_test_files()
    # split_and_save_i3d()    

    # in_dir = os.path.join(dir_meta,'test_data','TH14_test_set_mp4')
    # out_dir = os.path.join(dir_meta,'test_data','rgb_10_fps_256')

    # script_extract_frames(in_dir, out_dir)
    # checking_frame_extraction(in_dir, out_dir)

    # verify length of 



if __name__=='__main__':
    main()