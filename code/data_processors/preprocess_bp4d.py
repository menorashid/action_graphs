# import cv2
import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import scipy.misc
import numpy as np
import random
import skimage.transform
import multiprocessing
# import face_alignment
import dlib
import cv2
import shutil
import skimage.transform

def saveCroppedFace((in_file, out_file, im_size, savegray, idx_file_curr)):
    if idx_file_curr%100==0:
        print idx_file_curr

    classifier_path = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml';

    img = cv2.imread(in_file);
    
    gray  =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade  =  cv2.CascadeClassifier(classifier_path)
    faces  =  face_cascade.detectMultiScale(gray)
    if len(faces)==0:
        print 'PROBLEM';
        return -1
    else:
        sizes=np.array([face_curr[2]*face_curr[3] for face_curr in faces]);
        faces=faces[np.argmax(sizes)];
        size_crop = np.max(sizes)

    [x,y,w,h] = faces;
    
    roi = gray[y:y+h, x:x+w]    
    if not savegray:
        roi = img[y:y+h, x:x+w]

    if im_size is not None:
        roi=cv2.resize(roi,tuple(im_size));
    cv2.imwrite(out_file,roi)

    return size_crop




def save_resized_images((in_file,out_file,im_size,savegray,idx_file_curr)):
    if idx_file_curr%100==0:
        print idx_file_curr

    img = cv2.imread(in_file);
    
    if savegray:
        gray  =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi=cv2.resize(gray,tuple(im_size));
    else:
        roi=cv2.resize(img,tuple(im_size));

    cv2.imwrite(out_file,roi)


def test_face_detector():
    
    face_detector_path = '../data/mmod_human_face_detector.dat'
    face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
    im_path = '../data/bp4d/preprocess_im_256_color_nodetect/M013/T3/267.jpg'
    # M003/T3/052.jpg'
    im = scipy.misc.imread(im_path)
    print im.shape
    bbox_all = face_detector(im, 1)
    boxes = []
    sizes = []
    for i,bbox in enumerate(bbox_all):
        bbox = bbox.rect
        crop_box = [bbox.top(),bbox.bottom(),bbox.left(),bbox.right()]
        sizes.append((crop_box[1]-crop_box[0])*(crop_box[3]-crop_box[2]))
        boxes.append(crop_box)

    best_box = boxes[np.argmax(sizes)]
    size_r = best_box[1]-best_box[0]
    size_c = best_box[3]-best_box[2]
    pad = [size_r//4,size_c//4]
    best_box = [max(0,best_box[0]-pad[0]),
                min(im.shape[0],best_box[1]+pad[0]),
                max(0,best_box[2]-pad[1]),
                min(im.shape[1],best_box[3]+pad[1])]
    print best_box
    im_crop = im[best_box[0]:best_box[1],best_box[2]:best_box[3]]
    print im_crop.shape

    scipy.misc.imsave('../scratch/im_crop.jpg',im_crop)

        # print bbox.right(),bbox.left(),bbox.top(),bbox.bottom()

def saveCroppedFace_NEW((in_file, out_file, im_size, savegray, idx_file_curr)):
    if idx_file_curr%100==0:
        print idx_file_curr

    classifier_path  = '../data/mmod_human_face_detector.dat'
    face_detector = dlib.cnn_face_detection_model_v1(classifier_path)

    im = scipy.misc.imread(in_file)
    bbox_all = face_detector(im, 1)
    boxes = []
    sizes = []
    if len(bbox_all)<1:
        print 'PROBLEM'
        return -1

    for i,bbox in enumerate(bbox_all):
        bbox = bbox.rect
        crop_box = [bbox.top(),bbox.bottom(),bbox.left(),bbox.right()]
        sizes.append((crop_box[1]-crop_box[0])*(crop_box[3]-crop_box[2]))
        boxes.append(crop_box)

    best_box = boxes[np.argmax(sizes)]
    size_crop = np.max(sizes)

    size_r = best_box[1]-best_box[0]
    size_c = best_box[3]-best_box[2]
    pad = [size_r//4,size_c//4]
    best_box = [max(0,best_box[0]-pad[0]),
                min(im.shape[0],best_box[1]+pad[0]),
                max(0,best_box[2]-pad[1]),
                min(im.shape[1],best_box[3]+pad[1])]
    
    im_crop = im[best_box[0]:best_box[1],best_box[2]:best_box[3]]
    

    if im_size is not None:
        roi=cv2.resize(im_crop,tuple(im_size));
    scipy.misc.imsave(out_file,roi)

    return size_crop
    

def saveCroppedFace_NEW_batch((file_pairs, im_size, savegray, idx_file_curr,save_all)):
    # if idx_file_curr%100==0:
    #     print idx_file_curr

    classifier_path  = '../data/mmod_human_face_detector.dat'
    face_detector = dlib.cnn_face_detection_model_v1(classifier_path)

    for idx_file_curr,(in_file,out_file) in enumerate(file_pairs):
        if idx_file_curr%10==0:
            print idx_file_curr

        im = scipy.misc.imread(in_file)
        bbox_all = face_detector(im, 1)
        boxes = []
        sizes = []
        if len(bbox_all)<1:
            print in_file
            print 'PROBLEM'
            continue

        for i,bbox in enumerate(bbox_all):
            bbox = bbox.rect
            crop_box = [bbox.top(),bbox.bottom(),bbox.left(),bbox.right()]
            sizes.append((crop_box[1]-crop_box[0])*(crop_box[3]-crop_box[2]))
            boxes.append(crop_box)

        # print len(boxes)
        if save_all:
            lefts = [box[2] for box in boxes]
            idx_sort = np.argsort(lefts)
            boxes = [boxes[idx_curr] for idx_curr in idx_sort]
            out_files = [out_file[:out_file.rindex('.')]+'_'+str(idx)+out_file[out_file.rindex('.'):] for idx in range(len(boxes))]
        else:
            out_files = [out_file]
            boxes = [boxes[np.argmax(sizes)]]

        for best_box,out_file in zip(boxes,out_files):  
            # print best_box, out_file

            # best_box = boxes[np.argmax(sizes)]
            # size_crop = np.max(sizes)

            size_r = best_box[1]-best_box[0]
            size_c = best_box[3]-best_box[2]
            pad = [size_r//4,size_c//4]
            best_box = [max(0,best_box[0]-pad[0]),
                        min(im.shape[0],best_box[1]+pad[0]),
                        max(0,best_box[2]-pad[1]),
                        min(im.shape[1],best_box[3]+pad[1])]
            
            im_crop = im[best_box[0]:best_box[1],best_box[2]:best_box[3]]
            
            if savegray:
                im_crop  =  cv2.cvtColor(im_crop, cv2.COLOR_RGB2GRAY)

            if im_size is not None:
                roi=cv2.resize(im_crop,tuple(im_size));
            scipy.misc.imsave(out_file,roi)



def script_save_bbox():
    dir_meta= '../data/bp4d'

    in_dir_meta = os.path.join(dir_meta,'preprocess_im_'+str(256)+'_color_nodetect')

    out_dir_meta = os.path.join(dir_meta,'preprocess_im_'+str(256)+'_color_nodetect_bbox')
    util.mkdir(out_dir_meta)
    

    im_list_in = glob.glob(os.path.join(in_dir_meta,'*','*','*.jpg'))
    print len(im_list_in)
    
    
    savegray = False
    args = []
    
    args = []
    file_pairs = []
    for idx_im_in,im_in in enumerate(im_list_in):
        out_file = im_in.replace(in_dir_meta,out_dir_meta).replace('.jpg','.npy')
        if os.path.exists(out_file):
            continue
        out_dir_curr = os.path.split(out_file)[0]
        util.makedirs(out_dir_curr)
        file_pairs.append((im_in,out_file))

    chunk_size = 5
    chunks = [file_pairs[x:x+chunk_size] for x in range(0, len(file_pairs), chunk_size)]
    args = [(chunk_curr,idx_im_in) for idx_im_in,chunk_curr in enumerate(chunks)]

    print len(args)
    # args = args[:1000]
    for arg in args:
        print arg
        size = save_best_bbox_batch(arg)
        raw_input()


    # pool = multiprocessing.Pool(4)
    # crop_sizes = pool.map(saveCroppedFace_NEW_batch,args)
    # content = []
    # out_im_all = [arg_curr[1] for arg_curr in args]
    # np.savez(os.path.join(dir_meta,'sizes_256.npz'),crop_sizes = np.array(crop_sizes),out_im_all = np.array(out_im_all))


def save_best_bbox_batch((file_pairs, idx_file_curr)):
    # if idx_file_curr%100==0:
    #     print idx_file_curr

    classifier_path  = '../data/mmod_human_face_detector.dat'
    face_detector = dlib.cnn_face_detection_model_v1(classifier_path)

    for idx_file_curr,(in_file,out_file) in enumerate(file_pairs):
        if idx_file_curr%10==0:
            print idx_file_curr

        im = scipy.misc.imread(in_file)
        bbox_all = face_detector(im, 1)
        boxes = []
        sizes = []
        if len(bbox_all)<1:
            print in_file
            print 'PROBLEM'
            continue

        for i,bbox in enumerate(bbox_all):
            bbox = bbox.rect
            crop_box = [bbox.top(),bbox.bottom(),bbox.left(),bbox.right()]
            sizes.append((crop_box[1]-crop_box[0])*(crop_box[3]-crop_box[2]))
            boxes.append(crop_box)

        best_box = boxes[np.argmax(sizes)]
        np.save(out_file,best_box)
  



def script_save_cropped_faces():
    dir_meta= '../data/bp4d'

    # in_dir_meta = os.path.join(dir_meta,'preprocess_im_'+str(256)+'_color_nodetect')
    # im_size = [110,110]
    # out_dir_meta = os.path.join(dir_meta,'preprocess_im_'+str(im_size)+'_color')
    # util.mkdir(out_dir_meta)

    in_dir_meta = os.path.join(dir_meta,'BP4D','BP4D-training')
    im_size = [256,256]
    out_dir_meta = os.path.join(dir_meta,'preprocess_im_'+str(im_size[0])+'_color')
    util.mkdir(out_dir_meta)
    

    im_list_in = glob.glob(os.path.join(in_dir_meta,'*','*','*.jpg'))
    print len(im_list_in)
    
    
    savegray = False
    args = []
    
    args = []
    file_pairs = []
    for idx_im_in,im_in in enumerate(im_list_in):
        out_file = im_in.replace(in_dir_meta,out_dir_meta)
        if os.path.exists(out_file):
            continue
        out_dir_curr = os.path.split(out_file)[0]
        util.makedirs(out_dir_curr)
        file_pairs.append((im_in,out_file))

    chunk_size = 500
    chunks = [file_pairs[x:x+chunk_size] for x in range(0, len(file_pairs), chunk_size)]
    args = [(chunk_curr,im_size,savegray,idx_im_in) for idx_im_in,chunk_curr in enumerate(chunks)]

    print len(args)
    # args = args[:1000]
    # for arg in args:
    #     print arg
    #     size = saveCroppedFace_NEW_batch(arg)
    #     # saveCroppedFace(arg)
    #     raw_input()


    pool = multiprocessing.Pool(4)
    crop_sizes = pool.map(saveCroppedFace_NEW_batch,args)
    content = []
    out_im_all = [arg_curr[1] for arg_curr in args]
    np.savez(os.path.join(dir_meta,'sizes_256.npz'),crop_sizes = np.array(crop_sizes),out_im_all = np.array(out_im_all))



def make_au_vec_per_frame(csv_file):
    lines = util.readLinesFromFile(csv_file)
    arr = []
    for line in lines:
        arr_curr = [int(val) for val in line.split(',')]
        arr.append(arr_curr)
    
    return np.array(arr)

def script_save_resize_faces():
    dir_meta= '../data/bp4d'
    im_size = [110,110]
    out_dir_meta = os.path.join(dir_meta,'preprocess_im_'+str(im_size[0])+'_color_nodetect')
    in_dir_meta = os.path.join(dir_meta,'BP4D','BP4D-training')
    # in_dir_meta = os.path.join(dir_meta,'preprocess_im_'+str(256)+'_color_nodetect')

    im_list_in = glob.glob(os.path.join(in_dir_meta,'*','*','*.jpg'))

    
    savegray = False
    args = []
    for idx_im_in,im_in in enumerate(im_list_in):
        out_file = im_in.replace(in_dir_meta,out_dir_meta)
        if os.path.exists(out_file):
            continue

        out_dir_curr = os.path.split(out_file)[0]
        # print out_dir_curr
        util.makedirs(out_dir_curr)
        args.append((im_in,out_file,im_size,savegray,idx_im_in))

    print len(args)
    # args = args[:10]
    # for arg in args:
        # print arg
        # save_resized_images(arg)
    #     size = saveCroppedFace(arg)
        # raw_input()



    pool = multiprocessing.Pool(4)
    pool.map(save_resized_images,args)
    # crop_sizes = 
    # content = []
    # out_im_all = [arg_curr[1] for arg_curr in args]
    # np.savez(os.path.join(dir_meta,'sizes.npz'),crop_sizes = np.array(crop_sizes),out_im_all = np.array(out_im_all))

def make_anno_files():
    au_keep = [1,2,4,6,7,10,12,14,15,17,23,24]
    out_dir = '../data/bp4d/anno_text'
    util.mkdir(out_dir)
    im_dir_meta = '../data/bp4d/BP4D/BP4D-training'

    # csv_file = '../data/bp4d/AUCoding/AUCoding/F001_T1.csv'
    csv_files = glob.glob('../data/bp4d/AUCoding/AUCoding/*.csv')
    print len(csv_files)
    total_lines = 0
    for idx_csv_file,csv_file in enumerate(csv_files):
        print idx_csv_file
        arr = make_au_vec_per_frame(csv_file)
        # print arr.shape
        # print arr[0,:]
        idx_keep = np.array([1 if val in au_keep else 0 for val in arr[0,:] ])
        idx_keep[0]=1

        # print idx_keep
        # print arr[0,idx_keep>0]


        rel_cols = arr[1:,idx_keep>0]

        # print rel_cols.shape
        
        ims = rel_cols[:,0]
        rest = rel_cols[:,1:]
        assert np.all(np.unique(rest)==np.array([0,1]))

        out_file = os.path.split(csv_file)[1][:-4]
        # print out_file

        subj,sess = out_file.split('_')

        im_dir_curr = os.path.join(im_dir_meta,subj,sess)
        examples = glob.glob(os.path.join(im_dir_curr,'*.jpg'))
        example = os.path.split(examples[0])[1][:-4]
        num_zeros = len(example)
        # print len(example)

        out_lines = []
        for im_idx, im in enumerate(ims):

            im_str = str(im)
            im_str = '0'*(num_zeros-len(im_str))+im_str
            im_file = os.path.join(im_dir_curr,im_str+'.jpg')

            anno = rest[im_idx]

            if not os.path.exists(im_file) or np.sum(anno)<1:
                continue
            
            anno_str = [str(val) for val in anno]
            assert len(anno_str)==len(au_keep)
            out_line = [im_file]+anno_str
            out_line = ' '.join(out_line)
            out_lines.append(out_line)

        total_lines +=len(out_lines)

        out_file_anno = os.path.join(out_dir,out_file+'.txt')
        util.writeFile(out_file_anno,out_lines)
        print out_file_anno,len(out_lines),total_lines


def make_train_test_subs():
    dir_meta = '../data/bp4d'
    im_dir_meta = os.path.join(dir_meta,'BP4D','BP4D-training')
    out_dir_subs = os.path.join(dir_meta,'subs')
    util.mkdir(out_dir_subs)

    subs = [os.path.split(dir_curr)[1] for dir_curr in glob.glob(os.path.join(im_dir_meta,'*'))]
    print subs
    print len(subs)
    subs.sort()
    print subs
    num_splits = 3
    folds = []
    for fold_num in range(num_splits):
        fold_curr = subs[fold_num::num_splits]
        folds.append(fold_curr)
    
    for fold_num in range(num_splits):
        train_folds = []
        for idx_fold,fold_curr in enumerate(folds):
            if idx_fold!=fold_num:
                train_folds = train_folds+fold_curr
        test_folds = folds[fold_num]
        out_file_train = os.path.join(out_dir_subs,'train_'+str(fold_num)+'.txt')
        out_file_test = os.path.join(out_dir_subs,'test_'+str(fold_num)+'.txt')
        assert len(train_folds)+len(test_folds)==len(list(set(train_folds+test_folds)))

        print fold_num, len(train_folds),len(test_folds)
        print out_file_train, out_file_test
        util.writeFile(out_file_train, train_folds)
        util.writeFile(out_file_test, test_folds)


def write_train_file(out_file_train, out_dir_annos, out_dir_im, train_folds, replace_str):

    all_anno_files = []
    for sub_curr in train_folds:
        all_anno_files= all_anno_files+glob.glob(os.path.join(out_dir_annos,sub_curr+'*.txt'))
    
    all_lines = []
    for anno_file in all_anno_files:
        all_lines = all_lines+util.readLinesFromFile(anno_file)

    out_lines = []
    total_missing = 0
    for line_curr in all_lines:
        out_line = line_curr.replace(replace_str,out_dir_im)
        im_out = out_line.split(' ')[0]
        if os.path.exists(im_out):
        # assert os.path.exists(im_out)
            out_lines.append(out_line)
        else:
            # print im_out
            total_missing+=1

    print total_missing
    print len(out_lines)
    print out_lines[0]
    random.shuffle(out_lines)
    util.writeFile(out_file_train,out_lines)
    

def save_color_as_gray((in_file,out_file,im_size,idx_file_curr)):
    if idx_file_curr%1000 ==0:
        print idx_file_curr

    img = cv2.imread(in_file);
    gray  =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if im_size is not None:
        gray=cv2.resize(gray,tuple(im_size));
    cv2.imwrite(out_file,gray)

    

def script_make_im_gray():
    dir_meta = '../data/bp4d'
    # out_dir_im = os.path.join(dir_meta, 'preprocess_im_110_color_align')
    # out_dir_files = os.path.join(dir_meta, 'train_test_files_110_color_align')
    # out_dir_files_new = os.path.join(dir_meta, 'train_test_files_110_gray_align')
    # out_dir_im_new = os.path.join(dir_meta, 'preprocess_im_110_gray_align')


    out_dir_im = os.path.join(dir_meta, 'preprocess_im_110_color_nodetect')
    out_dir_files = os.path.join(dir_meta, 'train_test_files_110_color_nodetect')
    out_dir_files_new = os.path.join(dir_meta, 'train_test_files_110_gray_nodetect')
    out_dir_im_new = os.path.join(dir_meta, 'preprocess_im_110_gray_nodetect')
    util.mkdir(out_dir_files_new)

    num_folds = 3
    im_size = None
    # [96,96]
    all_im = []
    for fold_curr in range(num_folds):
        train_file = os.path.join(out_dir_files,'train_'+str(fold_curr)+'.txt')
        test_file = os.path.join(out_dir_files,'test_'+str(fold_curr)+'.txt')
        all_data = util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)
        all_im = all_im + [line_curr.split(' ')[0] for line_curr in all_data]

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
    #     save_color_as_gray(arg)
    #     raw_input()


    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(save_color_as_gray,args)
    





def make_train_test_files():
    dir_meta = '../data/bp4d'
    out_dir_subs = os.path.join(dir_meta,'subs')
    out_dir_annos = os.path.join(dir_meta, 'anno_text')

    # out_dir_im = os.path.join(dir_meta, 'preprocess_im_110_color_nodetect')
    # out_dir_files = os.path.join(dir_meta, 'train_test_files_110_color_nodetect')

    # out_dir_im = os.path.join(dir_meta, 'preprocess_im_110_color')
    # out_dir_files = os.path.join(dir_meta, 'train_test_files_110_color')

    # out_dir_im = os.path.join(dir_meta, 'preprocess_im_96_gray')
    # out_dir_files = os.path.join(dir_meta, 'train_test_files_96_gray')

    # out_dir_im = os.path.join(dir_meta, 'preprocess_im_110_color_align')
    # out_dir_files = os.path.join(dir_meta, 'train_test_files_110_color_align')

    # out_dir_im = os.path.join(dir_meta, 'preprocess_im_110_gray_align')
    # out_dir_files = os.path.join(dir_meta, 'train_test_files_110_gray_align')

    # out_dir_im = os.path.join(dir_meta, 'preprocess_im_256_color_align')
    # out_dir_files = os.path.join(dir_meta, 'train_test_files_256_color_align')

    out_dir_im = os.path.join(dir_meta, 'preprocess_im_110_gray_nodetect')
    out_dir_files = os.path.join(dir_meta, 'train_test_files_110_gray_nodetect')


    replace_str = '../data/bp4d/BP4D/BP4D-training'
    util.mkdir(out_dir_files)
    num_folds = 3

    for fold_num in range(num_folds):
        for file_pre_str in ['train','test']:
            train_sub_file = os.path.join(out_dir_subs,file_pre_str+'_'+str(fold_num)+'.txt')
            train_folds = util.readLinesFromFile(train_sub_file)
            out_file_train = os.path.join(out_dir_files,file_pre_str+'_'+str(fold_num)+'.txt')
            write_train_file(out_file_train, out_dir_annos, out_dir_im, train_folds, replace_str)


def save_mean_std_im(im_rel_all, out_file_mean,out_file_std):
    im_all = []
    for im_file in im_rel_all:
        im = scipy.misc.imread(im_file)
        im = im[:,:,np.newaxis]
        im_all.append(im)

    print im_all[0].shape
    im_all = np.concatenate(im_all,2)
    

    print im_all.shape, np.min(im_all),np.max(im_all)
    mean_val = np.mean(im_all,axis=2)
    print mean_val.shape,np.min(mean_val),np.max(mean_val)

    std_val = np.std(im_all,axis=2)
    print std_val.shape,np.min(std_val),np.max(std_val)
    cv2.imwrite(out_file_mean,mean_val)
    cv2.imwrite(out_file_std,std_val)


def make_select_mean_files(dir_files,num_folds):

    num_per_video = 200

    for fold_num in range(num_folds):
        print fold_num

        train_file = os.path.join(dir_files,'train_'+str(fold_num)+'.txt')
        im_files = util.readLinesFromFile(train_file)
        im_files = [line_curr.split(' ')[0] for line_curr in im_files]
        print len(im_files)
        num_videos = [os.path.split(im_file)[0] for im_file in im_files]
        num_videos = list(set(num_videos))
        print len(num_videos)

        im_rel_all = []

        for video_curr in num_videos:
            im_rel = [im_file for im_file in im_files if im_file.startswith(video_curr)]
            random.shuffle(im_rel)
            im_rel_all = im_rel_all+im_rel[:num_per_video]

        print len(im_rel_all)

        out_file_mean = os.path.join(dir_files,'train_'+str(fold_num)+'_mean.png')
        out_file_std = os.path.join(dir_files,'train_'+str(fold_num)+'_std.png')
        save_mean_std_im(im_rel_all, out_file_mean,out_file_std)


def save_kp_for_alignment():
    dir_meta = '../data/bp4d'
    out_dir_im = os.path.join(dir_meta, 'preprocess_im_110_color')
    out_dir_files = os.path.join(dir_meta, 'train_test_files_110_color')
    out_dir_kp = os.path.join(dir_meta,'preprocess_im_110_color_kp')
    util.mkdir(out_dir_kp)

    all_im = []
    for fold_num in range(3):
        for file_pre in ['train','test']:
            file_curr = os.path.join(out_dir_files,file_pre+'_'+str(fold_num)+'.txt')
            im_list_curr = [line.split(' ')[0] for line in util.readLinesFromFile(file_curr)]
            all_im.extend(im_list_curr)
    all_im = list(set(all_im))
    print len(all_im)

    batch_size = 128
    batches = [all_im[x:x+batch_size] for x in range(0, len(all_im), batch_size)]
    total = 0

    for b in batches:
        total +=len(b)

    assert total==len(all_im)

    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)
    for idx_im_list, im_list in enumerate(batches):
        print 'batch',idx_im_list,'of',len(batches)
        preds = fa.get_landmarks_simple(im_list)
        # print preds.shape,len(im_list)
        for idx_im_curr,im_curr in enumerate(im_list):
            pred_curr = preds[idx_im_curr]
            # print pred_curr.shape
            out_file_pred = im_curr.replace(out_dir_im,out_dir_kp).replace('.jpg','.npy')
            # print out_file_pred,im_curr
            dir_curr = os.path.split(out_file_pred)[0]
            util.makedirs(dir_curr)
            # raw_input()
            np.save(out_file_pred,pred_curr)

def save_registered_face((avg_pts_file,mat_file,im_file,out_file,idx)):
    if not idx%100:
        print idx

    avg_pts = np.load(avg_pts_file)

    pts = np.load(mat_file)
    
    im = scipy.misc.imread(im_file)
    
    tform = skimage.transform.estimate_transform('similarity', pts, avg_pts)
    im_new = skimage.transform.warp(im, tform.inverse, output_shape=(im.shape[0],im.shape[1]), order=1, mode='edge')

    
    scipy.misc.imsave(out_file,im_new)

def save_avg_kp_for_alignment():
    import matplotlib.pyplot as plt

    dir_meta = '../data/bp4d'
    kp_dir = os.path.join(dir_meta,'preprocess_im_110_color_kp')
    out_file_avg_kp = os.path.join(kp_dir,'avg_kp.npy')


    list_of_nps = glob.glob(os.path.join(kp_dir,'*','*','*.npy'))
    random.shuffle(list_of_nps)

    print len(list_of_nps)
    # list_of_nps = list_of_nps[:10]
    # print len(list_of_nps)s

    for idx_np_curr,np_curr in enumerate(list_of_nps):
        if idx_np_curr==0:
            kp = np.load(np_curr).astype(np.float32)
        else:
            kp = kp+np.load(np_curr).astype(np.float32)

    avg_kp = kp/len(list_of_nps)
    print np.min(avg_kp),np.max(avg_kp)

    np.save(out_file_avg_kp,avg_kp)

    plt.figure()
    plt.plot(avg_kp[:,0],avg_kp[:,1])
    plt.savefig(os.path.join(kp_dir,'avg_kp.jpg'))
    plt.close()


def script_save_align_im():
    dir_meta = '../data/bp4d'
    kp_dir = os.path.join(dir_meta,'preprocess_im_110_color_kp')
    im_dir_in = os.path.join(dir_meta,'preprocess_im_110_color')
    im_dir_out = os.path.join(dir_meta,'preprocess_im_110_color_align')


    # out_dir = '../scratch/check_align_bp4d'
    # util.mkdir(out_dir)

    list_of_nps = glob.glob(os.path.join(kp_dir,'*','*','*.npy'))
    # random.shuffle(list_of_nps)
    # np_curr = list_of_nps[0]
    avg_pts_file= os.path.join(kp_dir,'avg_kp.npy')

    args = []
    for idx_np_curr,np_curr in enumerate(list_of_nps):
        im_curr = np_curr.replace(kp_dir,im_dir_in).replace('.npy','.jpg')
        assert os.path.exists(im_curr)
        out_file = im_curr.replace(im_dir_in,im_dir_out)
        if os.path.exists(out_file):
            continue
        out_dir_curr = os.path.split(out_file)[0]
        util.makedirs(out_dir_curr)

        args.append((avg_pts_file,np_curr,im_curr,out_file,idx_np_curr))

    print len(args)
    # for arg in args:
    #     print arg
    #     save_registered_face(arg)
        

    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # pool.map(save_registered_face, args)


def rough_work():



    # input = io.imread('../data/bp4d/BP4D/BP4D-training/F001/T1/2440.jpg')
    dir_im = '../data/bp4d/preprocess_im_110_color/F001/T1'
    im_list = glob.glob(os.path.join(dir_im,'*.jpg'))
    # print im_list
    im_list = im_list[:10]



    preds = fa.get_landmarks_simple(im_list)

    out_dir_check = os.path.join('../scratch/check_kp_fan')
    util.mkdir(out_dir_check)

    for idx_im_curr, im_curr in enumerate(im_list):
        im_curr =scipy.misc.imread(im_curr)
        pts_rel = preds[idx_im_curr]
        for pt_curr in pts_rel:
            cv2.circle(im_curr, (int(pt_curr[0]),int(pt_curr[1])), 2, (255,255,255),-1)
        out_file_curr = os.path.join(out_dir_check,str(idx_im_curr)+'.jpg')
        scipy.misc.imsave(out_file_curr,im_curr)

    visualize.writeHTMLForFolder(out_dir_check)



def save_kp_orginal_im():

    dir_meta = '../data/bp4d'
    dir_im_meta = os.path.join(dir_meta, 'BP4D','BP4D-training')
    dir_kp = os.path.join(dir_meta,'kp_org_im')

    dir_im_meta = os.path.join(dir_meta, 'preprocess_im_256_color_nodetect')
    dir_kp = os.path.join(dir_meta,'kp_256')    
    
    ims_all = glob.glob(os.path.join(dir_im_meta,'*','*','*.jpg'))
    print len(ims_all)

    # raw_input()
    # args = []


    args = []
    for idx_im_curr, im_curr in enumerate(ims_all):
        # print idx_im_curr
        
        out_file_curr = im_curr.replace(dir_im_meta, dir_kp).replace('.jpg','.npy')

        if os.path.exists(out_file_curr):
            continue

        out_dir_curr = os.path.split(out_file_curr)[0]
        util.makedirs(out_dir_curr)

        args.append((im_curr,out_file_curr,idx_im_curr)) 

    # args = args[:30000]


    chunk_size = len(args)//4
    args = [args[x:x+chunk_size] for x in range(0, len(args), chunk_size)]
    print len(args)
    print len(args[0])
    print sum([len(val) for val in args])

    pool = multiprocessing.Pool(4)
    pool.map(save_align_mp,args)

def save_align_mp((args)):

    print len(args)

    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)    

    for arg in args:
        im_curr, out_file_curr, idx_arg = arg
        if idx_arg%100==0:
            print idx_arg
        try:
            pred = fa.get_landmarks(im_curr, False)    
            if pred is not None:
                np.save(out_file_curr,pred[0])
        except:
            pass


def save_align_im_diff_scale((kp_in_file,im_org_file,avg_pts_file,out_file, out_scale,idx)):
    if not idx%100:
        print idx

    kp = np.load(kp_in_file)
    kp = kp/256.

    im_org = scipy.misc.imread(im_org_file)
    
    kp[:,0] = kp[:,0]*im_org.shape[1]
    kp[:,1] = kp[:,1]*im_org.shape[0]

    
    avg_pts = np.load(avg_pts_file)
    
    tform = skimage.transform.estimate_transform('similarity', kp, avg_pts)
    im_new = skimage.transform.warp(im_org, tform.inverse, output_shape=(out_scale[0],out_scale[1]), order=1, mode='edge')
    
    scipy.misc.imsave(out_file,im_new)


def make_avg_kp_256():
    dir_meta = '../data/bp4d'
    avg_kp_file = os.path.join(dir_meta, 'preprocess_im_110_color_kp/avg_kp.npy')
    # params = [192,32,32]
    # params = [192,32,56]
    params = [128,64,64]
    out_file = os.path.join(dir_meta, 'avg_kp_256_'+'_'.join([str(val) for val in params])+'.npy')
    
    avg_kp = np.load(avg_kp_file)
    avg_kp = avg_kp-np.min(avg_kp,0,keepdims=True)
    print avg_kp.shape,np.min(avg_kp,0),np.max(avg_kp,0)
    # print np.max(avg_kp,0,keepdims=True)
    avg_kp = avg_kp/np.max(avg_kp,0,keepdims=True)
    avg_kp = avg_kp*params[0]
    print np.min(avg_kp,0),np.max(avg_kp,0)
    # avg_kp = avg_kp+params[1]
    avg_kp[:,0] = avg_kp[:,0]+params[1]
    print np.min(avg_kp,0),np.max(avg_kp,0)
    avg_kp[:,1] = avg_kp[:,1]+params[2]
    print np.min(avg_kp,0),np.max(avg_kp,0)


    np.save( out_file, avg_kp)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(avg_kp[:,0],avg_kp[:,1])
    plt.savefig('../scratch/kp_avg_sc.jpg')
    plt.close()


def script_save_align_im_diff_scale():

    dir_meta = '../data/bp4d'
    dir_im_org = os.path.join(dir_meta, 'BP4D','BP4D-training')
    dir_kp = os.path.join(dir_meta,'kp_256')    
    out_dir_im = os.path.join(dir_meta, 'preprocess_im_256_color_align')
    np_done = glob.glob(os.path.join(dir_kp,'*','*','*.npy'))
    
    print len(np_done)
    

    params = [192,32,56]
    avg_kp_file = os.path.join(dir_meta, 'avg_kp_256_'+'_'.join([str(val) for val in params])+'.npy')

    args = []
    for idx_kp_file, kp_in_file in enumerate(np_done):
        # kp_in_file = np_done[-1]
        im_org_file = kp_in_file.replace(dir_kp, dir_im_org).replace('.npy','.jpg')
        out_file = im_org_file.replace(dir_im_org,out_dir_im)
        out_scale = [256,256]
        if os.path.exists(out_file):
            continue
        out_dir_curr = os.path.split(out_file)[0]
        util.makedirs(out_dir_curr)
        args.append((kp_in_file,im_org_file,avg_kp_file,out_file, out_scale,idx_kp_file))

    print len(args)
    # return
    import time
    for idx_arg,arg in enumerate(args):
        # print arg
        # t= time.time()
        if idx_arg%100==0:
            print idx_arg
        save_align_im_diff_scale(arg)
        # print time.time()-t
        # break

    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # pool.map(save_align_im_diff_scale,args)


def sanity_check():

    dir_meta = '../data/bp4d'
    out_dir_files = os.path.join(dir_meta, 'train_test_files_110_color_nodetect')
    for fold_num in range(3):
        train_file = os.path.join(out_dir_files,'train_'+str(fold_num)+'.txt')
        test_file = os.path.join(out_dir_files,'test_'+str(fold_num)+'.txt')
        train_data = util.readLinesFromFile(train_file)
        test_data = util.readLinesFromFile(test_file)
        train_folders = [os.path.split(line_curr.split(' ')[0])[0] for line_curr in train_data]
        test_folders = [os.path.split(line_curr.split(' ')[0])[0] for line_curr in test_data]
        train_folds = list(set(train_folders))
        test_folds = list(set(test_folders))
        print len(train_folds),len(test_folds),len(set(train_folds+test_folds)),len(train_folds)+len(test_folds)
        print np.in1d(test_folds,train_folds)


def main():
    # make_train_test_files()
    # script_make_im_gray()
    # script_save_align_im_diff_scale()
    # save_kp_orginal_im()

    # make_avg_kp_256()







    
    # script_save_bbox()
    # file_curr = '../data/bp4d/sizes_256.npz'

    # sizes =  np.load(file_curr)
    # print sizes.keys()
    # print sizes['crop_sizes'].shape


    # save_kp_orginal_im()
    # make_train_test_files()
    # script_save_align_im()

    # script_make_im_gray()
    # test_face_detector()

    # script_save_cropped_faces()
    # make_train_test_files()

    dir_meta = '../data/bp4d'
    out_dir_files = os.path.join(dir_meta, 'train_test_files_110_gray_nodetect')
    make_select_mean_files(out_dir_files,3)
    # make_select_mean_files(dir_files,num_folds)


    return


    # im = '../data/bp4d/preprocess_im_256_color_nodetect/M013/T3/324.jpg'
    # bbox = '../data/bp4d/preprocess_im_256_color_nodetect_bbox/M013/T3/324.npy'

    # out_file = '../scratch/see_bbox.jpg'
    # # util.mkdir(out_dir)

    # im = scipy.misc.imread(im)
    # bbox = np.load(bbox)
    # cv2.rectangle(im,(bbox[2],bbox[0]),(bbox[3],bbox[1]),(255,255,255),2)
    # scipy.misc.imsave(out_file,im)

    # make_train_test_files()
    # make_train_test_subs()
    # script_save_resize_faces()

    # return
    dir_meta = '../data/bp4d'

    dir_im_meta = os.path.join(dir_meta, 'BP4D','BP4D-training')
    dir_kp = os.path.join(dir_meta,'kp_256')    

    np_done = glob.glob(os.path.join(dir_kp,'*','*','*.npy'))
    for np_curr in np_done:
        im_org = np_curr.replace(dir_kp,dir_im_meta).replace('.npy','.jpg')
        im_org = scipy.misc.imread(im_org)

        kp = np.load(np_curr)
        kp = kp/256.
        kp[:,0] = kp[:,0]*im_org.shape[1]
        kp[:,1] = kp[:,1]*im_org.shape[0]

        out_file_curr = '../scratch/check_kp_big.jpg'
        for pt_curr in kp:
            cv2.circle(im_org, (int(pt_curr[0]),int(pt_curr[1])), 2, (255,255,255),-1)
        
        print out_file_curr
        scipy.misc.imsave(out_file_curr,im_org)
        
        break



        
    



if __name__=='__main__':
    main()