import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import scipy.misc
import numpy as np
import random
import cv2
import multiprocessing

def make_train_test_split():
    pass

def make_fold_files():
    dir_meta = '../data/Oulu_CASIA'
    out_dir_files = os.path.join(dir_meta,'train_test_files')
    util.mkdir(out_dir_files)

    dir_meta_subjects = glob.glob(os.path.join(dir_meta,'PreProcess_Img/NI_Acropped','*'))
    range_subjects = ['P'+'0'*(3-len(str(num)))+str(num) for num in range(1,81)]

    num_folds = 10
    folds = []
    for i in range(num_folds):
        fold_curr = range_subjects[i::num_folds]
        folds.append(fold_curr)

    for i in range(num_folds):
        out_file_train = os.path.join(out_dir_files,'train_subs_'+str(i)+'.txt')
        out_file_test = os.path.join(out_dir_files,'test_subs_'+str(i)+'.txt')
        train_subs =[]
        for idx_fold_curr,fold_curr in enumerate(folds):
            if idx_fold_curr!=i:
                train_subs = train_subs+ fold_curr 
        test_subs = folds[i]
        
        util.writeFile(out_file_train,train_subs)
        util.writeFile(out_file_test,test_subs)

def make_fold_files_with_val():
    dir_meta = '../data/Oulu_CASIA'
    out_dir_files = os.path.join(dir_meta,'subs')
    util.mkdir(out_dir_files)

    dir_meta_subjects = glob.glob(os.path.join(dir_meta,'PreProcess_Img/NI_Acropped','*'))
    range_subjects = ['P'+'0'*(3-len(str(num)))+str(num) for num in range(1,81)]

    num_folds = 10
    folds = []
    for i in range(num_folds):
        fold_curr = range_subjects[i::num_folds]
        folds.append(fold_curr)

    for i in range(num_folds):
        out_file_train = os.path.join(out_dir_files,'train_subs_'+str(i)+'.txt')
        out_file_val = os.path.join(out_dir_files,'val_subs_'+str(i)+'.txt')
        out_file_test = os.path.join(out_dir_files,'test_subs_'+str(i)+'.txt')
        train_subs =[]
        # print i, (i+1)%10
        val_fold = (i+1)%num_folds
        for idx_fold_curr,fold_curr in enumerate(folds):
            if idx_fold_curr!=i and idx_fold_curr!=val_fold:
                train_subs = train_subs+ fold_curr 
        test_subs = folds[i]
        val_subs = folds[val_fold]
        
        util.writeFile(out_file_train,train_subs)
        util.writeFile(out_file_test,test_subs)
        util.writeFile(out_file_val,val_subs)

def save_mean_std_im(dir_files):
    im_resize = [96,96]

    for split_num in range(0,10):
        train_file = os.path.join(dir_files,'train_'+str(split_num)+'.txt')
        out_file_mean = os.path.join(dir_files,'train_'+str(split_num)+'_mean.png')
        out_file_std = os.path.join(dir_files,'train_'+str(split_num)+'_std.png')

        lines = util.readLinesFromFile(train_file)
        im_all = []
        for line in lines:
            im = line.split(' ')[0]
            im = scipy.misc.imresize(scipy.misc.imread(im),(im_resize[0],im_resize[1])).astype(np.float32)
            # im = im/255.
            im = im[:,:,np.newaxis]
            im_all.append(im)

        # print len(im_all)
        print im_all[0].shape
        im_all = np.concatenate(im_all,2)
        

        print im_all.shape, np.min(im_all),np.max(im_all)
        mean_val = np.mean(im_all,axis=2)
        print mean_val.shape,np.min(mean_val),np.max(mean_val)

        std_val = np.std(im_all,axis=2)
        print std_val.shape,np.min(std_val),np.max(std_val)
        cv2.imwrite(out_file_mean,mean_val)
        cv2.imwrite(out_file_std,std_val)

        # print mean_val,std_val
        # mean_std = np.array([mean_val,std_val])
        # print mean_std.shape, mean_std
        # np.save(out_file,mean_std)


def save_mean_std_vals(dir_files):
    im_resize = [96,96]

    for split_num in range(0,10):
        train_file = os.path.join(dir_files,'train_'+str(split_num)+'.txt')
        out_file = os.path.join(dir_files,'train_'+str(split_num)+'_mean_std_val_0_1.npy')

        lines = util.readLinesFromFile(train_file)
        im_all = []
        for line in lines:
            im = line.split(' ')[0]
            im = scipy.misc.imresize(scipy.misc.imread(im),(im_resize[0],im_resize[1])).astype(np.float32)
            im = im/255.
            im = im[:,:,np.newaxis]
            im_all.append(im)

        # print len(im_all)
        print im_all[0].shape
        im_all = np.concatenate(im_all,2)
        

        print im_all.shape, np.min(im_all),np.max(im_all)
        mean_val = np.mean(im_all)
        std_val = np.std(im_all)
        print mean_val,std_val
        mean_std = np.array([mean_val,std_val])
        print mean_std.shape, mean_std
        np.save(out_file,mean_std)

def write_train_test_files_no_neutral(val=False):
    dir_meta = '../data/Oulu_CASIA'
    # out_dir_files = os.path.join(dir_meta,'train_test_files_preprocess_maheen_vl_color_256')
    out_dir_files = os.path.join(dir_meta,'train_test_files_preprocess_vl')
    util.mkdir(out_dir_files)
    # subs_dir = out_dir_files
    if val:
        subs_dir = os.path.join(dir_meta,'subs')
    else:
        subs_dir = os.path.join(dir_meta,'train_test_files')

    out_dir_single_im = os.path.join(out_dir_files,'three_im_no_neutral_just_strong_'+str(val))
    util.mkdir(out_dir_single_im)
    
    expressions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

    # mid_dir = ['PreProcess_Img/NI_Acropped/*','PreProcess_Img/VL_Acropped/Strong']
    # mid_dir = 'preprocess_maheen/NI'
    # mid_dir = 'preprocess_maheen/VL_256'
    mid_dir = 'PreProcess_Img/VL_Acropped'
    
    dir_meta_subjects = glob.glob(os.path.join(dir_meta,mid_dir,'Strong'))

    dir_ims = glob.glob(os.path.join(dir_meta,mid_dir,'Strong','*','*'))
    

    num_folds = 10
    num_select = 3
    type_files = ['train','test']
    if val:
        type_files.append('val')

    for pre_str in type_files:
        subs_file_pre = pre_str+'_subs_'
        out_file_pre = pre_str+'_'
        for fold_num in range(num_folds):
            train_subs = util.readLinesFromFile(os.path.join(subs_dir,subs_file_pre+str(fold_num)+'.txt'))
            dir_ims_curr = [dir_curr for dir_curr in dir_ims if dir_curr.split('/')[-2] in train_subs]

            out_file_curr = os.path.join(out_dir_single_im,out_file_pre+str(fold_num)+'.txt')
            lines_all = []

            all_lens =[]
            
            for dir_im in dir_ims_curr:
                ims = glob.glob(os.path.join(dir_im,'*.jpeg'))
                all_lens.append( len(ims))
                ims.sort()
                if len(ims)==0:
                    print dir_im
                exp = str(expressions.index(dir_im.split('/')[-1]))
                exp_im = ims[len(ims)-num_select:]
                lines = [im_curr+' '+exp for im_curr in exp_im]
                lines_all.extend(lines)
            
            print min(all_lens),max(all_lens)
            assert len(lines_all) == len(set(lines_all))
            print out_file_curr,len(lines_all)
            random.shuffle(lines_all)
            util.writeFile(out_file_curr,lines_all)

    save_mean_std_vals(out_dir_single_im)

def write_train_test_files():
    dir_meta = '../data/Oulu_CASIA'
    out_dir_files = os.path.join(dir_meta,'train_test_files_preprocess_maheen_vl_gray')
    util.mkdir(out_dir_files)
    out_dir_single_im = os.path.join(out_dir_files,'three_im_balance_neutral')
    util.mkdir(out_dir_single_im)
    
    expressions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

    
    dir_meta_subjects = glob.glob(os.path.join(dir_meta,'preprocess_maheen/VL_gray','*'))
        # 'PreProcess_Img/NI_Acropped','*'))

    dir_ims = glob.glob(os.path.join(dir_meta,'preprocess_maheen/VL_gray','*','*','*'))

    num_folds = 10
    num_select = 3
    
    balance = True
    
    for pre_str in ['train','test']:
        subs_file_pre = pre_str+'_subs_'
        out_file_pre = pre_str+'_'

        for fold_num in range(num_folds):
            train_subs = util.readLinesFromFile(os.path.join(out_dir_files,subs_file_pre+str(fold_num)+'.txt'))
            dir_ims_curr = [dir_curr for dir_curr in dir_ims if dir_curr.split('/')[-2] in train_subs]

            out_file_curr = os.path.join(out_dir_single_im,out_file_pre+str(fold_num)+'.txt')
            lines_all = []

            all_lens =[]
            
            for idx_dir_im,dir_im in enumerate(dir_ims_curr):
                ims = glob.glob(os.path.join(dir_im,'*.jpeg'))
                all_lens.append( len(ims))
                ims.sort()
                if len(ims)==0:
                    print dir_im
                exp = str(expressions.index(dir_im.split('/')[-1])+1)

                neutral = ims[:num_select]
                exp_im = ims[len(ims)-num_select:]

                if balance and os.path.split(dir_im)[1]!='Anger':
                    neutral = []
                # print dir_im, len(neutral)

                lines = [im_curr+' 0' for im_curr in neutral]+[im_curr+' '+exp for im_curr in exp_im]
                lines_all.extend(lines)
            
            print min(all_lens),max(all_lens)
            assert len(lines_all) == len(set(lines_all))
            print out_file_curr,len(lines_all)
            random.shuffle(lines_all)
            util.writeFile(out_file_curr,lines_all)
    
    save_mean_std_vals(out_dir_single_im)

def verify_distribution():
    dir_meta = '../data/Oulu_CASIA'
    out_dir_files = os.path.join(dir_meta,'train_test_files')
    out_dir_single_im = os.path.join(out_dir_files,'single_im_balance_neutral')

    num_folds = 10
    for split_num in range(num_folds):
        print split_num
        lines = util.readLinesFromFile(os.path.join(out_dir_single_im,'train_'+str(split_num)+'.txt'))
        classes = [int(file_curr.split(' ')[1]) for file_curr in lines]
        classes_uni = list(set(classes))
        classes_uni.sort()
        for class_curr in classes_uni:
            print class_curr, classes.count(class_curr)
        print '___'

# def saveCroppedFace(in_file,out_file,im_size=None, classifier_path=None,savegray=True):
def saveCroppedFace((in_file, out_file, im_size, savegray, idx_file_curr)):
    if idx_file_curr%100==0:
        print idx_file_curr

    # if classifier_path==None:
    classifier_path = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml';

    img = cv2.imread(in_file);
    
    gray  =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade  =  cv2.CascadeClassifier(classifier_path)
    faces  =  face_cascade.detectMultiScale(gray)
    if len(faces)==0:
        print 'PROBLEM';
        return in_file;
    else:
        # print len(faces),
        sizes=np.array([face_curr[2]*face_curr[3] for face_curr in faces]);
        faces=faces[np.argmax(sizes)];
        # print np.max(sizes);

    [x,y,w,h] = faces;
    padding = [h*5/100.,w*0/100.]
    padding = [int(val) for val in padding]

    start_r = max(0,y-padding[0])
    end_r = min(img.shape[0],y+h+padding[0])

    start_c = max(0,x-padding[1])
    end_c = min(img.shape[1],x+w+padding[1])
    # print x,y,w,h
    # print padding
    # print start_r,end_r,start_c,end_c
    # raw_input()
    
    # if padding is not None:
    #     # y = y-padding[0]
    #     h= h+padding[0]
    #     # x = x-padding[1]
    #     w= w+padding[1]
    
    # roi = gray[y:y+h, x:x+w]    
    # if not savegray:
    #     roi = img[y:y+h, x:x+w]
    
    roi = gray[start_r:end_r, start_c:end_c]    
    if not savegray:
        roi = img[start_r:end_r, start_c:end_c,:]    
    

    if im_size is not None:
        roi=cv2.resize(roi,tuple(im_size));
    cv2.imwrite(out_file,roi)

def save_cropped_images_script():
    dir_meta = '../data/Oulu_CASIA/OriginalImg/NI'
    out_dir_meta = '../data/Oulu_CASIA/preprocess_maheen/NI'

    dir_meta = '../data/Oulu_CASIA/OriginalImg/VL'
    out_dir_meta = '../data/Oulu_CASIA/preprocess_maheen/VL'; savegray=False
    out_dir_meta = '../data/Oulu_CASIA/preprocess_maheen/VL_256'; savegray=False
    
    util.makedirs(out_dir_meta)
    all_im = glob.glob(os.path.join(dir_meta,'Strong','*','*','*.jpeg'))
    print len(all_im)
    im_size = [256,256]
    problem_files = []

    args = []
    for idx_file_curr,file_curr in enumerate(all_im):
        # if idx_file_curr%100==0:
        #     print idx_file_curr,len(all_im)
        out_file = file_curr.replace(dir_meta,out_dir_meta)
        # if os.path.exists(out_file):
        #     continue
        out_dir_curr = os.path.split(out_file)[0]
        util.makedirs(out_dir_curr)
        args.append((file_curr,out_file,im_size,savegray,idx_file_curr))

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(saveCroppedFace,args)

    # for arg in args:
    #     saveCroppedFace(arg)
    #     print arg
    #     raw_input()
        # file_curr,out_file,im_size,savegray=savegray)

    for idx_file_curr,file_curr in enumerate(all_im):
        out_file = file_curr.replace(dir_meta,out_dir_meta)
        out_dir_curr = os.path.split(out_file)[0]
        if not os.path.exists(out_file):
            print file_curr
            problem_files.append(file_curr)
    print len(problem_files)


def main():
    # make_fold_files_with_val()
    # return

    # write_train_test_files_no_neutral(val=False)
    dir_files = '../data/Oulu_CASIA/train_test_files_preprocess_vl/three_im_no_neutral_just_strong_False'
    save_mean_std_im(dir_files)
    # return
    # save_cropped_images_script()
    return
    dir_server = '/disk3'
    str_replace = ['..',os.path.join(dir_server,'maheen_data','eccv_18')]

    split_num = 0
    dir_files = '../data/Oulu_CASIA/train_test_files_preprocess_maheen_vl_color_256/three_im_no_neutral_just_strong_False'
    train_file = os.path.join(dir_files,'train_'+str(split_num)+'.txt')
    test_file =  os.path.join(dir_files,'test_'+str(split_num)+'.txt')
    all_files = util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)
    all_files = [file_curr.split(' ')[0] for file_curr in all_files]
    all_files = [util.getRelPath(file_curr.replace(str_replace[0],str_replace[1]),dir_server) for file_curr in all_files]
    captions = ['' for file_curr in all_files]
    print len(all_files)
    all_files = np.array(all_files)
    all_files = np.reshape(all_files,(479,3))
    captions = np.reshape(np.array(captions),(479,3))
    print all_files.shape



    # print all_files[:10]
    # print captions[:10]
    out_file_html = os.path.join(dir_files,str(split_num)+'.html')
    visualize.writeHTML(out_file_html,all_files,captions,256,256)
    print out_file_html.replace('..','http://vision3.idav.ucdavis.edu:1000/maheen_data/eccv_18')




    # write_train_test_files()
    # write_train_test_files_no_neutral()



        # break


    
    
if __name__=='__main__':
    main()