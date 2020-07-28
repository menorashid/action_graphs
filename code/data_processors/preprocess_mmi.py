import sys
sys.path.append('./')
import os
import preprocess_bp4d

from helpers import util, visualize
import glob
import numpy as np
import subprocess
import scipy.io

import matplotlib.pyplot as plt
import skimage.transform
import multiprocessing
import random
import cv2

def parse_meta_info():
    dir_meta = '../data/mmi'
    video_dir_meta = os.path.join(dir_meta,'Sessions')
    dirs = [dir_curr for dir_curr in glob.glob(os.path.join(video_dir_meta,'*')) if os.path.isdir(dir_curr)]
    print len(dirs)
    # xml_files = glob.glob(os.path.join(video_dir_meta,'*','*.xml'))
    # dirs = []
    # views = []
    annos = np.zeros((len(dirs),2))

    strs_to_find = ['view','"Emotion" Value']
    for idx_dir_curr,dir_curr in enumerate(dirs):
        sess_file = os.path.join(dir_curr,'session.xml')
        assert os.path.exists(sess_file)

        anno_file = glob.glob(os.path.join(dir_curr,'*-*.xml'))
        if len(anno_file)>1:
            anno_file = [file_curr for file_curr in anno_file if 'aucs' not in file_curr]
            assert len(anno_file)==1
        anno_file = anno_file[0]

        for idx_file,(str_to_find,xml_file) in enumerate(zip(strs_to_find,[sess_file,anno_file])):
            with open(xml_file,'r') as f:
                lines = f.read()
            idx_view = lines.find(str_to_find)
            view_curr = lines[idx_view+len(str_to_find)+2:idx_view+len(str_to_find)+3]
            view_curr = int(view_curr)          
            annos [idx_dir_curr, idx_file] = view_curr

    print annos.shape
    print np.unique(annos[:,0])
    print np.unique(annos[:,1])
    bin_curr = annos[:,1]==9
    for dir_curr in np.array(dirs)[bin_curr]:
        print dir_curr

    return dirs,annos

def extract_frames(video_file):
    print video_file
    out_dir_curr = video_file[:video_file.rindex('.')]
    util.mkdir(out_dir_curr)
    out_file_format = os.path.join(out_dir_curr,os.path.split(out_dir_curr)[1]+'_%05d.jpg')
    command = []
    command.extend(['ffmpeg','-i',video_file])
    command.append(out_file_format)
    command.append('-hide_banner')
    command = ' '.join(command)
    subprocess.call(command, shell=True)


def script_extract_frames():
    dir_meta = '../data/mmi'
    video_dir_meta = os.path.join(dir_meta,'Sessions')
    videos = glob.glob(os.path.join(video_dir_meta,'*','*.avi'))

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(extract_frames,videos)

def make_anno_file():
    dir_meta = '../data/mmi'
    dirs,annos = parse_meta_info()
    annos = annos.astype(int)
    annos_full = []
    out_file_annos = os.path.join(dir_meta, 'annos_all.txt')


    print len(dirs), annos.shape

    for idx_dir_curr, dir_curr in enumerate(dirs):
        video_name = glob.glob(os.path.join(dir_curr,'*.avi'))
        assert len(video_name)==1
        video_name = video_name[0]
        dir_frame = os.path.split(video_name)[1]
        dir_frame = dir_frame[:dir_frame.rindex('.')]
        dir_frame = os.path.join(dir_curr,dir_frame)
        frames = glob.glob(os.path.join(dir_frame,'*.jpg'))
        if len(frames)<3:
            print 'continuing',video_name,len(frames)
            continue
        frames.sort()
        mid_frames = len(frames)//2
        # print len(frames)
        # print video_name
        mid_frames = range(mid_frames-2,mid_frames+3)
        for frame_num in mid_frames:
            # print frame_num, len(frames)
            info_curr = [frames[frame_num],annos[idx_dir_curr,0],annos[idx_dir_curr,1]]
            annos_full.append(' '.join([str(val) for val in info_curr]))

    print len(annos_full)
    print annos_full[0]

    util.writeFile(out_file_annos, annos_full)

def make_html_for_flip_ver():
    dir_meta = '../data/mmi'
    dir_server = '/disk3'
    str_replace = ['..',os.path.join(dir_server,'maheen_data/eccv_18')]
    click_str = 'http://vision3.idav.ucdavis.edu:1000'

    anno_file = os.path.join(dir_meta,'annos_all_oriented.txt')
    out_file_html = os.path.join(dir_meta, 'annos_all_oriented.html')

    annos = util.readLinesFromFile(anno_file)
    ims = [file_curr.split(' ')[0] for file_curr in annos]
    dirs_rel = list(set([os.path.split(file_curr)[0] for file_curr in ims]))
    ims_html = []
    captions_html = []
    for dir_curr in dirs_rel:
        ims_row = [file_curr for file_curr in ims if file_curr.startswith(dir_curr)]
        ims_row = [util.getRelPath(im_curr.replace(str_replace[0],str_replace[1]),dir_server) for im_curr in ims_row]
        captions_row = [os.path.split(im_curr)[1] for im_curr in ims_row]
        captions_html.append(captions_row)
        ims_html.append(ims_row)

    visualize.writeHTML(out_file_html,ims_html,captions_html,96,96)


def save_flipped_images():
    dir_meta = '../data/mmi'
    dir_server = '/disk3'
    str_replace = ['..',os.path.join(dir_server,'maheen_data/eccv_18')]
    click_str = 'http://vision3.idav.ucdavis.edu:1000'

    anno_file = os.path.join(dir_meta,'annos_all.txt')
    anno_file_new = os.path.join(dir_meta,'annos_all_oriented.txt')

    no_flip = ['S054','S053']
    opp_flip = ['S021']
    annos = util.readLinesFromFile(anno_file)
    annos_new = []

    for line_curr in annos:
        im_file,view,emo = line_curr.split(' ')

        if int(view)==2:
            out_im_file = im_file

        elif int(view)==1:
            assert opp_flip[0] in im_file
            # continue
            out_dir_curr = os.path.split(im_file)[0]+'_oriented'
            util.mkdir(out_dir_curr)
            out_im_file = os.path.join(out_dir_curr,os.path.split(im_file)[1])
            
            if os.path.exists(out_im_file):
                continue

            im = scipy.misc.imread(im_file)
            im = np.transpose(im,(1,0,2))
            im = im[::-1,:,:]
            
            scipy.misc.imsave(out_im_file,im)
            

        else :
            if no_flip[0] in im_file or no_flip[1] in im_file:
                out_im_file = im_file
                continue
            else:
                out_dir_curr = os.path.split(im_file)[0]+'_oriented'
                util.mkdir(out_dir_curr)
                out_im_file = os.path.join(out_dir_curr,os.path.split(im_file)[1])
                if os.path.exists(out_im_file):
                    im = scipy.misc.imread(im_file)
                    im = np.transpose(im,(1,0,2))
                    im = im[:,::-1,:]
                    scipy.misc.imsave(out_im_file,im)
                
                # raw_input()

        annos_new.append(' '.join([out_im_file,view,emo]))
    util.writeFile(anno_file_new,annos_new)
    

    # out_file_html = os.path.join(dir_meta, 'annos_all.html')

def detect_frames_single_head():
    dir_meta = '../data/mmi'
    dir_server = '/disk3'
    str_replace = ['..',os.path.join(dir_server,'maheen_data/eccv_18')]
    click_str = 'http://vision3.idav.ucdavis.edu:1000'
    anno_file_new = os.path.join(dir_meta,'annos_all_oriented.txt')

    im_size = [110,110]
    savegray = True
    out_dir_im = os.path.join(dir_meta,'preprocess_im_gray_110')
    util.mkdir(out_dir_im)
    str_replace_dir = [os.path.join(dir_meta,'Sessions'),out_dir_im]
    annos_all = util.readLinesFromFile(anno_file_new)

    file_pairs = []
    for idx_anno_curr,anno_curr in enumerate(annos_all):
        im_file, view, emo = anno_curr.split(' ')
        out_file = im_file.replace(str_replace_dir[0],str_replace_dir[1])
        if os.path.exists(out_file) and int(view)!=2:
            continue

        out_dir_curr = os.path.split(out_file)[0]
        util.makedirs(out_dir_curr)
        file_pairs.append((im_file,out_file))
        

    args = []
    chunk_size = 50
    chunks = [file_pairs[x:x+chunk_size] for x in range(0, len(file_pairs), chunk_size)]
    args = [(chunk_curr,im_size,savegray,idx_im_in,True) for idx_im_in,chunk_curr in enumerate(chunks)]
    print len(args)

    for arg in args:
        print arg[0]
        size = preprocess_bp4d.saveCroppedFace_NEW_batch(arg)
    #     # saveCroppedFace(arg)
        # break

    # pool = multiprocessing.Pool(4)
    # pool.map(preprocess_bp4d.saveCroppedFace_NEW_batch,args)


def make_train_test_subs():
    dir_meta = '../data/mmi'
    anno_file_new = os.path.join(dir_meta,'annos_all_oriented_pruned.txt')
    annos = util.readLinesFromFile(anno_file_new)
    subs = []
    views = []
    emos = []

    for anno_curr in annos:
        im_file, view, emo = anno_curr.split(' ')
        sub_curr = os.path.split(im_file)[1]
        sub_curr = sub_curr[:sub_curr.index('-')]
        subs.append(sub_curr)
        views.append(int(view))
        emos.append(int(emo))

    subs = np.array(subs)
    views = np.array(views)
    emos = np.array(emos)

    print len(np.unique(subs))
    for sub in np.unique(subs):
        views_curr = list(np.unique(views[subs==sub]))
        # if 2 in views_curr:
        #     assert len(views_curr)==1
        print sub, views_curr, np.sum(subs==sub), np.unique(emos[subs==sub])
        # print np.unique(views[subs==sub])

def make_pruned_anno():
    dir_meta = '../data/mmi'
    anno_file = os.path.join(dir_meta,'annos_all_oriented.txt')
    anno_file_new = os.path.join(dir_meta,'annos_all_oriented_pruned.txt')
    new_lines = []
    for line_curr in util.readLinesFromFile(anno_file):
        file,view,emo = line_curr.split(' ')
        emo = int(emo)
        if 'S006' in line_curr or emo==9:
            continue
        new_lines.append(line_curr)
    util.writeFile(anno_file_new,new_lines)
    



def save_mean_std_im(dir_files):
    
    for split_num in range(0,2):
        train_file = os.path.join(dir_files,'train_'+str(split_num)+'.txt')
        out_file_mean = os.path.join(dir_files,'train_'+str(split_num)+'_mean.png')
        out_file_std = os.path.join(dir_files,'train_'+str(split_num)+'_std.png')

        lines = util.readLinesFromFile(train_file)
        im_all = []
        for line in lines:
            im = line.split(' ')[0]
            im = scipy.misc.imread(im).astype(np.float32)
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


def make_train_test_files():
    dir_meta = '../data/mmi'
    out_dir_train = os.path.join(dir_meta,'train_test_files')
    util.mkdir(out_dir_train)
    anno_file = os.path.join(dir_meta,'annos_all_oriented_pruned.txt')
    key_side = {'S001':0,'S002':0,'S003':1,'S005':1,'S015':1,'S016':1}

    new_im_dir = os.path.join(dir_meta,'preprocess_im_gray_110')
    old_im_dir = os.path.join(dir_meta,'Sessions')
    # util.mkdir(out_dir_im)
    # str_replace_dir = [,out_dir_im]

    scheme = '0'
    test_subs = ['S001','S002','S003','S005','S015','S016','S021']
    train_subs = ['S028','S030','S031','S032','S033','S034','S035','S036','S037','S038','S039','S040','S041','S042','S043','S044','S045','S046','S047','S048','S049','S050']

    # scheme = '1'
    # test_subs = ['S002','S003','S005','S015','S016','S021']
    # train_subs = ['S001','S028','S030','S031','S032','S033','S034','S035','S036','S037','S038','S039','S040','S041','S042','S043','S044','S045','S046','S047','S048','S049','S050']

    annos = util.readLinesFromFile(anno_file)
    
    out_file_train = os.path.join(out_dir_train,'train_'+scheme+'.txt')
    out_file_test_side = os.path.join(out_dir_train,'test_side_'+scheme+'.txt')
    out_file_test_front = os.path.join(out_dir_train,'test_front_'+scheme+'.txt')

    train_lines = []
    test_side = []
    test_front = []

    for anno_curr in annos:
        im_file, view, emo = anno_curr.split(' ')
        im_file = im_file.replace(old_im_dir,new_im_dir)
        view = int(view)
        emo = int(emo)-1
        sub_curr = os.path.split(im_file)[1]
        sub_curr = sub_curr[:sub_curr.index('-')]
        if sub_curr in train_subs:
            if view == 2:
                assert scheme=='1'
                im_file = im_file[:im_file.rindex('.')]
                train_lines.append([im_file+'_0.jpg',emo])
                train_lines.append([im_file+'_1.jpg',emo])
            else:
                assert view==0
                train_lines.append([im_file,emo])
        else:
            assert sub_curr in test_subs
            if view == 2:
                im_file = im_file[:im_file.rindex('.')]
                side_num = key_side[sub_curr]
                front_num = (side_num+1)%2
                test_side.append([im_file+'_'+str(side_num)+'.jpg',emo])
                test_front.append([im_file+'_'+str(front_num)+'.jpg',emo])
            else:
                assert view==1
                test_side.append([im_file,emo])

    
    pairs = [(out_file_train,train_lines),(out_file_test_side,test_side),(out_file_test_front,test_front)]

    for out_file_curr, lines_curr in pairs:
        out_lines = []

        for line_curr in lines_curr:
            # print line_curr[0]
            assert os.path.exists(line_curr[0])
            out_lines.append(' '.join([str(val) for val in line_curr]))

        print len(out_lines),out_file_curr
        util.writeFile(out_file_curr,out_lines)







def verify_html():
    dir_meta = '../data/mmi'
    dir_server = '/disk3'
    str_replace = ['..',os.path.join(dir_server,'maheen_data/eccv_18')]
    click_str = 'http://vision3.idav.ucdavis.edu:1000'

    file_pres = ['train','test_side','test_front']
    folds = [0,1]
    out_dir_train = os.path.join(dir_meta,'train_test_files')
    for file_pre in file_pres:
        for fold in folds:
            anno_file = os.path.join(out_dir_train,file_pre+'_'+str(fold)+'.txt')
            out_file_html = anno_file[:anno_file.rindex('.')]+'.html'


            # anno_file = os.path.join(dir_meta,'annos_all_oriented.txt')
            # out_file_html = os.path.join(dir_meta, 'annos_all_oriented.html')

            annos = util.readLinesFromFile(anno_file)
            ims = [file_curr.split(' ')[0] for file_curr in annos]
            dirs_rel = list(set([os.path.split(file_curr)[0] for file_curr in ims]))
            ims_html = []
            captions_html = []
            for dir_curr in dirs_rel:
                ims_row = [file_curr for file_curr in ims if file_curr.startswith(dir_curr)]
                ims_row = [util.getRelPath(im_curr.replace(str_replace[0],str_replace[1]),dir_server) for im_curr in ims_row]
                captions_row = [os.path.split(im_curr)[1] for im_curr in ims_row]
                captions_html.append(captions_row)
                ims_html.append(ims_row)

            visualize.writeHTML(out_file_html,ims_html,captions_html,96,96)
            print out_file_html.replace(dir_server,click_str)




def main():
    
    verify_html()

    # dir_files = '../data/mmi/train_test_files'
    # save_mean_std_im(dir_files)

    # make_train_test_files()

    # make_pruned_anno()
    # make_train_test_subs()
    # detect_frames_single_head()

    # save_flipped_images()
    # make_html_for_flip_ver()
    # extract_frames('../data/mmi/Sessions/224/S002-106.avi')
    # make_anno_file()

    # script_extract_frames()
    # parse_meta_info()

    print 'hello'

if __name__=='__main__':
    main()