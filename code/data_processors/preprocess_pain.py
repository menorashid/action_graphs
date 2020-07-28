import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import scipy.misc
import numpy as np
import random
import face_alignment
import skimage.transform
import multiprocessing
import dlib
import cv2
import shutil
import urllib
from preprocess_bp4d import save_align_mp,save_best_bbox_batch


def parse_facs_file(file_curr):
    lines = util.readLinesFromFile(file_curr)
    lines = [[int(float(val)) for val in line_curr.split()] for line_curr in lines]
    return np.array(lines)


def create_sess_facs_anno(out_file_curr, dir_curr, facs_to_keep):
    files_all = glob.glob(os.path.join(dir_curr,'*.txt'))
    
    strs_all = []

    for file_curr in files_all:
        anno = parse_facs_file(file_curr)
    
        if anno.size>0 and np.sum(np.isin(anno[:,0],facs_to_keep))>0:
            str_curr = []
            for fac_curr in facs_to_keep:
                anno_rel = anno[anno[:,0]==fac_curr,:2]
                if anno_rel.size>0:
                    assert anno_rel.shape[0]==1
                    str_curr.append(anno_rel[0,1])
                else:
                    str_curr.append(0)

        else:
            str_curr = [0]*len(facs_to_keep)
        
        str_curr = ' '.join([str(val) for val in [file_curr]+str_curr])
    
        strs_all.append(str_curr)

    util.writeFile(out_file_curr,strs_all)

def create_sess_pspi_anno(out_file_curr, dir_curr):
    files_all = glob.glob(os.path.join(dir_curr,'*.txt'))
    
    strs_all = []
    count_pain = 0
    for file_curr in files_all:
        anno_curr = util.readLinesFromFile(file_curr)
        assert len(anno_curr)==1
        
        anno_curr = int(float(anno_curr[0]))
        count_pain+=anno_curr>0

        str_curr = ' '.join([str(val) for val in [file_curr,anno_curr]])    
        strs_all.append(str_curr)

    # print count_pain,len(files_all)
    util.writeFile(out_file_curr,strs_all)
    return count_pain

def script_create_sess_facs_anno():
    dir_meta = '../data/pain'
    frame_dir = os.path.join(dir_meta, 'Frame_Labels','FACS')
    dirs = glob.glob(os.path.join(frame_dir,'*','*'))
    facs_to_keep = [4,6,7,9,10,43]
    
    anno_dir = os.path.join(dir_meta,'anno_au')
    util.mkdir(anno_dir)

    str_replace = [frame_dir,anno_dir]

    
    for idx_dir_curr,dir_curr in enumerate(dirs):
        out_file_curr = dir_curr.replace(str_replace[0],str_replace[1])+'.txt'
        util.makedirs(os.path.split(out_file_curr)[0])
        print out_file_curr, idx_dir_curr,'of',len(dirs)
        create_sess_facs_anno(out_file_curr, dir_curr, facs_to_keep)

def script_create_sess_pspi_anno():
    dir_meta = '../data/pain'
    frame_dir = os.path.join(dir_meta, 'Frame_Labels','PSPI')
    dirs = glob.glob(os.path.join(frame_dir,'*','*'))
    
    anno_dir = os.path.join(dir_meta,'anno_pspi')
    util.mkdir(anno_dir)

    str_replace = [frame_dir,anno_dir]

    num_pains = 0
    for idx_dir_curr,dir_curr in enumerate(dirs):
        out_file_curr = dir_curr.replace(str_replace[0],str_replace[1])+'.txt'
        util.makedirs(os.path.split(out_file_curr)[0])
        print out_file_curr, idx_dir_curr,'of',len(dirs)
        count_pain = create_sess_pspi_anno(out_file_curr, dir_curr)
        num_pains +=count_pain>0
    print num_pains, len(dirs)


def looking_at_gt_pain():
    dir_meta = '../data/pain'
    gt_pain_anno_dir = os.path.join(dir_meta, 'Sequence_Labels')
    out_dir_meta = os.path.join(gt_pain_anno_dir,'gt_avg')
    util.mkdir(out_dir_meta)
    gt_pain_anno_dirs = [os.path.join(gt_pain_anno_dir,dir_curr) for dir_curr in ['AFF','VAS','SEN']]
    min_maxs = [[1,14],[0,10],[1,14]]
    
    out_range = [0,10]
    
    sequence_names = [dir_curr.replace(gt_pain_anno_dirs[0]+'/','') for dir_curr in glob.glob(os.path.join(gt_pain_anno_dirs[0],'*','*.txt'))]
    
    # for gt_pain_anno_dir in gt_pain_anno_dirs:
    for sequence_name in sequence_names:
        pain_levels = []
        for gt_pain_anno_dir,min_max in zip(gt_pain_anno_dirs,min_maxs):    
            # print sequence_name,gt_pain_anno_dir
            file_curr = os.path.join(gt_pain_anno_dir,sequence_name)
            # print file_curr
            pain_val = int(float(util.readLinesFromFile(file_curr)[0]))
            pain_val = (pain_val-min_max[0])/float(min_max[1]-min_max[0]) * (out_range[1]-out_range[0])+out_range[0]
            pain_levels.append(pain_val)
        # print pain_levels
        avg_pain = np.mean(pain_levels)
        out_file_curr = os.path.join(out_dir_meta,sequence_name)
        util.makedirs(os.path.split(out_file_curr)[0])
        util.writeFile(out_file_curr,[str(avg_pain)])



        # pain_vals = np.array(pain_vals)
        # print os.path.split(gt_pain_anno_dir)[1],pain_vals.size,np.min(pain_vals),np.max(pain_vals)

    # dirs = glob.glob(os.path.join(frame_dir,'*'))

    # total_sess = 0
    # for dir_curr in dirs:
    #   files = glob.glob(os.path.join(dir_curr,'*.txt'))
    #   total_sess+=len(files)
    #   pain_levels = [int(float(util.readLinesFromFile(file_curr)[0])) for file_curr in files]
    #   assert np.max(pain_levels)<=10
    #   pain_counts = [pain_levels.count(val) for val in range(10)]
    #   print 'Subject %s, num sessions %d' % (os.path.split(dir_curr)[1],len(files))
    #   print ' '.join([str(val) for val in pain_counts])
    #   print '__'
    # print total_sess


def save_cropped_face_from_bbox((in_file,out_file,bbox_file, im_size, savegray,idx_file_curr)):
    
    
    if idx_file_curr%100==0:
        print idx_file_curr

    im = scipy.misc.imread(in_file)
    try:
        best_box = np.load(bbox_file)
    except:
        print 'PROBLEM',bbox_file
        return

    
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



def looking_at_im():
    dir_meta = '../data/pain'
    dir_im = os.path.join(dir_meta,'Images')
    all_im = glob.glob(os.path.join(dir_im,'*','*','*.png'))
    print len(all_im)
    im_curr = scipy.misc.imread(all_im[48000])
    print im_curr.shape


def script_save_best_bbox():

    dir_meta= '../data/pain'

    dir_im = os.path.join(dir_meta,'Images')
    
    out_dir_meta = os.path.join(dir_meta,'im_best_bbox')
    util.mkdir(out_dir_meta)
    

    im_list_in = glob.glob(os.path.join(dir_im,'*','*','*.png'))
    print len(im_list_in)
    # im_list_in = im_list_in[:len(im_list_in)//2]
    # im_list_in = im_list_in[len(im_list_in)//2:]
    # print len(im_list_in)
    
    savegray = False
    args = []
    
    args = []
    file_pairs = []
    for idx_im_in,im_in in enumerate(im_list_in):
        out_file = im_in.replace(dir_im,out_dir_meta).replace('.png','.npy')
        if os.path.exists(out_file):
            continue
        out_dir_curr = os.path.split(out_file)[0]
        util.makedirs(out_dir_curr)
        file_pairs.append((im_in,out_file))

    print len(args)

    # chunk_size = 50
    # chunks = [file_pairs[x:x+chunk_size] for x in range(0, len(file_pairs), chunk_size)]
    # args = [(chunk_curr,idx_im_in) for idx_im_in,chunk_curr in enumerate(chunks)]

    # print len(args)
    # # args = args[:1000]
    # for arg in args:
    #     print arg
    #     size = save_best_bbox_batch(arg)


    # raw_input()

    # pool = multiprocessing.Pool(4)
    # pool.map(save_best_bbox_batch,args)
    # pool.close()
    # pool.join()


def get_threshold_pain_au_only( au_dir, sub, gt_pain_dir, strs_replace, threshold=1):

    pain_files = glob.glob(os.path.join(gt_pain_dir,sub,'*.txt'))
    # pain_levels = np.array([float(util.readLinesFromFile(pain_file)[0]) for pain_file in pain_files])
    # bin_pain = list(pain_levels>=threshold)

    num_skipped = 0
    out_lines = []
    for idx_pain_file, pain_file in enumerate(pain_files):
        pain_level = float(util.readLinesFromFile(pain_file)[0])
        # print pain_file, pain_level
        if pain_level>=threshold:
            au_file_curr = os.path.join(au_dir,sub,os.path.split(pain_file)[1])
            # print au_file_curr
            assert os.path.exists(au_file_curr)

            for line_curr in util.readLinesFromFile(au_file_curr):
                line_curr = line_curr.split(' ',1)
                im_curr = line_curr[0]
                for str_replace in strs_replace:
                    im_curr = im_curr.replace(str_replace[0],str_replace[1])
                
                if not os.path.exists(im_curr):
                    num_skipped+=1
                    # print 'DOES NOT EXIST %s' % im_curr
                    continue

                line_new = ' '.join([im_curr,line_curr[1]])
                # print line_curr,line_new
                out_lines +=[line_new]
                # raw_input()

            
        else:
            pass
            # print 'skipping %s with pain level %.3f' % (pain_file, pain_level)
    
    # print len(out_lines),len(set(out_lines))
    # print out_lines[0]
    # print list(set(out_lines))[0]
    # raw_input()
    # print num_skipped
    return out_lines, num_skipped


def write_train_test_au():
    dir_meta = '../data/pain'
    au_dir = os.path.join(dir_meta, 'anno_au')
    facs_dir = os.path.join(dir_meta, 'Frame_Labels','FACS')
    im_dir = os.path.join(dir_meta, 'preprocess_256_color')
    gt_pain_dir = os.path.join(dir_meta, 'Sequence_Labels', 'gt_avg')

    train_test_dir = os.path.join(dir_meta,'train_test_files_loo_1_thresh_au_only')
    util.mkdir(train_test_dir)
    
    strs_replace = [[facs_dir,im_dir],['_facs.txt','.jpg']]
    people_dirs = [os.path.split(dir_curr)[1] for dir_curr in glob.glob(os.path.join(au_dir,'*'))]
    people_dirs.sort()

    for idx_test_dir, test_dir in enumerate(people_dirs):

        train_file = os.path.join(train_test_dir,'train_'+str(idx_test_dir)+'.txt')
        test_file = os.path.join(train_test_dir,'test_'+str(idx_test_dir)+'.txt')

        train_lines = []
        num_skipped_all = 0
        for idx_people_dir, people_dir in enumerate(people_dirs):
            if idx_people_dir!=idx_test_dir:
        # train_dirs = [dir_curr for dir_curr in people_dirs if dir_curr!=test_dir]

                train_lines_curr, num_skipped = get_threshold_pain_au_only(au_dir,people_dir,gt_pain_dir, strs_replace)
                num_skipped_all += num_skipped
                train_lines += train_lines_curr

            else:
                test_lines, _ = get_threshold_pain_au_only(au_dir,people_dir,gt_pain_dir, strs_replace,threshold=0)

        print num_skipped_all
        print train_file, len(train_lines)
        print test_file, len(test_lines)
        
        # random.shuffle(train_lines)
        # random.shuffle(test_lines)

        util.writeFile(train_file, train_lines)
        util.writeFile(test_file, test_lines)





def script_check_bbox():
    bbox_file = '../data/pain/im_best_bbox/064-ak064/ak064t1afunaff/ak064t1afunaff019.npy'
    im_file = '../data/pain/Images/064-ak064/ak064t1afunaff/ak064t1afunaff019.png'
    im = scipy.misc.imread(im_file)
    bbox = np.load(bbox_file)
    print im.shape
    print bbox.shape

    out_file = '../scratch/pain_im.jpg'
    scipy.misc.imsave(out_file,im)

    out_file = '../scratch/pain_im_bbox.jpg'
    cv2.rectangle(im,(bbox[2],bbox[0]),(bbox[3],bbox[1]),(255,255,255),2)
    scipy.misc.imsave(out_file,im)


def script_save_cropped_faces():
    dir_meta= '../data/pain'
    in_dir_im = os.path.join(dir_meta,'Images')
    bbox_dir = os.path.join(dir_meta,'im_best_bbox')
    
    size_im = [256,256]
    savegray = False
    out_dir_im = os.path.join(dir_meta,'preprocess_'+str(size_im[0])+'_color')
    util.mkdir(out_dir_im)

    im_list_in = glob.glob(os.path.join(in_dir_im,'*','*','*.png'))

    args = []
    
    for idx_file_curr,in_file_curr in enumerate(im_list_in):
        out_file_curr = in_file_curr.replace(in_dir_im,out_dir_im).replace('.png','.jpg')
        bbox_file = in_file_curr.replace(in_dir_im,bbox_dir).replace('.png','.npy')
        if os.path.exists(out_file_curr) or not os.path.exists(bbox_file):
            continue
        out_dir_curr = os.path.split(out_file_curr)[0]
        util.makedirs(out_dir_curr)
        arg_curr = (in_file_curr,out_file_curr,bbox_file,size_im,savegray,idx_file_curr)
        args.append(arg_curr)

    print len(args)
    for arg_curr in args:
        # print arg_curr
        # raw_input()
        save_cropped_face_from_bbox(arg_curr)
    #     break

    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # pool.map(save_cropped_face_from_bbox,args)
    # pool.close()
    # pool.join()
    

def main():

    # write_train_test_au()
    # script_save_cropped_faces()
    # script_save_best_bbox()
    # looking_at_im()
    # looking_at_gt_pain()
    # script_create_sess_pspi_anno()
    # script_create_sess_facs_anno()


    return

    dir_meta = '../data/pain'
    frame_dir = os.path.join(dir_meta, 'Frame_Labels')
    dirs = glob.glob(os.path.join(frame_dir,'FACS','*'))

    facs_to_keep = [4,6,7,9,10,43]

    sum_frames = 0
    all_facs = []
    all_intensities = []
    for dir_curr in dirs:
        frame_files = glob.glob(os.path.join(dir_curr,'*','*.txt'))
        for frame_file in frame_files:
            anno = parse_facs_file(frame_file)

            if len(anno)>0:
                all_facs += list(anno[:,0])
                all_intensities += list(anno[:,1])


    all_facs = np.array(all_facs)
    all_intensities = np.array(all_intensities)
    
    print all_facs.shape

    for fac_curr in np.unique(all_facs):
        print fac_curr, np.sum(all_facs==fac_curr),np.unique(all_intensities[all_facs==fac_curr])




                # print anno
        #   break
        # break
    # print uni_facs

    #   print len(frame_files)
    #   sum_frames += len(frame_files)
    #   print frame_files[0]

    # print sum_frames


    # print len(dirs)
    

if __name__=='__main__':
    main()