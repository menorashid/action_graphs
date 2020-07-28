# import cv2
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
from preprocess_bp4d import save_align_mp

def download_image((url,out_file,idx)):
    if idx%100==0:
        print idx
    try:
        urllib.urlretrieve(url, out_file)
    except:
        print 'ERROR',url,out_file


def script_download_image():
    # idx_url_file,url_file,url_files,str_replace):
    dir_meta = '../data/emotionet'
    out_dir_im = os.path.join(dir_meta,'im')
    util.mkdir(out_dir_im)
    str_replace = ['http://cbcsnas01.ece.ohio-state.edu/EmotioNet/Images',out_dir_im]

    dir_url_files = os.path.join(dir_meta,'emotioNet_challenge_files_server')
    url_files = glob.glob(os.path.join(dir_url_files,'*.txt'))
    url_files.sort()

    
    args = []        
    out_files = []
    for idx_url_file, url_file in enumerate(url_files):
        print 'On file %d of %d' %(idx_url_file,len(url_files)) 
        im = [line_curr.split('\t')[0] for line_curr in util.readLinesFromFile(url_file)]
        # out_files = [im_curr.replace(str_replace[0],str_replace[1]) for im_curr in im]


        for idx_im_curr,im_curr in enumerate(im):
            out_file_curr = im_curr.replace(str_replace[0],str_replace[1])
            if os.path.exists(out_file_curr):
                out_files.append(out_file_curr)
                continue
            out_dir_curr = os.path.split(out_file_curr)[0]
            util.makedirs(out_dir_curr)
            args.append((im_curr, out_file_curr, idx_im_curr))

        print len(args)
        print url_file

    print len(args)
    print len(out_files)
    return out_files

def save_resized_images((in_file,out_file,im_size,savegray,idx_file_curr)):
    if idx_file_curr%100==0:
        print idx_file_curr

    img = cv2.imread(in_file);
    
    if len(img.shape)<3:
        img = img[:,:,np.newaxis]
        img = np.concatenate((img,img,img),2)
            
    if savegray:
        gray  =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi=cv2.resize(gray,tuple(im_size));
    else:
        roi=cv2.resize(img,tuple(im_size));

    cv2.imwrite(out_file,roi)
    

def script_save_resize_faces():
    dir_meta = '../data/emotionet'
    im_size = [256,256]
    im_file_list = os.path.join(dir_meta,'im_list.txt')
    
    in_dir_meta = os.path.join(dir_meta,'im')
    out_dir_meta = os.path.join(dir_meta,'preprocess_im_'+str(im_size[0])+'_color_nodetect')
    im_list_in = util.readLinesFromFile(im_file_list)


    
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

    # for arg in args:
    #     print arg
    #     save_resized_images(arg)
        # break

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(save_resized_images,args)
    
    

def script_save_kp():
    # out_file_list = os.path.join(dir_meta,'im_list.txt')
    # all_im = util.readLinesFromFile(out_file_list)

    dir_meta = '../data/emotionet'
    # im_size = [256,256]
    # out_dir_im = os.path.join(dir_meta,'preprocess_im_'+str(im_size[0])+'_color_nodetect')
    # out_dir_kp = out_dir_im.replace('_im_','_kp_')
    # str_im_rep = None
    # print out_dir_kp

    # im_file_list = out_dir_im+'_list_1.txt'
    # all_im = util.readLinesFromFile(im_file_list)    
    
    out_dir_im = os.path.join(dir_meta,'im')
    out_dir_kp = out_dir_im.replace('im','kp')
    print out_dir_kp
    str_im_rep = os.path.join(dir_meta,'preprocess_im_256_color_nodetect')

    print out_dir_kp

    im_file_list = os.path.join(dir_meta,'missing_256')
    all_im = util.readLinesFromFile(im_file_list) 
    print len(all_im)
    if str_im_rep is not None:
        all_im = [im_curr.replace(str_im_rep,out_dir_im) for im_curr in all_im]


    # chunk_size_im = len(all_im)//4
    # all_im_list = [all_im[x:x+chunk_size_im] for x in range(0, len(all_im), chunk_size_im)]
    # print len(all_im_list)
    # idx_to_do = 3
    # print idx_to_do
    # print len(all_im_list[idx_to_do])
    # arr_lens = [len(val) for val in all_im_list] 
    # print arr_lens
    # print sum(arr_lens)
    # # raw_input()
    # # return
    # all_im = all_im_list[idx_to_do]

    args = []
    for idx_im_curr,im_curr in enumerate(all_im):
        out_file_curr = im_curr.replace(out_dir_im,out_dir_kp).replace('.jpg','.npy')
        if os.path.exists(out_file_curr):
            print out_file_curr
            continue
        out_dir_curr = os.path.split(out_file_curr)[0]
        util.makedirs(out_dir_curr)

        args.append((im_curr,out_file_curr,idx_im_curr))
    #     # im_curr, out_file_curr, idx_arg

    print len(args)
    # chunk_size = max(3,len(args)//3)
    # args = [args[x:x+chunk_size] for x in range(0, len(args), chunk_size)]
    # print len(args)
    # print len(args[0])
    # print sum([len(val) for val in args])

    # pool = multiprocessing.Pool(4)
    # # for arg in args:
    # # print args
    # # save_align_mp(args)
    # # #     raw_input()
    # pool.map(save_align_mp,args)


def make_missing_np_list():
    dir_meta = '../data/emotionet'
    im_size =[256,256]
    out_dir_im = os.path.join(dir_meta,'preprocess_im_'+str(im_size[0])+'_color_nodetect')
    out_dir_kp = out_dir_im.replace('_im_','_kp_')
    
    print out_dir_kp

    im_file_list = out_dir_im+'_list_1.txt'
    out_file_missing = os.path.join(dir_meta,'missing_'+str(im_size[0]))
    all_im = util.readLinesFromFile(im_file_list)    
    print len(all_im)

    missing_list = []
    for idx_im_curr,im_curr in enumerate(all_im):
        out_file_curr = im_curr.replace(out_dir_im,out_dir_kp).replace('.jpg','.npy')
        if not os.path.exists(out_file_curr):
            missing_list.append(im_curr)

    print len(missing_list)
    print missing_list[0]
    util.writeFile(out_file_missing,missing_list)
    #     # im_curr, out_file_curr, idx_arg



def make_im_list():
    dir_meta = '../data/emotionet'
    im_size =[256,256]
    out_dir_meta = os.path.join(dir_meta,'preprocess_im_'+str(im_size[0])+'_color_nodetect')
    im_file_list = out_dir_meta+'_list_1.txt'
    im_list = glob.glob(os.path.join(out_dir_meta,'*','*.jpg'))
    print len(im_list)
    im_list.sort()

    util.writeFile(im_file_list,im_list)


def save_im_kp(im, kp, out_file):
    radius = int(max(im.shape[0],im.shape[1])*0.01)
    for kp_curr in kp:
        kp_curr = (int(kp_curr[0]),int(kp_curr[1]))
        cv2.circle(im,kp_curr, radius, (255,0,0),-1)
    cv2.imwrite(out_file,im)


def save_resize_kp_mp((im_org_file,im_size,kp_in_file,kp_out_file,idx_im_curr)):
    if idx_im_curr%100==0:
        print idx_im_curr

    # try:
    im_org = cv2.imread(im_org_file)
    # scipy.misc.imread(im_org_file)
    kp = np.load(kp_in_file)
    # print kp.shape
    # print im_org.shape
    # print im_org_file
    kp_org = kp/float(im_size[0])
    kp_org[:,0] = kp_org[:,0]*im_org.shape[1]
    kp_org[:,1] = kp_org[:,1]*im_org.shape[0]
    np.save(kp_out_file,kp_org)
    # except:
    #     pass

def save_resize_kp():
    dir_meta = '../data/emotionet'
    im_size = [256,256]
    out_dir_im = os.path.join(dir_meta,'preprocess_im_'+str(im_size[0])+'_color_nodetect')
    out_dir_kp = out_dir_im.replace('_im_','_kp_')
    im_file_list = out_dir_im+'_list_1.txt'
    
    out_dir_im_org = os.path.join(dir_meta,'im')
    out_dir_kp_org = os.path.join(dir_meta,'kp')
    all_im = util.readLinesFromFile(im_file_list)   
    print len(all_im)
    out_file_exists = os.path.join(dir_meta,'exists.txt')


    # exists = []
    args = []
    for idx_im_curr, im_curr in enumerate(all_im):
        im_org_file = im_curr.replace(out_dir_im,out_dir_im_org)
        kp_in_file = im_curr.replace(out_dir_im,out_dir_kp).replace('.jpg','.npy')
        kp_out_file = kp_in_file.replace(out_dir_kp,out_dir_kp_org)

        util.makedirs(os.path.split(kp_out_file)[0])
        
        if os.path.exists(kp_out_file) or not os.path.exists(kp_in_file):
            # print kp_out_file
            # exists.append(im_org_file)
            continue
        
        args.append((im_org_file,im_size,kp_in_file,kp_out_file,idx_im_curr))
    
    print len(args)
    # print len(exists)
    # util.writeFile(out_file_exists,exists)
    # print args[0]
    # # out_dir_scratch = os.path.join('../scratch','emotionet_kp_rs')
    # # util.mkdir(out_dir_scratch)
    # for idx_arg, arg in enumerate(args):
    #     print idx_arg,arg
        # raw_input()
        # save_resize_kp_mp(arg)
        # raw_input()
        # out_file_curr = os.path.join('../scratch',str(idx_arg)+'.jpg')
        # save_im_kp(cv2.imread(arg[0]),np.load(arg[-2]),out_file_curr)
        # print out_file_curr
        # break
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(save_resize_kp_mp,args)


def script_save_align_im():
    dir_meta = '../data/emotionet'
    im_size = [256,256]
    str_replace_im = os.path.join(dir_meta,'preprocess_im_'+str(im_size[0])+'_color_nodetect')
    out_dir_im_align = os.path.join(dir_meta,'preprocess_im_'+str(im_size[0])+'_color_align')
    im_file_list = str_replace_im+'_list_1.txt'
    
    out_dir_im = os.path.join(dir_meta,'im')
    out_dir_kp = os.path.join(dir_meta,'kp')
    # out_dir_kp = str_replace_im.replace('_im_','_kp_')

    all_im = util.readLinesFromFile(im_file_list)   
    all_im = [im_curr.replace(str_replace_im,out_dir_im) for im_curr in all_im]

    avg_pts_file = '../data/bp4d/avg_kp_256_192_32_56.npy'
    out_scale = im_size[:]
    # exists = []
    args = []
    for idx_im_curr, im_curr in enumerate(all_im):
        kp_file = im_curr.replace(out_dir_im,out_dir_kp).replace('.jpg','.npy')
        
        out_file = im_curr.replace(out_dir_im,out_dir_im_align)
        
        # if not os.path.exists(kp_file):
        #     continue
        if os.path.exists(out_file) or not os.path.exists(kp_file):
            continue
        
        out_dir_curr = os.path.split(out_file)[0]
        util.makedirs(out_dir_curr)

        util.makedirs(os.path.split(out_file)[0])
        args.append((im_curr,kp_file,avg_pts_file,out_file,out_scale, idx_im_curr))
    
    print len(args)
    print args[0]
    # args = args[:10]
    # # util.mkdir(out_dir_scratch)
    # for idx_arg, arg in enumerate(args):
    #     print arg
    #     raw_input()
        # save_align_im(arg)
    #     raw_input()
        # break
    pool = multiprocessing.dummy.Pool(multiprocessing.cpu_count())
    pool.map(save_align_im,args)
    pool.close()
    pool.join()
    


    
def save_align_im((im_org_file,kp_org_file,avg_pts_file,out_file, out_scale , idx_im_curr)):


    if idx_im_curr%100==0:
        print idx_im_curr

    im_org = cv2.imread(im_org_file)
    kp = np.load(kp_org_file)

    kp = kp/float(256)
    kp[:,0] = kp[:,0]*im_org.shape[1]
    kp[:,1] = kp[:,1]*im_org.shape[0]

    avg_pts = np.load(avg_pts_file)
    
    tform = skimage.transform.estimate_transform('similarity', kp, avg_pts)
    im_new = skimage.transform.warp(im_org, tform.inverse, output_shape=(out_scale[0],out_scale[1]), order=1, mode='edge')
    im_new = im_new*255
    # print np.min(im_new),np.max(im_new)
    cv2.imwrite(out_file,im_new)

        


def test_resize_kp():
    dir_meta = '../data/emotionet'
    im_size = [256,256]
    out_dir_im = os.path.join(dir_meta,'preprocess_im_'+str(im_size[0])+'_color_nodetect')
    out_dir_kp = out_dir_im.replace('_im_','_kp_')
    
    out_dir_im_org = os.path.join(dir_meta,'im')

    im_file_list = out_dir_im+'_list_1.txt'
    all_im = util.readLinesFromFile(im_file_list)   

    out_dir_scratch = '../scratch/emotionet_kp'
    util.mkdir(out_dir_scratch)

    for idx_im_curr, im_curr in enumerate(all_im[:100]):
        im_org_file = im_curr.replace(out_dir_im,out_dir_im_org)
        kp_in_file = im_curr.replace(out_dir_im,out_dir_kp).replace('.jpg','.npy')
        if not os.path.exists(kp_in_file):
            print 'CONTINUING',kp_in_file
            continue
        im_org = scipy.misc.imread(im_org_file)
        im = scipy.misc.imread(im_curr)
        
        kp = np.load(kp_in_file)
        
        kp_org = kp/float(im_size[0])
        kp_org[:,0] = kp_org[:,0]*im_org.shape[1]
        kp_org[:,1] = kp_org[:,1]*im_org.shape[0]

        out_file = os.path.join(out_dir_scratch,str(idx_im_curr)+'.jpg')
        save_im_kp(im,kp,out_file)

        out_file_org = os.path.join(out_dir_scratch,str(idx_im_curr)+'_org.jpg')
        save_im_kp(im_org,kp_org,out_file_org)

    visualize.writeHTMLForFolder(out_dir_scratch)


def make_anno_file(im_dir_pre):
    dir_meta = '../data/emotionet';
    out_dir = os.path.join(dir_meta,'anno_files')
    util.mkdir(out_dir)

    dir_url_files = os.path.join(dir_meta,'emotioNet_challenge_files_server');
    url_files = glob.glob(os.path.join(dir_url_files,'*.txt'));


    aus_to_keep = [1,2,4,5,6,9,12,17,20,25,26]
    aus_to_keep = np.array(aus_to_keep)
    idx_to_keep = aus_to_keep-1

    im_str_replace = ['http://cbcsnas01.ece.ohio-state.edu/EmotioNet/Images',
                    '../data/emotionet/preprocess_im_256_color_align']

    sum_aus = np.zeros(idx_to_keep.shape);

    # for anno_file in url_files:
    for idx_url_file,url_file in enumerate(url_files):
        out_file = url_file.replace(dir_url_files,out_dir);

        print url_file, out_file
        print 'On file %d of %d' %(idx_url_file,len(url_files)) 

        lines = util.readLinesFromFile(url_file)
        lines = [line_curr.split('\t') for line_curr in lines]

        
        lines_to_print = []

        for line in lines:
            # print line[0]
            im_curr = line[0].replace(im_str_replace[0],im_str_replace[1])

            if os.path.exists(im_curr):
                
                anno_arr = [int(val) for idx_val,val in enumerate(line[2:]) if idx_val in idx_to_keep]
                assert np.max(anno_arr)<=1
                
                sum_aus = sum_aus+np.array(anno_arr);

                line_out = ' '.join([str(val) for val in [im_curr]+anno_arr])
                lines_to_print.append(line_out)

        print len(lines),len(lines_to_print), sum_aus
        util.writeFile(out_file,lines_to_print)
        

def reduce_training_data():
    dir_meta = '../data/emotionet';
    anno_dir = os.path.join(dir_meta,'anno_files')
    out_dir_train_test = os.path.join(dir_meta,'train_test_files_3_files');
    util.mkdir(out_dir_train_test)

    anno_files = glob.glob(os.path.join(anno_dir,'dataFile_*.txt'))

    print len(anno_files);
    anno_files = anno_files[:3]

    util.writeFile(os.path.join(out_dir_train_test,'dataFiles_used.txt'),anno_files)

    org_au = [1,2,4,5,6,9,12,17,20,25,26]
    aus_to_keep = [1,9,12]


    num_chunks = 3



    all_data = []
    for anno_file in anno_files:
        anno_curr = util.readLinesFromFile(anno_file)
        print anno_file, len(anno_curr)
        all_data = all_data + anno_curr

    im_files = [line_curr.split(' ')[0] for line_curr in all_data]
    anno_bin = [[int(val) for val in line_curr.split(' ')[1:]] for line_curr in all_data]
    anno_bin = np.array(anno_bin);
    print np.sum(anno_bin,0)

    print len(im_files)
    print len(anno_bin)
    print im_files[0]
    print anno_bin[0]


    chunk_size = len(all_data)//num_chunks
    print chunk_size

    chunks = [all_data[x:x+chunk_size] for x in range(0, len(all_data), chunk_size)]
    chunks[-2] = chunks[-2]+chunks[-1]
    chunks = chunks[:3]

    print len(chunks)
    for chunk in chunks:
        print len(chunk)


    for fold_num in range(num_chunks):

        out_file_train = os.path.join(out_dir_train_test,'train_'+str(fold_num)+'.txt')
        out_file_test = os.path.join(out_dir_train_test,'test_'+str(fold_num)+'.txt')

        # print len(fold_data)
        print fold_num
        train_data = []
        [train_data.extend(data_curr) for idx_data_curr, data_curr in enumerate(chunks) if idx_data_curr!=fold_num]
        test_data = chunks[fold_num]
        
        print out_file_train, len(train_data)
        print out_file_test, len(test_data)

        util.writeFile(out_file_train, train_data)
        util.writeFile(out_file_test, test_data)

        
        
 

    # for anno_file in anno_files:
    #     print anno_file




def check_class_weights():
    folds = range(3)
    dir_files = '../data/emotionet/train_test_files_toy'
    for fold in folds:
        train_file = os.path.join(dir_files,'train_'+str(fold)+'.txt')
        test_file =  os.path.join(dir_files,'test_'+str(fold)+'.txt')
        train_weights = util.get_class_weights_au(util.readLinesFromFile(train_file))
        test_weights = util.get_class_weights_au(util.readLinesFromFile(test_file))
        diff_weights = np.abs(train_weights-test_weights)
        print fold
        print np.min(diff_weights),np.max(diff_weights),np.mean(diff_weights)
        # print list(train_weights)
        # print list(test_weights)
        # print list(diff_weights)



        


def main():

    # reduce_training_data()
    check_class_weights()

    # print 'hello'
    # reduce_training_data()
    # make_anno_file(None);
    # script_save_align_im()
    # dir_meta = '../data/bp4d'
    

    # params = [192,32,56]
    # avg_kp_file = os.path.join(dir_meta, 'avg_kp_256_'+'_'.join([str(val) for val in params])+'.npy')
    # print avg_kp_file


    # save_resize_kp()
    # test_resize_kp()
    # script_save_kp()
    # make_missing_np_list()
    # make_im_list()
    # script_save_resize_faces()
    # script_save_kp()
    # dir_meta = '../data/emotionet'
    # out_file_list = os.path.join(dir_meta,'im_list.txt')
    # all_im = util.readLinesFromFile(out_file_list)
    # print len(all_im)
    # for im_curr in all_im:
    #     assert im_curr.endswith('.jpg')


    # out_files = script_download_image()
    # util.writeFile(out_file_list,out_files)


    # idx_url_file,url_file,url_files, str_replace)




if __name__=='__main__':
    main()
