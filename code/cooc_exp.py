import os
import numpy as np
import scipy.io
import glob
from helpers import util, visualize
import sklearn.metrics
from globals import class_names
import torch
import exp_mill_bl as emb
from debugging_graph import readTrainTestFile

from sklearn.cluster import KMeans
from sklearn.externals import joblib
import time
import multiprocessing

# def cos_videos(first_arr, second_arr):
    
#     print first_vid.shape

def load_video(first_vid, npy_dict, vid_lens):
    if first_vid not in npy_dict:
        first_vid_data = np.load(first_vid)
        npy_dict[first_vid] = first_vid_data/np.linalg.norm(first_vid_data, axis = 1, keepdims = True)
        vid_lens.append(first_vid_data.shape[0])


def save_dots():
    dir_train_test_files = '../data/ucf101/train_test_files'
    train_file = os.path.join(dir_train_test_files,'train.txt')
    test_file = os.path.join(dir_train_test_files,'test.txt')
    
    lines = util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)
    # lines = lines[:5]
    num_files = len(lines)
    print len(lines)
    print lines[0]

    out_dir = '../scratch/i3d_dists'
    util.mkdir(out_dir)

    npy_files = [line_curr.split(' ')[0] for line_curr in lines]
    # vid_names = []
    vid_lens = []
    arr_dots = []
    npy_dict = {}
    for first_vid_idx in range(num_files-1):
        t = time.time()
        print first_vid_idx

        arr_curr = []
        first_vid = npy_files[first_vid_idx]
        load_video(first_vid, npy_dict, vid_lens)

        
        # print vid_lens

        first_vid_data = npy_dict[first_vid]

        for second_vid_idx in range(first_vid_idx+1, num_files):

            second_vid = npy_files[second_vid_idx]
            load_video(second_vid, npy_dict, vid_lens)
            second_vid_data = npy_dict[second_vid]

            dists = np.matmul(first_vid_data, second_vid_data.T)
            
            arr_curr.append(dists)

        arr_curr = np.concatenate(arr_curr,axis = 1)

        out_file = os.path.join(out_dir,str(first_vid_idx)+'.npy')
        np.save(out_file, arr_curr)

        print time.time()-t
        raw_input()
        # print arr_curr.shape
    
    vid_lens = np.array(vid_lens)
    
    np.save(os.path.join(out_dir,'vid_lens.npy'),vid_lens)


def save_sort_idx():
    dir_train_test_files = '../data/ucf101/train_test_files'
    train_file = os.path.join(dir_train_test_files,'train.txt')
    test_file = os.path.join(dir_train_test_files,'test.txt')
    
    lines = util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)
    # lines = lines[:5]
    num_files = len(lines)
    print len(lines)
    print lines[0]

    out_dir = '../scratch/i3d_dists'
    util.mkdir(out_dir)

    npy_files = [line_curr.split(' ')[0] for line_curr in lines]
    
        

    vid_lens = np.load(os.path.join(out_dir,'vid_lens.npy'))
    # print vid_lens[213]
    # data = np.load(npy_files[213])
    # print data.shape
    # idx_vid_len = 213
    # start_idx = np.sum(vid_lens[:idx_vid_len])
    # end_idx = np.sum(vid_lens[:idx_vid_len+1])
    # print start_idx, end_idx
    # raw_input()

    arr_dots = []
    
    for first_vid_idx in range(num_files-1):
        # if first_vid_idx in [209,213]:
        file_curr = os.path.join(out_dir,str(first_vid_idx)+'.npy')
        arr_dots.append(np.load(file_curr))
        # else:
        #     arr_dots.append([])
    
    
    total_len = np.sum(vid_lens)
    vid_idx_arr = np.ones(sum(vid_lens))*-1
    # print 'len(arr_dots), total_len',len(arr_dots), total_len
    

    for idx_vid_len in range(len(vid_lens)):
        print 'idx_vid_len',idx_vid_len
        start_idx = np.sum(vid_lens[:idx_vid_len])
        end_idx = np.sum(vid_lens[:idx_vid_len+1])
        vid_idx_arr[start_idx:end_idx]=idx_vid_len

        # print 'start_idx, end_idx',start_idx, end_idx
        # print 'arr_dots[idx_vid_len].shape',arr_dots[idx_vid_len].shape
        
        new_dot = np.ones((vid_lens[idx_vid_len],total_len))*-1
        
        if idx_vid_len<(len(vid_lens)-1):
            new_dot[:,end_idx:] = arr_dots[idx_vid_len]

        for prev in range(idx_vid_len):
            print prev,
            # print 'vid_lens[:idx_vid_len+1]',vid_lens[:idx_vid_len+1]
            start_idx_prev = np.sum(vid_lens[:prev])
            end_idx_prev = np.sum(vid_lens[:prev+1])
            start_ac = start_idx - end_idx_prev
            end_ac = end_idx - end_idx_prev

            prev_select = arr_dots[prev][:,start_ac:end_ac].T
            new_dot[:,start_idx_prev:end_idx_prev] = prev_select
        print ''
        sort_idx = np.argsort(new_dot,axis = 1)[:,::-1]
        out_file = os.path.join(out_dir,str(idx_vid_len)+'_sort_idx.npy')
        np.save(out_file, sort_idx)

    out_file = os.path.join(out_dir, 'vid_idx_arr.npy')
    np.save(out_file, vid_idx_arr)



def save_sort_idx_just_train():
    dir_train_test_files = '../data/ucf101/train_test_files'
    train_file = os.path.join(dir_train_test_files,'train.txt')
    test_file = os.path.join(dir_train_test_files,'test.txt')
    
    train_lines = util.readLinesFromFile(train_file)
    lines = train_lines+util.readLinesFromFile(test_file)
    # lines = lines[:5]
    num_train = len(train_lines)
    num_files = len(lines)
    print 'num_files, num_train',num_files, num_train
    
    max_to_match_with = num_train
    range_to_sort = range(num_train,num_files)

    

    out_dir = '../scratch/i3d_dists_just_train'
    out_dir_in = '../scratch/i3d_dists'
    util.mkdir(out_dir)

    npy_files = [line_curr.split(' ')[0] for line_curr in lines]
    
    vid_lens = np.load(os.path.join(out_dir_in,'vid_lens.npy'))
    vid_lens = vid_lens[:num_files]
    total_len = np.sum(vid_lens)
    total_len_train = np.sum(vid_lens[:num_train])
    print 'total_len, total_len_train',total_len, total_len_train
    
    arr_dots = []
    for first_vid_idx in range(num_train):
        file_curr = os.path.join(out_dir_in,str(first_vid_idx)+'.npy')
        arr_curr = np.load(file_curr)
        # e_curr = np.sum(vid_lens[:first_vid_idx+1])
        # arr_curr = arr_curr[:,:(total_len - e_curr)]
        arr_dots.append(arr_curr)
        
    
    vid_idx_arr = np.ones(sum(vid_lens))*-1
    

    for idx_vid_len in range_to_sort:
        out_file = os.path.join(out_dir,str(idx_vid_len)+'_sort_idx.npy')
        if os.path.exists(out_file):
            continue
    # range(num_files):
        print 'idx_vid_len',idx_vid_len
        start_idx = np.sum(vid_lens[:idx_vid_len])
        end_idx = np.sum(vid_lens[:idx_vid_len+1])
        vid_idx_arr[start_idx:end_idx]=idx_vid_len
        
        # print 'start_idx,end_idx', start_idx, end_idx
        
        new_dot = np.ones((vid_lens[idx_vid_len],total_len_train))*-1
        
        if idx_vid_len<(num_train-1):
            # print 'loading self'
            arr_curr = arr_dots[idx_vid_len]
            # np.load(file_curr)
            # e_curr = np.sum(vid_lens[:first_vid_idx+1])
            arr_curr = arr_curr[:,:(total_len_train - end_idx)]
            new_dot[:,end_idx:] = arr_curr
            # arr_dots[idx_vid_len]

        for prev in range(min(max_to_match_with,idx_vid_len)):
            assert prev<num_train
            # print 'vid_lens[:idx_vid_len+1]',vid_lens[:idx_vid_len+1]
            start_idx_prev = np.sum(vid_lens[:prev])
            end_idx_prev = np.sum(vid_lens[:prev+1])
            start_ac = start_idx - end_idx_prev
            end_ac = end_idx - end_idx_prev

            print prev,
            # , start_ac, end_ac

            prev_select = arr_dots[prev][:,start_ac:end_ac].T

            # print arr_dots[prev].shape,prev_select.shape
            
            new_dot[:,start_idx_prev:end_idx_prev] = prev_select
        print ''
        sort_idx = np.argsort(new_dot,axis = 1)[:,::-1]
        out_file = os.path.join(out_dir,str(idx_vid_len)+'_sort_idx.npy')
        # print out_file
        # print os.path.exists(out_file)
        # print sort_idx.shape, new_dot.shape
        # raw_input()
        np.save(out_file, sort_idx)

    # out_file = os.path.join(out_dir, 'vid_idx_arr.npy')
    # np.save(out_file, vid_idx_arr)


def save_sort_idx_just_train_opv():
    dir_train_test_files = '../data/ucf101/train_test_files'
    train_file = os.path.join(dir_train_test_files,'train.txt')
    test_file = os.path.join(dir_train_test_files,'test.txt')
    
    train_lines = util.readLinesFromFile(train_file)
    lines = train_lines+util.readLinesFromFile(test_file)
    num_train = len(train_lines)
    num_files = len(lines)
    print 'num_files, num_train',num_files, num_train

    out_dir = '../scratch/i3d_dists_just_train'
    out_dir_in = '../scratch/i3d_dists'
    util.mkdir(out_dir)

    npy_files = [line_curr.split(' ')[0] for line_curr in lines]
    
    vid_lens = np.load(os.path.join(out_dir_in,'vid_lens.npy'))
    # vid_lens = vid_lens[:len(lines)]
    
    # total_len = np.sum(vid_lens)
    total_len = np.sum(vid_lens)
    total_len_train = np.sum(vid_lens[:num_train])
    vid_idx_arr_file = os.path.join(out_dir, 'vid_idx_arr.npy')
    vid_idx_arr = np.load(vid_idx_arr_file)
    print vid_idx_arr.shape
    
    arr_dots = []
    for first_vid_idx in range(num_train):
        print first_vid_idx
        file_curr = os.path.join(out_dir_in,str(first_vid_idx)+'.npy')
        arr_curr = np.load(file_curr)
        # print arr_curr.shape
        # raw_input()
        # e_curr = np.sum(vid_lens[:first_vid_idx+1])
        # arr_curr = arr_curr[:,:(total_len - e_curr)]
        arr_dots.append(arr_curr)
    

    

    args =[]
    for idx_vid_len in range(len(vid_lens)):
        out_file_curr = os.path.join(out_dir,str(idx_vid_len)+'_sort_idx_opv.npz')
        if os.path.exists(out_file_curr):
            continue
        arg = (arr_dots, vid_idx_arr, idx_vid_len, vid_lens, out_file_curr,num_train)

        args.append(arg)
    print len(args)
    
    for arg in args:
        save_sort_idx_opv_mp(arg)
        # print 'done'
        # break

    # pool = multiprocessing.Pool()
    # pool.map(save_sort_idx_opv_mp, args)
    # pool.close()
    # pool.join()


def save_sort_idx_opv_mp((arr_dots, vid_idx_arr, idx_vid_len,vid_lens,out_file_curr, num_train)):
    print idx_vid_len
    vid_idx_opv = np.unique(vid_idx_arr)
    # print vid_idx_opv
    # return
    new_dot = integrate_dots(arr_dots, idx_vid_len, vid_lens, num_train)
    # print new_dot.shape
    # raw_input()
    max_vals_all = []
    max_idx_all = []

    for vid_idx_curr in vid_idx_opv:
        rel_vid_idx = np.where(vid_idx_curr == vid_idx_arr)[0]

        rel_dots = new_dot[:, vid_idx_arr==vid_idx_curr]
        max_idx = np.argmax(rel_dots,axis = 1)[:,np.newaxis]
        max_idx_all.append(max_idx.flatten())

        max_vals = np.take_along_axis(rel_dots, max_idx, axis = 1)

        max_idx_new_dot = rel_vid_idx[max_idx]
        max_vals_all.append(max_vals.flatten())
        
    max_vals_all = np.array(max_vals_all).T
    max_idx_all = np.array(max_idx_all).T
    sort_idx = np.argsort(max_vals_all,axis = 1)[:,::-1]

    np.savez_compressed(out_file_curr,sort_idx = sort_idx, max_vals_all = max_vals_all, max_idx_all = max_idx_all)


def integrate_dots(arr_dots, idx_vid_len, vid_lens, num_train):
    # for idx_vid_len in range(len(vid_lens)):
    total_len = np.sum(vid_lens)
    total_len_train = np.sum(vid_lens[:num_train])
    # print 'idx_vid_len',idx_vid_len
    start_idx = np.sum(vid_lens[:idx_vid_len])
    end_idx = np.sum(vid_lens[:idx_vid_len+1])
    # vid_idx_arr[start_idx:end_idx]=idx_vid_len
    
    new_dot = np.ones((vid_lens[idx_vid_len],total_len_train))*-1
    
    print idx_vid_len

    if idx_vid_len<(num_train-1):
        # new_dot[:,end_idx:] = arr_dots[idx_vid_len]

        # print 'loading self'
        arr_curr = arr_dots[idx_vid_len]
        arr_curr = arr_curr[:,:(total_len_train - end_idx)]
        new_dot[:,end_idx:] = arr_curr
        # arr_dots[idx_vid_len]

    for prev in range(min(num_train,idx_vid_len)):
        
        assert prev<num_train
        # print 'vid_lens[:idx_vid_len+1]',vid_lens[:idx_vid_len+1]
        start_idx_prev = np.sum(vid_lens[:prev])
        end_idx_prev = np.sum(vid_lens[:prev+1])
        start_ac = start_idx - end_idx_prev
        end_ac = end_idx - end_idx_prev

        print prev,
        prev_select = arr_dots[prev][:,start_ac:end_ac].T
        new_dot[:,start_idx_prev:end_idx_prev] = prev_select

    return new_dot

def save_cooc_per_vid_mp((sort_idx_file, out_file, n, vid_idx_arr, idx_vid_len)):
    print idx_vid_len
    try:
        sort_idx = np.load(sort_idx_file)
    except:
        print 'error ',sort_idx_file
        return

    if sort_idx_file.endswith('.npz'):
        sort_idx = sort_idx['sort_idx']

    # print sort_idx.shape
    sort_idx = sort_idx[:,:n]
    num_segments = sort_idx.shape[0]

    arr_cooc = np.ones((num_segments,num_segments))
    for i in range(num_segments):
        for j in range(i+1,num_segments):
            vid1 = vid_idx_arr[sort_idx[i]]
            vid2 = vid_idx_arr[sort_idx[j]]            
            comm1 = np.sum(np.isin(vid1,vid2))
            comm2 = np.sum(np.isin(vid2,vid1))
            cooc = min(comm1,comm2)/float(n)
            arr_cooc[i,j] = cooc
            arr_cooc[j,i] = cooc

    # print arr_cooc.shape, 

    # print out_file
    np.savez_compressed(out_file, arr_cooc)


def get_cooc_inter_video(sort_idx_1_org, sort_idx_2_org, vid_idx_arr, n_vals):
    
    num_segments_1 = sort_idx_1_org.shape[0]
    num_segments_2 = sort_idx_2_org.shape[0]
    arr_cooc = np.ones((num_segments_1,num_segments_2, len(n_vals)))

    for idx_n, n in enumerate(n_vals):
        sort_idx_1 = sort_idx_1_org[:,:n]
        sort_idx_2 = sort_idx_2_org[:,:n]
        for i in range(num_segments_1):
            for j in range(num_segments_2):
                vid1 = vid_idx_arr[sort_idx_1[i]]
                vid2 = vid_idx_arr[sort_idx_2[j]]            
                comm1 = np.sum(np.isin(vid1,vid2))
                comm2 = np.sum(np.isin(vid2,vid1))
                cooc = min(comm1,comm2)/float(n)
                arr_cooc[i,j, idx_n] = cooc
                # arr_cooc[j,i] = cooc

    # print out_file
    return arr_cooc
    # np.savez_compressed(out_file, arr_cooc)

def get_cooc_per_vid(n = 10, just_train = False, one_per_vid = False):

    dir_train_test_files = '../data/ucf101/train_test_files'
    dir_gt_vecs = '../data/ucf101/gt_vecs/just_primary_corrected'
    train_file = os.path.join(dir_train_test_files,'train.txt')
    test_file = os.path.join(dir_train_test_files,'test.txt')
    
    
    if just_train:
        out_dir = '../scratch/i3d_dists_just_train'
        lines = util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)
    else:
        out_dir = '../scratch/i3d_dists'
        lines = util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)
    
    vid_idx_arr_file = os.path.join(out_dir, 'vid_idx_arr.npy')
    vid_idx_arr = np.load(vid_idx_arr_file)
    
    if one_per_vid:
        out_dir_coocs = os.path.join(out_dir,'arr_coocs_opv_'+str(n))
        vid_idx_arr = np.unique(vid_idx_arr)
        post_load = '_sort_idx_opv.npz'
        # func_curr = save_cooc_per_vid_one_per_vid_mp
    else:
        out_dir_coocs = os.path.join(out_dir,'arr_coocs_'+str(n))
        post_load = '_sort_idx.npy'
        
    func_curr = save_cooc_per_vid_mp
    util.mkdir(out_dir_coocs)

    

    npy_files = [line_curr.split(' ')[0] for line_curr in lines]
    # gt_files = [os.path.join(dir_gt_vecs, os.path.split(line_curr)[1]) for line_curr in npy_files]

    num_files = len(npy_files)

    # n = 10
    
    args = []

    for idx_vid_len in range(num_files):
        
        vid_name = os.path.split(npy_files[idx_vid_len])[1]
        out_file = os.path.join(out_dir_coocs, vid_name.replace('.npy','.npz'))

        sort_idx_file = os.path.join(out_dir,str(idx_vid_len)+post_load)

        if os.path.exists(out_file):
            continue

        arg_curr = (sort_idx_file, out_file, n, vid_idx_arr, idx_vid_len)
        args.append(arg_curr)

    print len(args)
   
    if len(args)>0:
        pool = multiprocessing.Pool()
        pool.map(func_curr, args)
        pool.close()
        pool.join()

def visualize_cooc_hists(n = 10, just_train = False, one_per_vid = 'reg'):

    dir_train_test_files = '../data/ucf101/train_test_files'
    dir_gt_vecs = '../data/ucf101/gt_vecs/just_primary_corrected'
    train_file = os.path.join(dir_train_test_files,'train.txt')
    test_file = os.path.join(dir_train_test_files,'test.txt')
    if just_train:
        out_dir = '../scratch/i3d_dists_just_train'
        lines = util.readLinesFromFile(train_file)
        # +util.readLinesFromFile(test_file)
    else:
        out_dir = '../scratch/i3d_dists'
        lines = util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)

    # out_dir = '../scratch/i3d_dists'
    if one_per_vid =='opv':
        out_dir_pre = 'arr_coocs_opv_'
    elif one_per_vid == 'opvpc':
        out_dir_pre = 'arr_coocs_per_class/'
    else:
        out_dir_pre = 'arr_coocs_'


    # # n = 10
    out_dir_coocs = os.path.join(out_dir,out_dir_pre+str(n))
    out_dir_hists = os.path.join(out_dir,out_dir_pre+str(n)+'_viz')

    # out_dir_coocs = os.path.join(out_dir,'arr_coocs_per_class',n)
    # out_dir_hists = os.path.join(out_dir,'arr_coocs_per_class',n+'_viz')
    util.mkdir(out_dir_hists)
    out_dir_cooc_viz = os.path.join(out_dir_hists, 'mat')
    out_dir_bg = os.path.join(out_dir_hists,'bg')
    out_dir_fg = os.path.join(out_dir_hists,'fg')
    out_dir_bg_nall = os.path.join(out_dir_hists,'bg_nall')
    out_dir_fg_nall = os.path.join(out_dir_hists,'fg_nall')
    
    out_dirs_all = [out_dir_cooc_viz, out_dir_fg,out_dir_bg, out_dir_fg_nall, out_dir_bg_nall]
    for out_dir_curr in out_dirs_all:
        util.mkdir(out_dir_curr)



    npy_files = [line_curr.split(' ')[0] for line_curr in lines]
    vid_names_per_class, class_id = get_vid_names_per_class(lines)

    gt_files = [os.path.join(dir_gt_vecs, os.path.split(line_curr)[1]) for line_curr in npy_files]
    
    num_files = len(npy_files)
    all_vals_all = []


    legend_entries = ['FG All','FG FG','BG All','BG BG','FG BG']
    xlabel = 'Cooc Value'
    ylabel = 'Frequency'
    num_bins = np.arange(0,1.1,.1)
    xtick_labels = ['%.1f'%val for val in num_bins]
    plot_idx_all = [None, [0,1],[2,3],[1,4],[3,4]]
    title_pres = ['Mat ','Foreground Hist for ', 'Background Hist for ','Foreground Hist for ', 'Background Hist for ']

    for idx_vid, gt_file in enumerate(gt_files):
        vid_name = os.path.split(gt_file)[1]
        gt_arr = np.load(gt_file)
        arr_cooc_file = os.path.join(out_dir_coocs, vid_name.replace('.npy','.npz'))
        arr_cooc = np.load(arr_cooc_file)['arr_0']
        just_vid_name = vid_name[:vid_name.rindex('.')]

        eye = -2*np.eye(arr_cooc.shape[0])
        arr_cooc_h = arr_cooc+eye

        fg_all = arr_cooc_h[gt_arr>0,:]
        fg_fg = fg_all[:,gt_arr>0]
        fg_bg = fg_all[:,gt_arr==0]
        bg_all = arr_cooc_h[gt_arr==0,:]
        bg_bg = bg_all[:,gt_arr==0]
        
        all_vals = [val.flatten() for val in [fg_all,fg_fg,bg_all,bg_bg,fg_bg]]
        all_vals_all.append(all_vals)

        # for idx_out_file, out_dir_curr in enumerate(out_dirs_all):
        #     out_file_curr = os.path.join(out_dir_curr, just_vid_name+'.jpg')
        #     if os.path.exists(out_file_curr):
        #         continue
            
        #     title = title_pres[idx_out_file]+just_vid_name
        #     plot_idx_curr = plot_idx_all[idx_out_file]
            
        #     if plot_idx_curr is None:
        #         visualize.saveMatAsImage(arr_cooc, out_file_curr, title = title)
        #     else:
        #         vals_curr = [all_vals[idx_curr] for idx_curr in plot_idx_curr]
        #         legend_entries_curr = [legend_entries[idx_curr] for idx_curr in plot_idx_curr]
        #         bins_all = [num_bins for idx_curr in plot_idx_curr]

        #         visualize.plotMultiHist(out_file_curr ,vals = vals_curr, num_bins = bins_all, legend_entries = legend_entries_curr, title = title, xlabel = xlabel, ylabel = ylabel, xticks = xtick_labels, density = True, align = 'mid')
        #     print out_file_curr

    print len(class_id)
    for class_idx, class_id_curr in enumerate(class_id.T):
        print class_id_curr
        vals_rel = [all_vals_all[idx] for idx in np.where(class_id_curr)[0]]
        print len(vals_rel)
        # 
        vals_rel_new = [[] for idx in range(len(vals_rel[0]))]
        print vals_rel_new
        for vals_all_curr in vals_rel:
            for idx_val_cell, val_cell in enumerate(vals_all_curr):
                vals_rel_new[idx_val_cell].append(val_cell)
        vals_rel_new = [np.concatenate(vals_row) for vals_row in vals_rel_new]
        vals_rel = vals_rel_new

        just_vid_name = class_names[class_idx]

        for idx_out_file, out_dir_curr in enumerate(out_dirs_all):
            out_file_curr = os.path.join(out_dir_curr, just_vid_name+'.jpg')
            title = title_pres[idx_out_file]+just_vid_name
            plot_idx_curr = plot_idx_all[idx_out_file]
            
            if plot_idx_curr is None:
                continue
                # visualize.saveMatAsImage(arr_cooc, out_file_curr, title = title)
            else:
                vals_curr = [vals_rel[idx_curr] for idx_curr in plot_idx_curr]
                legend_entries_curr = [legend_entries[idx_curr] for idx_curr in plot_idx_curr]
                bins_all = [num_bins for idx_curr in plot_idx_curr]

                visualize.plotMultiHist(out_file_curr ,vals = vals_curr, num_bins = bins_all, legend_entries = legend_entries_curr, title = title, xlabel = xlabel, ylabel = ylabel, xticks = xtick_labels, density = True, align = 'mid')
            print out_file_curr
            if idx_out_file == (len(out_dirs_all)-1):
                visualize.writeHTMLForFolder(out_dir_curr)

     

def plot_all_necessaries(just_vid_name, out_dir_cooc_viz, out_dir_fg, out_dir_bg, gt_arr_row, gt_arr_col, arr_cooc):
    # just_vid_name = vid_name[:vid_name.rindex('.')]

    out_file_cooc = os.path.join(out_dir_cooc_viz, just_vid_name+'.jpg')
    out_file_fg = os.path.join(out_dir_fg, just_vid_name+'.jpg')
    out_file_bg = os.path.join(out_dir_bg, just_vid_name+'.jpg')
    
    # eye = -2*np.eye(arr_cooc.shape[0])
    # arr_cooc_h = arr_cooc+eye

    
    fg_all = arr_cooc[gt_arr_row>0,:]
    fg_fg = fg_all[:,gt_arr_col>0]
    bg_all = arr_cooc[gt_arr_row==0,:]
    bg_bg = bg_all[:,gt_arr_col==0]
    
    num_bins = np.arange(0,1.1,.1)
    
    all_vals = [val.flatten() for val in [fg_all,fg_fg,bg_all,bg_bg]]
    
    legend_entries = ['FG All','FG FG','BG All','BG BG']
    xlabel = 'Cooc Value'
    ylabel = 'Frequency'
    xtick_labels = ['%.1f'%val for val in num_bins]
    

    title = 'Foreground Hist for '+just_vid_name
    visualize.plotMultiHist(out_file_fg ,vals = all_vals[:2], num_bins = [num_bins, num_bins], legend_entries = legend_entries[:2], title = title, xlabel = xlabel, ylabel = ylabel, xticks = xtick_labels, density = True, align = 'mid')

    title = 'Foreground Hist for '+just_vid_name
    visualize.plotMultiHist(out_file_bg ,vals = all_vals[2:], num_bins = [num_bins, num_bins], legend_entries = legend_entries[2:], title = title, xlabel = xlabel, ylabel = ylabel, xticks = xtick_labels, density = True, align = 'mid')
    
    title = 'Mat '+just_vid_name
    arr_cooc_mat = arr_cooc
    arr_cooc_mat[arr_cooc_mat<0]=0
    visualize.saveMatAsImage(arr_cooc_mat, out_file_cooc, title = title)
    print out_file_cooc


def make_cooc_html(just_train = False):
    # read lines from files

    dir_train_test_files = '../data/ucf101/train_test_files'
    train_file = os.path.join(dir_train_test_files,'train_just_primary_corrected.txt')
    test_file = os.path.join(dir_train_test_files,'test_just_primary_corrected.txt')
    
    if just_train:
        out_dir = '../scratch/i3d_dists_just_train'
        out_dir_html = '../scratch/i3d_dists_just_train_htmls'
        lines = util.readLinesFromFile(train_file)
    else:
        out_dir = '../scratch/i3d_dists'
        out_dir_html = '../scratch/i3d_dists_htmls'
        lines = util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)

    # get vid names per class
    npy_files = [line_curr.split(' ')[0] for line_curr in lines]
    vid_names_per_class, class_id = get_vid_names_per_class(lines)

    #make the html paths
    util.mkdir(out_dir_html)
    dir_server = '/disk2/maheen_data'
    str_replace = ['..',os.path.join(dir_server,'nn_net')]

    # make the pre lists
    n_vals = [10,25,50,100]
    post_dirs = ['mat','bg','fg']
    in_dirs = [[] for post_dir in post_dirs]
    in_dir_strs = [[] for post_dir in post_dirs]
    out_dir_htmls = [os.path.join(out_dir_html,post_dir) for post_dir in post_dirs]
    for idx_post_dir, post_dir in enumerate(post_dirs):
        for n_val in n_vals:
            dir_curr = os.path.join(out_dir, 'arr_coocs_'+str(n_val)+'_viz', post_dir)
            str_curr = 'N '+str(n_val)
            in_dirs[idx_post_dir].append((dir_curr, str_curr))
            

    print in_dirs

    for idx_html, out_dir_html in enumerate(out_dir_htmls):
        util.mkdir(out_dir_html)
        in_dirs_curr = in_dirs[idx_html]
        for idx_class, vid_names in enumerate(vid_names_per_class):
            out_file_html = os.path.join(out_dir_html,class_names[idx_class]+'.html')
            rows = []
            captions = []

            for vid_name in vid_names:
                row_curr = []; caption_curr = []
                for in_dir_curr, str_curr in in_dirs_curr:
                    file_curr = os.path.join(in_dir_curr, vid_name+'.jpg')
                    file_curr = util.getRelPath(file_curr.replace(str_replace[0],str_replace[1]), dir_server)
                    row_curr.append(file_curr)
                    caption_curr.append(str_curr+' '+vid_name)
                rows.append(row_curr)
                captions.append(caption_curr)
                # (out_file_html,im_paths,captions,height=height,width=width)
            visualize.writeHTML(out_file_html, rows, captions, height = 330, width = 400)

        #mat rows
        #fg rows
        #bg rows
        # for each n value
            #append to each of the rows
            #add caption
        #add to all rows
    #write htmls

def get_vid_names_per_class(lines):
    line_splits = [line.split(' ') for line in lines]
    npy_files = [line_split[0] for line_split in line_splits]
    class_id = [[int(val) for val in line_split[1:]] for line_split in line_splits]
    class_id = np.array(class_id)
    just_vid_names = np.array([os.path.split(file_curr)[1][:-4] for file_curr in npy_files])
    vid_names_per_class = []
    for idx_class in range(class_id.shape[1]):
        vid_names_per_class.append(just_vid_names[class_id[:,idx_class]>0])
    return vid_names_per_class, class_id


def get_coocs_per_class(just_train = False):

    # load files
    dir_train_test_files = '../data/ucf101/train_test_files'
    train_file = os.path.join(dir_train_test_files,'train_just_primary_corrected.txt')
    test_file = os.path.join(dir_train_test_files,'test_just_primary_corrected.txt')
    dir_gt_vecs = '../data/ucf101/gt_vecs/just_primary_corrected'

    if just_train:
        out_dir = '../scratch/i3d_dists_just_train'
        out_dir_coocs = '../scratch/i3d_dists_just_train/arr_coocs_pairwise'
        util.mkdir(out_dir_coocs)
        lines = util.readLinesFromFile(train_file)
    else:
        out_dir = '../scratch/i3d_dists'
        lines = util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)

    npy_files = [line_curr.split(' ')[0] for line_curr in lines]
    gt_files = [os.path.join(dir_gt_vecs, os.path.split(line_curr)[1]) for line_curr in npy_files]

    # get vid names per class
    vid_names_per_class, class_id = get_vid_names_per_class(lines)

    # load video idx
    vid_idx_arr_file = os.path.join(out_dir, 'vid_idx_arr.npy')
    vid_idx_arr = np.load(vid_idx_arr_file)
    print vid_idx_arr.shape

    #make giant matrix to fill up 
    # giant_cooc_matrix = np.zeros((vid_idx_arr.shape[0],vid_idx_arr.shape[1]))

    # for every class
    #for every video
    #for every other video
    #exclude other videos idx
    #get cooc for these two vids
    #fill up rel part in giant matrix
    #use giant gt to get hists
    #done
    n_vals = [10,25,50,100]

    args = []
    idx_test = 0
    for idx_class in range(class_id.shape[1]):
        rel_vid_ids = list(np.where(class_id[:,idx_class])[0])
        for idx_1, vid_id_1 in enumerate(rel_vid_ids):
            sort_idx_1_file = os.path.join(out_dir, str(vid_id_1)+'_sort_idx.npy') 
            for idx_2 in range(idx_1+1,len(rel_vid_ids)):
                vid_id_2 = rel_vid_ids[idx_2]
                sort_idx_2_file = os.path.join(out_dir, str(vid_id_2)+'_sort_idx.npy') 
                out_file = '_'.join([str(val) for val in [idx_class, vid_id_1, vid_id_2]])+'.npz'
                out_file = os.path.join(out_dir_coocs,out_file)
                if os.path.exists(out_file):
                    continue

                args.append((sort_idx_1_file, sort_idx_2_file, vid_id_1, vid_id_2, vid_idx_arr, n_vals, out_file, idx_test))
                idx_test+=1

    print len(args)

    raw_input()
    # for arg in args:
    #     print arg[-2]
    #     save_cooc_inter_video_mp(arg)
    #     raw_input()
    
    # pool = multiprocessing.Pool()
    # pool.map(save_cooc_inter_video_mp, args)
    # pool.close()
    # pool.join()

def save_hists_per_class(just_train = False):

    # load files
    dir_train_test_files = '../data/ucf101/train_test_files'
    train_file = os.path.join(dir_train_test_files,'train_just_primary_corrected.txt')
    test_file = os.path.join(dir_train_test_files,'test_just_primary_corrected.txt')
    dir_gt_vecs = '../data/ucf101/gt_vecs/just_primary_corrected'

    if just_train:
        out_dir = '../scratch/i3d_dists_just_train'
        out_dir_coocs = '../scratch/i3d_dists_just_train/arr_coocs_pairwise'

        out_dir_viz = '../scratch/i3d_dists_just_train/arr_coocs_pairwise_viz'
        util.mkdir(out_dir_viz)

        lines = util.readLinesFromFile(train_file)
    else:
        out_dir = '../scratch/i3d_dists'
        lines = util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)

    npy_files = [line_curr.split(' ')[0] for line_curr in lines]
    gt_files = [os.path.join(dir_gt_vecs, os.path.split(line_curr)[1]) for line_curr in npy_files]

    # get vid names per class
    vid_names_per_class, class_id = get_vid_names_per_class(lines)

    # load video idx
    vid_idx_arr_file = os.path.join(out_dir, 'vid_idx_arr.npy')
    vid_idx_arr = np.load(vid_idx_arr_file)
    print vid_idx_arr.shape

    # load arrs
    # transpose and fit in other place 
    # make per video hists
    # make giant per class hist

    n_vals = [10,25,50,100]
    
    mat_dirs = [os.path.join(out_dir_viz, 'n_'+str(n_val),'mat') for n_val in n_vals]
    fg_dirs = [os.path.join(out_dir_viz, 'n_'+str(n_val),'fg') for n_val in n_vals]
    bg_dirs = [os.path.join(out_dir_viz, 'n_'+str(n_val),'bg') for n_val in n_vals]
    [util.makedirs(dir_curr) for dirs in [mat_dirs, fg_dirs, bg_dirs] for dir_curr in dirs ]
    args = []
    idx_test = 0

    # for every class
    for idx_class in range(class_id.shape[1]):
        # get vid ids
        rel_vid_ids = list(np.where(class_id[:,idx_class])[0])
        rel_vid_idx_arr = vid_idx_arr[np.isin(vid_idx_arr, rel_vid_ids)]
        total_segs = rel_vid_idx_arr.shape[0]
    
        # make giant mat with vid_idx_arr
        giant_cooc = np.ones((total_segs, total_segs,len(n_vals)))*-1

        # make giant gt arr 
        giant_gt = [np.load(gt_files[rel_vid_id]) for rel_vid_id in rel_vid_ids]
        giant_gt = np.concatenate(giant_gt, axis = 0)
        
        assert giant_gt.shape[0]==rel_vid_idx_arr.shape[0]

        for idx_1, vid_id_1 in enumerate(rel_vid_ids):
            sort_idx_1_file = os.path.join(out_dir, str(vid_id_1)+'_sort_idx.npy') 
            for idx_2 in range(idx_1+1,len(rel_vid_ids)):
                vid_id_2 = rel_vid_ids[idx_2]
                sort_idx_2_file = os.path.join(out_dir, str(vid_id_2)+'_sort_idx.npy') 
                out_file = '_'.join([str(val) for val in [idx_class, vid_id_1, vid_id_2]])+'.npz'
                out_file = os.path.join(out_dir_coocs,out_file)
                assert os.path.exists(out_file)

                cooc = np.load(out_file)['arr_0']

                idx_arr_1 = np.where(rel_vid_idx_arr==vid_id_1)[0]
                idx_arr_2 = np.where(rel_vid_idx_arr==vid_id_2)[0]
                start_1 = np.min(idx_arr_1); end_1 = np.max(idx_arr_1)+1
                start_2 = np.min(idx_arr_2); end_2 = np.max(idx_arr_2)+1

                giant_cooc[start_1:end_1, start_2:end_2,:] = cooc
                giant_cooc[ start_2:end_2,start_1:end_1,:] = np.transpose(cooc,(1,0,2))


        for vid_id in rel_vid_ids:
            just_vid_name = os.path.split(gt_files[vid_id])[1]
            just_vid_name = just_vid_name[:just_vid_name.rindex('.')]
            bin_keep = rel_vid_idx_arr==vid_id
            gt_arr_row = giant_gt[bin_keep]
            arr_cooc = giant_cooc[bin_keep,:,:]
            for idx_n_val, n_val in enumerate(n_vals):
                plot_all_necessaries(just_vid_name, 
                    mat_dirs[idx_n_val],
                    fg_dirs[idx_n_val], 
                    bg_dirs[idx_n_val],
                    gt_arr_row, 
                    giant_gt, 
                    arr_cooc[:,:,idx_n_val])
        
        just_vid_name = class_names[idx_class]
        for idx_n_val, n_val in enumerate(n_vals):
            plot_all_necessaries(just_vid_name, 
                    mat_dirs[idx_n_val],
                    fg_dirs[idx_n_val], 
                    bg_dirs[idx_n_val],
                    giant_gt, 
                    giant_gt, 
                    giant_cooc[:,:,idx_n_val])


def save_cooc_inter_video_mp((sort_idx_1_file, sort_idx_2_file, vid_id_1, vid_id_2, vid_idx_arr, n_vals, out_file, idx_test)):
    print idx_test
    sort_idx_1 = np.load(sort_idx_1_file)            
    sort_idx_2 = np.load(sort_idx_2_file)
    idx_exclude = np.logical_or(vid_idx_arr==vid_id_1,vid_idx_arr==vid_id_2)
    idx_exclude = np.where(idx_exclude)[0]
    bin_include_2 = np.isin(sort_idx_2, idx_exclude, invert = True)
    bin_include_1 = np.isin(sort_idx_1, idx_exclude, invert = True)
    shape_1 = sort_idx_1.shape
    shape_2 = sort_idx_2.shape

    sort_idx_inc_1 = np.reshape(sort_idx_1.flatten()[bin_include_1.flatten()],(shape_1[0],-1))
    sort_idx_inc_2 = np.reshape(sort_idx_2.flatten()[bin_include_2.flatten()],(shape_2[0],-1))

    arr_cooc_all = get_cooc_inter_video(sort_idx_inc_1, sort_idx_inc_2, vid_idx_arr, n_vals)
    
    np.savez_compressed(out_file, arr_cooc_all)



def compress_files():
    in_dir = '../scratch/i3d_dists'
    out_dir = '../scratch_new/i3d_dists'
    util.mkdir(out_dir)
    npy_files = glob.glob(os.path.join(in_dir, '*_sort_idx.npy'))

    for idx_file_curr, file_curr in enumerate(npy_files):
        print idx_file_curr, len(npy_files)
        data_curr = np.load(file_curr)
        out_file = os.path.join(out_dir, os.path.split(file_curr)[1])
        out_file = out_file[:out_file.rindex('.')]+'.npz'
        np.savez_compressed(out_file,data_curr)
        

def create_comparative_html_fast():
    just_train = True
    # load files
    dir_train_test_files = '../data/ucf101/train_test_files'
    train_file = os.path.join(dir_train_test_files,'train_just_primary_corrected.txt')
    test_file = os.path.join(dir_train_test_files,'test_just_primary_corrected.txt')
    dir_gt_vecs = '../data/ucf101/gt_vecs/just_primary_corrected'

    if just_train:
        out_dir = '../scratch/i3d_dists_just_train'
        out_dir_coocs = '../scratch/i3d_dists_just_train/arr_coocs_pairwise'

        out_dir_viz = '../scratch/i3d_dists_just_train/arr_coocs_pairwise_viz'
        util.mkdir(out_dir_viz)

        lines = util.readLinesFromFile(train_file)
    else:
        out_dir = '../scratch/i3d_dists'
        lines = util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)

    npy_files = [line_curr.split(' ')[0] for line_curr in lines]
    gt_files = [os.path.join(dir_gt_vecs, os.path.split(line_curr)[1]) for line_curr in npy_files]

    # get vid names per class
    vid_names_per_class, class_id = get_vid_names_per_class(lines)
    # print vid_names_per_class

    out_dir_html = out_dir+'_pairwise_htmls'
    util.mkdir(out_dir_html)
    dir_server = '/disk2/maheen_data'
    out_dir_viz = out_dir_viz.replace('..',os.path.join(dir_server,'nn_net'))


    for dir_type in ['fg','bg','mat']:
        out_dir_html_curr = os.path.join(out_dir_html,dir_type)
        util.mkdir(out_dir_html_curr)
        n_strs = ['n_10','n_25','n_50','n_100']
        folders = [os.path.join(out_dir_viz,n_str,dir_type) for n_str in n_strs]
        for idx_class, vid_names_curr in enumerate(vid_names_per_class):
            class_name = class_names[idx_class]
            img_names = [class_name+'.jpg']+[vid_name+'.jpg' for vid_name in vid_names_curr]
            out_file_html = os.path.join(out_dir_html_curr,class_name+'.html')
            visualize.writeHTMLForDifferentFolders(out_file_html,folders,n_strs,img_names,rel_path_replace=dir_server,height=330,width=400) 


def cooc_sanity_check():
    dir_cooc_meta = '../scratch/i3d_dists_just_train'
    num_neighbors = 100
    dir_cooc = os.path.join(dir_cooc_meta,'arr_coocs_'+str(num_neighbors))
    all_files = glob.glob(os.path.join(dir_cooc, 'video_*.npz'))
    print len(all_files)

    dir_i3d_feat = '../data/i3d_features'
    dir_i3d_feat_val = os.path.join(dir_i3d_feat, 'Thumos14-I3D-JOINTFeatures_val')
    dir_i3d_feat_test = os.path.join(dir_i3d_feat, 'Thumos14-I3D-JOINTFeatures_test')

    for idx_file_curr, file_curr in enumerate(all_files):
        print file_curr
        just_name = os.path.split(file_curr)[1]
        just_name = just_name[:just_name.rindex('.')]
        cooc = np.load(file_curr)['arr_0']
        
        if 'validation' in os.path.split(file_curr)[1]:
            feat_file = os.path.join(dir_i3d_feat_val, just_name+'.npy')
        else:
            feat_file = os.path.join(dir_i3d_feat_test, just_name+'.npy')

        features = np.load(feat_file)

        assert cooc.shape[0]==features.shape[0]



def get_cooc_per_class_per_vid_mp((idx_vid, class_bin, sort_idx_file, out_file_info,num_train)):

    (out_dir_meta,out_file) = out_file_info
    print idx_vid
    # print class_bin.shape
    # print class_bin[idx_vid]

    sort_idx = np.load(sort_idx_file)
    if sort_idx_file.endswith('.npz'):
        sort_idx =  sort_idx['sort_idx']

    sort_idx_idx = os.path.split(sort_idx_file)[1]
    sort_idx_idx = int(sort_idx_idx.split('_')[0])
    assert sort_idx_idx==idx_vid


    for class_id_curr in range(class_bin.shape[1]):

        class_str = class_names[class_id_curr]
        out_dir_curr = os.path.join(out_dir_meta,class_str)
        util.mkdir(out_dir_curr)
        out_file_mat = os.path.join(out_dir_curr, out_file)
        
        if os.path.exists(out_file_mat):
            continue

        idx_rel = np.where(class_bin[:num_train,class_id_curr])[0]
        

        num_neighbors = len(idx_rel)

        if idx_vid<num_train and class_bin[idx_vid,class_id_curr]:
            assert np.isin(idx_vid,idx_rel)
            assert not np.isin(idx_vid, sort_idx[:,:num_neighbors])
            num_neighbors = num_neighbors-1
        else:
            assert not np.isin(idx_vid,idx_rel)
            assert not np.isin(idx_vid, sort_idx[:,:num_neighbors])

     
        top_neighbors = sort_idx[:,:num_neighbors]
        cooc_curr = np.zeros((sort_idx.shape[0], sort_idx.shape[0]))
        bin_rel_top = np.isin(top_neighbors, idx_rel)

        for row_idx in range(cooc_curr.shape[0]):

            cooc_curr[row_idx,row_idx] = 1.
            row_top = top_neighbors[row_idx,bin_rel_top[row_idx]]
            
            if not np.any(bin_rel_top[row_idx]):
                # print 'no rel row!',bin_rel_top[row_idx],top_neighbors[row_idx]
                continue

            for col_idx in range(row_idx+1, cooc_curr.shape[1]):
                if not np.any(bin_rel_top[col_idx]):
                    # print 'no rel col!',bin_rel_top[col_idx],top_neighbors[col_idx]
                    continue

                # print ' top_neighbors[row_idx]', top_neighbors[row_idx]
                # print ' top_neighbors[col_idx]', top_neighbors[col_idx]

                col_top = top_neighbors[col_idx,bin_rel_top[col_idx]]

                common_top = np.intersect1d( row_top,col_top )

                cooc_val = common_top.size/float(num_neighbors)

                cooc_curr[row_idx, col_idx] = cooc_val
                cooc_curr[col_idx, row_idx] = cooc_val

        # out_file_im = '../scratch/'+'_'.join([str(val) for val in ['check',idx_vid,class_id_curr,'.jpg']])
        # visualize.saveMatAsImage(cooc_curr,out_file_im)
        # raw_input()

        np.savez_compressed(out_file_mat, cooc_curr)

def get_cooc_per_class_per_vid():
    dir_train_test_files = '../data/ucf101/train_test_files'
    dir_gt_vecs = '../data/ucf101/gt_vecs/just_primary_corrected'
    train_file = os.path.join(dir_train_test_files,'train.txt')
    test_file = os.path.join(dir_train_test_files,'test.txt')
    
    out_dir = '../scratch/i3d_dists_just_train'
    lines_train = util.readLinesFromFile(train_file)
    num_train = len(lines_train)
    lines = lines_train+util.readLinesFromFile(test_file)

    out_dir_cooc = '../scratch/i3d_dists_just_train/arr_coocs_per_class'

    vid_idx_arr_file = os.path.join(out_dir, 'vid_idx_arr.npy')
    vid_idx_arr = np.load(vid_idx_arr_file)
    
    vid_idx_arr = np.unique(vid_idx_arr)
    post_load = '_sort_idx_opv.npz'
    
    class_bin = [[int(val) for val in line_curr.split(' ')[1:]] for line_curr in lines]
    class_bin = np.array(class_bin)
    vid_names = [os.path.split(line_curr.split(' ')[0])[1] for line_curr in lines]
    print len(vid_names)
    print class_bin.shape

    num_files = len(vid_names)

    args = []

    for idx_vid_len in range(num_files):
        
        vid_name = vid_names[idx_vid_len]
        sort_idx_file = os.path.join(out_dir,str(idx_vid_len)+post_load)
        vid_name = vid_name[:vid_name.rindex('.')]
        out_file_info= (out_dir_cooc, vid_name+'.npz')
        # if idx_vid_len<num_train:
        #     is_train = True
        # else:
        #     is_train = False
        arg_curr = (idx_vid_len, class_bin, sort_idx_file, out_file_info, num_train)
        args.append(arg_curr)

    # for arg in args:
    #     get_cooc_per_class_per_vid_mp(arg)

    print len(args)
    pool = multiprocessing.Pool()
    pool.map(get_cooc_per_class_per_vid_mp, args)
    pool.close()
    pool.join()


def main():
    for class_name_curr in class_names[1:]:
        visualize_cooc_hists(n = class_name_curr, just_train = True, one_per_vid = 'opvpc')
    # visualize_cooc_hists(just_train = True)
    # visualize.writeHTMLForFolder('../scratch')

    # save_sort_idx_just_train_opv()
    # get_cooc_per_class_per_vid()

    # return
    # print 'heelo'
    # raw_input()
    # save_sort_idx_just_train()
    # return
    # save_sort_idx_just_train_opv()
    # get_cooc_per_vid(n = 10, just_train = True, one_per_vid = True)
    # save_sort_idx_one_per_vid(just_train = True)

    # just_train = True

    # nn_net/scratch/i3d_dists_just_train_htmls/fg/BaseballPitch.html

    # save_hists_per_class(just_train = True)
    # get_coocs_per_class(just_train = True)
    # compress_files()

    # one_per_vid = False
    # just_train = True
    # for n in [10,25,50]:
    #     get_cooc_per_vid(n = n, just_train = just_train, one_per_vid = one_per_vid)
    #     visualize_cooc_hists(n = n, just_train = just_train, one_per_vid = one_per_vid)


    #     visualize.writeHTMLForFolder('../scratch/i3d_dists_just_train/arr_coocs_'+str(n)+'_viz/fg_nall')

    # save_sort_idx_just_train()
    # make_cooc_html(just_train = True)

    # visualize_cooc_hists()
    # get_cooc_per_vid()
    # save_dots()

    # return 
    


    print 'hello'

if __name__=='__main__':
    main()