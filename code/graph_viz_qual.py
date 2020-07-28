import numpy as np
import os
import glob
from helpers import visualize,util
from globals import class_names
import cooc_exp

def get_start_end(bin_keep):
    bin_keep = bin_keep.astype(int)
    bin_keep_rot = np.roll(bin_keep, 1)
    bin_keep_rot[0] = 0
    diff = bin_keep - bin_keep_rot
    idx_start_all = list(np.where(diff==1)[0])
    idx_end_all = list(np.where(diff==-1)[0])
    if len(idx_start_all)>len(idx_end_all):
        assert len(idx_start_all)-1==len(idx_end_all)
        idx_end_all.append(bin_keep.shape[0])
    
    assert len(idx_start_all)==len(idx_end_all)
    return idx_start_all, idx_end_all

def visualize_edge_hists(out_dir_coocs,out_dir_hists):
    # n = 10, just_train = False, one_per_vid = 'reg'):

    dir_train_test_files = '../data/ucf101/train_test_files'
    dir_gt_vecs = '../data/ucf101/gt_vecs/just_primary_corrected'
    test_file = os.path.join(dir_train_test_files,'test.txt')

    lines = util.readLinesFromFile(test_file)


    # if just_train:
    #     out_dir = '../scratch/i3d_dists_just_train'
    #     lines = util.readLinesFromFile(train_file)
    #     # +util.readLinesFromFile(test_file)
    # else:
    #     out_dir = '../scratch/i3d_dists'
    #     lines = util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)

    # out_dir = '../scratch/i3d_dists'
    # if one_per_vid =='opv':
    #     out_dir_pre = 'arr_coocs_opv_'
    # elif one_per_vid == 'opvpc':
    #     out_dir_pre = 'arr_coocs_per_class/'
    # else:
    #     out_dir_pre = 'arr_coocs_'


    # # n = 10
    
    # out_dir_hists = os.path.join(out_dir,out_dir_pre+str(n)+'_viz')

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
    vid_names_per_class, class_id = cooc_exp.get_vid_names_per_class(lines)

    gt_files = [os.path.join(dir_gt_vecs, os.path.split(line_curr)[1]) for line_curr in npy_files]
    
    num_files = len(npy_files)
    all_vals_all = []


    legend_entries = ['FG All','FG FG','BG All','BG BG','FG BG']
    xlabel = 'Edge Weight'
    ylabel = 'Frequency'
    num_bins = np.arange(0.5,1.1,.1)
    xtick_labels = ['%.1f'%val for val in num_bins[:-1]]+['']
    plot_idx_all = [None, [0,1],[2,3],[1,4],[3,4]]
    title_pres = ['Mat ','Foreground Hist for ', 'Background Hist for ','Foreground Hist for ', 'Background Hist for ']

    for idx_vid, gt_file in enumerate(gt_files):
        vid_name = os.path.split(gt_file)[1]
        gt_arr = np.load(gt_file)
        # print os.path.join(out_dir_coocs,'*',vid_name)
        arr_cooc_file = glob.glob(os.path.join(out_dir_coocs,'*',vid_name))
        # print arr_cooc_file
        assert len(arr_cooc_file)==1
        arr_cooc_file = arr_cooc_file[0]
        # arr_cooc_file = os.path.join(out_dir_coocs, vid_name)
            # .replace('.npy','.npz'))
        arr_cooc = np.load(arr_cooc_file)
        # ['arr_0']
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

    # print len(class_id)
    for class_idx, class_id_curr in enumerate(class_id.T):
        # print class_id_curr
        vals_rel = [all_vals_all[idx] for idx in np.where(class_id_curr)[0]]
        # print len(vals_rel)
        # 
        vals_rel_new = [[] for idx in range(len(vals_rel[0]))]
        # print vals_rel_new
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
                # print len(vals_rel), len(vals_rel[0]), type(vals_rel[0])
                vals_curr = [vals_rel[idx_curr]*10 for idx_curr in plot_idx_curr]
                legend_entries_curr = [legend_entries[idx_curr] for idx_curr in plot_idx_curr]
                bins_all = [num_bins*10 for idx_curr in plot_idx_curr]
                # print bins_all
                # print num_bins

                visualize.plotMultiHist(out_file_curr ,vals = vals_curr, num_bins = bins_all, legend_entries = legend_entries_curr, title = title, xlabel = xlabel, ylabel = ylabel, xticks = xtick_labels, density = True, align = 'mid')
            print out_file_curr
            # raw_input()
            # if idx_out_file == (len(out_dirs_all)-1):
            visualize.writeHTMLForFolder(out_dir_curr)

def def_visualize_edge_hists():
    dir_sparsity = '../experiments/graph_multi_video_with_L1/graph_multi_video_with_L1_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_0.00__ablation/results_model_249_0.0_0.5/viz_sim_mat'
    
    dir_l1 = '../experiments/graph_multi_video_with_L1/graph_multi_video_with_L1_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00__ablation/results_model_249_0.0_0.5/viz_sim_mat'

    dir_mcasl = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_500_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_0.50__noLimit/results_model_249_0_0.5/viz_sim_mat'

    dir_gt = '../data/ucf101/gt_vecs/just_primary_corrected/'
    anno_file = '../data/ucf101/train_test_files/test_just_primary_corrected.txt'

    dir_curr = dir_sparsity
    out_dir_hists = '../scratch/sparse_edges_hists_0.5'
    visualize_edge_hists(dir_curr,out_dir_hists)

    dir_curr = dir_l1
    out_dir_hists = '../scratch/l1_edges_hists_0.5'
    visualize_edge_hists(dir_curr,out_dir_hists)

    dir_curr = dir_mcasl
    out_dir_hists = '../scratch/mcasl_edges_hists_0.5'
    visualize_edge_hists(dir_curr,out_dir_hists)


def make_html_for_folder():

    dir_sparsity = '../scratch/sparse_edges_hists_0.5'
    # '../experiments/graph_multi_video_with_L1/graph_multi_video_with_L1_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_0.00__ablation/results_model_249_0.0_0.5/viz_sim_mat'
    
    dir_l1 = '../scratch/l1_edges_hists_0.5'
    # '../experiments/graph_multi_video_with_L1/graph_multi_video_with_L1_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00__ablation/results_model_249_0.0_0.5/viz_sim_mat'

    rel_path_replace = ['..','/data/maheen/nn_net']
    folder_inner = ['fg_nall','bg_nall']
    folders = [os.path.join(dir_curr,folder_curr).replace(rel_path_replace[0],rel_path_replace[1]) for dir_curr in [dir_sparsity,dir_l1] for folder_curr in folder_inner]
    captions = [' '.join([dir_curr,folder_curr]) for dir_curr in ['sparse','l1'] for folder_curr in folder_inner]

    img_names = [class_name_curr+'.jpg' for class_name_curr in class_names]
    out_file_html = '../scratch/bg_fg_comparsion.html'
    visualize.writeHTMLForDifferentFolders(out_file_html,folders,captions,img_names,rel_path_replace='/data/maheen',height=200,width=200)


def getting_percent_stats():



def main():
    def_visualize_edge_hists()
    
    return
    dir_mcasl = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_500_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_0.50__noLimit/results_model_249_0_0.5/viz_sim_mat'

    dir_gt = '../data/ucf101/gt_vecs/just_primary_corrected/'
    # vid_name = 'CricketBowling/video_test_0000273'

    vid_name = 'CleanAndJerk/video_test_0000635'
    graph = np.load(os.path.join(dir_mcasl,vid_name+'.npy'))
    # graph[graph<0.5]=0
    gt = np.load(os.path.join(dir_gt,os.path.split(vid_name)[1]+'.npy'))
    start, end = get_start_end(gt>0)


    # print graph[20:30,0:30]
    # raw_input()

    print start
    print end

    for idx_start,start_curr in enumerate(start):
        end_curr = end[idx_start]
        mid_curr = end_curr-1
        # (end_curr +start_curr)//2
        # start_curr+1
        # 

        # print graph[190:200,:20]
        # raw_input()

        row = graph[mid_curr,:]
        idx_sort = np.argsort(row)[:15]
        # [::-1][:15]
        # [:15]
        # [::-1][:15]
        # 

        print start_curr, end_curr

        print idx_sort
        print row[idx_sort]
        frame_num = (idx_sort+1)*160//25
        print frame_num
        
        raw_input()
        # rel_rows = graph[idx_start:idx_end,:]



    # type_hist = 'fg_fg'
    # type_hist = 'fg_bg'
    # type_hist = 'bg_bg'
    # # type_hist = 'fg_fg'

    # dir_curr = dir_sparsity
    # get_edge_hists(dir_curr,dir_gt,anno_file,type_hist)




if __name__=='__main__':
    main()



