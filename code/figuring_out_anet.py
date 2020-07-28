import os
import glob
import numpy as np
from helpers import util, visualize
from analysis import evaluate_thumos as et
import globals as globals
import multiprocessing
import scipy.special
# from globals import class_names_activitynet
def softmax(x, axis = 0 ):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis = axis, keepdims = True))
    return e_x / e_x.sum(axis = axis, keepdims = True)

def read_anno_file(anno_file):
    lines = util.readLinesFromFile(anno_file)
    anno_files = []
    labels = []
    for train_file_curr in lines:
        anno = train_file_curr.split(' ')
        label = anno[1:]
        anno_files.append(anno[0])
        labels.append([int(label) for label in label])

    labels = np.array(labels).astype(float)
    # anno_files = np.array(anno_files)
    return anno_files, labels

def get_threshold_val(out_files, bin_out_files, threshold):

    min_max =  np.zeros((len(bin_out_files), 2))

    for class_idx, bin_class in enumerate(bin_out_files[:1]):   
        rel_files = out_files[bin_class]
        for idx_rel_file, rel_file in enumerate(rel_files):
            pred_vals = np.load(rel_file)[:,class_idx]
            min_curr = np.min(pred_vals)
            max_curr = np.max(pred_vals)
            if idx_rel_file==0:
                min_val = min_curr
                max_val = max_curr
            else:
                min_val = min(min_val, min_curr)
                max_cal = max(max_val, max_curr)

        min_max[class_idx,0] = min_val
        min_max[class_idx,1] = max_val

    threshold = min_max[:,0]+ (min_max[:,1]-min_max[:,0])*threshold
    # np.sum(min_max, axis = 1)2.
    return threshold

def get_min_max_mp(rel_file):
    pred_vals = np.load(rel_file)
    min_curr = np.min(pred_vals, axis = 0, keepdims = True)
    max_curr = np.max(pred_vals, axis = 0, keepdims = True)
    return min_curr, max_curr

def get_min_max_all(out_files, threshold, per_class = False):
    pool = multiprocessing.Pool()

    min_max = pool.map(get_min_max_mp, out_files)
    min_vals = np.concatenate([min_max[idx][0] for idx in range(len(min_max))], axis = 0)
    max_vals = np.concatenate([min_max[idx][1] for idx in range(len(min_max))], axis =0)
    # print min_vals.shape, max_vals.shape

    if per_class:
        min_vals = np.min(min_vals, axis = 0)
        max_vals = np.max(max_vals, axis = 0)
    else:
        min_vals = np.min(min_vals)
        max_vals = np.max(max_vals)

    thresh = min_vals+(max_vals - min_vals)*threshold
    
    return thresh

def get_totals((out_file, threshes)):
    
    pred_vals = np.load(out_file)
    pred_vals_sq = np.square(pred_vals)

    if threshes is None:
        threshes = [np.max(pred_vals)]
    
    n_t = len(threshes)
    n_classes = pred_vals.shape[1]

    x = np.zeros((n_t, 2, n_classes))
    x_sq = np.zeros((n_t, 2, n_classes))
    n = np.zeros((n_t, 2, n_classes))

    for idx_t, thresh in enumerate(threshes):
        bin_less = pred_vals<=thresh
        bin_more = np.logical_not(bin_less)

        for idx_bin, bin_curr in enumerate([bin_less, bin_more]):        
            n[idx_t,idx_bin,:] = np.sum(bin_curr, axis = 0)
            x_sq[idx_t,idx_bin,:] = np.sum(pred_vals_sq*bin_curr, axis = 0)
            x[idx_t,idx_bin,:] = np.sum(pred_vals*bin_curr, axis = 0)

    return x_sq, x, n

def get_totals_class_thresh((out_file, threshes_all)):
    
    pred_vals_all = np.load(out_file)
    pred_vals_sq_all = np.square(pred_vals_all)

    n_t = threshes_all.shape[0]
    n_classes = pred_vals_all.shape[1]
    assert threshes_all.shape[1]==n_classes

    x = np.zeros((n_t, 2, n_classes))
    x_sq = np.zeros((n_t, 2, n_classes))
    n = np.zeros((n_t, 2, n_classes))

    for idx_c in range(n_classes):
        
        # print 'threshes_all.shape',threshes_all.shape
        # print 'pred_vals_all.shape',pred_vals_all.shape
        # print 'pred_vals_sq_all.shape',pred_vals_sq_all.shape

        threshes = threshes_all[:,idx_c]
        pred_vals = pred_vals_all[:,idx_c]
        pred_vals_sq = pred_vals_sq_all[:,idx_c]
        
        # print 'threshes.shape',threshes.shape
        # print 'pred_vals.shape',pred_vals.shape
        # print 'pred_vals_sq.shape',pred_vals_sq.shape

        for idx_t, thresh in enumerate(threshes):
            bin_less = pred_vals<=thresh
            bin_more = np.logical_not(bin_less)
            # print 'bin_less.shape, bin_more.shape',bin_less.shape, bin_more.shape    
            # print 'np.sum(bin_less), np.sum(bin_more)',np.sum(bin_less), np.sum(bin_more)    

            for idx_bin, bin_curr in enumerate([bin_less, bin_more]):        
                n[idx_t,idx_bin,idx_c] = np.sum(bin_curr, axis = 0)
                x_sq[idx_t,idx_bin,idx_c] = np.sum(pred_vals_sq*bin_curr, axis = 0)
                x[idx_t,idx_bin,idx_c] = np.sum(pred_vals*bin_curr, axis = 0)

        

    return x_sq, x, n

def get_z_test_all(out_files, threshold):
    
    pool = multiprocessing.Pool()
    min_max = pool.map(get_min_max_mp, out_files)
    pool.close()
    pool.join()

    min_vals = np.concatenate([min_max[idx][0] for idx in range(len(min_max))], axis = 0)
    max_vals = np.concatenate([min_max[idx][1] for idx in range(len(min_max))], axis =0)
    
    min_vals = np.min(min_vals, axis =0)
    max_vals = np.max(max_vals, axis = 0)
    print 'min_vals.shape',min_vals.shape
    print 'max_vals.shape',max_vals.shape
    threshes = []
    for idx in range(min_vals.shape[0]):
        min_val = min_vals[idx]
        max_val = max_vals[idx]
        thresh_curr = np.linspace(min_val, max_val, num=10, endpoint=False)[1:]
        threshes.append(thresh_curr[:,np.newaxis])
    threshes = np.concatenate(threshes, axis = 1)
    
    args = []
    args = [(out_file, threshes) for out_file in out_files]
    pool = multiprocessing.Pool()
    results = pool.map(get_totals_class_thresh,args)
    print 'len(results)',len(results)
    

    # min_val = np.min(min_vals)
    # max_val = np.max(max_vals)
    # threshes = np.linspace(min_val, max_val, num = 10, endpoint = False)[1:]
    # args = []
    # args = [(out_file, threshes) for out_file in out_files]
    # results = pool.map(get_totals,args)
    # print len(results)

    np_res = []
    for idx in range(3):
        arr_curr = np.sum(np.concatenate([res[idx][np.newaxis,:] for res in results], axis = 0) , axis = 0)
        np_res.append(arr_curr)
    
    [x_sq, x, n] = np_res

    n = n.astype(float)
    mean = x/n
    variance = x_sq/n - np.square(mean)
    std = np.sqrt(variance)
    numo = np.abs(mean[:,0,:]-mean[:,1,:])
    deno = variance/n
    deno = np.sqrt(deno[:,0,:]+deno[:,1,:])

    z = numo/deno
    # print z
    thresh_to_ret = []
    for idx_c in range(z.shape[1]):
        thresh_keep = np.sum(n[:,:,idx_c]==0,axis = 1)==0
        thresh_rel = threshes[thresh_keep,idx_c]

        # print threshes
        # print z.shape
        # print z[thresh_keep, idx_c]
        idx_max =  np.argmax(z[thresh_keep, idx_c])
        thresh = thresh_rel[idx_max]
        thresh_to_ret.append(thresh)
        # print idx_c, np.sum(thresh_keep), thresh
        # print thresh_rel
        # print z[:,idx_c]
    return np.array(thresh_to_ret)

def get_bin_counts((out_file, threshes_all)):
    
    pred_vals_all = np.load(out_file)
    pred_vals_sq_all = np.square(pred_vals_all)

    n_t = threshes_all.shape[0] -1
    n_classes = pred_vals_all.shape[1]
    assert threshes_all.shape[1]==n_classes

    x = np.zeros((n_t, n_classes))
    x_sq = np.zeros((n_t, n_classes))
    n = np.zeros((n_t, n_classes))

    for idx_c in range(n_classes):
        
        threshes = threshes_all[:,idx_c]
        pred_vals = pred_vals_all[:,idx_c]
        pred_vals_sq = pred_vals_sq_all[:,idx_c]
    
        for idx_t, thresh in enumerate(threshes[:-1]):
            bin_curr = np.logical_and(pred_vals>=thresh, pred_vals<threshes[idx_t+1])
            # bin_less = pred_vals<=thresh
            # bin_more = np.logical_not(bin_less)
            # print 'bin_less.shape, bin_more.shape',bin_less.shape, bin_more.shape    
            # print 'np.sum(bin_less), np.sum(bin_more)',np.sum(bin_less), np.sum(bin_more)    

            # for idx_bin, bin_curr in enumerate([bin_less, bin_more]):        
            n[idx_t,idx_c] = np.sum(bin_curr, axis = 0)
            x_sq[idx_t,idx_c] = np.sum(pred_vals_sq*bin_curr, axis = 0)
            x[idx_t,idx_c] = np.sum(pred_vals*bin_curr, axis = 0)

        

    return x_sq, x, n

def get_class_hists_all(out_files, n_bins, single_thresh = False):
    
    pool = multiprocessing.Pool()
    min_max = pool.map(get_min_max_mp, out_files)
    pool.close()
    pool.join()

    min_vals = np.concatenate([min_max[idx][0] for idx in range(len(min_max))], axis = 0)
    max_vals = np.concatenate([min_max[idx][1] for idx in range(len(min_max))], axis =0)
    
    min_vals = np.min(min_vals, axis =0)
    max_vals = np.max(max_vals, axis = 0)
    print 'min_vals.shape',min_vals.shape
    print 'max_vals.shape',max_vals.shape

    if single_thresh:
        min_val = np.min(min_vals)
        max_val = np.max(max_vals)
        threshes = np.linspace(min_val, max_val, num = n_bins, endpoint = True)
        threshes = np.tile(threshes, (min_vals.shape[0],1)).T
        # print threshes.shape
        # print threshes[:3,:3]
        # raw_input()
    else:
        threshes = []
        for idx in range(min_vals.shape[0]):
            min_val = min_vals[idx]
            max_val = max_vals[idx]
            thresh_curr = np.linspace(min_val, max_val, num=n_bins, endpoint=True)
            threshes.append(thresh_curr[:,np.newaxis])
        threshes = np.concatenate(threshes, axis = 1)
        # print threshes.shape
        # print threshes[:,:3]
    
    args = []
    args = [(out_file, threshes) for out_file in out_files]
    pool = multiprocessing.Pool()
    # for arg in args:
    #     get_bin_counts(arg)
    results = pool.map(get_bin_counts,args)
    pool.close()
    pool.join()

    print 'len(results)',len(results)


    np_res = []
    for idx in range(3):
        arr_curr = np.sum(np.concatenate([res[idx][np.newaxis,:] for res in results], axis = 0) , axis = 0)
        np_res.append(arr_curr)
    
    [x_sq, x, n] = np_res
    print n.shape, threshes.shape
    return n, threshes
   

def get_z_test(pred_vals):
    # pred_vals = pred_vals-np.min(pred_vals)

    pred_max = np.max(pred_vals)
    pred_min = np.min(pred_vals)
    inc = (pred_max - pred_min)*0.1
    threshes = np.arange(pred_min+inc,pred_max,inc)
    # print threshes
    # print pred_max, pred_min

    diffs = []
    for thresh in threshes:
        a = pred_vals[pred_vals<=thresh]
        b = pred_vals[pred_vals>thresh]
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        a_sig2 = np.square(np.std(a)/np.sqrt(a.size))
        b_sig2 = np.square(np.std(b)/np.sqrt(b.size))
        
        # print thresh
        # print a.size, b.size, a_mean, b_mean, a_sig2, b_sig2 
        

        diffs.append(np.abs(a_mean-b_mean)/np.sqrt(a_sig2+b_sig2))
    diffs = np.array(diffs)
    thresh = threshes[np.argmax(diffs)]
    return thresh
    # print thresh
    # print diffs
    # print vals.shape
    # print threshes.shape
    # raw_input()

def get_otsu_thresh(pred_vals, n_bins):
    hist, bin_edges = np.histogram(pred_vals, n_bins)
    otsu_val = otsu_method(hist)
    idx_max = np.argmax(otsu_val)
    new_thresh = (bin_edges[idx_max]+bin_edges[idx_max+1])/2
    return new_thresh

def visualizing_threshes():
    anno_file = '../data/activitynet/train_test_files/val.txt'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_True_BinaryCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00__cwOld_MulNumClasses_numSim_64/results_model_249_original_class_0_0.5_-2/outf'
    # out_dir = '../scratch/looking_at_anet/viz_pred_gt'


    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_128/results_model_249_original_class_0_0.5_-2/outf'
    # dataset = 'anet'
    
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_static_median_mean_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_128/results_model_249_original_class_0_0.5_-2/outf'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_BinaryCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_128/results_model_249_original_class_-1_-4_-2/outf'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_BinaryCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_3.00_1.00_1.00_changingSparsityAbs_128/results_model_249_original_class_-1_-4_-2/outf'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_True_graph_sum_True_deno_8_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_fs_diff_changingSparsityAbs_128/results_model_149_original_class_-1_-4_-2/outf'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_True_graph_sum_True_deno_8_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00__classicwithsig_128/results_model_99_original_class_0_-0.1_-2/outf'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_True_graph_sum_True_deno_1_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_128/results_model_249_original_class_-1_-0.1_-2/outf'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_self_determination_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_BinaryCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_mindeno8_changingSparsityAbs_128/results_model_249_original_class_-1_-2_-2/outf'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_True_graph_sum_True_deno_random_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_128/results_model_149_original_class_-1_-0.1_-2/outf'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_random_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_BinaryCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_128/results_model_149_original_class_-1_0.5_-2/outf'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_random_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_BinaryCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_weighted_changingSparsityAbs_128/results_model_49_original_class_-1_-2_-2/outf'
    # dataset = 'anet'

    anno_file = '../data/ucf101/train_test_files/test.txt'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_0/results_model_249_original_class_0_0.5_-2/outf'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_False_BinaryCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_0/results_model_249_original_class_0.0_0.5_-2/outf'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_True_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_500_step_500_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_0/results_model_249_original_class_0.0_0.5_-2/outf'
    res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_False_BinaryCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_hardtanh01_changingSparsityAbs_0/results_model_249_original_class_-1_-0.1_-2/outf'
    dataset = 'ucf'
    
    fps_stuff = 16./25.
    threshold = 0.75

    anno_file = '../data/charades/train_test_files/i3d_charades_both_test.txt'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_True_graph_sum_True_deno_8_n_classes_157_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_charades_i3d_charades_both/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_numSim_128/results_model_249_original_class_0_-0.1_-2/outf'
    res_dir = '../experiments/graph_multi_video_with_L1_retF_tanh/graph_multi_video_with_L1_retF_tanh_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_True_graph_sum_True_deno_8_n_classes_157_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_charades_i3d_charades_both/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_numSim_128/results_model_249_original_class_justraw_0_-0.9_-2/outf'
    dataset = 'charades'
    # fps_stuff = 1./6.
    # out_dir = '../scratch/looking_at_'+dataset+'/viz_pred_gt_percent_abs_0.5_thresh_ztest_per_class_all'
    # out_dir = '../scratch/looking_at_'+dataset+'/viz_pred_gt_percent_abs_0.5_thresh_otsu_individual'
    # out_dir = '../scratch/looking_at_'+dataset+'/viz_pred_median_mean_thresh_otsu_individual'
    out_dir = '../scratch/looking_at_'+dataset+'/viz_pred_percent_0.5_mce_tanh'
    # _bce_3_1_1_249'
    util.makedirs(out_dir)
    
    


    # anno_npz = '../data/activitynet/gt_npys/val_pruned.npz'
    
   

    
    

    out_files = glob.glob(os.path.join(res_dir,'*.npy'))
    out_files = np.array(out_files)

    # class_thresholds = get_z_test_all(out_files, threshold)
    

    anno_files, labels = read_anno_file(anno_file)

    anno_jnames = np.array([os.path.split(anno_file)[1] for anno_file in anno_files])
    out_jnames = np.array([os.path.split(out_file)[1] for out_file in out_files])
    
    num_classes = labels.shape[1]

    bin_out_files = []
    for class_idx in range(num_classes):
        rel_bin = labels[:,class_idx]>0
        rel_anno_jnames = anno_jnames[rel_bin]
        rel_bin_out_files = np.in1d(out_jnames, rel_anno_jnames)
        bin_out_files.append(rel_bin_out_files)

    # print len(bin_out_files), bin_out_files[0].shape, np.sum(bin_out_files[0])
    if dataset =='anet':
        gt_vid_names, gt_class_names, gt_time_intervals = et.load_activitynet_gt(False)
        class_names = globals.class_names_activitynet
    elif dataset == 'ucf':
        gt_vid_names, gt_class_names, gt_time_intervals = et.load_ucf_gt(False)
        class_names = globals.class_names
    elif dataset == 'charades':
        # gt_vid_names, gt_class_names, gt_time_intervals = et.load_ucf_gt(False)
        # class_names = globals.class_names
        class_names = globals.class_names_charades
        gt_vid_names, gt_class_names, gt_time_intervals = et.load_charades_gt(False)
        # overlap_thresh_all = np.arange(0.1,0.2,0.1)
        # aps = np.zeros((len(class_names)+1,1))
        # fps_stuff = 16./25.
    
    gt_vid_names = np.array(gt_vid_names)
    gt_class_names = np.array(gt_class_names)
    gt_time_intervals = np.array(gt_time_intervals)


    # get threshold for each class
    # class_thresholds = get_min_max_all(out_files, threshold)
    # print class_thresholds.shape
    # return

    # return
    # class_thresholds = get_threshold_val(out_files, bin_out_files, threshold = threshold)

    n_bins = 10
    for class_idx, bin_class in enumerate(bin_out_files):
        rel_files = out_files[bin_class]
        rel_class_name = class_names[class_idx]
        
        out_dir_curr = os.path.join(out_dir,rel_class_name)
        util.mkdir(out_dir_curr)

        bin_class = gt_class_names == rel_class_name

        for rel_file in rel_files:
            pred_vals = np.load(rel_file)[:,class_idx]

            max_det_conf = np.max(pred_vals)
            min_det_conf = np.min(pred_vals)
            
            old_thresh = min_det_conf+ (max_det_conf - min_det_conf)/2.
            # hist, bin_edges = np.histogram(pred_vals, n_bins)
            # otsu_val = otsu_method(hist)

            # print bin_edges, min_det_conf, max_det_conf
            # print otsu_val, np.argmax(otsu_val)
            # idx_max = np.argmax(otsu_val)
            # print bin_edges[idx_max]
            # new_thresh = (bin_edges[idx_max]+bin_edges[idx_max+1])/2
            # print new_thresh
            # raw_input()
            # # bin_edges = bin_edges[1:]
            
            # print hist.shape, bin_edges.shape, otsu_val.shape
            # new_thresh = bin_edges[np.argmax(otsu_val)]
            new_thresh = get_otsu_thresh(pred_vals, n_bins)




            # new_thresh = get_z_test(pred_vals)

            # new_thresh = class_thresholds[class_idx]


            out_shape_curr = len(pred_vals)

            rel_name = os.path.split(rel_file)[1][:-4]
            bin_vid = gt_vid_names == rel_name
            
            rel_gt_time = gt_time_intervals[np.logical_and(bin_vid,bin_class)]
            # print pred_vals.shape
            # print rel_gt_time
            # print rel_name
            # raw_input()
            det_times = np.array(range(0,out_shape_curr))*fps_stuff
            gt_vals = np.zeros(det_times.shape)

            for gt_time_curr in rel_gt_time:
                idx_start = np.argmin(np.abs(det_times-gt_time_curr[0]))
                idx_end = np.argmin(np.abs(det_times-gt_time_curr[1]))
                gt_vals[idx_start:idx_end] = max_det_conf

            gt_vals[gt_vals==0] = min_det_conf


            out_file_viz = os.path.join(out_dir_curr,rel_name+'.jpg')
            out_file_hist = os.path.join(out_dir_curr,rel_name+'_hist.jpg')

            plot_arr = [ (det_times, pred_vals),(det_times, gt_vals)]
            plot_arr += [ (det_times, np.ones(det_times.shape)*old_thresh),(det_times, np.ones(det_times.shape)*new_thresh)]
            legend_entries = ['Pred','GT']
            legend_entries += ['Old','New ']
            title = 'det conf over time'
            # print out_file_viz


            visualize.hist(pred_vals,out_file_hist,bins=n_bins,normed=True,xlabel='Value',ylabel='Frequency',title=title)
            visualize.plotSimple(plot_arr,out_file = out_file_viz, title = title,xlabel = 'Time',ylabel = 'Detection Confidence',legend_entries=legend_entries)
            # raw_input()
            
        visualize.writeHTMLForFolder(out_dir_curr)
        print out_dir_curr
        # break


def otsu_method(hist_vals):
    total = float(np.sum(hist_vals))
    otsu_val = np.zeros((hist_vals.size-1,))
    for idx_split in range(len(hist_vals)-1):
        d1 = hist_vals[:idx_split+1]
        d2 = hist_vals[idx_split+1:]
        
        w1 = np.sum(d1)/total
        w2 = np.sum(d2)/total
        
        d1_n = d1/float(np.sum(d1))
        d2_n = d2/float(np.sum(d2))

        var1 = np.var(d1_n)
        var2 = np.var(d2_n)
        otsu_val[idx_split] = w1*var1 + w2*var2

    return otsu_val


    

def get_bin_out_files(anno_file, out_files):
    anno_files, labels = read_anno_file(anno_file)
    anno_jnames = np.array([os.path.split(anno_file)[1] for anno_file in anno_files])
    out_jnames = np.array([os.path.split(out_file)[1] for out_file in out_files])
    
    num_classes = labels.shape[1]
    bin_out_files = []
    for class_idx in range(num_classes):
        rel_bin = labels[:,class_idx]>0
        rel_anno_jnames = anno_jnames[rel_bin]
        rel_bin_out_files = np.in1d(out_jnames, rel_anno_jnames)
        bin_out_files.append(rel_bin_out_files)

    return bin_out_files

def save_counts_threshes(out_files, bin_out_files, class_names, out_dir, n_bins = 10, single = False, single_thresh = False):
    if single:
        counts_all, threshes_all = get_class_hists_all(out_files, n_bins, single_thresh = single_thresh)
    else:

        counts_all = []
        threshes_all = []
        for class_idx in range(len(class_names)):
            
            out_files_class = out_files[bin_out_files[class_idx]]
            counts, threshes = get_class_hists_all(out_files_class, n_bins)
            counts_all.append(counts[:,class_idx:class_idx+1])
            threshes_all.append(threshes[:,class_idx:class_idx+1])

        counts_all = np.concatenate(counts_all, axis =1 )
        threshes_all = np.concatenate(threshes_all, axis = 1)

    print counts_all.shape, threshes_all.shape
    np.save(os.path.join(out_dir, 'counts_all.npy'), counts_all)
    np.save(os.path.join(out_dir, 'threshes_all.npy'), threshes_all)

def viz_hists(out_dir, class_names, counts, threshes):
    widths = threshes[1:,:] - threshes[:-1,:]
    for class_idx in range(len(class_names)):
        out_file = os.path.join(out_dir, class_names[class_idx].replace(' ','_')+'.jpg')
        y_vals = counts[:,class_idx]
        x_vals = threshes[:-1,class_idx]

        w = widths[:,class_idx]
        visualize.plotBars(out_file,x_vals,w,y_vals,'r',xlabel='Val',ylabel='Count',title=class_names[class_idx])
        print out_file

    out_file = os.path.join(out_dir, 'total.jpg')
    y_vals = np.sum(counts, axis = 1)
    x_vals = threshes[:-1,0]
    w = widths[:,0]
    print y_vals.shape, x_vals.shape, widths
    visualize.plotBars(out_file,x_vals,w,y_vals,'r',xlabel='Val',ylabel='Count',title='Total')
    print out_file

    visualize.writeHTMLForFolder(out_dir)
    
    
def visualizing_pmf():
    anno_file = '../data/activitynet/train_test_files/val.txt'
    
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_static_median_mean_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_128/results_model_249_original_class_0_0.5_-2/outf'

    res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_128/results_model_249_original_class_0_0.5_-2/outf'    
    dataset = 'anet'

    # anno_file = '../data/ucf101/train_test_files/test.txt'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_0/results_model_249_original_class_0_0.5_-2/outf'
    # dataset = 'ucf'

    res_dir_pmf = os.path.join(os.path.split(res_dir)[0],'outpmf')
    out_dir = '../scratch/looking_at_'+dataset+'/viz_pmf_percent_0.5/pmf_dist_pred_true'
    util.makedirs(out_dir)
    
    fps_stuff = 16./25.
    threshold = 0.5
    pmf_thresh = 0

    out_files = glob.glob(os.path.join(res_dir,'*.npy'))
    out_files = np.array(out_files)

    # class_thresholds = get_z_test_all(out_files, threshold)
    

    anno_files, labels = read_anno_file(anno_file)

    anno_jnames = np.array([os.path.split(anno_file)[1] for anno_file in anno_files])
    out_jnames = np.array([os.path.split(out_file)[1] for out_file in out_files])
    
    num_classes = labels.shape[1]
    bin_out_files = []
    for class_idx in range(num_classes):
        rel_bin = labels[:,class_idx]>0
        rel_anno_jnames = anno_jnames[rel_bin]
        rel_bin_out_files = np.in1d(out_jnames, rel_anno_jnames)
        bin_out_files.append(rel_bin_out_files)

    # print len(bin_out_files), bin_out_files[0].shape, np.sum(bin_out_files[0])
    if dataset =='anet':
        gt_vid_names, gt_class_names, gt_time_intervals = et.load_activitynet_gt(False)
        class_names = globals.class_names_activitynet
    elif dataset == 'ucf':
        gt_vid_names, gt_class_names, gt_time_intervals = et.load_ucf_gt(False)
        class_names = globals.class_names
    
    gt_vid_names = np.array(gt_vid_names)
    gt_class_names = np.array(gt_class_names)
    gt_time_intervals = np.array(gt_time_intervals)

    pmf_files = [os.path.join(res_dir_pmf,os.path.split(file_curr)[1]) for file_curr in out_files]
    pmf_all = [np.load(file_curr)[np.newaxis,:] for file_curr in pmf_files]
    pmf_all = np.concatenate(pmf_all,axis = 0)

    pmf_smax = softmax(pmf_all,axis = 1)

    for class_idx in range(num_classes):
        pmf_plot = []
        post_pend = []
        bin_curr = pmf_all[:,class_idx]>0
        # for idx in range(10):
        #     print pmf_all[bin_curr,:][idx]
        #     print pmf_smax[bin_curr,:][idx]
        #     print '___'
        #     raw_input()

        pmf_plot.append( pmf_smax[bin_curr,class_idx])
        post_pend.append('_smax_true')
        pmf_plot.append( pmf_smax[np.logical_not(bin_curr),class_idx])
        post_pend.append('_smax_false')
        pmf_plot.append( pmf_all[bin_curr,class_idx])
        post_pend.append('_true')
        pmf_plot.append( pmf_all[np.logical_not(bin_curr),class_idx])
        post_pend.append('_false')

        for pmf_curr, post_pend_curr in zip(pmf_plot, post_pend):

            out_file_curr = os.path.join(out_dir, class_names[class_idx]+post_pend_curr+'.jpg')
            title = class_names[class_idx]+post_pend_curr.replace('_',' ').title()
            visualize.hist(pmf_curr,out_file_curr,bins=10,normed=True,xlabel='Value',ylabel='Frequency',title=title)

            print out_file_curr

    visualize.writeHTMLForFolder(out_dir)


    return
    
    bin_pred_files = []
    for class_idx in range(num_classes):
        bin_pred = pmf_all[:,class_idx]>pmf_thresh
        bin_pred_files.append(bin_pred)

    new_threshes = []
    print '[',
    for class_idx in range(num_classes):
        out_files_curr = out_files[bin_pred_files[class_idx]]
        # print out_files_curr.shape
        class_thresholds = get_min_max_all(out_files_curr, threshold, per_class = True)
        # print class_thresholds
        new_thresh = class_thresholds[class_idx]
        print new_thresh,',',
        new_threshes.append(new_thresh)
        # raw_input()
    print ']'
    return

    counts_file = os.path.join(out_dir, 'counts_all.npy')
    threshes_file = os.path.join(out_dir, 'threshes_all.npy')

    if not os.path.exists(counts_file):
        save_counts_threshes(out_files, bin_pred_files, class_names, out_dir, n_bins = 11, single = False, single_thresh = False)

    counts = np.load(counts_file)
    print counts.shape
    threshes = np.load(threshes_file)
    print threshes.shape

    # viz_hists(out_dir, class_names, counts, threshes)

    print counts[:3,:3]
    print threshes[:3,:3]

    new_threshes = []
    print '[',
    for class_idx in range(counts.shape[1]):
        hist = counts[:,class_idx]
        bin_edges = threshes[:,class_idx]
        hist = hist[bin_edges[:-1]<=0]
        otsu_val = otsu_method(hist)
        idx_max = np.argmax(otsu_val)
        new_thresh = (bin_edges[idx_max]+bin_edges[idx_max+1])/2
        print new_thresh,',',
        new_threshes.append(new_thresh)
        # raw_input()
    print ']'




    # get threshold for each class
    # class_thresholds = get_min_max_all(out_files, threshold)
    # print class_thresholds.shape
    # return

    # return
    # class_thresholds = get_threshold_val(out_files, bin_out_files, threshold = threshold)
    # n_bins = 10


    # for class_idx, bin_class in enumerate(bin_out_files):
    #     rel_files = out_files[bin_class]
    #     for rel_file in rel_files:
    #         pmf_file = os.path.join(res_dir_pmf,os.path.split(rel_file)[1])
    #         print pmf_file, os.path.exists(pmf_file)
    #         pmf = np.load(pmf_file)
    #         print pmf
    #         print class_idx, pmf[class_idx]

    #         # print 
    #         raw_input()


    # n_bins = 10
    # for class_idx, bin_class in enumerate(bin_out_files):
    #     rel_files = out_files[bin_class]
    #     rel_class_name = class_names[class_idx]
        
    #     out_dir_curr = os.path.join(out_dir,rel_class_name)
    #     util.mkdir(out_dir_curr)

    #     bin_class = gt_class_names == rel_class_name

    #     for rel_file in rel_files:
    #         pred_vals = np.load(rel_file)[:,class_idx]
    #         max_det_conf = np.max(pred_vals)
    #         min_det_conf = np.min(pred_vals)
            
    #         old_thresh = min_det_conf+ (max_det_conf - min_det_conf)/2.
    #         # hist, bin_edges = np.histogram(pred_vals, n_bins)
    #         # otsu_val = otsu_method(hist)

    #         # print bin_edges, min_det_conf, max_det_conf
    #         # print otsu_val, np.argmax(otsu_val)
    #         # idx_max = np.argmax(otsu_val)
    #         # print bin_edges[idx_max]
    #         # new_thresh = (bin_edges[idx_max]+bin_edges[idx_max+1])/2
    #         # print new_thresh
    #         # raw_input()
    #         # # bin_edges = bin_edges[1:]
            
    #         # print hist.shape, bin_edges.shape, otsu_val.shape
    #         # new_thresh = bin_edges[np.argmax(otsu_val)]
    #         new_thresh = get_otsu_thresh(pred_vals, n_bins)




    #         # new_thresh = get_z_test(pred_vals)

    #         # new_thresh = class_thresholds[class_idx]


    #         out_shape_curr = len(pred_vals)

    #         rel_name = os.path.split(rel_file)[1][:-4]
    #         bin_vid = gt_vid_names == rel_name
            
    #         rel_gt_time = gt_time_intervals[np.logical_and(bin_vid,bin_class)]

    #         det_times = np.array(range(0,out_shape_curr))*fps_stuff
    #         gt_vals = np.zeros(det_times.shape)

    #         for gt_time_curr in rel_gt_time:
    #             idx_start = np.argmin(np.abs(det_times-gt_time_curr[0]))
    #             idx_end = np.argmin(np.abs(det_times-gt_time_curr[1]))
    #             gt_vals[idx_start:idx_end] = max_det_conf

    #         gt_vals[gt_vals==0] = min_det_conf


    #         out_file_viz = os.path.join(out_dir_curr,rel_name+'.jpg')
    #         out_file_hist = os.path.join(out_dir_curr,rel_name+'_hist.jpg')

    #         plot_arr = [ (det_times, pred_vals),(det_times, gt_vals)]
    #         plot_arr += [ (det_times, np.ones(det_times.shape)*old_thresh),(det_times, np.ones(det_times.shape)*new_thresh)]
    #         legend_entries = ['Pred','GT']
    #         legend_entries += ['Old','New ']
    #         title = 'det conf over time'
    #         # print out_file_viz


    #         visualize.hist(pred_vals,out_file_hist,bins=n_bins,normed=True,xlabel='Value',ylabel='Frequency',title=title)
    #         visualize.plotSimple(plot_arr,out_file = out_file_viz, title = title,xlabel = 'Time',ylabel = 'Detection Confidence',legend_entries=legend_entries)
    #         # raw_input()
            
    #     visualize.writeHTMLForFolder(out_dir_curr)
    #     print out_dir_curr
    #     # break



def main():
    # visualizing_pmf()
    visualizing_threshes()
    return

    anno_file = '../data/activitynet/train_test_files/val.txt'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_True_BinaryCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00__cwOld_MulNumClasses_numSim_64/results_model_249_original_class_0_0.5_-2/outf'
    # out_dir = '../scratch/looking_at_anet/viz_pred_gt'


    res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_100_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_activitynet/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_128/results_model_249_original_class_0_0.5_-2/outf'
    dataset = 'anet'
    class_names = globals.class_names_activitynet

    # anno_file = '../data/ucf101/train_test_files/test.txt'
    # res_dir = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_percent_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00_changingSparsityAbs_0/results_model_249_original_class_0_0.5_-2/outf'
    # dataset = 'ucf'
    # class_names = globals.class_names

    # out_dir = '../scratch/looking_at_'+dataset+'/viz_pred_gt_percent_abs_0.5_thresh_ztest_per_class_all'
    out_dir = '../scratch/looking_at_'+dataset+'/threshold_hists_multi_multi_thresh'
    util.makedirs(out_dir)
    
    # counts_all = np.load(os.path.join(out_dir,'counts_all.npy'))
    # for class_idx in range(counts_all.shape[1]):
    #     hist_curr = counts_all[:,class_idx]
    #     otsu_val = otsu_method(hist_curr)
    #     hist_val = hist_curr/float(np.sum(hist_curr))
    #     x = np.array(list(range(hist_val.size)))
    #     xAndYs = [(x, hist_val), (x[:-1], otsu_val)]
    #     out_file = os.path.join(out_dir, class_names[class_idx]+'_otsu.jpg')
    #     visualize.plotSimple(xAndYs,out_file,title='Otsu'+class_names[class_idx],xlabel='num',ylabel='freq',legend_entries=['hist','otsu'])

    # visualize.writeHTMLForFolder(out_dir)




    # return    


    # anno_npz = '../data/activitynet/gt_npys/val_pruned.npz'
    
    fps_stuff = 16./25.
    threshold = 0.75

    
    
    counts_file = os.path.join(out_dir, 'counts_all.npy')
    threshes_file = os.path.join(out_dir, 'threshes_all.npy')
    out_files = glob.glob(os.path.join(res_dir,'*.npy'))
    out_files = np.array(out_files)

    threshes = get_min_max_all(out_files, 0.5, per_class = True)
    print '[',
    for thresh in threshes:
        print thresh,',',
    print ']'
    # print threshes

    return

    bin_out_files = get_bin_out_files(anno_file, out_files)
    if not os.path.exists(counts_file):
        save_counts_threshes(out_files, bin_out_files, class_names, out_dir, n_bins = 11, single = False, single_thresh = False)

    counts = np.load(counts_file)
    print counts.shape
    threshes = np.load(threshes_file)
    print threshes.shape
    print counts[:3,:3]
    print threshes[:3,:3]

    new_threshes = []
    print '[',
    for class_idx in range(counts.shape[1]):
        hist = counts[:,class_idx]
        bin_edges = threshes[:,class_idx]
        otsu_val = otsu_method(hist)
        idx_max = np.argmax(otsu_val)
        new_thresh = (bin_edges[idx_max]+bin_edges[idx_max+1])/2
        print new_thresh,',',
        new_threshes.append(new_thresh)
    print ']'

    print new_threshes

    # return
    # viz_hists(out_dir, class_names, counts, threshes)

    # hist_vals = np.sum(counts, axis = 1)
    # otsu_vals = otsu_method(hist_vals)
    # idx_max = np.argmax(otsu_vals)

    # print otsu_vals
    # print hist_vals.shape, threshes.shape, otsu_vals.shape
    # for idx, otsu_val in enumerate(otsu_vals):
    #     print otsu_val, threshes[idx,0], threshes[idx+1,0]
    # # print otsu_vals
    # # print 
    # print idx_max, (threshes[idx_max,0]+threshes[idx_max+1,0])/2.



    # visualize.writeHTMLForFolder(out_dir)
    # class_thresholds = get_z_test_all(out_files, threshold)
    # counts, threshes = get_class_hists_all(out_files, 10)
    # widths = threshes[1:,:] - threshes[:-1,:]


    

    



    return

    

    # print len(bin_out_files), bin_out_files[0].shape, np.sum(bin_out_files[0])
    if dataset =='anet':
        gt_vid_names, gt_class_names, gt_time_intervals = et.load_activitynet_gt(False)
        class_names = globals.class_names_activitynet
    elif dataset == 'ucf':
        gt_vid_names, gt_class_names, gt_time_intervals = et.load_ucf_gt(False)
        class_names = globals.class_names
    
    gt_vid_names = np.array(gt_vid_names)
    gt_class_names = np.array(gt_class_names)
    gt_time_intervals = np.array(gt_time_intervals)



    


       


if __name__=='__main__':
    main()  
