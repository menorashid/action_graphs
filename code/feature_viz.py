import os
import numpy as np
import scipy.io
import glob
from helpers import util, visualize
import sklearn.metrics
from globals import class_names
import torch
import exp_mill_bl as emb
from debugging_graph import get_gt_vector, readTrainTestFile, softmax
from analysis import evaluate_thumos as et
from train_test_mill import merge_detections


def plot_classes(feat_dir, gt_vec_dir, out_dir, class_idx_keep):
    # g_str = 'graph_2_nononlin_b'
    # lin_str = 'lin_2_nononlin'
    # graph_feat_dir = os.path.join('../scratch',g_str)
    # lin_feat_dir = os.path.join('../scratch',lin_str)
    # out_dir = '../scratch/comparing_features'
    n_classes = len(class_names)

    if class_idx_keep is None:
        class_idx_keep = range(n_classes)
    
    util.mkdir(out_dir)

    vid_names = glob.glob(os.path.join(feat_dir,'*.npy'))
    vid_names = [os.path.split(file_curr)[1][:-4] for file_curr in vid_names]

    graph_features = [[] for i in range(n_classes)]
    
    for vid_name in vid_names:
        gt_file = os.path.join(gt_vec_dir, vid_name+'.npz')
        npz_data = np.load(gt_file)
        gt_vecs = npz_data['gt_vecs']
        gt_classes = npz_data['gt_classes']
        
        graph_data = np.load(os.path.join(feat_dir, vid_name+'.npy'))
        
        for idx_gt, gt_vec in enumerate(gt_vecs):
            class_curr = int(gt_classes[idx_gt])
            if class_curr not in class_idx_keep:
                continue
            bin_curr = gt_vec>0
            graph_data_rel = graph_data[bin_curr,:]
            
            graph_features[class_curr].append(graph_data_rel)


    class_names_curr = np.array(class_names)[class_idx_keep]
    graph_features = [graph_features[idx] for idx in class_idx_keep]

    class_names_keep = [class_names_curr[idx] for idx,f in enumerate(graph_features) if len(f)>0]
    features =  [np.concatenate(f,axis = 0) for f in graph_features if len(f)>0]

    xAndYs = [(f[:,0],f[:,1]) for f in features]

    title = 'Features'
    out_file = '_'.join(class_names_keep)+'.jpg'
    out_file = os.path.join(out_dir,out_file)

    xlabel = 'x'
    ylabel = 'y'
    legend_entries = class_names_keep
    visualize.plotSimple(xAndYs,out_file=out_file,title=title,xlabel=xlabel,ylabel=ylabel,legend_entries=legend_entries,outside=True, noline= True)
    print out_file



def main():
        

    gt_vec_dir = '../experiments/graph_multi_video_same_F_ens_dll_moredepth/graph_multi_video_same_F_ens_dll_moredepth_aft_nonlin_HT_L2_non_lin_HT_num_graphs_1_sparsify_0.5_graph_size_2_sigmoid_True_deno_0.5_n_classes_20_in_out_2048_256_feat_dim_2048_512_method_cos_zero_self_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropy_300_step_300_0.1_0.001_0.001_0.001_ABS_bias_retry/model_299_graph_etc_0'

    dir_results = '../experiments/graph_multi_video_same_i3dF/graph_multi_video_same_i3dF_aft_nonlin_HT_l2_sparsify_0.5_non_lin_None_method_cos_zero_self_deno_8_n_classes_20_in_out_2048_2_graph_size_2_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropy_100_step_100_0.1_0.001_0.01_ABS_bias_wb/results_model_99_0_0.5'
    
    dir_results = '../experiments/graph_multi_video_same_i3dF/graph_multi_video_same_i3dF_aft_nonlin_HT_l2_non_lin_None_sparsify_False_graph_size_2_method_cos_zero_self_deno_8_n_classes_20_in_out_2048_2_graph_sum_True_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_100_step_100_0.1_0.001_0.001_lw_1.00_0.50_ABS_bias_wb/results_model_99_0_0.5'
    
    feat_dir = os.path.join(dir_results, 'outf')
    out_dir = os.path.join(dir_results, 'viz_outf')
    util.mkdir(out_dir)
    
    class_idx_keep = range(10,16)

    plot_classes(feat_dir, gt_vec_dir, out_dir, class_idx_keep)


    return

    g_str = 'graph_2_nononlin_b'
    lin_str = 'lin_2_nononlin'
    graph_feat_dir = os.path.join('../scratch',g_str)
    lin_feat_dir = os.path.join('../scratch',lin_str)
    out_dir = '../scratch/comparing_features'
    util.mkdir(out_dir)

    vid_names = glob.glob(os.path.join(graph_feat_dir,'*.npy'))
    vid_names = [os.path.split(file_curr)[1][:-4] for file_curr in vid_names]

    graph_features = [[] for i in range(20)]
    lin_features = [[] for i in range(20)]

    for vid_name in vid_names:
        gt_file = os.path.join(gt_vec_dir, vid_name+'.npz')
        npz_data = np.load(gt_file)
        gt_vecs = npz_data['gt_vecs']
        gt_classes = npz_data['gt_classes']
        
        graph_data = np.load(os.path.join(graph_feat_dir, vid_name+'.npy'))
        lin_data = np.load(os.path.join(lin_feat_dir, vid_name+'.npy'))
        
        if gt_classes.size>1:
            continue

        bin_curr = gt_vecs[0]>0
        graph_data_rel = graph_data[bin_curr,:]
        lin_data_rel = lin_data[bin_curr,:]
        print graph_data_rel.shape
        print lin_data_rel.shape
        class_curr = int(gt_classes[0])
        
        lin_features[class_curr].append(lin_data_rel)
        graph_features[class_curr].append(graph_data_rel)

    # lin_features = [np.concatenate(lin_feat,axis = 0) for lin_feat in lin_features]
    # graph_features = [np.concatenate(graph_feat,axis = 0) for graph_feat in graph_features]

    titles = [lin_str,g_str]
    class_names_curr = class_names[10:16]
    lin_features = lin_features[10:16]
    graph_features = graph_features[10:16]

    for idx_features,features in enumerate([lin_features,graph_features]):
        class_names_keep = [class_names_curr[idx] for idx,f in enumerate(features) if len(f)>0]
        features =  [np.concatenate(f,axis = 0) for f in features if len(f)>0]

        xAndYs = [(f[:,0],f[:,1]) for f in features]

        title = titles[idx_features]
        out_file = os.path.join(out_dir,title+'.jpg')
        xlabel = 'x'
        ylabel = 'y'
        legend_entries = class_names_keep
        visualize.plotSimple(xAndYs,out_file=out_file,title=title,xlabel=xlabel,ylabel=ylabel,legend_entries=legend_entries,outside=True, noline= True)
        print out_file



    # for class_curr in range(20):
    #     graph_feat = graph_features[class_curr]
    #     lin_feat = lin_features[class_curr]
    #     print len(graph_feat), len(lin_feat)
    #     graph_feat = np.concatenate(graph_feat, axis = 0)
    #     lin_feat = np.concatenate(lin_feat, axis = 0)
    #     out_file_curr = os.path.join(out_dir_viz, 'lin_'+class_names[class_curr]+'.jpg')

    




if __name__=='__main__':
    main()