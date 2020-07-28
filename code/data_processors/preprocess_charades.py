import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import scipy.misc
import numpy as np
import random
import scipy.io
import csv
import multiprocessing
import globals
import time

def load_annos():
    data_dir = '../data/charades'
    file_curr = os.path.join(data_dir,'Charades','Charades_v1_train.csv')

    with open(file_curr) as f:
        reader = csv.DictReader(f)
        for row in reader:
            print row.keys()
            print row['id']
            actions = row['actions'].split(';')
            print actions
            raw_input()


def save_all_anno():
    meta_dir = '../data/charades'
    anno_dir = os.path.join(meta_dir,'Charades')

    out_file = os.path.join(meta_dir,'annos_test_wmiss.npz')
    wmiss = True
    anno_file = os.path.join(anno_dir,'Charades_v1_test.csv')

    # anno_files = glob.glob(os.path.join(anno_dir,'*.txt'))
    # anno_files.sort()

    gt_class_names = []
    gt_time_intervals = []
    gt_vid_names = []
    problem = 0
    with open(anno_file) as f:
        reader = csv.DictReader(f)
        for idx_row, row in enumerate(reader):
            if idx_row%100==0:
                print idx_row, 1863

            vid_id = row['id']
            # print row['id']
            # print row.keys()
            # print row['length']

            # raw_input()
            actions = row['actions'].split(';')
            if actions[0]=='' and not wmiss:
                continue
            elif actions[0]=='' and wmiss:
                problem+=1
                gt_class_name = globals.class_names_charades[0]
                start_time = 0
                end_time = row['length']
                gt_class_names.append(gt_class_name)
                gt_time_intervals.append([float(start_time), float(end_time)])
                gt_vid_names.append(vid_id)
            else:
                for action_curr in actions:
                    gt_class_name, start_time, end_time = action_curr.split(' ')
                    gt_class_names.append(gt_class_name)
                    gt_time_intervals.append([float(start_time), float(end_time)])
                    gt_vid_names.append(vid_id)

    gt_vid_names = np.array(gt_vid_names)
    gt_time_intervals = np.array(gt_time_intervals)
    gt_class_names = np.array(gt_class_names)

    print gt_time_intervals.shape
    print gt_class_names.shape
    print gt_vid_names.shape
    print problem

    # np.savez(out_file, gt_time_intervals = gt_time_intervals ,gt_class_names = gt_class_names ,gt_vid_names = gt_vid_names)

def save_features_mp((in_dir_rgb, out_file, idx_vid)):
    print idx_vid

    files_rgb = glob.glob(os.path.join(in_dir_rgb,'*.txt'))
    files_rgb.sort()

    feat_curr = [np.loadtxt(file_curr) for file_curr in files_rgb]
    feat_curr = np.array(feat_curr)

    np.savez_compressed(out_file,feat_curr)


def save_features():
    meta_dir = '../data/charades'
    
    # rgb_dir = os.path.join(meta_dir, 'Charades_v1_features_rgb')
    # out_dir = os.path.join(meta_dir,'vgg16_rgb_features_npz')
    # util.mkdir(out_dir)
    
    rgb_dir = os.path.join(meta_dir, 'Charades_v1_features_flow')
    out_dir = os.path.join(meta_dir,'vgg16_flow_features_npz')
    util.mkdir(out_dir)
    
    vid_ids = [os.path.split(dir_curr)[1] for dir_curr in glob.glob(os.path.join(rgb_dir,'*')) if os.path.isdir(dir_curr)]
    print len(vid_ids)

    args = []
    for idx_vid, vid_id in enumerate(vid_ids):
        in_dir_rgb = os.path.join(rgb_dir, vid_id)
        # in_dir_flow = os.path.join(flow_dir, vid_id)
        out_file = os.path.join(out_dir, vid_id+'.npz')
        if os.path.exists(out_file):
            continue

        args.append((in_dir_rgb,  out_file, idx_vid))

    print len(args)
    # for arg in args:
    #     save_features_mp(arg)
    #     break

    pool = multiprocessing.Pool()
    pool.map(save_features_mp, args)

def save_features_i3d_charades():
    meta_dir = '../data/charades'
    vid_ids = np.array(util.readLinesFromFile(os.path.join(meta_dir,'vid_ids.txt')))
    rgb_dir = os.path.join(meta_dir,'i3d_charades_rgb')
    flow_dir = os.path.join(meta_dir,'i3d_charades_flow')
    
    for dir_curr in [rgb_dir, flow_dir]:
        vids_exist = glob.glob(os.path.join(dir_curr,'*.npy'))
        vids_exist = [os.path.split(vid_curr)[1][:-4] for vid_curr in vids_exist]
        missing = vid_ids[np.logical_not(np.in1d(vid_ids, vids_exist))]
        print dir_curr
        print missing.shape
        # out_file_missing = os.path.join(meta_dir, 'vid_ids_missing.txt')
        # util.writeFile(out_file_missing, list(missing))
        # print vids_exist[:10]
        # raw_input()
    


def make_train_test_files(anno_file, dir_features, out_file, post_pend = '.npz'):
    loaded = np.load(anno_file)

    gt_time_intervals = loaded['gt_time_intervals']
    gt_class_names = loaded['gt_class_names']
    gt_vid_names = loaded['gt_vid_names']

    # class_names = list(np.unique(gt_class_names))
    # class_names.sort()
    class_names = globals.class_names_charades
    print class_names

    num_classes = len(class_names)
    print num_classes

    lines_to_write = []

    missing = 0
    for vid_name in np.unique(gt_vid_names):

        feature_file = os.path.join(dir_features, vid_name+post_pend)
        if not os.path.exists(feature_file):
            missing+=1
            

        rel_classes = np.unique(gt_class_names[gt_vid_names==vid_name])
        bin_curr = np.zeros(num_classes).astype(int)
        for rel_class in rel_classes:
            bin_curr[class_names.index(rel_class)] = 1

        line_curr = ' '.join([str(val) for val in [feature_file]+list(bin_curr)])
        lines_to_write.append(line_curr)
        # print lines_to_write
        # raw_input()

    print out_file, len(lines_to_write), missing

    util.writeFile(out_file, lines_to_write)


def check_len():
    meta_dir = '../data/charades'
    train_test_dir = os.path.join(meta_dir, 'train_test_files')
    util.mkdir(train_test_dir)
    
    rgb_dir = os.path.join(meta_dir, 'Charades_v1_features_rgb')
    anno_dir = os.path.join(meta_dir, 'vgg16_rgb_features_npz')
    # anno_file = os.path.join(meta_dir, 'annos_test.npz')
    # out_file = os.path.join(train_test_dir, 'vgg_16_rgb_test.txt')
    
    anno_file = os.path.join(meta_dir, 'annos_test.npz')
    loaded = np.load(anno_file)

    gt_time_intervals = loaded['gt_time_intervals']
    gt_class_names = loaded['gt_class_names']
    gt_vid_names = loaded['gt_vid_names']

    annos_all = []

    for vid_name in np.unique(gt_vid_names):
        anno_len = len(glob.glob(os.path.join(rgb_dir, vid_name, '*.txt')))
        annos_all.append(anno_len)

    annos_all = np.array(annos_all)
    print annos_all.shape
    print np.min(annos_all), np.max(annos_all), np.mean(annos_all)

def script_save_train_test_files():
    meta_dir = '../data/charades'
    train_test_dir = os.path.join(meta_dir, 'train_test_files')
    util.mkdir(train_test_dir)
    
    anno_files = [os.path.join(meta_dir, 'annos_test.npz'),
                os.path.join(meta_dir, 'annos_train.npz')]

    # anno_dir = os.path.join(meta_dir, 'vgg16_rgb_features_npy')
    # out_files = [os.path.join(train_test_dir, 'vgg_16_rgb_npy_test.txt'),
                # os.path.join(train_test_dir, 'vgg_16_rgb_npy_train.txt')]

    # anno_dir = os.path.join(meta_dir, 'i3d_rgb')
    # out_files = [os.path.join(train_test_dir, 'i3d_rgb_npy_test.txt'),
    #             os.path.join(train_test_dir, 'i3d_rgb_npy_train.txt')]

    # anno_dir = os.path.join(meta_dir, 'i3d_flow')
    # out_files = [os.path.join(train_test_dir, 'i3d_flow_npy_test.txt'),
    #             os.path.join(train_test_dir, 'i3d_flow_npy_train.txt')]

    # anno_dir = os.path.join(meta_dir, 'i3d_both')
    # out_files = [os.path.join(train_test_dir, 'i3d_both_test_wmiss.txt'),
    #             os.path.join(train_test_dir, 'i3d_both_train_wmiss.txt')]

    anno_dir = os.path.join(meta_dir, 'i3d_charades_both')
    out_files = [os.path.join(train_test_dir, 'i3d_charades_both_test.txt'),
                os.path.join(train_test_dir, 'i3d_charades_both_train_wmiss.txt')]


    post_pend = '.npy'
    
    # 
    # out_file = 
    for anno_file, out_file in zip(anno_files, out_files):
        make_train_test_files(anno_file, anno_dir, out_file, post_pend)


def check_fps_vgg():
    meta_dir = '../data/charades'
    anno_dir = os.path.join(meta_dir,'Charades')
    dir_feat = os.path.join(meta_dir, 'Charades_v1_features_rgb')
    # out_file = os.path.join(meta_dir,'annos_test.npz')
    anno_file = os.path.join(anno_dir,'Charades_v1_test.csv')

    gt_length = {}
    
    with open(anno_file) as f:
        reader = csv.DictReader(f)
        for idx_row, row in enumerate(reader):
            vid_id = row['id']
            gt_length[vid_id]=float(row['length'])

    fps = []
    for vid_id in gt_length.keys():
        num_annos = len(glob.glob(os.path.join(dir_feat,vid_id,'*.txt')))
        fps.append(float(num_annos)/gt_length[vid_id])

    fps = np.array(fps)
    print np.mean(fps), np.min(fps), np.max(fps)


def save_npys():
    
    meta_dir = '../data/charades'
    out_dir = os.path.join(meta_dir, 'vgg16_rgb_features_npy')
    util.mkdir(out_dir)
    
    train_file = os.path.join(meta_dir, 'train_test_files','vgg_16_rgb_test.txt')
    lines = util.readLinesFromFile(train_file)
    
    # t = time.time()
    for idx_line_curr, line_curr in enumerate(lines):
        if idx_line_curr%100==0:
            print idx_line_curr, len(lines)

        file_curr = line_curr.split(' ')[0]
        out_file = os.path.join(out_dir, os.path.split(file_curr)[1]).replace('.npz','.npy')
        if not os.path.exists(out_file):
            feat = np.load(file_curr)['arr_0']
            np.save(out_file, feat)
    # print time.time() - t

    # t = time.time()
    # for line_curr in lines[:130]:
    #     file_curr = line_curr.split(' ')[0]
    #     feat = np.load(file_curr.replace('npz','npy'))
    #         # ['arr_0']
    #     # out_file = os.path.join(out_dir, os.path.split(file_curr)[1]).replace('.npz','.npy')
    #     # print file_curr, out_file
    #     # np.save(out_file, feat)
    # print time.time() - t


def merge_i3d_rgb_flow_features():
    meta_dir = '../data/charades'
    flow_dir = os.path.join(meta_dir, 'i3d_charades_flow')
    rgb_dir = os.path.join(meta_dir, 'i3d_charades_rgb')
    out_dir = os.path.join(meta_dir, 'i3d_charades_both')
    util.mkdir(out_dir)

    out_file_diffs = os.path.join(meta_dir, 'i3d_charades_diffs.txt')
    diffs = []

    files = glob.glob(os.path.join(flow_dir, '*.npy'))
    for flow_file in files:

        rgb_file = os.path.join(rgb_dir, os.path.split(flow_file)[1])
        out_file = os.path.join(out_dir, os.path.split(flow_file)[1])
        if os.path.exists(out_file):
            continue

        print out_file
        try:
            feat_rgb = np.load(rgb_file)
        except:
            print 'Error', rgb_file
            diffs.append('Error '+ rgb_file)
            continue

        feat_flow = np.load(flow_file)
            
        # print np.min(feat_flow), np.max(feat_flow)
        # print np.min(feat_rgb), np.max(feat_rgb)
        # print feat_flow.shape, feat_rgb.shape
        if feat_rgb.shape[0]!=feat_flow.shape[0]:
            print feat_rgb.shape, feat_flow.shape
            diff_curr = feat_rgb.shape[0]-feat_flow.shape[0]
            num_keep = min(feat_rgb.shape[0], feat_flow.shape[0])
            feat_flow = feat_flow[:num_keep,:]
            feat_rgb = feat_rgb[:num_keep,:]
            print feat_rgb.shape, feat_flow.shape
            diffs.append(rgb_file+' '+str(diff_curr))

            # raw_input()

        out_feat = np.concatenate([feat_rgb, feat_flow],axis = 1)
        np.save(out_file, out_feat)
        # raw_input()

    util.writeFile(out_file_diffs, diffs)

    
def create_charades_det_file(dir_out):
    # dir_out = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_157_dropout_0.8_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_charades_i3d_both/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.0001_0.0001_0.0001_lw_1.00_1.00_1.00__numSim_64_sumnomean_noExclusiveCASL_NEW_noMax/results_model_249_original_class_0_0.5_-2/outf'

    # dir_out = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_157_dropout_0.8_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_charades_i3d_both/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00__numSim_64_sumnomean_noExclusiveCASL_NEW_noMax/results_model_99_original_class_0_0.5_-2/outf'

    # dir_out = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_157_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_charades_i3d_both/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_0.10_1.00_1.00__numSim_64_sumnomean_noExclusiveCASL_NEW_noMax/results_model_99_original_class_0_0.5_-2/outf'

    # dir_out = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_157_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_charades_i3d_both/all_classes_False_just_primary_False_limit_None_cw_False_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_0.10_1.00_1.00__numSim_64_sumnomean_noExclusiveCASL_NEW_noMax/results_model_249_original_class_0_0.5_-2/outf'

    # dir_out = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_157_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_charades_i3d_both/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00__numSim_32_sumnomean_noExclusiveCASL_NEW_noMax/results_model_249_original_class_0_0.5_-2/outf'

    # dir_out = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_157_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_charades_i3d_both/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00__numSim_128_sumnomean_noExclusiveCASL_NEW_noMax/results_model_249_original_class_0_0.5_-2/outf'

    # dir_out = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_157_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_charades_i3d_both/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00__numSim_0_sumnomean_noExclusiveCASL_NEW_noMax/results_model_249_original_class_0_0.5_-2/outf'

    feat_dir = '../data/charades/i3d_both'
    vids = glob.glob(os.path.join(dir_out,'*.npy'))

    out_file = os.path.join(dir_out,'collated_for_matlab.txt')
    str_rep = 'maheenrashid@169.237.118.15:/home/maheenrashid/nn_net'
    vid_lines = []
    for vid_curr in vids:
        vid_name = os.path.split(vid_curr)[1][:-4]
        preds = np.load(vid_curr)
        feats = np.load(os.path.join(feat_dir, vid_name+'.npy'))
        num_feats = preds.shape[0]
        assert feats.shape[0]==preds.shape[0]
        equal_pts = np.clip(np.round(np.linspace(0,num_feats, 25, endpoint = False)),0,num_feats-1).astype(int)

        
        for idx_idx,idx in enumerate(equal_pts):
            pred_str = [vid_name,str(idx_idx)]
            pred_str += [str(val) for val in preds[idx,:]]
            # print len(pred_str)
            pred_str = ' '.join(pred_str)
            vid_lines.append(pred_str)


        # print equal_pts

        # print preds.shape, feats.shape
        # raw_input()
    util.writeFile(out_file, vid_lines)
    print out_file.replace('..',str_rep)


def main():
    print 'hello'

    # save_features_i3d_charades()
    # create_charades_det_file()
    # merge_i3d_rgb_flow_features()
    script_save_train_test_files()
    # save_npys()
    # check_fps_vgg()
    # save_all_anno()
    # save_features()
    # check_len()

    


if __name__=='__main__':
    main()