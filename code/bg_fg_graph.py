from exp_mill_bl import train_simple_mill_all_classes
import torch
import random
import numpy as np
import glob
import os
from helpers import util
from debugging_graph import readTrainTestFile, get_gt_vector
import multiprocessing

def graph_l1_experiment_best_yet():
    print 'hey girl'
    # raw_input()
    model_name = 'graph_multi_video_with_L1_retF'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1,0.5]
    plot_losses = True

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # loss_weights = [1,1]
    # plot_losses = False

    lr = [0.001,0.001, 0.001]
    multibranch = 1
    # loss_weights = [1,1]
    
    branch_to_test = -2
    print 'branch_to_test',branch_to_test
    attention = True

    k_vec = None

    gt_vec = False
    just_primary = False

    seed = 999
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    
    
    epoch_stuff = [250,500]
    dataset = 'ucf'
    limit  = None
    save_after = 50
    
    test_mode = True
    
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = [249]
    retrain = False
    viz_mode = True
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=True
 
    network_params['graph_size'] = 2
    network_params['method'] = 'cos'
    network_params['sparsify'] = 0.5
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = False
    post_pend = '_noLimit'
    first_thresh=0
    class_weights = True
    test_after = 10
    all_classes = False
    
    second_thresh = 0.5
    det_class = -1
    train_simple_mill_all_classes (model_name = model_name,
                        lr = lr,
                        dataset = dataset,
                        network_params = network_params,
                        limit = limit, 
                        epoch_stuff= epoch_stuff,
                        batch_size = 32,
                        batch_size_val = 32,
                        save_after = save_after,
                        test_mode = test_mode,
                        class_weights = class_weights,
                        test_after = test_after,
                        all_classes = all_classes,
                        just_primary = just_primary,
                        model_nums = model_nums,
                        retrain = retrain,
                        viz_mode = viz_mode,
                        second_thresh = second_thresh,
                        first_thresh = first_thresh,
                        det_class = det_class,
                        post_pend = post_pend,
                        viz_sim = viz_sim,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec,
                        attention = attention,
                        test_pair = False,
                        test_post_pend = test_post_pend,
                        test_method = test_method,
                        criterion_str = criterion_str,
                        plot_losses = plot_losses)




def graph_ablation_study():
    print 'hey girl'
    # raw_input()
    # model_name = 'graph_multi_video_with_L1_retF'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    # loss_weights = [1.,0.,0.5]
    # plot_losses = True
    save_outfs = False
    model_name = 'graph_multi_video_with_L1'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    loss_weights = [1.,0.]
    plot_losses = False

    lr = [0.001,0.001, 0.001]
    multibranch = 1
    # loss_weights = [1,1]
    
    branch_to_test = -2
    print 'branch_to_test',branch_to_test
    attention = True

    k_vec = None

    gt_vec = False
    just_primary = False

    seed = 999
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    
    
    epoch_stuff = [250,250]
    dataset = 'ucf'
    num_similar = 0
    batch_size = 32
    batch_size_val = 32
    limit  = None
    save_after = 50
    
    test_mode = True
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = None
    retrain = False
    viz_mode = True
    viz_sim = True

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}

    network_params['deno'] = 8
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,1024]
    # network_params['feat_ret']=True
 
    network_params['graph_size'] = 2
    network_params['method'] = 'cos'
    network_params['sparsify'] = 0.5
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = False
    post_pend = '_ablation'
    first_thresh=0.
    class_weights = True
    test_after = 10
    all_classes = False
    
    second_thresh = 0.5
    det_class = -1
    train_simple_mill_all_classes (model_name = model_name,
                        lr = lr,
                        dataset = dataset,
                        network_params = network_params,
                        limit = limit, 
                        epoch_stuff= epoch_stuff,
                        batch_size = batch_size,
                        batch_size_val = batch_size_val,
                        save_after = save_after,
                        test_mode = test_mode,
                        class_weights = class_weights,
                        test_after = test_after,
                        all_classes = all_classes,
                        just_primary = just_primary,
                        model_nums = model_nums,
                        retrain = retrain,
                        viz_mode = viz_mode,
                        second_thresh = second_thresh,
                        first_thresh = first_thresh,
                        det_class = det_class,
                        post_pend = post_pend,
                        viz_sim = viz_sim,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec,
                        attention = attention,
                        test_pair = False,
                        test_post_pend = test_post_pend,
                        test_method = test_method,
                        criterion_str = criterion_str,
                        plot_losses = plot_losses,
                        num_similar = num_similar,
                        save_outfs = save_outfs)


def process_one_graph((graph_file, vid_names, annos,graph_idx)):
	if graph_idx%50==0:
		print graph_idx
	vid_name = os.path.split(graph_file)[1]
	idx_anno = vid_names.index(vid_name)

	anno_curr = annos[idx_anno,:]
	# print anno_curr
	class_idx_all = np.where(anno_curr)[0]
	# print class_idx_all
	# return [],[]
	# print vid_name
	
	good_edges_all = []
	bad_edges_all = []

	for class_idx in class_idx_all:
		graph = np.load(graph_file)
		# print graph[:3,:3]

		out_shape_curr = graph.shape[0]
		gt_vals, det_times = get_gt_vector(vid_name[:-4], out_shape_curr, class_idx, dataset = 'ucf')
		# print gt_vals
		# print class_idx
		# print gt_vals.shape, np.unique(gt_vals)
		graph = np.abs(graph)
		# graph = graph*(1-np.eye(out_shape_curr))
		# print graph[:3,:3]
		graph = np.triu(graph,1)
		# print graph[:3,:3]
		# raw_input()

		rel_rows = graph[gt_vals>0,:]
		num_gt = np.sum(gt_vals)
		num_bg = np.sum(gt_vals<1)
		# print num_gt, num_bg, gt_vals.size, num_gt+num_bg

		total = num_gt*(num_gt-1)/2.
		# within gt
		rel_cells = rel_rows[:,gt_vals>0]
		# total = rel_cells.size
		kept = np.sum(rel_cells>0.5)
		good_edges = kept/total
		# float(total)
		# print good_edges

		# if np.isnan(good_edges):
		# 	print gt_vals,num_gt, total, kept

		# 	raw_input()

		# without gt
		rel_cells = rel_rows[:,gt_vals<1]
		total = num_gt * num_bg
		kept = np.sum(rel_cells>0.5)
		bad_edges = kept/float(total)
		# print bad_edges
		
		if np.isnan(bad_edges) or np.isnan(good_edges):
			continue

		good_edges_all.append(good_edges)
		bad_edges_all.append(bad_edges)
		# raw_input()

	return good_edges_all, bad_edges_all

def main():
	print 'hello'

	# graph_l1_experiment_best_yet()

	# return
	sparse = '../experiments/graph_multi_video_with_L1/graph_multi_video_with_L1_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_0.00__ablation/results_model_249_0.0_0.5/viz_sim_mat'

	sparse_mcasl = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_0.00_0.50__ablation/results_model_249_0.0_0.5/viz_sim_mat'

	sparse_l1 = '../experiments/graph_multi_video_with_L1/graph_multi_video_with_L1_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00__ablation/results_model_249_0.0_0.5/viz_sim_mat'

	sparse_l1_mcasl = '../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_500_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_0.50__noLimit/results_model_249_0_0.5/viz_sim_mat'

	test_file = '../data/ucf101/train_test_files/test.txt'
	# test_file = util.readLinesFromFile(test_file)
	# print test_file[0]
	vid_files, annos = readTrainTestFile(test_file)
	annos = np.array(annos)
	vid_files = [os.path.split(vid_file)[1] for vid_file in vid_files]
	print vid_files[0]

	meta_dirs = [sparse, sparse_l1, sparse_mcasl,sparse_l1_mcasl]
	str_meta_dirs = ['sparse', 'sparse_l1', 'sparse_mcasl','sparse_l1_mcasl']

	
	for str_meta_dir, meta_dir in zip(str_meta_dirs, meta_dirs):

		graph_files = glob.glob(os.path.join(meta_dir, '*','*.npy'))
		graph_files = list(set(graph_files))
		# print len(graph_files)
		# print graph_files[0]

		# ge_all = []
		# be_all = []

		args = []

		for graph_idx in range(len(graph_files)):
			# if graph_idx%50==0:
			# 	print graph_idx
			args.append((graph_files[graph_idx],vid_files,annos,graph_idx))

		pool = multiprocessing.Pool()
		returned_vals = pool.map(process_one_graph,args)
		# for arg in args:
		# 	process_one_graph(arg)

		ge_all = [ge_c for ge_b in returned_vals for ge_c in ge_b[0] ]
		be_all = [ge_c for ge_b in returned_vals for ge_c in ge_b[1] ]
		# print zip(returned_vals)
		# print returned_vals
		# print ge_all
		# print be_all
		# 	ge, be = process_one_graph(graph_files[graph_idx], vid_files, annos)
		# 	ge_all+=ge
		# 	be_all+=be

		print len(ge_all)
		print len(be_all)
		# print ge_all
		# print be_all

		# print np.array(ge_all), np.min(ge_all), np.max(ge_all)
		# print np.array(be_all), np.min(be_all), np.max(be_all)

		print str_meta_dir, np.mean(ge_all), np.mean(be_all)


		# break


	# graph_ablation_study()
	# graph_l1_experiment_best_yet()

if __name__=='__main__':
	main()