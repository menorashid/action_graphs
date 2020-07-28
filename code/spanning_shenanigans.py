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

def saving_graphs_etc(model_file = None, graph_num = None,k_vec = None,sparsify = False):
    # model_file = '../experiments/graph_multi_video_multi_F_joint_train_normalize_True_True_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_feat_dim_2048_64_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.5_0.5_ABS/model_299.pt'
    if model_file is None:
        model_file = '../experiments/graph_multi_video_multi_F_joint_train_normalize_True_True_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_feat_dim_2048_64_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.5_0.5_ABS_gk6/model_299.pt'


    model = torch.load(model_file).cuda()
    model.eval()

    train_data, test_train_data, test_data, n_classes, trim_preds = emb.get_data('ucf', 500, False, just_primary = False, gt_vec = False, k_vec = k_vec)

    # test_data = train_data
    # test_bool = False
    # test_data.feature_limit = None

    batch_size = 1
    test_bool = True
    print 'test_bool',test_bool


    out_dir_meta = model_file[:model_file.rindex('.')]
    out_dir_meta = out_dir_meta+'_graph_etc'
    if graph_num is not None:
        out_dir_meta+='_'+str(graph_num)
    util.mkdir(out_dir_meta)
    print out_dir_meta

    anno_file = test_data.anno_file
    print anno_file
    # raw_input()
    if graph_num is None:
        graph_num = 0

    vid_names, annos = readTrainTestFile(anno_file,k_vec)

    test_dataloader = torch.utils.data.DataLoader(test_data, 
                        batch_size = batch_size,
                        collate_fn = test_data.collate_fn,
                        shuffle = False, 
                        num_workers = 1)

    import torch.nn.functional as F
    # preds = []
    # labels = []

    for idx_data, data in enumerate(test_dataloader):
        gt_classes = np.where(annos[idx_data])[0]
        vid_name = os.path.split(vid_names[idx_data])[1]
        vid_name = vid_name[:vid_name.rindex('.')]
        # if vid_name!='video_test_0000273':
        #     continue
        
        label = data['label'].cpu().data.numpy()
        if k_vec is None:
            to_pass_in = data['features']
        else:
            to_pass_in = [data['features'],data['gt_vec']]
            
        affinity = model.get_similarity(to_pass_in,idx_graph_layer = graph_num, sparsify = sparsify, nosum = True)

        x_all, pmf = model(to_pass_in)
        if len(x_all)>4:
            x_all = [x_all]
            pmf = [pmf]
        # print len(x_all)
        # raw_input()
        # .shape
        # assert len(pmf)==1

        x_all = torch.cat([x_all_curr.unsqueeze(0) for x_all_curr in x_all],0)
        x_all = x_all.data.cpu().numpy()
        affinity = affinity.data.cpu().numpy()

        gt_vecs = []    
        for gt_class in gt_classes:
            gt_vec,_ = get_gt_vector(vid_name, x_all.shape[1], gt_class, test = test_bool)

            gt_vecs.append(gt_vec)
        gt_vecs = np.array(gt_vecs)
        
        out_file = os.path.join(out_dir_meta, vid_name+'.npz')
        print out_file
        
        # print gt_vecs.shape
        # raw_input()

        np.savez_compressed(out_file,
            gt_vecs = gt_vecs,
            affinity = affinity,
            x_all = x_all,
            gt_classes = gt_classes)


def get_spanned_nodes(gt_class, x_all, affinity, thresh, deno = 8, oracle = 'topk', gt_vec = None):

    # turn off self connections
    # make undirected
    affinity = np.triu(affinity,k=1)

    # oracle gives best start node
    if oracle=='gt':
        assert gt_vec is not None
        rel_cols = np.array(affinity)
        rel_cols[:,gt_vec==0]= 0
        start_node = np.argmax(np.sum(rel_cols, axis = 1))
        nodes_kept = [start_node]

    elif oracle=='topk':

        nodes_kept = np.argsort(x_all[:,gt_class])[::-1]
        k = max(1,x_all.shape[0]//deno)
        nodes_kept = list(nodes_kept[:k])
        
    elif oracle=='conf':
        deno = np.max(x_all[:,gt_class]) - deno
        nodes_kept = np.where(x_all[:,gt_class]>=deno)[0]
        # k = max(1,x_all.shape[0]//deno)
        nodes_kept = list(nodes_kept)
        # print x_all[nodes_kept,gt_class]
        # raw_input()
    else:
        raise ValueError('oracle '+oracle+' not good')



    edge_thresh = 1
    gt_count = []
    edge_threshes = [1]*len(nodes_kept)
    iter_inserted = [0]*len(nodes_kept)

    # affinity[:,start_node]= thresh
    # affinity[start_node,:]= thresh
    # idx = 0
    iter_curr = 1
    while edge_thresh>=thresh:
        
        # deactivate connections to kept nodes
        affinity[:,nodes_kept]= thresh-1

        # get rel graph
        rel_graph = affinity[nodes_kept,:]

        # pick max neighbor
        max_idx = np.argmax(rel_graph,axis = 1)
        max_vals = np.array([rel_graph[idx,max_idx[idx]] for idx in range(max_idx.size)])

        max_idx_idx = np.argmax(max_vals)
        edge_thresh = max_vals[max_idx_idx]
        max_neighbor = max_idx[max_idx_idx]
        
        # check if max neighbor worth keeping
        if edge_thresh<thresh:
            break

        # add max neighbor
        nodes_kept.append(max_neighbor)


        # check gt count
        if gt_vec is not None:
            gt_count.append( np.sum(gt_vec[nodes_kept]))
        
        # check edge thresh
        edge_threshes.append(edge_thresh)
        iter_inserted.append(iter_curr)
        iter_curr+=1

        # if gt_count[-1]>=np.sum(gt_vec)*0.9:
        #     break
        # print 'gt_count[-1], np.sum(gt_vec)', gt_count[-1], np.sum(gt_vec)
        # print 'edge_threshes[-1]',edge_threshes[-1]
        # if gt_count[-1]==np.sum(gt_vec):
        #     raw_input()
        
        

    # print len(nodes_kept)
    # print len(gt_count)
    # print len(edge_threshes)

    # print 'gt_count[-1], np.sum(gt_vec)', gt_count[-1], np.sum(gt_vec)
    # print 'edge_threshes[-1]',edge_threshes[-1]
    return nodes_kept, edge_threshes, gt_count, iter_inserted


def get_spanned_nodes_expansion_level(gt_class, x_all, affinity, thresh, deno = 8, oracle = 'topk', gt_vec = None, expansion_level = 1):

    # turn off self connections
    # make undirected
    affinity = np.triu(affinity,k=1)
    # affinity = np.abs(affinity)

    # oracle gives best start node
    if oracle=='gt':
        assert gt_vec is not None
        rel_cols = np.array(affinity)
        rel_cols[:,gt_vec==0]= 0
        start_node = np.argmax(np.sum(rel_cols, axis = 1))
        nodes_kept = [start_node]

    elif oracle=='topk':

        nodes_kept = np.argsort(x_all[:,gt_class])[::-1]
        k = max(1,x_all.shape[0]//deno)
        nodes_kept = list(nodes_kept[:k])
        
    elif oracle=='conf':
        deno = np.max(x_all[:,gt_class]) - deno
        nodes_kept = np.where(x_all[:,gt_class]>=deno)[0]
        # k = max(1,x_all.shape[0]//deno)
        nodes_kept = list(nodes_kept)
        # print x_all[nodes_kept,gt_class]
        # raw_input()
    else:
        raise ValueError('oracle '+oracle+' not good')



    edge_thresh = 1
    gt_count = []
    edge_threshes = [1]*len(nodes_kept)
    iter_inserted = [0]*len(nodes_kept)
    # print len(nodes_kept)

    for iter_curr in range(expansion_level):
        affinity[:,nodes_kept]= 0
        # thresh-1
        rel_graph = affinity[nodes_kept,:]
        # print np.min(rel_graph), np.max(rel_graph)

        nodes_to_keep = np.where(rel_graph>=thresh)[0]
        edge_thresh = rel_graph[rel_graph>=thresh]

        edge_threshes.extend(list(edge_thresh))
        nodes_kept.extend(list(nodes_to_keep))
        iter_inserted.extend([iter_curr]*len(nodes_to_keep))
        if gt_vec is not None:
            gt_count.append( np.sum(gt_vec[nodes_kept]))
        
        if edge_thresh.size>0:
            print len(nodes_to_keep), iter_curr, np.min(edge_thresh), np.max(edge_thresh)

    return nodes_kept, edge_threshes, gt_count, iter_inserted


def trying_it_out():
    # dir_graphs = '../experiments/graph_multi_video_multi_F_joint_train_normalize_True_True_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_feat_dim_2048_64_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.5_0.5_ABS/model_299_graph_etc'

    # out_dir_meta = '../scratch/spanning_shenanigans'
    # util.mkdir(out_dir_meta)

    model_file='../experiments/graph_multi_video_multi_F_joint_train_normalize_True_True_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_64_feat_dim_2048_64_64_64_gk_2_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_1_1_1_ABS/model_299.pt'
    

    graph_num = 1
    dir_graphs = model_file[:model_file.rindex('.')]
    dir_graphs = dir_graphs+'_graph_etc'
    if graph_num is not None:
        dir_graphs +='_'+str(graph_num)


    # out_dir_meta = '../scratch/spanning_shenanigans'
    out_dir_meta = os.path.join(os.path.split(dir_graphs)[0],'spanning_shenanigans')
    util.mkdir(out_dir_meta)

    vid_name = 'video_test_0000273'
    # oracle = False
    # idx = 1
    # thresh = 0.85

    oracle = False
    thresh = 0.9999
    branch_pred = 1
    idx = 0

    str_dir = '_'.join([str(val) for val in [vid_name, oracle, idx, thresh]])
    out_dir_curr = os.path.join(out_dir_meta, str_dir)
    util.mkdir(out_dir_curr)
    

    npz_file = os.path.join(dir_graphs, vid_name+'.npz')
    npz_data = np.load(npz_file)
    gt_vecs = npz_data['gt_vecs']
    affinity = npz_data['affinity']
    x_all = npz_data['x_all']
    gt_classes = npz_data['gt_classes']

    
    gt_vec = gt_vecs[idx]
    gt_vec[gt_vec>0]=1

    gt_class = gt_classes[idx]
    
    nodes_kept, edge_threshes, gt_count,_ = get_spanned_nodes(gt_class, x_all[branch_pred], affinity, thresh , deno = 8, oracle = oracle, gt_vec = gt_vec)

    precision_all = []
    recall_all = []

    for idx_n in range(len(nodes_kept)):
        idx_str = str(idx_n)
        idx_str = '0'*(4-len(idx_str))+idx_str
        out_file_curr = os.path.join(out_dir_curr, idx_str+'.jpg')

        nodes_kept_curr = nodes_kept[:idx_n+1]
        pred = np.zeros(gt_vec.shape)
        pred[nodes_kept_curr]=1

        precision = sklearn.metrics.precision_score(gt_vec, pred, labels = [1])
        recall = sklearn.metrics.recall_score(gt_vec, pred, labels = [1])
        # print precision, recall
        # raw_input()
        precision_all.append(precision)
        recall_all.append(recall)    
        
        x_axis = np.array(range(gt_vec.size))
        title_curr = ' '.join(['prec','%.2f'%precision,'rec','%.2f'%recall])

        visualize.plotSimple([(x_axis,pred),(x_axis,gt_vec)],out_file = out_file_curr,title = title_curr ,xlabel = 'time',ylabel = 'det conf',legend_entries=['Det','GT'])

    out_file_curr = os.path.join(out_dir_curr, 'prec_rec.jpg')
    visualize.plotSimple([(recall_all, precision_all)],out_file = out_file_curr,title = 'prec_rec' ,xlabel = 'Recall',ylabel = 'Precision')

    visualize.writeHTMLForFolder(out_dir_curr)
    print out_dir_curr


    
def getting_detection_results(model_file = None, graph_num = None,
    oracle = False,
    thresh = 0.8,
    branch_pred = 0,
    gt_it = False,
    add_identity = False,
    merge_with = 'max',
    deno = 8, 
    expansion_level = 0):

    # model_file = '../experiments/graph_multi_video_multi_F_joint_train_normalize_True_True_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_feat_dim_2048_64_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.5_0.5_ABS/model_299.pt'
    if model_file is None:
        model_file = '../experiments/graph_multi_video_multi_F_joint_train_normalize_True_True_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_feat_dim_2048_64_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.5_0.5_ABS/model_299.pt'


    dir_graphs = model_file[:model_file.rindex('.')]
    dir_graphs = dir_graphs+'_graph_etc'
    if graph_num is not None:
        dir_graphs +='_'+str(graph_num)

    print dir_graphs
    # out_dir_meta = '../scratch/spanning_shenanigans'
    out_dir_meta = os.path.join(os.path.split(dir_graphs)[0],'spanning_shenanigans')
    util.mkdir(out_dir_meta)
    
    out_dir = '_'.join([str(val) for val in ['eval',oracle,thresh,branch_pred,gt_it, merge_with,deno, expansion_level]])
    out_dir = os.path.join(out_dir_meta,out_dir)
    util.mkdir(out_dir)
    out_dir_span = os.path.join(out_dir,'span')
    util.mkdir(out_dir_span)

    vid_files = glob.glob(os.path.join(dir_graphs,'*test*.npz'))
    
    det_vid_names = []
    det_events_class = []
    det_conf_all = []
    det_time_intervals_all = []

    for vid_file in vid_files:
        vid_name = os.path.split(vid_file)[1]
        vid_name = vid_name[:vid_name.rindex('.')]
        npz_data = np.load(vid_file)
        gt_vecs = npz_data['gt_vecs']
        affinity = npz_data['affinity']
        x_all = npz_data['x_all'][branch_pred]
        gt_classes = npz_data['gt_classes']

        len_video = x_all.shape[0]

        start_seq = np.array(range(0,len_video))*16./25.
        end_seq = np.array(range(1,len_video+1))*16./25.
        det_time_intervals_meta = np.concatenate([start_seq[:,np.newaxis],end_seq[:,np.newaxis]],axis=1)
        
        nodes_kept_arr = []
        edge_threshes_arr = []
        gt_count_arr = []
        iter_inserted_arr = []

        for idx in range(gt_classes.size):

            gt_vec = gt_vecs[idx]
            gt_vec[gt_vec>0]=1

            gt_class = gt_classes[idx]
            # print vid_name, idx, gt_class
            
            # nodes_kept, edge_threshes, gt_count, iter_inserted = get_spanned_nodes(gt_class, x_all, affinity, thresh , deno = deno, oracle = oracle, gt_vec = gt_vec)
            if expansion_level>0:
                nodes_kept, edge_threshes, gt_count, iter_inserted = get_spanned_nodes_expansion_level(gt_class, x_all, affinity, thresh , deno = deno, oracle = oracle, gt_vec = gt_vec,expansion_level = expansion_level)
            else:
                nodes_kept, edge_threshes, gt_count, iter_inserted = get_spanned_nodes(gt_class, x_all, affinity, thresh , deno = deno, oracle = oracle, gt_vec = gt_vec)

            nodes_kept_arr.append(nodes_kept)
            edge_threshes_arr.append(edge_threshes)
            gt_count_arr.append(gt_count)
            iter_inserted_arr.append(iter_inserted)


            det_conf = np.zeros((len_video,))
            
            edge_threshes = np.array(edge_threshes)
            iter_inserted = np.array(iter_inserted)
            iter_inserted = (len_video-iter_inserted)/float(len_video)

            if gt_it:
                det_conf[nodes_kept] = x_all[nodes_kept,gt_class]
            else:
                det_conf[nodes_kept] = edge_threshes
            
            print np.min(det_conf), np.max(det_conf)

            bin_keep = np.zeros((len_video,))
            bin_keep[nodes_kept] = 1
            bin_keep = bin_keep>0

            # array(edge_threshes)
            det_conf, det_time_intervals = merge_detections(bin_keep, det_conf, det_time_intervals_meta, merge_with = merge_with)
                # print class_idx, labels[idx_sample][class_idx],np.min(det_conf), np.max(det_conf),len(det_time_intervals)

                # det_time_intervals = det_time_intervals_meta
                
            det_vid_names.extend([vid_name]*det_conf.shape[0])
            det_events_class.extend([gt_class]*det_conf.shape[0])
            det_conf_all.append(det_conf)
            det_time_intervals_all.append(det_time_intervals)
        
        out_file_span = os.path.join(out_dir_span,vid_name+'.npy')
        nodes_kept_arr = np.array(nodes_kept_arr)
        edge_threshes_arr = np.array(edge_threshes_arr)
        gt_count_arr = np.array(gt_count_arr)
        iter_inserted_arr = np.array(iter_inserted_arr)
        
        np.savez_compressed(out_file_span, 
                    nodes_kept_arr = nodes_kept_arr,
                    edge_threshes_arr = edge_threshes_arr,
                    gt_count_arr = gt_count_arr,
                    iter_inserted_arr = iter_inserted_arr,
                    gt_vecs = npz_data['gt_vecs'],
                    affinity = npz_data['affinity'],
                    x_all = npz_data['x_all'],
                    gt_classes = npz_data['gt_classes'])

    det_conf = np.concatenate(det_conf_all,axis =0)
    det_time_intervals = np.concatenate(det_time_intervals_all,axis = 0)
    det_events_class = np.array(det_events_class)
    
    log_arr = []
    aps = et.test_overlap(det_vid_names, det_conf, det_time_intervals,det_events_class,log_arr = log_arr, dataset = 'ucf')
    out_file = os.path.join(out_dir,'log.txt')
    print out_file

    util.writeFile(out_file, log_arr)





def save_graphs_to_look_at(model_file, graph_nums):
    out_dir_meta = model_file[:model_file.rindex('.')]
    out_dir_meta_meta = out_dir_meta+'_graph_etc'
    out_dir_viz = out_dir_meta_meta+'_viz'
    util.mkdir(out_dir_viz)
    for graph_num in graph_nums:
        out_dir_meta=out_dir_meta_meta+'_'+str(graph_num)
        assert os.path.exists(out_dir_meta)
        vid_files = glob.glob(os.path.join(out_dir_meta,'*test*.npz'))
        

        for vid_file in vid_files:

            npz_data = np.load(vid_file)
            vid_file = os.path.split(vid_file)[1]
            affinity = npz_data['affinity']
            
            gt_vecs = npz_data['gt_vecs']
            gt_classes = npz_data['gt_classes']
            x_all = npz_data['x_all']
            
            
            plotter = []
            legend_entries = []
            for gt_idx,gt_class in enumerate(gt_classes):
                gt_vec = gt_vecs[gt_idx]
                val_rel =x_all[0,:,gt_class]
                gt_vec = gt_vec/np.max(gt_vec)
                gt_vec = gt_vec* np.max(val_rel)
                # (gt_idx+1)
                x_axis = range(gt_vec.size)
                plotter.append((x_axis,gt_vec))
                plotter.append((x_axis,val_rel))
                legend_entries.append(class_names[gt_class])
                legend_entries.append(class_names[gt_class]+' pred')

            out_file = os.path.join(out_dir_viz, vid_file[:vid_file.rindex('.')]+'_gt.jpg')
            visualize.plotSimple(plotter,out_file = out_file,xlabel = 'time',ylabel = '',legend_entries=legend_entries,outside = True)


            out_file = os.path.join(out_dir_viz, vid_file[:vid_file.rindex('.')]+'_'+str(graph_num)+'.jpg')
            visualize.saveMatAsImage(affinity, out_file)

            visualize.writeHTMLForFolder(out_dir_viz)
            # print out_dir_viz


def get_l2_diff(aff, perfectG, threshes):
    diffs_curr =[]
    for thresh in threshes:
        aff[aff<thresh]=0
        # aff[aff>thresh]=1
        diff_curr = np.sqrt(np.sum(np.power(perfectG - aff,2)))
        diffs_curr.append(diff_curr)
    return diffs_curr


def get_distance_from_perfect(model_file, graph_num):

    out_dir_meta = model_file[:model_file.rindex('.')]
    out_dir_meta_meta = out_dir_meta+'_graph_etc'
    

    out_dir_meta=out_dir_meta_meta+'_'+str(graph_num)
    out_dir_viz = out_dir_meta+'_dist_perfectG'
    print out_dir_viz

    util.mkdir(out_dir_viz)
    assert os.path.exists(out_dir_meta)
    vid_files = glob.glob(os.path.join(out_dir_meta,'*validation*.npz'))
    
    class_collations = [[] for idx in range(len(class_names))]
    class_collations_pos = [[] for idx in range(len(class_names))]
    viz = True
    threshes = np.arange(0.1,1.1,0.1)
    print threshes

    for vid_file in vid_files:
        print vid_file
        npz_data = np.load(vid_file)
        vid_file = os.path.split(vid_file)[1]
        affinity = npz_data['affinity']
        
        gt_vecs = npz_data['gt_vecs']
        gt_classes = npz_data['gt_classes']        

        if viz:
            out_file = os.path.join(out_dir_viz, vid_file[:vid_file.rindex('.')]+'_'+str(graph_num)+'.jpg')
            visualize.saveMatAsImage(affinity, out_file)

            plotter = []
            legend_entries = []
            for gt_idx,gt_class in enumerate(gt_classes):
                gt_vec = gt_vecs[gt_idx]
                gt_vec = gt_vec/np.max(gt_vec)
                gt_vec = gt_vec* (gt_idx+1)
                x_axis = range(gt_vec.size)
                plotter.append((x_axis,gt_vec))
                legend_entries.append(class_names[gt_class])

            out_file = os.path.join(out_dir_viz, vid_file[:vid_file.rindex('.')]+'_gt.jpg')
            visualize.plotSimple(plotter,out_file = out_file,xlabel = 'time',ylabel = '',legend_entries=legend_entries)


        for idx_gt,gt_vec in enumerate(gt_vecs):
            gt_class = gt_classes[idx_gt]
            class_name = class_names[gt_class]
            gt_vec = gt_vec[:,np.newaxis]
            perfectG = np.dot(gt_vec,gt_vec.T)
            aff = np.array(affinity)
            aff_just_pos = aff*perfectG
            diff = get_l2_diff(aff, perfectG, threshes)
            diff_pos = get_l2_diff(aff_just_pos, perfectG, threshes)

            class_collations[gt_class].append(diff)
            class_collations_pos[gt_class].append(diff_pos)

            plotter = [(threshes, diff),(threshes, diff_pos)]
            legend_entries = ['All','Pos']
            out_file = os.path.join(out_dir_viz, vid_file[:vid_file.rindex('.')]+'_'+class_name+'_diff.jpg')
            visualize.plotSimple(plotter,out_file = out_file,xlabel = 'Thresh',ylabel = 'Diff',legend_entries=legend_entries)

            if viz:
                out_file = os.path.join(out_dir_viz, vid_file[:vid_file.rindex('.')]+'_'+class_name+'_perfectG.jpg')
                visualize.saveMatAsImage(perfectG, out_file)

        visualize.writeHTMLForFolder(out_dir_viz)

    for idx_class in range(len(class_names)):

        class_name = class_names[idx_class]
        cc = np.array(class_collations[idx_class])
        ccp = np.array(class_collations_pos[idx_class])
        cc = np.mean(cc,axis = 0)
        ccp = np.mean(ccp, axis = 0)
        plotter = [(threshes, cc),(threshes, ccp)]
        legend_entries = ['All','Pos']
        out_file = os.path.join(out_dir_viz, 'average_'+class_name+'_diff.jpg')
        visualize.plotSimple(plotter,out_file = out_file,title = class_name, xlabel = 'Thresh',ylabel = 'Diff',legend_entries=legend_entries)

    visualize.writeHTMLForFolder(out_dir_viz)



def trying_it_out_expansion_levels(model_file, graph_num = None,
    oracle = False,
    thresh = 0.8,
    branch_pred = 0,
    gt_it = False,
    add_identity = False,
    merge_with = 'max',
    deno = 8,
    expansion_level = 1):

    pass

    
def script_for_ens_best_yet():
    # oracle = False
    # oracle = 'topk'
    # deno = 8
    oracle = 'conf'
    deno = 0.01
    gt_it = False
    expansion_level = 1
    merge_with = 'max'

    dir_model = '../experiments/graph_multi_video_same_F_ens_dll/graph_multi_video_same_F_ens_dll_aft_nonlin_HT_l2_non_lin_HT_sparsify_0.75_0.5_0.25_graph_size_2_sigmoid_True_deno_8_n_classes_20_in_out_2048_256_feat_dim_2048_512_method_cos_zero_self_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranch_500_step_500_0.1_0.001_0.001_ABS_bias/model_499.pt'
    threshes = [0.75,0.5,0.25]
    # [0.25,0.5,0.75]
    # 
    for num in range(1,2):
        thresh = threshes[num]
        # saving_graphs_etc(dir_model, num,k_vec = None, sparsify = True)

    # get_distance_from_perfect(dir_model, graph_num)

    # dir_model = None
    # graph_num = 0

        getting_detection_results(dir_model, num, oracle = oracle, thresh = thresh, gt_it = gt_it,branch_pred = num, merge_with =merge_with, deno = deno, expansion_level = expansion_level)
        # raw_input()
    
def scratch():


    # trying_it_out()

    # return
    # dir_model = '../experiments/graph_multi_video_multi_F_joint_train_normalize_True_True_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_128_feat_dim_2048_128_gk_2_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.5_0.5_ABS/model_299.pt'

    # dir_model='../experiments/graph_multi_video_multi_F_joint_train_normalize_True_True_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_64_feat_dim_2048_64_64_64_gk_2_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_1_1_1_ABS/model_299.pt'
    
    # dir_model='../experiments/graph_multi_video_multi_F_joint_train_normalize_True_True_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_feat_dim_2048_64_gk_2_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.5_0.5_ABS/model_299.pt'

    # dir_model='../experiments/graph_multi_video_multi_F_joint_train_normalize_True_True_non_lin_HT_sparsify_False_graph_size_2_deno_8_n_classes_20_in_out_2048_64_feat_dim_2048_64_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.5_0.5_ABS/model_299.pt'

    # dir_model='../experiments/graph_multi_video_multi_F_joint_train_gaft_normalize_True_True_aft_nonlin_LN_HT_non_lin_HT_sparsify_False_graph_size_2_deno_8_n_classes_20_in_out_2048_64_64_feat_dim_2048_64_64_64_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.33_0.33_0.33_ABS/model_299.pt'


    # dir_model = '../experiments/graph_multi_video_multi_F_joint_train_gaft_normalize_True_True_aft_nonlin_LN_HT_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_64_feat_dim_2048_64_64_64_gk_2_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.33_0.33_0.33_ABS/model_299.pt'

    # dir_model = '../experiments/graph_multi_video_multi_F_joint_train_gaft_normalize_True_True_aft_nonlin_HT_l2_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_feat_dim_2048_8_gk_8_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.50_0.50_ABS_bias/model_299.pt'

    # dir_model = '../experiments/graph_multi_video_multi_F_joint_train_gaft_normalize_True_True_aft_nonlin_HT_l2_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_feat_dim_2048_8_gk_8_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.00_1.00_ABS_bias/model_299.pt'

    graph_num = 0
    oracle = False
    thresh = 0.8
    branch_pred = 0
    gt_it = False


    # dir_model='../experiments/graph_multi_video_multi_F_joint_train_gaft_normalize_True_True_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_feat_dim_2048_64_gk_8_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.50_0.50_ABS_BN/model_299.pt'

    dir_model = '../experiments/graph_multi_video_i3dF_gaft_normalize_True_True_aft_nonlin_HT_l2_non_lin_None_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_feat_dim_2048_64_gk_8_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropy_300_step_300_0.1_0.001_ABS_bias/model_299.pt'
    
    # dir_model = '../experiments/graph_multi_video_multi_F_joint_train_rlbn_aft_nonlin_HT_BN_non_lin_HT_sparsify_True_graph_size_2_deno_8_n_classes_20_in_out_2048_64_feat_dim_2048_64_gk_8_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_200_step_200_0.1_0.001_0.001_lw_0.50_0.50_ABS_trs_false_af_false/model_199.pt'

    dir_model = '../experiments/graph_multi_video_cooc_aft_nonlin_HT_l2_non_lin_None_sparsify_False_graph_size_rand_deno_8_n_classes_20_in_out_2048_256_feat_dim_100_all_method_affinity_dict_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.001_ABS_bias/model_99.pt'

    dir_model = '../experiments/graph_multi_video_cooc/graph_multi_video_cooc_aft_nonlin_HT_l2_non_lin_None_sparsify_False_graph_size_rand_deno_8_n_classes_20_in_out_2048_256_feat_dim_100_TennisSwingneg_method_affinity_dict_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.001_ABS_bias/model_99.pt'

    dir_model = '../experiments/graph_multi_video_same_F_ens_dll/graph_multi_video_same_F_ens_dll_aft_nonlin_HT_l2_non_lin_HT_sparsify_0.8_0.6_graph_size_2_deno_8_n_classes_20_in_out_2048_128_feat_dim_2048_256_method_cos_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_lw_0.50_0.50_ABS_bias/model_299.pt'


    dir_model = '../experiments/graph_multi_video_same_F_ens_dll/graph_multi_video_same_F_ens_dll_aft_nonlin_HT_l2_non_lin_HT_sparsify_0.75_0.5_0.25_graph_size_2_sigmoid_True_deno_8_n_classes_20_in_out_2048_256_feat_dim_2048_512_method_cos_zero_self_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranch_300_step_300_0.1_0.001_0.001_ABS_bias/model_299.pt'


    dir_model = '../experiments/graph_multi_video_same_F_ens_dll_moredepth_concat_sim/graph_multi_video_same_F_ens_dll_moredepth_concat_sim_aft_nonlin_HT_L2_num_graphs_1_num_branches_1_non_lin_aft_RL_graph_size_1_sigmoid_True_deno_8_n_classes_20_in_out_2048_128_feat_dim_2048_64_non_lin_HT_scaling_method_n_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.001_0.001_ABS_bias/model_99.pt'
    dir_model = '../experiments/graph_multi_video_same_F_ens_dll_moredepth_concat_sim/graph_multi_video_same_F_ens_dll_moredepth_concat_sim_aft_nonlin_HT_L2_num_graphs_1_num_branches_1_non_lin_aft_RL_graph_size_1_sigmoid_False_deno_8_n_classes_20_in_out_2048_128_feat_dim_2048_64_non_lin_HT_scaling_method_n_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.001_0.001_ABS_bias_t/model_9.pt'

    dir_model = '../experiments/graph_multi_video_same_F_ens_dll_moredepth_concat_sim/graph_multi_video_same_F_ens_dll_moredepth_concat_sim_aft_nonlin_HT_L2_num_graphs_1_num_branches_1_non_lin_aft_RL_graph_size_1_sigmoid_False_deno_8_n_classes_20_in_out_2048_128_feat_dim_2048_64_non_lin_HT_scaling_method_n_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.001_0.001_ABS_bias_sym/model_39.pt'
    dir_model = '../experiments/graph_multi_video_same_F_ens_dll_moredepth_concat_sim/graph_multi_video_same_F_ens_dll_moredepth_concat_sim_aft_nonlin_HT_L2_num_graphs_1_num_branches_1_non_lin_aft_RL_graph_size_1_sigmoid_False_deno_8_n_classes_20_in_out_2048_128_feat_dim_2048_64_non_lin_HT_scaling_method_sum_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.001_0.001_ABS_bias_sym/model_9.pt'
    dir_model = '../experiments/graph_multi_video_same_F_ens_dll_moredepth_concat_sim/graph_multi_video_same_F_ens_dll_moredepth_concat_sim_aft_nonlin_HT_L2_num_graphs_1_num_branches_1_non_lin_aft_HT_graph_size_1_sigmoid_False_deno_8_n_classes_20_in_out_2048_128_feat_dim_2048_64_non_lin_HT_scaling_method_n_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.001_0.001_ABS_bias_sym/model_9.pt'

    dir_model = '../experiments/graph_multi_video_same_F_ens_dll_moredepth_concat_sim/graph_multi_video_same_F_ens_dll_moredepth_concat_sim_aft_nonlin_HT_L2_num_graphs_1_num_branches_1_non_lin_aft_RL_graph_size_1_sigmoid_True_deno_8_n_classes_20_in_out_2048_128_feat_dim_2048_64_non_lin_HT_scaling_method_n_ucf/all_classes_False_just_primary_False_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.001_0.001_ABS_bias_sym/model_9.pt'
    num = 0
    # for num in range(3):

    dir_model = '../experiments/graph_multi_video_same_F_ens_dll/graph_multi_video_same_F_ens_dll_aft_nonlin_HT_l2_non_lin_HT_sparsify_None_graph_size_1_sigmoid_True_deno_8_n_classes_20_in_out_2048_128_feat_dim_2048_256_method_cos_learn_thresh_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropy_100_step_100_0.1_0.001_0.001_ABS_bias/model_99.pt'

    dir_model = '../experiments/graph_multi_video_same_F_ens_dll_moredepth/graph_multi_video_same_F_ens_dll_moredepth_aft_nonlin_HT_L2_non_lin_HT_num_graphs_1_sparsify_0.5_graph_size_2_sigmoid_True_deno_0.5_n_classes_20_in_out_2048_256_feat_dim_2048_512_method_cos_zero_self_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropy_300_step_300_0.1_0.001_0.001_0.001_ABS_bias_retry/model_299.pt'
    
    dir_model = '../experiments/graph_multi_video_same_i3dF/graph_multi_video_same_i3dF_aft_nonlin_HT_l2_sparsify_0.5_non_lin_None_method_cos_zero_self_deno_8_n_classes_20_in_out_2048_2_graph_size_2_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropy_100_step_100_0.1_0.001_0.01_ABS_bias_wb/model_99.pt'

    # saving_graphs_etc(dir_model, num,k_vec = None, sparsify = True)
    save_graphs_to_look_at(dir_model, [num])

    # get_distance_from_perfect(dir_model, graph_num)

    # dir_model = None
    # graph_num = 0

    # getting_detection_results(dir_model, graph_num, oracle = oracle, thresh = thresh, branch_pred = branch_pred, gt_it = gt_it)
    # trying_it_out()

    # print 'hello'
    

def main():
    scratch()

    # script_for_ens_best_yet()


    

if __name__=='__main__':
    main()