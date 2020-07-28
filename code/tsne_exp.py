import sys
sys.path.append('../topicsne/')

from matplotlib.patches import Ellipse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
import glob
from helpers import visualize,util


from sklearn import manifold, datasets
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform

from wrapper import Wrapper
from vtsne import VTSNE

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



def get_rel_features(feat_file, gt_file, sample_rate = 1):
    feats = np.load(feat_file)
    bin_gt = np.load(gt_file)
    # bin_keep = bin_gt>0
    # idx_start, idx_end = get_start_end(bin_keep)
    # feats_to_keep =[]
    # for idx_idx, idx_start_curr in enumerate(idx_start):
    #     feats_rel = np.mean(feats[idx_start_curr:idx_end[idx_idx],:],axis = 0,keep_dims = True)
    #     print feats_rel.shape

    assert len(np.unique(bin_gt))==2
    feats_to_keep = feats[bin_gt>0,:]
    feats_to_keep = feats_to_keep[::sample_rate,:]
    
    return feats_to_keep


def get_feats_class_idx(dir_curr,dir_gt,anno_file,per_vid = 0):
    annos = util.readLinesFromFile(anno_file)
    annos = [line_curr.split(' ') for line_curr in annos]
    npy_files = [anno[0] for anno in annos]
    annos = np.array([[int(val) for val in anno[1:]] for anno in annos])
    feats_all = []
    class_idx_all = []

    for idx_anno, anno_file in enumerate(npy_files):
        class_idx = np.where(annos[idx_anno,:])[0]
        vid_name = os.path.split(anno_file)[1]

        feat_file = os.path.join(dir_curr,vid_name)
        gt_file = os.path.join(dir_gt, vid_name)

        rel_feats = get_rel_features(feat_file, gt_file)
        if per_vid>0:
            rel_feats_idx = np.random.random_integers(0,rel_feats.shape[0]-1,per_vid)
            print rel_feats_idx
            print rel_feats.shape
            rel_feats = rel_feats[rel_feats_idx,:]
            print rel_feats.shape

        class_idx = np.ones(rel_feats.shape[0])*class_idx
        feats_all.append(rel_feats)
        class_idx_all.append(class_idx)

    feats_all = np.concatenate(feats_all, axis = 0)
    class_idx_all = np.concatenate(class_idx_all, axis = 0)

    return feats_all, class_idx_all

def preprocess_feats_etc(feats, y, metric = 'cosine', perplexity = 30):
    n_points = feats.shape[0]
    print 'pairwise distancing'
    distances2 = pairwise_distances(feats, metric=metric)
    # , squared=True)
    print 'joint probbing'
    pij = manifold.t_sne._joint_probabilities(distances2, perplexity, False)
    print pij.shape

    print 'squareforming'
    pij = squareform(pij)
    print pij.shape

    return n_points, pij, y


def per_class_fg_tsne():
    dir_sparsity = '../experiments/graph_multi_video_with_L1/graph_multi_video_with_L1_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_0.00__ablation/results_model_249_original_class_0.0_0.5_-2/outphi'
    dir_l1 = '../experiments/graph_multi_video_with_L1/graph_multi_video_with_L1_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_20_in_out_2048_1024_feat_dim_2048_1024_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00__ablation/results_model_249_original_class_0.0_0.5_-2/outphi'
    dir_gt = '../data/ucf101/gt_vecs/just_primary_corrected/'
    anno_file = '../data/ucf101/train_test_files/test_just_primary_corrected.txt'
    

    np.random.seed(999)

    # dir_curr = dir_l1
    # per_vid = 3
    # out_dir_viz = '../scratch/model_l1_tsne_phi_viz_per_vid_'+str(per_vid)+'_100'
    # util.mkdir(out_dir_viz)

    dir_curr = dir_sparsity
    per_vid = 3
    out_dir_viz = '../scratch/model_sparse_tsne_phi_viz_per_vid_'+str(per_vid)+'_100'
    util.mkdir(out_dir_viz)


    metric = 'cosine'
    perplexity = 100
    draw_ellipse = True
    viz_after = 100
    num_epochs = 500
    feats, class_idx = get_feats_class_idx(dir_curr, dir_gt, anno_file, per_vid = per_vid)

    np.savez_compressed(os.path.join(out_dir_viz,'feats_class_idx.npz'),feats = feats, class_idx = class_idx)
    
    print feats.shape, class_idx.shape
    # raw_input()
    print 'preprocessing'
    n_points, pij2d, y = preprocess_feats_etc(feats, class_idx, metric = metric, perplexity = perplexity)
    


    
    i, j = np.indices(pij2d.shape)
    i = i.ravel()
    j = j.ravel()
    pij = pij2d.ravel().astype('float32')
    idx = i != j
    i, j, pij = i[idx], j[idx], pij[idx]

    n_topics = 2
    n_dim = 2
    
    model = VTSNE(n_points, n_topics, n_dim)
    wrap = Wrapper(model, batchsize=1024, epochs=1)
    for itr in range(num_epochs):
        wrap.fit(pij, i, j)

        # Visualize the results
        if itr%viz_after == 0 or itr==(num_epochs-1):
            embed = model.logits.weight.cpu().data.numpy()
            out_file_viz = os.path.join(out_dir_viz,'scatter_{:03d}.jpg'.format(itr))
            np.save(out_file_viz.replace('.jpg','.npy'),embed)
            
            f = plt.figure()
            if not draw_ellipse:
                plt.scatter(embed[:, 0], embed[:, 1], c=y * 1.0 / y.max())
                plt.axis('off')
                plt.savefig(out_file_viz, bbox_inches='tight')
                plt.close(f)
            else:
                # Visualize with ellipses
                var = np.sqrt(model.logits_lv.weight.clone().exp_().cpu().data.numpy())
                ax = plt.gca()
                for xy, (w, h), c in zip(embed, var, y):
                    e = Ellipse(xy=xy, width=w, height=h, ec=None, lw=0.0)
                    e.set_facecolor(plt.cm.Paired(c * 1.0 / y.max()))
                    e.set_alpha(0.5)
                    ax.add_artist(e)
                ax.set_xlim(-9, 9)
                ax.set_ylim(-9, 9)
                plt.axis('off')
                plt.savefig(out_file_viz, bbox_inches='tight')
                plt.close(f)
            visualize.writeHTMLForFolder(out_dir_viz)

def main():
    pass



    
if __name__=='__main__':
    main()