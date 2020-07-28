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


def make_clusters(train_file, n_per_video, n_clusters,out_dir):
    npy_files, anno_all = readTrainTestFile(train_file)
    feat_to_keep = []
    for npy_file in npy_files:
        feat_curr = np.load(npy_file)
        feat_to_keep.append(feat_curr[::n_per_video,:])

    feat_to_keep = np.concatenate(feat_to_keep, axis = 0)
    mean = np.mean(feat_to_keep,axis = 0, keepdims = True)
    std = np.std(feat_to_keep,axis = 0, keepdims = True)
    print mean.shape, std.shape

    feat_to_keep = (feat_to_keep -mean)/std
    print 'fitting'
    kmeans = KMeans(n_clusters = n_clusters).fit(feat_to_keep)
    out_file_mean_std = os.path.join(out_dir,'mean_std.npz')
    np.savez_compressed(out_file_mean_std, mean = mean, std = std)
    out_file_kmeans = os.path.join(out_dir,'kmeans.joblib')
    joblib.dump(kmeans,out_file_kmeans)


def save_kmean_labels(file_curr, k_means_dir, out_dir):
    npy_files, anno_all = readTrainTestFile(file_curr)
    out_file_mean_std = os.path.join(k_means_dir,'mean_std.npz')
    # np.savez_compressed(out_file_mean_std, mean = mean, std = std)
    out_file_kmeans = os.path.join(k_means_dir,'kmeans.joblib')
    kmeans = joblib.load(out_file_kmeans)
    mean_std = np.load(out_file_mean_std)
    mean = mean_std['mean']
    std = mean_std['std']
    for npy_file in npy_files:
        feat_curr = np.load(npy_file)
        feat_curr = (feat_curr-mean)/std
        # print feat_curr.shape
        labels_curr = kmeans.predict(feat_curr)
        print labels_curr.shape, np.min(labels_curr), np.max(labels_curr)

        out_file = os.path.join(out_dir, os.path.split(npy_file)[1])
        print out_file
        np.save(out_file, labels_curr)

def set_k_mul(k_count,labels):
    k = k_count.shape[0]
    for i in range(k):
        i_count = np.sum(labels==i)
        k_count[i,i] += i_count
        for j in range(i+1,k):
            j_count = np.sum(labels==j)
            j_count = i_count*j_count
            k_count[i,j] += j_count
            k_count[j,i] += j_count
    return k_count

def normalize_k_mul(k_count):
    k = k_count.shape[0]
    for i in range(k):
        self_count = max(1,k_count[i,i])
        k_count[i,:] = k_count[i,:]/self_count
        k_count[:,i] = k_count[:,i]/self_count
        k_count[i,i] = 1
    return k_count

def set_k_int(k_count,labels):
    k = k_count.shape[0]
    for i in range(k):
        i_count = np.sum(labels==i)
        k_count[i,i] += i_count
        for j in range(i+1,k):
            j_count = np.sum(labels==j)
            j_count = min(i_count,j_count)
            k_count[i,j] += j_count
            k_count[j,i] += j_count
    return k_count

def normalize_k_union(k_count):
    k = k_count.shape[0]
    # print k_count[10:12,10:12]
    for i in range(k):
        for j in range(i+1,k):
            div = k_count[i,i]+k_count[j,j]
            if div==0:
                assert k_count[i,j]==0
                assert k_count[j,i]==0
            else:
                k_count[i,j] = k_count[i,j]/div
                k_count[j,i] = k_count[j,i]/div
            # k_count[:,i] = k_count[:,i]/self_count
        k_count[i,i] = 1
    # print k_count[10:12,10:12]

    return k_count

def getting_edge_weights(file_curr, out_dir_labels,out_dir,k, set_k = set_k_mul, normalize_k = normalize_k_mul):
    npy_files, anno_all = readTrainTestFile(file_curr)
    k_count = np.zeros((len(class_names),k,k))
    k_count_big = np.zeros((k,k))

    for npy_file,anno_curr in zip(npy_files,anno_all):
        label_file = os.path.join(out_dir_labels, os.path.split(npy_file)[1])
        labels = np.load(label_file)
        
        k_count_big = set_k(k_count_big,labels)

        for gt_idx in np.where(anno_curr)[0]:
            k_count[gt_idx] = set_k(k_count[gt_idx],labels)

        
    k_count_big = normalize_k(k_count_big)
    print k_count_big.shape
    
    out_file = os.path.join(out_dir,'all_classes_mul.npy')
    np.save(out_file, k_count_big )

    out_file = os.path.join(out_dir,'all_classes_mul.jpg')
    visualize.saveMatAsImage(k_count_big, out_file)

    for class_idx in range(len(class_names)):
        k_count[class_idx] = normalize_k(k_count[class_idx])
        class_name = class_names[class_idx]
        
        out_file = os.path.join(out_dir,class_name+'.npy')
        np.save(out_file, k_count[class_idx])

        out_file = os.path.join(out_dir,class_name+'.jpg')
        visualize.saveMatAsImage(k_count[class_idx], out_file)
    visualize.writeHTMLForFolder(out_dir)

        


def get_gt_vector(vid_name, out_shape_curr, class_idx, loaded):
    
    class_name = class_names[class_idx]
    gt_vid_names_all = loaded['gtvideonames'][0]
    gt_class_names = loaded['gt_events_class'][0]
    gt_time_intervals = loaded['gt_time_intervals'][0]
    gt_time_intervals = np.array([a[0] for a in gt_time_intervals])
    
    # print class_name
    bin_keep = np.array(gt_vid_names_all) == vid_name
    # print np.where(bin_keep)[0]
    # print gt_vid_names_all[bin_keep]
    # print 'bef',gt_time_intervals[bin_keep]
    # print gt_class_names[bin_keep], class_name
    bin_keep = np.logical_and(bin_keep,gt_class_names==class_name)
    
    # print np.where(bin_keep)[0]
    # print 'aft',gt_time_intervals[bin_keep]

    gt_time_intervals = gt_time_intervals[bin_keep]
    # print gt_time_intervals
    # print gt_class_names[bin_keep]
    # print np.sum(gt_class_names==class_name)


    det_times = np.array(range(0,out_shape_curr))*16./25.
    
    gt_vals = np.zeros(det_times.shape)
    for gt_time_curr in gt_time_intervals:
        idx_start = np.argmin(np.abs(det_times-gt_time_curr[0]))
        idx_end = np.argmin(np.abs(det_times-gt_time_curr[1]))
        gt_vals[idx_start:idx_end] = 1

    # if gt_return:
    #     return gt_vals, det_times,gt_time_intervals
    # else:
    return gt_vals, det_times


def double_check_anno(file_curr, out_file_curr, is_test ):
    npy_files, anno_all = readTrainTestFile(file_curr)
    out_lines = []

    class_name = class_names[0]
    if is_test:
        mat_file = os.path.join('../TH14evalkit','mat_files', class_name+'_test.mat')
    else:
        mat_file = os.path.join('../TH14evalkit', class_name+'.mat')

    loaded = scipy.io.loadmat(mat_file)


    for idx_npy_file, (npy_file, anno) in enumerate(zip(npy_files,anno_all)):
        if idx_npy_file%10==0:
            print idx_npy_file,len(npy_files)
        # anno_curr = np.where(anno)[0]
        vid_name = os.path.split(npy_file)[1]
        vid_name = vid_name[:vid_name.rindex('.')]
        out_shape_curr = np.load(npy_file).shape[0]

        found = []
        for class_idx in range(20):
            gt_vec, _ = get_gt_vector(vid_name, out_shape_curr, class_idx, loaded)

            if np.sum(gt_vec)>0:
                found.append(1)
            else:
                found.append(0)

        # print anno
        # print found
        line_curr = ' '.join([str(val) for val in [npy_file]+found])
        out_lines.append(line_curr)
        if not np.all(np.array(anno)==np.array(found)):
            print vid_name
            print anno
            print found

    if out_file_curr is not None:
        util.writeFile(out_file_curr,out_lines)

        # raw_input()


def script_correct_anno():
    dir_train_test_files = '../data/ucf101/train_test_files'
    
    train_file = os.path.join(dir_train_test_files,'train_corrected.txt')
    out_train_file = os.path.join(dir_train_test_files,'train_ultra_correct.txt')
    # double_check_anno(train_file, out_train_file, False)
    # print 'new file!', out_train_file
    # double_check_anno(out_train_file, None, False)

    # test_file = os.path.join(dir_train_test_files,'test_corrected.txt')
    out_test_file = os.path.join(dir_train_test_files,'test_ultra_correct.txt')
    # double_check_anno(test_file, out_test_file, True)
    # print 'new file!', out_test_file
    double_check_anno(out_test_file, None, True)


def write_train_test_files(train_file, post_pend, out_dir_labels):
    out_file = train_file[:train_file.rindex('.')]+'_'+post_pend+'.txt'
    npy_files, anno_all = readTrainTestFile(train_file)
    out_lines = []
    for npy_file, anno_curr in zip(npy_files, anno_all):
        label_file = os.path.join(out_dir_labels,os.path.split(npy_file)[1])
        assert os.path.exists(label_file)
        line_curr = ' '.join([str(val) for val in [npy_file, label_file]+ anno_curr])
        # print line_curr
        out_lines.append(line_curr)
    
    util.writeFile(out_file, out_lines) 

    

def save_neg_cooc_graphs(out_dir):
    all_file = os.path.join(out_dir,'all_classes_mul.npy')
    all_cooc = np.load(all_file)
    for class_name in class_names:
        in_file = os.path.join(out_dir,class_name+'.npy')
        curr_cooc = np.load(in_file)
        out_cooc = curr_cooc - all_cooc
        out_cooc = out_cooc + np.eye(out_cooc.shape[0])
        out_file = os.path.join(out_dir,class_name+'neg.jpg')
        visualize.saveMatAsImage(out_cooc, out_file)
        # print 'curr_cooc',curr_cooc.shape,np.min(curr_cooc),np.max(curr_cooc)
        # print 'out_cooc',out_cooc.shape,np.min(out_cooc),np.max(out_cooc)
        # print 'all_cooc',all_cooc.shape,np.min(all_cooc),np.max(all_cooc)
        # print out_file
        out_file = os.path.join(out_dir,class_name+'neg.npy')
        np.save(out_file, out_cooc)
    visualize.writeHTMLForFolder(out_dir)

def save_neg_exp_cooc_graphs(out_dir):
    for class_name in class_names:
        in_file = os.path.join(out_dir,class_name+'neg.npy')
        curr_cooc = np.load(in_file)
        print np.min(curr_cooc),np.max(curr_cooc)
        out_cooc = np.exp(curr_cooc-1)
        print np.min(out_cooc),np.max(out_cooc)

        
        out_file = os.path.join(out_dir,class_name+'negexp.jpg')
        visualize.saveMatAsImage(out_cooc, out_file)
        # print out_file
        # print 'curr_cooc',curr_cooc.shape,np.min(curr_cooc),np.max(curr_cooc)
        # print 'out_cooc',out_cooc.shape,np.min(out_cooc),np.max(out_cooc)
        # print 'all_cooc',all_cooc.shape,np.min(all_cooc),np.max(all_cooc)
        
        out_file = os.path.join(out_dir,class_name+'negexp.npy')
        print out_file
        np.save(out_file, out_cooc)
        # raw_input()
    visualize.writeHTMLForFolder(out_dir)


def main():
    dir_train_test_files = '../data/ucf101/train_test_files'
    
    train_file = os.path.join(dir_train_test_files,'train_ultra_correct.txt')
    test_file = os.path.join(dir_train_test_files,'test_ultra_correct.txt')
    n_per_video = 3
    k = 100
    post_pend = 'k_'+str(k)

    out_dir_meta = '../data/ucf101/kmeans'
    util.mkdir(out_dir_meta)
    dir_curr = '_'.join([str(val) for val in [n_per_video,k]])
    dir_curr = os.path.join(out_dir_meta, dir_curr)
    util.mkdir(dir_curr)
    out_dir_labels = os.path.join(dir_curr,'npy_labeled')
    out_dir_edges = out_dir_labels
    # out_dir_edges =  os.path.join(out_dir_labels,'int_union')
    util.mkdir(out_dir_labels)
    util.mkdir(out_dir_edges)

    # make_clusters(train_file, n_per_video,k, dir_curr)
    # save_kmean_labels(train_file,dir_curr, out_dir_labels)
    # save_kmean_labels(test_file,dir_curr, out_dir_labels)

    # getting_edge_weights(train_file, out_dir_labels, out_dir_edges, k)
    # getting_edge_weights(train_file, out_dir_labels, out_dir_edges, k, set_k = set_k_int, normalize_k = normalize_k_union)
    # save_neg_cooc_graphs(out_dir_edges)
    save_neg_exp_cooc_graphs(out_dir_edges)
    
    # write_train_test_files(train_file, post_pend, out_dir_labels)
    # write_train_test_files(test_file, post_pend, out_dir_labels)


    print 'hello'

if __name__=='__main__':
    main()