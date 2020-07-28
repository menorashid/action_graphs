import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import numpy as np
import scipy.io
import json
from pprint import pprint
from globals import class_names_activitynet

dir_meta = '../data/activitynet'
anno_dir = os.path.join(dir_meta,'annos')
json_file = os.path.join(anno_dir,'activity_net.v1-2.min.json')

def get_i3d():
    i3d_file = '../data/i3d_features/ActivityNet1.2-I3D-JOINTFeatures.npy'
    features = np.load(i3d_file)
    return features
    
def split_into_datasets(json_file):
    with open(json_file) as f:
        data = json.load(f)
    
    print data['version']
    database = data['database']
    train_data = []
    
    for k in database:
        if str(database[k]['subset'])=='testing':
            continue

        train_data.append(database[k])
    return train_data


def get_rel_anno_info():
    train_data = split_into_datasets(json_file)

    ids = []
    times_all = []
    labels_all = []
    train_val = []
    durations = []

    for vid in train_data:
        id_curr = str(vid['url'])
        id_curr = id_curr[id_curr.rindex('?')+1:].replace('=','_')
        ids.append(id_curr)
        durations.append(vid['duration'])
        labels = []
        times = []
        
        train_val.append(str(vid['subset']))

        for segment in vid['annotations']:
            time_curr = segment['segment']
            class_curr = segment['label']
            times.append(time_curr)
            labels.append(class_curr)

        # print len(times)
        times = np.array(times)
        labels_all.append(labels)
        times_all.append(times)

    labels = []
    for labels_curr in labels_all:
        labels = labels+labels_curr
    labels = list(set(labels))
    labels.sort()

    return ids, durations, train_val, labels_all, times_all, labels
    
    # # for k in labels:
    # #     print k
    # print len(labels)

    # for id_curr in ids:
    #     if id_curr.startswith('v')
    #         count = count+1


def save_npys():
    out_dir_features = os.path.join(dir_meta,'i3d')
    util.mkdir(out_dir_features)
    out_dir_train = os.path.join(out_dir_features,'train_data')
    out_dir_val = os.path.join(out_dir_features,'val_data')
    util.mkdir(out_dir_train)
    util.mkdir(out_dir_val)
    out_dirs = [out_dir_train,out_dir_val]

    features = get_i3d()
    ids, durations, train_val, labels_all, times_all, labels = get_rel_anno_info()
    assert len(features)==len(ids)

    train_val = np.array(train_val)
    train_val_bool = np.zeros(train_val.shape).astype(int)
    train_val_bool[train_val=='validation']= 1

    for idx_id_curr,id_curr in enumerate(ids):
        out_file_curr = os.path.join(out_dirs[train_val_bool[idx_id_curr]],id_curr+'.npy')
        
        if os.path.exists(out_file_curr):
            continue

        features_curr = features[idx_id_curr]
        duration_curr = durations[idx_id_curr]
        feature_len = features_curr.shape[0]
        pred_len = duration_curr*25//16
        
        diff = np.abs(pred_len - feature_len)
        # diffs.append(diff)
        if diff>2:
            # print 'Continuing',diff, feature_len, pred_len, duration_curr, train_val_bool[idx_id_curr]
            print id_curr, train_val_bool[idx_id_curr]
            continue

        # assert diff<=2

        
        # np.save(out_file_curr, features_curr)

def write_train_test_files(select_labels = None):
    out_dir_features = os.path.join(dir_meta,'i3d')
    util.mkdir(out_dir_features)
    out_dir_train = os.path.join(out_dir_features,'train_data')
    out_dir_val = os.path.join(out_dir_features,'val_data')
    util.mkdir(out_dir_train)
    util.mkdir(out_dir_val)
    out_dirs = [out_dir_train,out_dir_val]

    out_dir_anno =os.path.join(dir_meta,'train_test_files')
    util.mkdir(out_dir_anno)

    anno_files = [os.path.join(out_dir_anno,'train_select.txt'),os.path.join(out_dir_anno,'val_select.txt')]

    # anno_files = [os.path.join(out_dir_anno,'train.txt'),os.path.join(out_dir_anno,'val.txt')]

    ids, durations, train_val, labels_all, times_all,labels = get_rel_anno_info()

    if select_labels is not None:
        labels = select_labels

    print len(ids),len(set(ids))

    for dir_curr,anno_file in zip(out_dirs,anno_files):
        anno_lines = []

        npy_files = glob.glob(os.path.join(dir_curr,'*.npy'))
        for npy_file in npy_files:
            anno_curr = np.zeros((len(labels),))

            id_curr = os.path.split(npy_file)[1]
            id_curr = id_curr[:id_curr.rindex('.')]
            idx_id_curr = ids.index(id_curr)
            labels_curr = labels_all[idx_id_curr]
            labels_curr = list(set(labels_curr))

            # if len(labels_curr)>1:
            #     print labels_curr
            if select_labels is not None:
                for label_curr in labels_curr:
                    if label_curr not in labels:
                        continue
                    anno_curr[labels.index(str(label_curr))] = 1
                
                if np.sum(anno_curr)==0:
                    continue
            else:
                for label_curr in labels_curr:
                    anno_curr[labels.index(str(label_curr))] = 1
                assert np.sum(anno_curr)>0
                assert np.sum(anno_curr)==len(labels_curr)

            anno_curr = [str(int(val)) for val in anno_curr]
            line_curr = ' '.join([npy_file]+anno_curr)
            anno_lines.append(line_curr)

        print anno_file, len(anno_lines)
        util.writeFile(anno_file, anno_lines)



def write_gt_numpys():
    out_dir_gt = os.path.join(dir_meta,'gt_npys')
    util.mkdir(out_dir_gt)
    
    required_str = 'val'
    out_file = os.path.join(out_dir_gt,required_str+'_pruned.npz')
    
    ids, durations, train_val, labels_all, times_all,labels = get_rel_anno_info()

    print np.unique(train_val)
    # gt_vid_names = loaded['gtvideonames'][0]
    # gt_class_names = loaded['gt_events_class'][0]
    # gt_time_intervals = loaded['gt_time_intervals'][0]
    

    vids_to_discard = ['v_v2zVnmbPmeQ',
                        'v__MWyhJS4KbM',
                        'v_6DXH6kwMe-Q',
                        'v_a0Zlu4AvdnI',
                        'v_5UlxCwq-LOs',
                        'v_Si6LZFiQT3k',
                        'v_0dkIbKXXFzI']


    gt_vid_names =[]
    gt_class_names = []
    gt_time_intervals = []

    for idx_vid in range(len(ids)):
        if not train_val[idx_vid].startswith(required_str):
            # print 'continuing',train_val[idx_vid]
            continue

        time_rel = times_all[idx_vid]
        label_rel = labels_all[idx_vid]

        num_instances = len(time_rel)
        
        id_curr = ids[idx_vid]

        if id_curr in vids_to_discard:
            print 'problem vid found',id_curr
            continue


        id_curr = [id_curr for idx in range(num_instances)]
        
        assert len(label_rel)==len(time_rel)==len(id_curr)

        gt_vid_names+=id_curr
        
        gt_time_intervals.append(time_rel)
        
        label_rel = [str(label_curr) for label_curr in label_rel]
        gt_class_names += label_rel

    gt_time_intervals = np.concatenate(gt_time_intervals, axis = 0)
    gt_class_names = np.array(gt_class_names)
    gt_vid_names = np.array(gt_vid_names)
    print gt_class_names[0], gt_class_names.shape
    print gt_vid_names[0], gt_vid_names.shape
    print gt_time_intervals.shape,gt_time_intervals[0]

    print out_file
    np.savez(out_file,gt_class_names = gt_class_names, gt_vid_names = gt_vid_names, gt_time_intervals = gt_time_intervals)

    # data = np.load(out_file)
    # gt_class_names = data['gt_class_names']
    # gt_vid_names = data['gt_vid_names']
    # gt_time_intervals = data['gt_time_intervals']
    # print gt_class_names.shape,gt_class_names[0]
    # print gt_vid_names.shape,gt_vid_names[0]
    # print gt_time_intervals.shape,gt_time_intervals[0]


def delete_useless_videos():
    ids, durations, train_val, labels_all, times_all, labels= get_rel_anno_info()
    # ids, durations, train_val, labels_all, times_all 
    print type(ids), ids[0]

    videos = glob.glob('../data/videos/*.mp4')
    to_del = []
    to_keep = []
    for vid in videos:
        vid_id = os.path.split(vid)[1].replace('.mp4','')
        if vid_id in ids:
            to_keep.append(vid)
        else:
            to_del.append(vid)

    print len(to_keep), len(to_del)
    util.writeFile('../data/to_del_vids.txt',to_del)
    util.writeFile('../data/to_keep_vids.txt',to_keep)




def write_gt_numpys_select(select_labels):
    out_dir_gt = os.path.join(dir_meta,'gt_npys')
    util.mkdir(out_dir_gt)
    
    required_str = 'train'
    out_file = os.path.join(out_dir_gt,required_str+'_select.npz')
    
    ids, durations, train_val, labels_all, times_all,labels = get_rel_anno_info()

    print np.unique(train_val)
    # gt_vid_names = loaded['gtvideonames'][0]
    # gt_class_names = loaded['gt_events_class'][0]
    # gt_time_intervals = loaded['gt_time_intervals'][0]
    

    vids_to_discard = ['v_v2zVnmbPmeQ',
                        'v__MWyhJS4KbM',
                        'v_6DXH6kwMe-Q',
                        'v_a0Zlu4AvdnI',
                        'v_5UlxCwq-LOs',
                        'v_Si6LZFiQT3k',
                        'v_0dkIbKXXFzI']


    gt_vid_names =[]
    gt_class_names = []
    gt_time_intervals = []

    for idx_vid in range(len(ids)):
        if not train_val[idx_vid].startswith(required_str):
            # print 'continuing',train_val[idx_vid]
            continue

        time_rel = times_all[idx_vid]
        label_rel = labels_all[idx_vid]

        num_instances = len(time_rel)
        
        id_curr = ids[idx_vid]

        if id_curr in vids_to_discard:
            print 'problem vid found',id_curr
            continue


        id_curr = [id_curr for idx in range(num_instances)]
        
        assert len(label_rel)==len(time_rel)==len(id_curr)

        for idx_anno,label_rel_rel in enumerate(label_rel):
            label_rel_rel = str(label_rel_rel)
            if label_rel_rel in select_labels:
                gt_vid_names.append(id_curr[idx_anno])
                # +=id_curr
                # print time_rel,time_rel[idx_anno]
                gt_time_intervals.append(list(time_rel[idx_anno]))
                # print gt_time_intervals
                # raw_input()
                gt_class_names.append(label_rel_rel)
                # label_rel = [str(label_curr) for label_curr in label_rel]
                # gt_class_names += label_rel
    gt_time_intervals = np.array(gt_time_intervals)
    # gt_time_intervals = np.concatenate(gt_time_intervals, axis = 0)
    gt_class_names = np.array(gt_class_names)
    gt_vid_names = np.array(gt_vid_names)
    print gt_class_names[0], gt_class_names.shape
    print gt_vid_names[0], gt_vid_names.shape
    print gt_time_intervals.shape,gt_time_intervals[0]

    print out_file
    np.savez(out_file,gt_class_names = gt_class_names, gt_vid_names = gt_vid_names, gt_time_intervals = gt_time_intervals)

    # data = np.load(out_file)
    # gt_class_names = data['gt_class_names']
    # gt_vid_names = data['gt_vid_names']
    # gt_time_intervals = data['gt_time_intervals']
    # print gt_class_names.shape,gt_class_names[0]
    # print gt_vid_names.shape,gt_vid_names[0]
    # print gt_time_intervals.shape,gt_time_intervals[0]


def main():

    # print 'hello'
    
    # save_npys()

        # print vid_id
        # raw_input()





    # features = get_i3d()
    # print features.shape
    # raw_input()

    # write_gt_numpys()
    
    ids, durations, train_val, labels_all, times_all,labels = get_rel_anno_info()

    

    labels_all = [[str(val) for val in label_inner] for label_inner in labels_all]
    # print labels_all[:10]
    # raw_input()
    # ids = np.array(ids)
    # durations = np.array(durations)
    # labels_all= np.array(labels_all)
    # times_all = np.array(times_all)

    in_to_vec = [ids, durations, labels_all]
    to_vec = [[],[],[]]
    time_vec = []


    for idx_t, t in enumerate(times_all):
        # if train_val[idx_t]=='training':
        #     continue

        for idx_t_in,t_in in enumerate(t):
            time_vec.append(t_in)
            for idx_vec_curr, vec_curr in enumerate(to_vec):
                if idx_vec_curr == 2:
                    vec_curr.append(in_to_vec[idx_vec_curr][idx_t][idx_t_in])
                else:
                    vec_curr.append(in_to_vec[idx_vec_curr][idx_t])
            
    vecs = [np.array(vec_curr) for vec_curr in to_vec+[time_vec]]
    # for vec in vecs:
    #     print vec.shape, vec[0]

    [ids, durations, labels_all, times_all] = vecs

    percentage_of_videos = (times_all[:,1]-times_all[:,0])/durations

    labels_to_keep = []
    thresh = 0.5

    for label in labels:
        bin_rel = labels_all==str(label)
        rel_percent = np.sum(percentage_of_videos[bin_rel])/float(np.sum(bin_rel))
        if rel_percent<thresh:
            labels_to_keep.append(label)

    labels_to_keep.sort()
    for label in labels_to_keep:
        print label

    write_gt_numpys_select(labels_to_keep)

    # print len(labels_to_keep), len(labels)

    # write_train_test_files(select_labels = labels_to_keep)

    # for idx_lab, lab in enumerate(labels):
    #     print lab, class_names_activitynet[idx_lab]
    #     assert lab ==class_names_activitynet[idx_lab]




        # print label, rel_percent


    # print times_all[0]
    # print ids.shape, durations.shape, labels_all.shape, times_all.shape

    # percentage_of_videos = times_all[:,1]-times_all[:,0]/durations
    # print percentage_of_videos
    # raw_input()



    # for label_curr in np.unique(labels_all):
    #     rel_videos = labels_all[


    # ids = np.array(ids)
    # durations = np.array(durations)
    # np.savez('../data/activitynet/ids_durations.npz',durations = durations, ids = ids)

    # print type(labels),labels[0]
    # out_file_labels = os.path.join('../data/activitynet','gt_labels_sorted.txt')
    # util.writeFile(out_file_labels,labels)

    # write_train_test_files()
    # save_npys()
    # pass
    # explore_features()

    



    
    
    


if __name__=='__main__':
    main()