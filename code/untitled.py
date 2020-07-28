def main():
    anno_file = '../data/activitynet/train_test_files/val.txt'
    anno_file = '../data/ucf101/train_test_files/test.txt'
    anno_file = '../data/charades/train_test_files/i3d_charades_both_test.txt'

    
    
    


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


