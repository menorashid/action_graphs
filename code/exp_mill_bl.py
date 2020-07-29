# from train_test_mill import *
from train_model_new import *
import models
# from criterions import *
import criterions
import os
from helpers import util,visualize
from dataset import *
import numpy as np
import torch
from globals import * 

def get_data(dataset, limit, all_classes, just_primary, gt_vec, k_vec, test_pair = False, num_similar = 0):

    if dataset =='ucf':
        dir_files = '../data/ucf101/train_test_files'
        n_classes = 20
        trim_preds = None
        post_pends = ['','','']
        train_file = os.path.join(dir_files, 'train')
        test_train_file = os.path.join(dir_files, 'test')
        if test_pair:
            test_file = os.path.join(dir_files, 'test_pair_rand20')
        else:
            test_file = os.path.join(dir_files, 'test')
            
        files = [train_file, test_train_file, test_file]

        if all_classes:
            n_classes = 101
            post_pends = [pp+val for pp,val in zip(post_pends,['_all','_all',''])]
            classes_all_list = util.readLinesFromFile(os.path.join(dir_files,'classes_all_list.txt'))
            classes_rel_list = util.readLinesFromFile(os.path.join(dir_files,'classes_rel_list.txt'))
            trim_preds = [classes_all_list,classes_rel_list]

        if just_primary:
            post_pends = [pp+val for pp,val in zip(post_pends,['_just_primary','_just_primary','_just_primary'])]
        
        # post_pends = [pp+val for pp,val in zip(post_pends,['_corrected','_corrected','_corrected'])]
        # if not all_classes:
        #     post_pends = [pp+val for pp,val in zip(post_pends,['_ultra_correct','_ultra_correct','_ultra_correct'])]
                

        if gt_vec:
            post_pends = [pp+val for pp,val in zip(post_pends,['_gt_vec','_gt_vec','_gt_vec'])]

        if k_vec is not None:
            # print 
            post_pends = [pp+val for pp,val in zip(post_pends,['_'+k_vec]*3)]


        post_pends = [pp+'.txt' for pp in post_pends]
        files = [file_curr+pp for file_curr,pp in zip(files,post_pends)]
        
        train_file, test_train_file, test_file = files
        # test_file = '../data/ucf101/train_test_files/test_onlyMultiFromGt.txt'
        if gt_vec or (k_vec is not None):
            train_data = UCF_dataset_gt_vec(train_file, limit)
            test_train_data = UCF_dataset_gt_vec(test_train_file, limit)
            test_data = UCF_dataset_gt_vec(test_file, None)
        else:
            # all_classes
            # train_data = UCF_dataset(train_file, limit)
            # test_train_data = UCF_dataset(test_train_file, limit)

            if num_similar>0:
                train_data = UCF_dataset_withNumSimilar(train_file, limit, num_similar = num_similar)
                # test_train_data =  UCF_dataset_withNumSimilar(test_train_file, limit, num_similar = num_similar)
            else:
                train_data = UCF_dataset(train_file, limit)
                # test_train_data = UCF_dataset(test_train_file, limit)

            test_train_data =  UCF_dataset(test_train_file, limit)
            test_data = UCF_dataset(test_file, None)
    elif dataset =='activitynet':
        dir_files = '../data/activitynet/train_test_files'
        n_classes = 100
        trim_preds = None
        post_pends = ['','','']
        train_file = os.path.join(dir_files, 'train')
        test_train_file = os.path.join(dir_files, 'val')
        test_file = os.path.join(dir_files, 'val')
            
        files = [train_file, test_train_file, test_file]

        post_pends = [pp+'.txt' for pp in post_pends]
        files = [file_curr+pp for file_curr,pp in zip(files,post_pends)]
        
        train_file, test_train_file, test_file = files
        if num_similar>0:
            train_data = UCF_dataset_withNumSimilar(train_file, limit, num_similar = num_similar)
            # test_train_data =  UCF_dataset_withNumSimilar(test_train_file, limit, num_similar = num_similar)
        else:
            train_data = UCF_dataset(train_file, limit)
            # test_train_data = UCF_dataset(test_train_file, limit)
        test_train_data =  UCF_dataset(test_train_file, limit)
        test_data =  UCF_dataset(test_file, None)
    elif dataset.startswith('charades'):
        pre_pend = '_'.join(dataset.split('_')[1:])
        dir_files = '../data/charades/train_test_files'
        n_classes = 157
        trim_preds = None
        post_pends = ['_wmiss','_wmiss','_wmiss']
        train_file = os.path.join(dir_files, pre_pend+'_train')
        test_train_file = os.path.join(dir_files, pre_pend+'_test')
        test_file = os.path.join(dir_files, pre_pend+'_test')
        
        files = [train_file, test_train_file, test_file]

        
        post_pends = [pp+'.txt' for pp in post_pends]
        files = [file_curr+pp for file_curr,pp in zip(files,post_pends)]
        
        train_file, test_train_file, test_file = files
        # train_data = UCF_dataset(train_file, limit)
        if num_similar>0:
            train_data = UCF_dataset_withNumSimilar(train_file, limit, num_similar = num_similar, just_one = False)
            # test_train_data =  UCF_dataset_withNumSimilar(test_train_file, limit, num_similar = num_similar)
        else:
            train_data = UCF_dataset(train_file, limit)
        test_train_data = UCF_dataset(test_train_file, limit)
        test_data = UCF_dataset(test_file, None)

    return train_data, test_train_data, test_data, n_classes, trim_preds

def get_criterion(criterion_str,attention,class_weights_val,  loss_weights, multibranch, num_similar = 0):

    args = {'class_weights' : class_weights_val, 
            'loss_weights' : loss_weights, 
            'num_branches' : multibranch} 

    if num_similar>0:
        args['num_similar'] = num_similar

    if criterion_str is None:
        if attention:
            criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
        elif multibranch>1:
            criterion_str = 'MultiCrossEntropyMultiBranch'
        else:
            criterion_str = 'MultiCrossEntropy'
    
    criterion_class = getattr(criterions, criterion_str)
    criterion = criterion_class(**args)


    return criterion, criterion_str

def train_simple_mill_all_classes(model_name,
                                    lr,
                                    dataset,
                                    network_params,
                                    limit,
                                    epoch_stuff=[30,60],
                                    res=False,
                                    class_weights = False,
                                    batch_size = 32,
                                    batch_size_val = 32,
                                    save_after = 1,
                                    model_file = None,
                                    gpu_id = 0,
                                    exp = False,
                                    test_mode = False,
                                    test_after = 1,
                                    all_classes = False,
                                    just_primary = False,
                                    model_nums = None,
                                    retrain = False,
                                    viz_mode = False,
                                    det_class = -1,
                                    second_thresh = 0.5,
                                    first_thresh = 0,
                                    post_pend = '',
                                    viz_sim = False,
                                    test_post_pend = '', 
                                    multibranch = 1,
                                    loss_weights = None,
                                    branch_to_test = 0,
                                    gt_vec = False,
                                    k_vec = None,
                                    attention = False,
                                    save_outfs = False,
                                    test_pair = False, 
                                    criterion_str= None, 
                                    test_method = 'original',
                                    plot_losses = False,
                                    num_similar = 0,
                                    det_test = False):

    num_epochs = epoch_stuff[1]

    if model_file is not None:
        [model_file, epoch_start] = model_file
    else:
        epoch_start = 0

    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr

    train_data, test_train_data, test_data, n_classes, trim_preds = get_data(dataset, limit, all_classes, just_primary, gt_vec, k_vec, test_pair = test_pair, num_similar = num_similar)
    
    network_params['n_classes']=n_classes

    train_file = train_data.anno_file
    

    if class_weights :
        pos_weight = util.get_pos_class_weight(util.readLinesFromFile(train_file),n_classes)
        class_weights_val = util.get_class_weights_au(util.readLinesFromFile(train_file),n_classes)
        class_weights_val = [pos_weight, class_weights_val]
    else:
        class_weights_val = None

    
    criterion, criterion_str = get_criterion(criterion_str,attention,class_weights_val,  loss_weights, multibranch, num_similar = num_similar)
    
    init = False

    
    out_dir_meta = os.path.join('../experiments',model_name)
    util.mkdir(out_dir_meta)

    out_dir_meta_str = [model_name]
    for k in network_params.keys():
        out_dir_meta_str.append(k)
        if type(network_params[k])==type([]):
            out_dir_meta_str.extend(network_params[k])
        else:
            out_dir_meta_str.append(network_params[k])
    out_dir_meta_str.append(dataset)
    out_dir_meta_str = '_'.join([str(val) for val in out_dir_meta_str])
    
    out_dir_meta = os.path.join(out_dir_meta,out_dir_meta_str)
    # print out_dir_meta
    util.mkdir(out_dir_meta)
    


    strs_append_list = ['all_classes',all_classes,'just_primary',just_primary,'limit',limit,'cw',class_weights, criterion_str, num_epochs]+dec_after+lr
    
    if loss_weights is not None:
        strs_append_list += ['lw']+['%.2f' % val for val in loss_weights]
    
    strs_append_list+=[post_pend] if len(post_pend)>0 else []
    strs_append = '_'.join([str(val) for val in strs_append_list])

    out_dir_train =  os.path.join(out_dir_meta,strs_append)
    final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
    
    if os.path.exists(final_model_file) and not test_mode and not retrain:
        print 'skipping',final_model_file
        return 
    else:
        print 'not skipping', final_model_file

    test_params_core = dict(
                trim_preds = trim_preds,
                second_thresh = second_thresh,
                first_thresh = first_thresh,
                multibranch = multibranch,
                branch_to_test = branch_to_test,
                dataset = dataset, 
                test_pair = test_pair,
                save_outfs = False,
                test_method = test_method)
    train_params = dict(out_dir_train = out_dir_train,
                train_data = train_data,
                test_data = test_train_data,
                test_args = test_params_core,
                batch_size = batch_size,
                batch_size_val = batch_size_val,
                num_epochs = num_epochs,
                save_after = save_after,
                disp_after = 1,
                plot_after = 1,
                test_after = test_after,
                lr = lr,
                dec_after = dec_after,
                model_name = model_name,
                criterion = criterion,
                gpu_id = gpu_id,
                num_workers = 0,
                model_file = model_file,
                epoch_start = epoch_start,
                network_params = network_params, 
                multibranch = multibranch,
                plot_losses = plot_losses,
                det_test = det_test)
    

    if not test_mode:
        train_model_new(**train_params)
    
    if model_nums is None :
        model_nums = [num_epochs-1] 
    
    for model_num in model_nums:

        print 'MODEL NUM',model_num
        # if save_outfs:
        #     save_outfs = os.path.join(out_dir_train, str(model_num)+'_out')
        #     util.mkdir(save_outfs)

        test_params = dict(out_dir_train = out_dir_train,
                model_num = model_num,
                test_data = test_data,
                batch_size_val = batch_size_val,
                criterion = criterion,
                gpu_id = gpu_id,
                num_workers = 0,
                trim_preds = trim_preds,
                visualize = False,
                det_class = det_class,
                second_thresh = second_thresh,
                first_thresh = first_thresh,
                post_pend=test_post_pend,
                multibranch = multibranch,
                branch_to_test =branch_to_test,
                dataset = dataset, 
                save_outfs = save_outfs,
                test_pair = test_pair,
                test_method = test_method)
        test_model(**test_params)
        if viz_mode:
            test_params = dict(out_dir_train = out_dir_train,
                    model_num = model_num,
                    test_data = test_data,
                    batch_size_val = batch_size_val,
                    criterion = criterion,
                    gpu_id = gpu_id,
                    num_workers = 0,
                    trim_preds = trim_preds,
                    visualize = True,
                    det_class = det_class,
                    second_thresh = second_thresh,
                    first_thresh = first_thresh,
                    post_pend=test_post_pend,
                    multibranch = multibranch,
                    branch_to_test =branch_to_test,
                    dataset = dataset)
            test_model(**test_params)
            # test_params = dict(out_dir_train = out_dir_train,
            #         model_num = model_num,
            #         test_data = test_data,
            #         batch_size_val = batch_size_val,
            #         gpu_id = gpu_id,
            #         num_workers = 0,
            #         second_thresh = second_thresh,
            #         first_thresh = first_thresh,
            #         dataset = dataset)
            # print 'visualizing'
            # visualize_sim_mat(**test_params)


def main():
    pass

if __name__=='__main__':
    main()
