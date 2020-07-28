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
    elif dataset =='activitynet_select':
        dir_files = '../data/activitynet/train_test_files'
        n_classes = 65
        trim_preds = None
        post_pends = ['','','']
        train_file = os.path.join(dir_files, 'train_select')
        test_train_file = os.path.join(dir_files, 'val_select')
        test_file = os.path.join(dir_files, 'val_select')
            
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
    elif dataset =='ucf_untf':
        dir_files = '../data/ucf101/train_test_files'
        n_classes = 20
        trim_preds = None
        post_pends = ['','','']
        train_file = os.path.join(dir_files, 'train_untf')
        test_train_file = os.path.join(dir_files, 'test_untf')
        test_file = os.path.join(dir_files, 'test_untf')
        
        files = [train_file, test_train_file, test_file]

        
        post_pends = [pp+'.txt' for pp in post_pends]
        files = [file_curr+pp for file_curr,pp in zip(files,post_pends)]
        
        train_file, test_train_file, test_file = files
        train_data = UCF_dataset(train_file, limit)
        test_train_data = UCF_dataset(test_train_file, limit)
        test_data = UCF_dataset(test_file, None)
    elif dataset == 'ucf_cooc_per_class':
        cooc_str = '_cooc_per_class'
        dir_files = '../data/ucf101/train_test_files'
        n_classes = 20
        trim_preds = None
        post_pends = ['','','']
        train_file = os.path.join(dir_files, 'train'+cooc_str)
        test_train_file = os.path.join(dir_files, 'test'+cooc_str)
        test_file = os.path.join(dir_files, 'test'+cooc_str)
        
        files = [train_file, test_train_file, test_file]
        post_pends = [pp+'.txt' for pp in post_pends]
        files = [file_curr+pp for file_curr,pp in zip(files,post_pends)]
        
        train_file, test_train_file, test_file = files
        train_data = UCF_dataset_cooc_per_class_graph(train_file, limit)
        test_train_data = UCF_dataset_cooc_per_class_graph(test_train_file, limit)
        test_data = UCF_dataset_cooc_per_class_graph(test_file, None)

    elif dataset.startswith('ucf_cooc'):
        cooc_number = '_'.join(dataset.split('_')[2:])
        cooc_str = '_cooc_'+cooc_number
        dir_files = '../data/ucf101/train_test_files'
        n_classes = 20
        trim_preds = None
        post_pends = ['','','']
        train_file = os.path.join(dir_files, 'train'+cooc_str)
        test_train_file = os.path.join(dir_files, 'test'+cooc_str)
        test_file = os.path.join(dir_files, 'test'+cooc_str)
        
        files = [train_file, test_train_file, test_file]

        
        post_pends = [pp+'.txt' for pp in post_pends]
        files = [file_curr+pp for file_curr,pp in zip(files,post_pends)]
        
        train_file, test_train_file, test_file = files
        train_data = UCF_dataset_cooc_graph(train_file, limit)
        test_train_data = UCF_dataset_cooc_graph(test_train_file, limit)
        test_data = UCF_dataset_cooc_graph(test_file, None)

    elif dataset == 'multithumos':
        cooc_number = '_'.join(dataset.split('_')[2:])
        cooc_str = '_cooc_'+cooc_number
        dir_files = '../data/multithumos/train_test_files'
        n_classes = 65
        trim_preds = None
        post_pends = ['','','']
        train_file = os.path.join(dir_files, 'train')
        test_train_file = os.path.join(dir_files, 'test')
        test_file = os.path.join(dir_files, 'test')
        
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

    #       if attention:
    #         if multibranch>1:
    #             criterion = MultiCrossEntropyMultiBranchWithL1(class_weights= class_weights_val, loss_weights = loss_weights, num_branches = multibranch)
    #             criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    #         else:
    #             criterion = MultiCrossEntropyMultiBranchWithL1(class_weights= class_weights_val, loss_weights = loss_weights, num_branches = 1)
    #             criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    #     else:
    #     if multibranch>1:
    #         criterion = MultiCrossEntropyMultiBranch(class_weights= class_weights_val, loss_weights = loss_weights, num_branches = multibranch)
    #         criterion_str = 'MultiCrossEntropyMultiBranch'
    #     else:
    #         criterion = MultiCrossEntropy(class_weights= class_weights_val)
    #         criterion_str = 'MultiCrossEntropy'
    # else:

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

    # test_mode = test_mode or viz_mode or viz_sim
    if model_file is not None:
        [model_file, epoch_start] = model_file
    else:
        epoch_start = 0

    # print model_file, epoch_start
    # raw_input()
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr

    train_data, test_train_data, test_data, n_classes, trim_preds = get_data(dataset, limit, all_classes, just_primary, gt_vec, k_vec, test_pair = test_pair, num_similar = num_similar)
    
    network_params['n_classes']=n_classes

    train_file = train_data.anno_file
    
    # print train_file
    # print test_data.anno_file
    # print class_weights
    # raw_input()

    if class_weights :
    # and 'BinaryCrossEntropy' in criterion_str:
        # print 'bce class_weights'
        # class_weights_val = util.get_class_weights_bce(util.readLinesFromFile(train_file),n_classes)
    # elif class_weights:
        # print 'new class_weights'
        pos_weight = util.get_pos_class_weight(util.readLinesFromFile(train_file),n_classes)
        # print 'pos_weight', pos_weight
        class_weights_val = util.get_class_weights_au(util.readLinesFromFile(train_file),n_classes)
        # print 'class_weights_val', class_weights_val
        class_weights_val = [pos_weight, class_weights_val]
        # raw_input()
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
    # print out_dir_train
    # raw_input()


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
        # test_model(**test_params)
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



def ens_moredepth_concat_sim_experiments():
    model_name = 'graph_multi_video_same_F_ens_dll_moredepth_concat_sim'

    lr = [0.001, 0.001]
    multibranch = 1
    loss_weights = None
    # [1/float(multibranch)]*multibranch
    branch_to_test = -1

    k_vec = None

    gt_vec = False
    just_primary = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)
    
    epoch_stuff = [100,100]
    dataset = 'ucf'
    limit  = 500
    save_after = 10
    
    test_mode = False

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,128]
    network_params['feat_dim'] = [2048,64]
    network_params['num_graphs'] = 1
    network_params['graph_size'] = 1
    network_params['num_branches'] = multibranch
    network_params['non_lin'] = 'HT'
    network_params['non_lin_aft'] = 'RL'
    network_params['aft_nonlin']='HT_L2'
    network_params['scaling_method']='n'
    network_params['sigmoid'] = True
    post_pend = 'ABS_bias_sym'
    
    first_thresh=0.

    class_weights = True
    test_after = 5
    
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
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec)



def ens_moredepth_experiments():
    model_name = 'graph_multi_video_same_F_ens_dll_moredepth'

    lr = [0.001,0.001, 0.01]
    multibranch = 1
    loss_weights = None
    # [1/float(multibranch)]*multibranch
    branch_to_test = -1

    k_vec = None

    gt_vec = False
    just_primary = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)
    
    epoch_stuff = [300,300]
    dataset = 'ucf'
    limit  = None
    save_after = 100
    
    test_mode = False

    model_nums = [99,199,299]
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,256]
    network_params['feat_dim'] = [2048,512]
    network_params['num_graphs'] = 1
    network_params['graph_size'] = 2
    network_params['method'] = 'cos_zero_self'
    network_params['sparsify'] = [0.5]
    # ,None,None]
    # network_params['layer_bef'] = [2048,512]
    network_params['non_lin'] = 'HT'
    network_params['aft_nonlin']='HT_L2'
    network_params['sigmoid'] = True
    post_pend = 'ABS_bias'
    
    first_thresh=0.1

    class_weights = True
    test_after = 5
    
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
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec)


def ens_experiments():
    model_name = 'graph_multi_video_same_F_ens_dll'
    # model_name = 'graph_multi_video_flexF_ofe'
    # _moredepth'
    # # lr = [0.001]
    lr = [0.001,0.001, 0.001]
    multibranch = 1
    loss_weights = [1,1]
    # [1/float(6)]*3+[1/2.]
    # 
    branch_to_test = -2
    attention = True

    # model_name = 'graph_multi_video_same_i3dF_ens_sll'
    # lr = [0.001]
    # model_name = 'graph_multi_video_diff_F_ens_sll'
    # lr = [0.001,0.001]
    # loss_weights = None
    # multibranch = 1
    # branch_to_test = 0

    k_vec = None

    gt_vec = False
    just_primary = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    
    epoch_stuff = [100,100]
    dataset = 'ucf'
    limit  = 500
    save_after = 10
    
    test_mode = True
    
    # test_method = 'wtalc'
    test_method = 'wtalc'
    
    test_post_pend = '_'+test_method+'_tp_fp_conf'
    model_nums = [99]
    # range(save_after-1,epoch_stuff[1],save_after)
    # 
    # [49,99]
    # 
    retrain = False
    viz_mode = False
    viz_sim = False
    # test_post_pend = ''

    post_pend = ''
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,512]
    network_params['feat_dim'] = [2048,1024]
    # network_params['num_graphs'] = 2
    # network_params['layer_bef'] = [2048,1024]
    # network_params['sameF'] = True
    network_params['graph_size'] = 2
    network_params['method'] = 'cos_zero_self'
    # network_params['sparsify'] = list(np.arange(0.5,1.0,0.1))[::-1]
    network_params['sparsify'] = [0.5]
    network_params['graph_sum'] = attention
    # network_params['just_graph'] = True
    # network_params['background'] = False
    # loss_weights = network_params['sparsify']
    # [0.9,0.8,0.7,0.6,0.5]
    # ,0.75,0.5]
    network_params['non_lin'] = 'HT'
    network_params['aft_nonlin']='HT_L2'
    network_params['sigmoid'] = True

    post_pend = 'ABS_bias'
    
    first_thresh=0.1

    class_weights = True
    test_after = 5
    
    all_classes = False
    # just_primary = False
    # gt_vec = False

    
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
                        # test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec,
                        attention = attention,
                        test_pair = False,
                        test_post_pend = test_post_pend,
                        test_method = test_method)

def ens_Fperg_experiments():
    model_name = 'graph_multi_video_Fperg_ens_dll_moredepth'
    lr = [0.001,0.001, 0.001]
    multibranch = 1
    loss_weights = [1,1]
    branch_to_test = -1
    attention = True

    
    k_vec = None
    gt_vec = False
    just_primary = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    
    epoch_stuff = [300,300]
    dataset = 'ucf'
    limit  = 500
    save_after = 50
    
    test_mode = True
    model_nums = [49,99,149,199,249,299]

    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''
    post_pend = ''
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,128,128]
    network_params['feat_dim'] = [64,64]
    network_params['num_graphs'] = 2
    network_params['graph_size'] = 2
    network_params['method'] = 'cos_zero_self'
    network_params['sparsify'] = [0.5]
    network_params['graph_sum'] = attention
    network_params['non_lin'] = 'HT'
    network_params['aft_nonlin']='HT_l2'
    network_params['sigmoid'] = True
    post_pend = 'ABS_bias'
    
    class_weights = True
    test_after = 5
    all_classes = False
    
    first_thresh=0.1
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
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec,
                        attention = attention)


def ens_att_experiments():
    model_name = 'graph_multi_video_attention_soft'
    lr = [0.001, 0.001, 0.001]
    multibranch = 1
    loss_weights = [1]*multibranch + [0.001]
    branch_to_test = -1
    attention = True

    k_vec = None

    gt_vec = False
    just_primary = False
    all_classes = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    
    epoch_stuff = [300,300]
    dataset = 'ucf'
    limit  = 500
    save_after = 100
    
    test_mode = False

    model_nums = None
    retrain = True
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,256]
    network_params['feat_dim'] = [2048,512]
    network_params['graph_size'] = 2
    network_params['method'] = 'cos'
    network_params['att'] = 256
    # network_params['sparsify'] = [0.5]
    network_params['non_lin'] = 'HT'
    network_params['aft_nonlin']='HT_l2'
    post_pend = 'ABS_bias'
    
    first_thresh=0
    second_thresh = 0.5
    det_class = -1
    
    class_weights = True
    test_after = 5
    
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
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec,
                        attention = attention)

def testing_exp():
    # model_name = 'graph_multi_video_multi_F_joint_train_gaft'
    # lr = [0.001,0.001]
    # multibranch = 2
    # loss_weights = [0,1]
    # branch_to_test = 1


    # model_name = 'graph_multi_video_i3dF_gaft'
    # lr = [0.001]
    # loss_weights = None
    # multibranch = 1
    # branch_to_test = 0


    # model_name = 'graph_multi_video_same_F'
    # lr = [0.001,0.001]
    # loss_weights = None
    # multibranch = 1
    # branch_to_test = 0
    # for idx_class in [1]:
    # range(18,20):
    k_vec = None

    model_name = 'graph_multi_video_cooc_ofe_olg'
    lr = [0.001]
    loss_weights = None
    multibranch = 1
    branch_to_test = 0
    k_vec = 'k_100'


    gt_vec = False
    just_primary = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    
    epoch_stuff = [300,300]
    dataset = 'ucf'
    limit  = 500
    save_after = 100
    
    test_mode = False

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    
    network_params = {}
    network_params['deno'] = 8
    # network_params['pretrained'] = 'ucf'
    network_params['in_out'] = [2048,16]
    network_params['feat_dim'] = '100'
    # [2048,32]
    # 
    network_params['post_pend'] = 'negexp'
    # [2048,32]
    network_params['graph_size'] = 2
    # network_params['gk'] = 8
    network_params['method'] = 'affinity_dict'
    # network_params['num_switch'] = [5,5]
    # network_params['focus'] = 0
    network_params['sparsify'] = False
    network_params['non_lin'] = None
    # network_params['normalize'] = [True, True]
    network_params['aft_nonlin']='HT_l2'
    # network_params['attention'] = False

    post_pend = 'ABS_bias'
    
    first_thresh=0

    

    class_weights = True
    test_after = 5
    
    all_classes = False
    # just_primary = False
    # gt_vec = False

    
    
    
    second_thresh =0.5
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
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec)

def super_simple_experiment():
    # model_name = 'just_mill_flexible'
    # model_name = 'graph_perfectG'
    # model_name = 'graph_pretrained_F_random'
    # model_name = 'graph_pretrained_F_ucf_64'
    # model_name = 'graph_pretrained_F_activitynet'
    # model_name = 'graph_multi_video_pretrained_F_ucf_64_zero_self'
    # model_name = 'graph_multi_video_pretrained_F_flexible'
    model_name = 'graph_multi_video_pretrained_F_flexible_alt_train_temp'
    # model_name = 'graph_pretrained_F'
    # model_name = 'graph_sim_direct_mill_cosine'
    # model_name = 'graph_sim_i3d_sim_mat_mill'
    # model_name = 'graph_sim_mill'
    # model_name = 'graph_same_G_multi_cat'
    # model_name = 'graph_2_G_multi_cat'
    # epoch_stuff = [25,25]
    # save_after = 5

    
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    # lr = [0.001,0.001]
    lr = [0.001,0.001]
    epoch_stuff = [500,500]
    dataset = 'ucf'
    limit  = 500
    save_after = 100
    
    test_mode = True
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    in_out = None

    network_params = {}
    network_params['deno'] = 8
    # network_params['layer_sizes'] = [2048,64]
    # ,2048,64]
    network_params['pretrained'] = 'default'
    network_params['in_out'] = [2048,64,2048,64]
    network_params['graph_size'] = 2
    # network_params['k']
    # graph_size = 1
    network_params['method'] = 'cos'
    network_params['num_switch'] = [5,5]
    network_params['focus'] = 0
    network_params['sparsify'] = True
    network_params['non_lin'] = 'HT'
    network_params['normalize'] = [True, True]
    
    post_pend = 'ABS_EASYLR'
    loss_weights = None
    multibranch = 1
    branch_to_test = 1

    # in_out = [2048,64]
    # post_pend = '_'.join([str(val) for val in in_out])
    # post_pend += '_seeded'
    # graph_size = None
    # post_pend += '_new_model_fix_ht_cos_norm'

    # graph_size = 32
    # post_pend += '_bw_32_bs_'+str(graph_size)
    first_thresh=0


    

    class_weights = True
    test_after = 5
    
    all_classes = False
    just_primary = False
    gt_vec = False

    model_nums = [99]
    
    
    second_thresh =0.5
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
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test)

def separate_supervision_experiment():
    # model_name = 'just_mill_2_1024'
    # model_name = 'graph_sim_direct_mill_cosine'
    # model_name = 'graph_sim_i3d_sim_mat_mill'
    # model_name = 'graph_sim_mill'
    # model_name = 'graph_same_G_multi_cat'
    # model_name = 'graph_same_G_multi_cat_separate_supervision_unit_norm'
    model_name = 'graph_same_G_sepsup_alt_train_2_layer'
    # epoch_stuff = [25,25]
    # save_after = 5


    lr = [1e-4,1e-4,1e-4,1e-4]
    epoch_stuff = [400,400]
    dataset = 'ucf'
    limit  = 500
    deno = 8
    save_after = 25
    
    loss_weights = None
    multibranch = 1
    num_switch = 5
    branch_to_test = 1
    test_mode = True

    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''
    post_pend = '512_1024'

    class_weights = True
    test_after = 10
    all_classes = False
    just_primary = False
    model_nums = [374]
    
    second_thresh =0.5
    det_class = -1
    train_simple_mill_all_classes (model_name = model_name,
                        lr = lr,
                        dataset = dataset,
                        deno = deno,
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
                        det_class = det_class,
                        post_pend = post_pend,
                        viz_sim = viz_sim,
                        test_post_pend = test_post_pend, 
                        loss_weights = loss_weights, 
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        num_switch = num_switch)

def create_comparative_viz(dirs, class_names, dir_strs, out_dir_html):

    for class_name in class_names:
        out_file_html = os.path.join(out_dir_html, class_name+'.html')
        ims_html = []
        captions_html = []

        im_list = glob.glob(os.path.join(dirs[0],class_name, '*.jpg'))
        im_list = [os.path.split(im_curr)[1] for im_curr in im_list]
        for im in im_list:
            row_curr = [util.getRelPath(os.path.join(dir_curr,class_name,im),dir_server) for dir_curr in dirs]
            caption_curr = [dir_str+' '+im[:im.rindex('.')] for dir_str in dir_strs]
            ims_html.append(row_curr)
            captions_html.append(caption_curr)

        visualize.writeHTML(out_file_html, ims_html, captions_html, height = 150, width = 200)

def scripts_comparative():
    # dir_meta= '../experiments/graph_sim_direct_mill_ucf/all_classes_False_just_primary_True_deno_8_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.0001'
    # dir_meta = dir_meta.replace(str_replace[0],str_replace[1])

    # dirs = ['results_model_24_0_0.5/viz_sim_mat', 'results_model_24_0_0.5/viz_-1_0_0.5']
    # dirs = [os.path.join(dir_meta, dir_curr) for dir_curr in dirs]

    # dir_meta_new= '../experiments/just_mill_2_1024_ucf/all_classes_False_just_primary_True_deno_8_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.0001'
    # dir_meta_new = dir_meta_new.replace(str_replace[0],str_replace[1])
    # dirs_new= ['results_model_99_0_0.5/viz_sim_mat', 'results_model_99_0_0.5/viz_-1_0_0.5']
    # dirs += [os.path.join(dir_meta_new, dir_curr) for dir_curr in dirs_new]


    # dir_meta_new= '../experiments/graph_sim_direct_mill_ucf/all_classes_False_just_primary_True_deno_8_limit_500_cw_True_MultiCrossEntropy_100_step_100_0.1_0.0001__noRelu'
    # dir_meta_new = dir_meta_new.replace(str_replace[0],str_replace[1])
    # dirs_new= ['results_model_24_0_0.5/viz_sim_mat', 'results_model_24_0_0.5/viz_-1_0_0.5']
    # dirs += [os.path.join(dir_meta_new, dir_curr) for dir_curr in dirs_new]


    # out_dir_html = os.path.join(dir_meta,'comparative_htmls')
    # util.mkdir(out_dir_html)

    
    # dir_strs = ['graph_sim_24','graph_pred_24','mill_2_layer_sim_99', 'mill_2_layer_pred_99','graph_sim_24_nr','graph_pred_24_nr']


    dir_meta = '../experiments/graph_same_G_multi_cat_separate_supervision_unit_norm_ucf/all_classes_False_just_primary_False_deno_8_limit_500_cw_True_MultiCrossEntropyMultiBranch_200_lw_1_1_step_200_0.1_0.0001_ht_cosine_normalizedG'

    dir_meta = dir_meta.replace(str_replace[0],str_replace[1])

    dirs = ['results_model_199_0_0.5_0/viz_-1_0_0.5', 'results_model_199_0_0.5_1/viz_-1_0_0.5','results_model_199_0_0.5/viz_sim_mat']
    dirs = [os.path.join(dir_meta, dir_curr) for dir_curr in dirs]
    out_dir_html = os.path.join(dir_meta,'comparative_htmls')
    util.mkdir(out_dir_html)
    dir_strs = ['pred_0','pred_1','sim']

    create_comparative_viz(dirs, class_names, dir_strs, out_dir_html) 



def exps_for_visualizing_W():
    model_name = 'graph_multi_video_same_i3dF'
    lr = [0.001,0.001]
    
    # model_name = 'just_mill_flexible'
    # lr = [0.001,0.001]
    # epoch_stuff = [100,100]

    multibranch = 1
    attention = True
    loss_weights = [1,0.5]
    branch_to_test = -1
    
    gt_vec = False
    just_primary = False
    all_classes = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    epoch_stuff = [100,100]
    dataset = 'ucf'
    limit  = None
    save_after = 50
    
    test_mode = False
    save_outfs = True

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    
    # network_params = {}
    # network_params['deno'] = 8
    # network_params['layer_sizes'] = [2048,2]
    

    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,2]
    # network_params['feat_dim'] = [2048,64]
    network_params['graph_size'] = 2
    network_params['method'] = 'cos_zero_self'
    network_params['sparsify'] = False
    network_params['graph_sum'] = attention

    network_params['non_lin'] = None
    network_params['aft_nonlin']='HT_l2'
    post_pend = 'ABS_bias_wb'
    
    first_thresh=0
    second_thresh = 0.5
    det_class = -1
    
    class_weights = True
    test_after = 5
    
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
                        test_post_pend = test_post_pend,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        save_outfs = save_outfs,
                        attention = attention
                        )



def ens_experiments_pool():
    model_name = 'graph_multi_video_same_F_ens_pool'
    # # lr = [0.001]
    lr = [0.001, 0.001,0.001,0.001]
    multibranch = 1
    loss_weights = None
    # [1/float(multibranch)]*multibranch
    # 
    branch_to_test = -1

    # model_name = 'graph_multi_video_same_i3dF_ens_sll'
    # lr = [0.001]
    # model_name = 'graph_multi_video_diff_F_ens_sll'
    # lr = [0.001,0.001]
    # loss_weights = None
    # multibranch = 1
    # branch_to_test = 0

    k_vec = None

    gt_vec = False
    just_primary = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    
    epoch_stuff = [100,100]
    dataset = 'ucf'
    limit  = 500
    save_after = 100
    
    test_mode = False

    model_nums = [99]
    retrain = False
    viz_mode = False
    viz_sim = False
    test_post_pend = ''

    post_pend = ''
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,256]
    network_params['feat_dim'] = [2048,512]
    # network_params['layer_bef'] = [2048,1024]
    network_params['graph_size'] = 2
    network_params['method'] = 'cos_zero_self'
    network_params['sparsify'] = [0.5,'lin']
    network_params['non_lin'] = 'HT'
    network_params['aft_nonlin']='HT_L2'
    network_params['pool_method']='avg_cat'
    network_params['sigmoid'] = True
    post_pend = 'ABS_bias'
    
    first_thresh=0.1

    class_weights = True
    test_after = 5
    
    all_classes = False
    # just_primary = False
    # gt_vec = False

    
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
                        test_post_pend = test_post_pend,
                        gt_vec = gt_vec,
                        loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec)


def wsddn_simply_experiments():
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    model_name = 'wsddn_just_rgb'
    lr = [0.001, 0.001, 0.001]
    criterion_str = 'Wsddn_Loss'
    loss_weights = [1,1]
    epoch_stuff = [100,100]

    # criterion_str = 'Wsddn_Loss'
    # loss_weights = None

    epoch_stuff = [10,10]
    retrain = False

    dataset = 'ucf_untf'
    limit  = 500
    save_after = 10
    test_after = 10
    class_weights = True


    network_params = {}
    network_params['deno'] = None
    network_params['in_out'] = [1024,256]
    network_params['ret_fc'] = 0
    post_pend = ''
    # post_pend = ''
    test_method = 'original'
    test_post_pend = '_x_det'+'_'+test_method
    test_mode = True
    viz_mode = True
    branch_to_test = -2
    second_thresh = 0.5
    first_thresh = 0.
    model_nums = None
    train_simple_mill_all_classes(model_name,
                                    lr,
                                    dataset,
                                    network_params,
                                    limit,
                                    epoch_stuff=epoch_stuff,
                                    class_weights = class_weights,
                                    save_after = save_after,
                                    test_after = test_after,
                                    post_pend = post_pend,
                                    branch_to_test = branch_to_test,
                                    test_mode= test_mode,
                                    second_thresh = second_thresh,
                                    criterion_str = criterion_str,
                                    retrain = retrain,
                                    model_nums = model_nums,
                                    test_post_pend = test_post_pend,
                                    viz_mode = viz_mode,
                                    loss_weights = loss_weights,
                                    first_thresh = first_thresh,
                                    test_method = test_method)
                                    

def simple_just_mill_flexible():
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    model_name = 'just_mill_flexible'
    lr = [0.001, 0.001]
    
    # criterion_str = 'Wsddn_Loss'
    # loss_weights = [1,0.01]

    criterion_str = 'MultiCrossEntropy'
    loss_weights = None

    epoch_stuff = [100,100]
    retrain = False

    dataset = 'ucf'
    limit  = 500
    save_after = 10
    test_after = 5
    class_weights = True


    network_params = {}
    network_params['deno'] = 8
    network_params['layer_sizes'] = [2048,2048]
    # network_params['ret_fc'] = True
    post_pend = ''
    test_method = 'original'
    test_post_pend = '_'+test_method
    test_mode = True
    viz_mode = False
    branch_to_test = -1
    second_thresh = 0.5
    first_thresh = 0.
    model_nums = None
    train_simple_mill_all_classes(model_name,
                                    lr,
                                    dataset,
                                    network_params,
                                    limit,
                                    epoch_stuff=epoch_stuff,
                                    class_weights = class_weights,
                                    save_after = save_after,
                                    test_after = test_after,
                                    post_pend = post_pend,
                                    branch_to_test = branch_to_test,
                                    test_mode= test_mode,
                                    second_thresh = second_thresh,
                                    criterion_str = criterion_str,
                                    retrain = retrain,
                                    model_nums = model_nums,
                                    test_post_pend = test_post_pend,
                                    viz_mode = viz_mode,
                                    loss_weights = loss_weights,
                                    test_method = test_method)

def graph_l1_experiment_untf():
    print 'hey girl'
    # raw_input()
    model_name = 'graph_multi_video_with_L1_retF'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1,1]
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
    
    
    
    epoch_stuff = [100,100]
    dataset = 'ucf_untf'
    limit  = 750
    save_after = 50
    
    test_mode = False
    
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = None
    retrain = False
    viz_mode = False
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
    post_pend = '_difflossweights'
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

def graph_l1_experiment():
    print 'hey girl'
    # raw_input()
    model_name = 'graph_multi_video_with_L1_retF'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1.,0.,0.3]
    plot_losses = True

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # loss_weights = [1.,0.]
    # plot_losses = False

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchFakeL1_CASL'
    # loss_weights = [1.,0.5]
    # plot_losses = True

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
    # -2*num_similar
    batch_size_val = 32
    # 32
    limit  = None
    save_after = 50
    
    test_mode = False
    
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = None
    retrain = False
    viz_mode = False
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
    post_pend = '_numSim_'+str(num_similar)+'_rest_'+str(batch_size)+'_test'
    # post_pend = '_htbefwithcasl'
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
                        num_similar = num_similar)


def graph_wsddn_experiment():
    model_name = 'graph_multi_video_wsddn'
    criterion_str = 'Wsddn_Loss'
    loss_weights = None

    lr = [0.001, 0.001,0.001, 0.001]
    multibranch = 1
    # loss_weights = [1,1]
    
    branch_to_test = -1
    # attention = True

    k_vec = None

    gt_vec = False
    just_primary = False

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)
    
    epoch_stuff = [200,200]
    dataset = 'ucf'
    limit  = 500
    save_after = 50
    
    test_mode = True
    
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    test_method = 'original'
    test_post_pend = '_'+test_method+'_x_det'

    model_nums = [49]
    # [99,199,299,399,499]

    retrain = False
    viz_mode = False
    viz_sim = False

    post_pend = 'det_HT'
    
    network_params = {}
    network_params['deno'] = None
    network_params['in_out'] = [2048,512]
    network_params['feat_dim'] = [2048,1024]
    network_params['graph_size'] = 1
    network_params['method'] = 'cos_zero_self'
    network_params['sparsify'] = 0.5
    # network_params['graph_sum'] = attention
    network_params['non_lin'] = 'HT'
    network_params['aft_nonlin']='HT_L2'
    network_params['sigmoid'] = False
    post_pend = 'ABS_bias'
    first_thresh=0.
    class_weights = True
    test_after = 5
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
                        # attention = attention,
                        test_pair = False,
                        test_post_pend = test_post_pend,
                        test_method = test_method,
                        criterion_str = criterion_str)


def comparing_best_worst():
    # print 'hello hello baby'
    # loaded = np.load('../scratch/graph_nol1_feats.npz')
    # loaded = np.load('../scratch/graph_nosparse_feats.npz')
    # loaded = np.load('../scratch/check_best_worst.npz')
    # loaded = np.load('../scratch/graph_l1_graph_out.npz')
    # loaded = np.load('../scratch/graph_l1_graphOut_noZero.npz')

    type_dist = 'best_second_best'

    loaded = np.load('../scratch/graph_l1_supervise_W_outW.npz')
    print loaded.keys()
    predictions = list(loaded['predictions'])
    det_vid_names = list(loaded['det_vid_names'])
    out_fs = list(loaded['out_fs'])
    dot_records_l1 =  wtalc.getBestWorstDot( out_fs,predictions, det_vid_names,  class_names_ucf)
    
    # raw_input()

    # loaded = np.load('../scratch/just_mill_feats.npz')
    # loaded = np.load('../scratch/graph_l1_feature_noZero.npz')
    loaded = np.load('../scratch/graph_l1_supervise_W_outF.npz')
    predictions = list(loaded['predictions'])
    det_vid_names = list(loaded['det_vid_names'])
    out_fs = list(loaded['out_fs'])
    dot_records_mill =  wtalc.getBestWorstDot( out_fs,predictions, det_vid_names,  class_names_ucf,type_dist = type_dist)
        
    classes = list(dot_records_l1.keys())
    
    # out_dir = '../scratch/best_worst_dot_nosparse'
    out_dir = '../scratch/graph_l1_supervise_W_F_W_comp_'+type_dist
    title = 'Comparing '+type_dist+' Distance'
    legend_vals = ['W','F']

    # out_dir = '../scratch/best_second_best_f_dot_out_noZero'
    # title = 'Comparing Best Second Best Distance'

    util.mkdir(out_dir)
    inc = 0.1
    for class_curr in classes:
        l1_vals = dot_records_l1[class_curr]
        mill_vals = dot_records_mill[class_curr]
        min_val = min(np.min(l1_vals), np.min(mill_vals))
        max_val = max(np.max(l1_vals), np.max(mill_vals))

        bins = np.arange(min_val, max_val+inc, inc)

        l1_vals,_ = np.histogram(l1_vals, bins)
        mill_vals,_ = np.histogram(mill_vals, bins)
        print l1_vals.shape
        print mill_vals.shape
        print bins.shape
        # l1_vals = [np.histogram(dot_records_l1[label_curr],bins) for label_curr in xtick_labels]
        # mill_vals = [np.histogram(dot_records_mill[label_curr],bins) for label_curr in xtick_labels]
        xtick_labels = ['%.2f'%val for val in bins[:-1]]
        dict_vals = {}
        dict_vals[legend_vals[0]]=l1_vals
        dict_vals[legend_vals[1]]= mill_vals
        
        
        xlabel = 'Class'
        ylabel = 'Frequency of Cosine Sim Result'
        out_file = os.path.join(out_dir,class_curr+'.jpg')
        colors = ['b','g']
        
        visualize.plotGroupBar(out_file,dict_vals,xtick_labels,legend_vals,colors, xlabel=xlabel,ylabel=ylabel,title=title,width=0.5,ylim=None,loc=None)
        print out_file

    visualize.writeHTMLForFolder(out_dir)


def graph_l1_supervise_W_experiment():
    # print 'hey girl'
    # raw_input()

    model_name = 'graph_multi_video_with_L1_supervise_W'
    branch_to_test = 1
    multibranch = 2
    loss_weights = [1,1,1]

    # model_name = 'wsddn_graph_multi_video_det'
    # multibranch = 1
    # branch_to_test = -1
    # loss_weights = [1,1]

    lr = [0.001,0.001, 0.001,0.001,0.001]
    
    
    
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
    
    
    
    epoch_stuff = [100,100]
    model_file = None
    # '../experiments/graph_multi_video_with_L1_supervise_W/graph_multi_video_with_L1_supervise_W_aft_nonlin_HT_L2_non_lin_HT_aft_nonlin_feat_HT_L2_sparsify_0.5_graph_size_2_sigmoid_True_in_out_feat_2048_2048_graph_sum_True_deno_8_n_classes_20_in_out_2048_512_feat_dim_2048_1024_method_cos_ucf/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_200_step_200_0.1_0.001_0.001_0.001_0.001_0.001_lw_1.00_1.00_1.00__noZeroSelf/model_199.pt'

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
    test_post_pend = '_'+test_method+'_graph_branch_no_softmax'

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['in_out_feat'] = [2048,2048]
    network_params['aft_nonlin_feat'] = 'HT_L2'
    network_params['deno'] = 8
    network_params['in_out'] = [2048,512]
    network_params['feat_dim'] = [2048,1024]
    network_params['graph_size'] = 2
    network_params['method'] = 'cos'
    network_params['sparsify'] = 0.5
    network_params['graph_sum'] = attention
    network_params['non_lin'] = 'HT'
    network_params['aft_nonlin']='HT_L2'
    network_params['sigmoid'] = False
    post_pend = '_noZeroSelf'
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
                        model_file = model_file)

def graph_l1_supervise_W_graph_direct_experiment():
    # print 'hey girl'
    # raw_input()
    model_name = 'graph_multi_video_with_L1_supervise_W_graph_direct'
    
    lr = [0.001,0.001, 0.001,0.001]
    multibranch = 2
    loss_weights = [1,1,1]
    
    branch_to_test = 1
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
    
    
    
    epoch_stuff = [100,100]
    dataset = 'ucf'
    limit  = None
    save_after = 50
    
    test_mode = False
    
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method+'_graph_branch'

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['in_out_feat'] = [2048,2048]
    network_params['aft_nonlin_feat'] = 'HT_L2'
    network_params['deno'] = 8
    network_params['in_out'] = [2048,20]
    network_params['feat_dim'] = [2048,1024]
    network_params['graph_size'] = 2
    network_params['method'] = 'cos'
    network_params['sparsify'] = 0.5
    network_params['graph_sum'] = attention
    network_params['non_lin'] = 'HT'
    network_params['aft_nonlin']=None
    network_params['sigmoid'] = False
    post_pend = '_noZeroSelf'
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
                        test_method = test_method)



def graph_cooc_experiment():
    # print 'hey girl'
    # raw_input()
    model_name = 'graph_multi_video_cog'
    criterion_str = 'MultiCrossEntropyMultiBranchFakeL1_CASL'
    # criterion_str = 'MultiCrossEntropyMultiBranch'
    # loss_weights = [1]
    # plot_losses = False

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    loss_weights = [1,0.5]
    plot_losses = True

    lr = [0.001,0.001]
    multibranch = 1
    # loss_weights = [1,1]
    
    branch_to_test = -2
    print 'branch_to_test',branch_to_test
    attention = False

    k_vec = None

    gt_vec = False
    just_primary = False

    seed = 999
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    
    
    epoch_stuff = [100,200]
    dataset = 'ucf_cooc_25'
    limit  = None
    save_after = 50
    
    test_mode = False
    
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,1024]
    network_params['feat_ret']=True
 
    # network_params['graph_size'] = 2
    # network_params['method'] = 'cos'
    network_params['sparsify'] = 'mid'
    # network_params['graph_sum'] = attention
    # network_params['non_lin'] = 'HT'
    network_params['aft_nonlin']='RL_L2'
    # network_params['sigmoid'] = False
    post_pend = '_regulardivsumrow'
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


def wtalc_us_experiment():
    print 'hey girl'
    # raw_input()
    model_name = 'wtalc_multi_video_with_L1'
    criterion_str = 'MultiCrossEntropyMultiBranchFakeL1_CASL'
    loss_weights = [1,0.5]
    plot_losses = True

    lr = [0.001,0.001]
    multibranch = 1
    
    branch_to_test = -2
    print 'branch_to_test',branch_to_test

    attention = False
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
    limit  = None
    save_after = 50
    
    test_mode = True
    
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = [249]
    retrain = False
    viz_mode = True
    viz_sim = False

    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,2048]
    network_params['feat_ret']=True
 
    network_params['aft_nonlin']='RL_L2'
    post_pend = '_caslexp'
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


def graph_sim_and_cooc_experiment():
    print 'hey girl'
    # raw_input()
    model_name = 'graph_multi_video_L1_retF_cog'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1.,0.5]
    plot_losses = True

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # loss_weights = [1,1]
    # plot_losses = False

    lr = [0.001,0.001, 0.001,0.001,0.001]
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
    
    
    
    epoch_stuff = [100,100]
    dataset = 'ucf_cooc_25'
    limit  = None
    save_after = 50
    
    test_mode = False
    
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=True
    network_params['graph_size'] = 2
    network_params['method'] = 'cos'
    network_params['sparsify'] = [0.5,'mid']
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = False
    network_params['smax'] = False
    post_pend = '_nosmaxcombonosig'
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


def graph_sim_and_cooc_per_class_merged_experiment():
    print 'hey girl'
    # raw_input()
    # model_name = 'graph_multi_video_L1_retF_cog_pcmfc_branched'
    # loss_weights = [1.,1.,1.,1.,0.5]
    # multibranch = 3
    model_name = 'graph_multi_video_L1_retF_cog_pcmfc'
    loss_weights = [1.,1.,0.5]
    multibranch = 1
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    plot_losses = True

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # loss_weights = [1,1]
    # plot_losses = False

    lr = [0.001,0.001, 0.001,0.001,0.001]
    
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
    
    
    
    epoch_stuff = [200,200]
    dataset = 'ucf_cooc_per_class_merged'
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

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,1024]
    network_params['in_out_gco'] = [2048,64]
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=True
    network_params['graph_size'] = 2
    network_params['method'] = 'cos'
    network_params['sparsify'] = [0.5,'mid']
    network_params['graph_sum'] = attention
    # network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    # network_params['sigmoid'] = True
    network_params['smax'] = 'add'
    post_pend = '_add'
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

def graph_cooc_per_class_experiment():
    # print 'hey girl'
    # raw_input()
    model_name = 'graph_multi_video_cog_per_class_merged_fc'
    # criterion_str = 'MultiCrossEntropyMultiBranch'
    # loss_weights = [1]
    # plot_losses = False

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # loss_weights = [1,1]
    # plot_losses = False

    lr = [0.001,0.001]
    multibranch = 1
    # loss_weights = [1,1]
    
    branch_to_test = -2
    print 'branch_to_test',branch_to_test
    attention = False

    k_vec = None

    gt_vec = False
    just_primary = False

    seed = 999
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    
    
    epoch_stuff = [100,100]
    dataset = 'ucf_cooc_per_class_merged'
    limit  = None
    save_after = 50
    
    test_mode = False
    
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,64]
    network_params['sparsify'] = 'mid'
    network_params['aft_nonlin']='RL_L2'
    post_pend = '_testing'
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
                        # loss_weights = loss_weights,
                        multibranch = multibranch,
                        branch_to_test = branch_to_test,
                        k_vec = k_vec,
                        attention = attention,
                        test_pair = False,
                        test_post_pend = test_post_pend,
                        test_method = test_method)
    # ,
    #                     criterion_str = criterion_str,
    #                     plot_losses = plot_losses)



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
    loss_weights = [1.,1.]
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
    viz_sim = False

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


def graph_size_study():
    print 'hey girl'
    # raw_input()
    model_name = 'graph_multi_video_with_L1_retF'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1.,1.,0.5]
    plot_losses = True

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # loss_weights = [1.,0.]
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
    
    
    
    epoch_stuff = [250,250]
    dataset = 'ucf'
    num_similar = 0
    batch_size = 32
    batch_size_val = 32
    limit  = 750
    save_after = 50
    
    test_mode = False
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}

    network_params['deno'] = 8
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=True
 
    network_params['graph_size'] = 32
    network_params['method'] = 'cos'
    network_params['sparsify'] = 0.5
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = False
    post_pend = '_multigraphexp'
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
                        num_similar = num_similar)



def main():
    graph_l1_experiment_best_yet()
    # wtalc_us_experiment()
    
    # graph_size_study()
    # graph_ablation_study()
    # graph_sim_and_cooc_per_class_merged_experiment()
    # graph_cooc_per_class_experiment()
    # graph_sim_and_cooc_experiment()
    
    # graph_cooc_experiment()
    # graph_l1_experiment()
    
    # graph_l1_supervise_W_graph_direct_experiment()
    # graph_l1_supervise_W_experiment()
    # comparing_best_worst()
    # return

    # numpy.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)

    # class_list = class_names_ucf
    # precisions, ap, class_names = wtalc.getTopKPrecRecall(1, predictions, det_vid_names, class_list)
    # iou = [1., 1.]
    # for idx,p in enumerate(precisions):
    #     p.append(ap[idx]) 

    # aps = np.array(precisions).T
    # print aps
    # print class_names

    # aps[:-1,:] = aps[:-1,:]*100
    # class_names.append('Average')
    # et.print_overlap(aps, class_names, iou, [])

    # simple_just_mill_flexible()
    # graph_wsddn_experiment()
    # graph_l1_experiment()

    # simple_just_mill_flexible()
    # wsddn_simply_experiments()
    # scripts_comparative()
    # separate_supervision_experiment()
    # super_simple_experiment()
    # testing_exp()
    # ens_experiments()
    # ens_Fperg_experiments()
    # ens_experiments_pool()
    # ens_moredepth_experiments()
    # ens_att_experiments()
    # ens_moredepth_concat_sim_experiments()
    # exps_for_visualizing_W()

if __name__=='__main__':
    main()
