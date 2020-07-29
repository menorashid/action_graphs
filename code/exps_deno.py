from exp_mill_bl import train_simple_mill_all_classes
import torch
import random
import numpy as np
import scipy

def thumos(deno):
    print 'hey girl'
    model_name = 'graph_multi_video_with_L1_retF_tanh'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1,1]
    plot_losses = True

    # learning rate for [\phi, graph layer, final linear layer]
    lr = [0.001,0.001, 0.001] 
    
    seed = 999
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # string to switch between datasets
    dataset = 'ucf'
    
    #num segments to include per video during training. in case of memory problems
    limit  = None

    # epoch_stuff = [num epochs after which to reduce lr, total num epochs]
    epoch_stuff = [250,250]
    save_after = 50
    # string to append to experiment folder
    post_pend = 'denoExp'

    test_after = 10 
    # set det_test to False if localization results on val not needed during training
    det_test = True

    # model numbers to test.
    model_nums = [249] 
    
    # set test mode to true to test a trained model
    test_mode = False
    
    save_outfs = False
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    # number of similar class videos to contain in each training batch. imp when num classes>batchsize 
    num_similar = 0
    
    retrain = False
    viz_mode = False
    viz_sim = False

    network_params = {}
    network_params['deno'] = deno
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=True

    # graph_size controls number of videos per graph during training (see supp Fig 1)
    network_params['graph_size'] = 1 
    network_params['method'] = 'cos'
    network_params['sparsify'] = 'percent_0.5'
    attention = True
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = True

    # default params. no need to change.
    first_thresh= 0
    class_weights = False
    all_classes = False
    second_thresh = -0.9
    det_class = -1
    multibranch = 1
    branch_to_test = -2
    k_vec = None
    gt_vec = False
    just_primary = False

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
                        plot_losses = plot_losses,
                        num_similar = num_similar,
                        save_outfs = save_outfs,
                        det_test = det_test)

def activitynet(deno, gs = 1):
    print 'hey girl'
    model_name = 'graph_multi_video_with_L1_retF_tanh'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1,1]
    plot_losses = True

    lr = [0.001,0.001, 0.001]
    
    num_similar = 128

    batch_size = 256
    batch_size_val = 256

    seed = 999
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    
    
    epoch_stuff = [250,250]
    dataset = 'activitynet'
    limit  = None
    save_after = 50
    test_after = 10
    
    test_mode = False
    save_outfs = False
    
    test_method = 'original'
    test_post_pend = '_'+test_method

    model_nums = [249]
    retrain = False
    viz_mode = False
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['deno'] = deno
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=True
    network_params['graph_size'] = gs
    network_params['method'] = 'cos'
    network_params['sparsify'] = 'percent_0.5'
    attention = True
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = True
    
    post_pend = 'denoExp'
    class_weights = False
    first_thresh=0
    all_classes = False
    second_thresh = -0.9
    det_class = -1
    multibranch = 1
    branch_to_test = -2
    k_vec = None
    gt_vec = False
    just_primary = False

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
                        save_outfs = save_outfs,
                        det_test = False)


def charades(deno,gs =1):
    print 'hey girl NEW'

    model_name = 'graph_multi_video_with_L1_retF_tanh'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1,1]

    dataset = 'charades_i3d_charades_both'
    class_weights = False
    num_similar = 128
    det_test = False
    plot_losses = True


    batch_size = 256
    batch_size_val = 256

    lr = [0.001,0.001, 0.001]
    multibranch = 1
    
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
    limit  = None
    save_after = 50
    test_mode = False
    save_outfs = False
    
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method

    model_nums = [249]
    retrain = False
    viz_mode = False
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['deno'] = deno
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=True
 
    network_params['graph_size'] = gs
    network_params['method'] = 'cos'
    network_params['sparsify'] = 'percent_0.5'
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = True
    
    post_pend = 'denoExp'
    
    first_thresh=0
    
    test_after = 50
    all_classes = False
    
    second_thresh = -0.9
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
                        det_test = det_test,
                        num_similar = num_similar,
                        save_outfs = save_outfs)



def main():
    
    for deno in [1, 2, 4, 8, 'random']:
        thumos(deno)
        # activitynet(deno)
        # charades(deno)




if __name__=='__main__':
    main()
