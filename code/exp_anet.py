from exp_mill_bl import *

def graph_debug_anet_experiment():
    print 'hey girl'
    # raw_input()
    # model_name = 'graph_multi_video_with_L1_retF'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    # loss_weights = [1.,1.,0.5]
    # plot_losses = True
    # add_feat_ret = True
    # num_similar =   64

    model_name = 'graph_multi_video_with_L1'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_withplot'
    loss_weights = [1.,1.]
    plot_losses = True
    add_feat_ret = False
    num_similar =   0

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
    dataset = 'activitynet'
    
    batch_size = 256
    # -2*num_similar
    batch_size_val = 256
    # 32
    limit  = None
    save_after = 50
    
    test_mode = True
    
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class_retry'

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}

    network_params['deno'] = 8
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,1024]
    if add_feat_ret:
        network_params['feat_ret']=True
 
    network_params['graph_size'] = 2
    network_params['method'] = 'cos'
    network_params['sparsify'] = 0.5
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = False
    post_pend = '_numSim_'+str(num_similar)+'_rest_'+str(batch_size)+'_cwfix_tuplefix'
    # post_pend = '_htbefwithcasl'
    first_thresh=0.
    class_weights = False
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


def wtalc_debug_anet_experiment():
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
    dataset = 'activitynet'
    num_similar = 64
    batch_size = 256
    limit  = None
    save_after = 50
    
    test_mode = True
    
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = None
    retrain = False
    viz_mode = False
    viz_sim = False

    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,2048]
    network_params['feat_ret']=True
 
    network_params['aft_nonlin']='RL_L2'
    post_pend = '_baseline'
    first_thresh=0.
    class_weights = False
    test_after = 10
    all_classes = False
    
    second_thresh = 1.
    det_class = -1
    train_simple_mill_all_classes (model_name = model_name,
                        lr = lr,
                        dataset = dataset,
                        network_params = network_params,
                        limit = limit, 
                        epoch_stuff= epoch_stuff,
                        batch_size = batch_size,
                        batch_size_val = batch_size,
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



def graph_best_yet():
    print 'hey girl'
    
    model_name = 'graph_multi_video_with_L1'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    loss_weights = [1.,0.]
    plot_losses = False

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
    dataset = 'activitynet'
    num_similar = 0
    batch_size = 256
    batch_size_val = 256
    limit  = None
    save_after = 10
    
    test_mode = True
    
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class_retry'

    model_nums = None
    retrain = False
    viz_mode = True
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}

    network_params['deno'] = 8
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,2048]
    # network_params['feat_ret']=True
 
    network_params['graph_size'] = 2
    network_params['method'] = 'cos'
    network_params['sparsify'] = 0.5
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = False
    post_pend = '_numSim_'+str(num_similar)+'_rest_'+str(batch_size)
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
    print 'hello'
    # graph_best_yet()
    # wtalc_debug_anet_experiment()
    graph_debug_anet_experiment()

if __name__=='__main__':
    main()