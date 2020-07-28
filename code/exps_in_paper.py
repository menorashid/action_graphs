from exp_mill_bl import train_simple_mill_all_classes
import torch
import random
import numpy as np

def graph_l1_experiment_best_yet():
    print 'hey girl'
    # raw_input()
    model_name = 'graph_multi_video_with_L1_retF'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    criterion_str = 'BinaryCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1,1]
    plot_losses = True

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # criterion_str = 'BinaryCrossEntropyMultiBranchWithL1'
    # loss_weights = [1,1.]
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
    
    test_mode = False
    
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = [249]
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
    post_pend = '_testAgain'
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


def graph_sparsity_exps():
    print 'hey girl'
    # raw_input()
    model_name = 'graph_multi_video_with_L1_retF'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,0.,0.]
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
    
    
    
    epoch_stuff = [250,250]
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
    network_params['sparsify'] = 0.6
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = False
    post_pend = '_vary_sparsity_thresh'
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


def graph_multithumos():
    print 'hey girl NEW'
    # raw_input()
    model_name = 'graph_multi_video_with_L1_retF'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1,1]

    # loss_weights = [1,1,0]
    dataset = 'multithumos'
    class_weights = False
    num_similar = 8
    det_test = True

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
    
    
    
    epoch_stuff = [500,500]
    # dataset = 'multithumos'
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

    model_nums = [149,249,499]
    retrain = True
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
    # network_params['dropout'] = 0.7
    post_pend = '_noLimit'
    post_pend = '_numSim_'+str(num_similar)+'_sumnomean_noExclusiveCASL_NEW_noMax'
    first_thresh=0
    test_after = 50
    all_classes = False
    
    second_thresh = 0.5
    det_class = -1
    print 'this is it! are you ready????'
    raw_input()
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
                        det_test = det_test,
                        num_similar = num_similar)



def graph_charades():
    print 'hey girl charades'
    # raw_input()
    model_name = 'graph_multi_video_with_L1_retF'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,0.5,1]

    # loss_weights = [1,1,0]
    # dataset = 'charades_vgg_16_rgb_npy'
    dataset = 'charades_i3d_both'
    class_weights = True
    num_similar = 32
    det_test = False

    batch_size = 256
    batch_size_val = 256

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
    
    model_file = None

    # model_file = ['../experiments/graph_multi_video_with_L1_retF/graph_multi_video_with_L1_retF_aft_nonlin_RL_L2_non_lin_None_sparsify_0.5_graph_size_2_sigmoid_False_graph_sum_True_deno_8_n_classes_157_dropout_0.8_in_out_2048_1024_feat_dim_2048_1024_feat_ret_True_method_cos_charades_i3d_both/all_classes_False_just_primary_False_limit_None_cw_True_MultiCrossEntropyMultiBranchWithL1_CASL_250_step_250_0.1_0.001_0.001_0.001_lw_1.00_1.00_1.00__numSim_32_sumnomean_noExclusiveCASL_NEW_noMax/model_249.pt',250]
    
    epoch_stuff = [250,250]
    # dataset = 'multithumos'
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

    model_nums = [99,249]
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
    # network_params['dropout'] = 0.8
    post_pend = '_noLimit'
    post_pend = '_numSim_'+str(num_similar)+'_sumnomean_noExclusiveCASL_NEW_noMax'
    # post_pend = '_numSim_'+str(num_similar)+'_'
    first_thresh=0
    test_after = 50
    all_classes = False
    
    second_thresh = 0.5
    det_class = -1
    # print 'this is it! are you ready????'
    # raw_input()
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
                        save_outfs = True,
                        model_file = model_file)


def graph_charades_other():
    print 'hey girl charades'
    # raw_input()
    model_name = 'graph_multi_video_with_L1_retF'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [0.1,1,1]

    # loss_weights = [1,1,0]
    # dataset = 'charades_vgg_16_rgb_npy'
    dataset = 'charades_i3d_both'
    class_weights = False
    num_similar = 64
    det_test = False

    batch_size = 256
    batch_size_val = 256

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
    
    
    
    epoch_stuff = [250,250]
    # dataset = 'multithumos'
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

    model_nums = [99]
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
    # network_params['dropout'] = 0.8
    post_pend = '_noLimit'
    post_pend = '_numSim_'+str(num_similar)+'_sumnomean_noExclusiveCASL_NEW_noMax'
    # post_pend = '_numSim_'+str(num_similar)+'_'
    first_thresh=0
    test_after = 50
    all_classes = False
    
    second_thresh = 0.5
    det_class = -1
    # print 'this is it! are you ready????'
    # raw_input()
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
                        save_outfs = True)

def graph_charades_32():
    print 'hey girl charades'
    # raw_input()
    model_name = 'graph_multi_video_with_L1_retF'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1,1]

    # loss_weights = [1,1,0]
    # dataset = 'charades_vgg_16_rgb_npy'
    dataset = 'charades_i3d_both'
    class_weights = True
    num_similar = 0
    det_test = False

    batch_size = 32
    batch_size_val = 32

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
    
    
    
    epoch_stuff = [250,250]
    # dataset = 'multithumos'
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
    # network_params['dropout'] = 0.8
    post_pend = '_noLimit'
    post_pend = '_numSim_'+str(num_similar)+'_sumnomean_noExclusiveCASL_NEW_noMax'
    # post_pend = '_numSim_'+str(num_similar)+'_'
    first_thresh=0
    test_after = 50
    all_classes = False
    
    second_thresh = 0.5
    det_class = -1
    # print 'this is it! are you ready????'
    # raw_input()
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
                        save_outfs = True)

def graph_charades_everything_sim():

    print 'hey girl charades'
    # raw_input()
    model_name = 'graph_multi_video_with_L1_retF'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1,1]

    # loss_weights = [1,1,0]
    # dataset = 'charades_vgg_16_rgb_npy'
    dataset = 'charades_i3d_both'
    class_weights = True
    num_similar = 128
    det_test = False

    batch_size = 256
    batch_size_val = 256

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
    
    
    
    epoch_stuff = [250,250]
    # dataset = 'multithumos'
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
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=True
 
    network_params['graph_size'] = 2
    network_params['method'] = 'cos'
    network_params['sparsify'] = 0.5
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = False
    # network_params['dropout'] = 0.8
    post_pend = '_noLimit'
    post_pend = '_numSim_'+str(num_similar)+'_sumnomean_noExclusiveCASL_NEW_noMax'
    # post_pend = '_numSim_'+str(num_similar)+'_'
    first_thresh=0
    test_after = 50
    all_classes = False
    
    second_thresh = 0.5
    det_class = -1
    # print 'this is it! are you ready????'
    # raw_input()
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
                        save_outfs = True)


def graph_i3d():
    print 'hey girl'
    # raw_input()
    model_name = 'graph_multi_video_with_L1_retF_i3d'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,0.,0.]
    plot_losses = True

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # loss_weights = [1,1]
    # plot_losses = False

    lr = [0.001,0.001,0.001]
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

    model_nums = [99]
    retrain = False
    viz_mode = False
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [1024,1024]
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=True
 
    network_params['graph_size'] = 2
    network_params['method'] = 'cos'
    network_params['sparsify'] = 0.5
    network_params['graph_sum'] = attention
    network_params['non_lin'] = 'RL'
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = False
    post_pend = '_linoninput'
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


def graph_norm_game():
    print 'hey girl'
    # raw_input()
    model_name = 'graph_multi_video_norm_game'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    loss_weights = [1,1]
    plot_losses = False

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # loss_weights = [1,1]
    # plot_losses = False

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
    dataset = 'ucf'
    limit  = None
    save_after = 50
    
    test_mode = False
    
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = [249]
    retrain = False
    viz_mode = False
    viz_sim = False

    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=False
 
    network_params['graph_size'] = 1
    network_params['method'] = 'cos'
    network_params['sparsify'] = None
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = False
    # network_params['nosum'] = 5
    # post_pend = '_rnsmaxways'
    network_params['nosum'] = 4
    post_pend = '_rnbothways'
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

def main():

    # graph_norm_game()
    # graph_i3d()
    # graph_sparsity_exps()
    graph_l1_experiment_best_yet()
    # graph_charades()
    # graph_charades_other()
    # graph_charades_32()
    # graph_charades_everything_sim()



if __name__=='__main__':
    main()
