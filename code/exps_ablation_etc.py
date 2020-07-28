from exp_mill_bl import train_simple_mill_all_classes
import torch
import random
import numpy as np
import scipy

def thumos_bce_actual(loss_weights,sparsify):
    print 'hey girl'
    model_name = 'graph_multi_video_with_L1_retF_tanh'
    criterion_str = 'BinaryCrossEntropyMultiBranchWithL1_CASL'
    # loss_weights = [1,1,0]
    plot_losses = True
    det_test = True

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # # criterion_str = 'BinaryCrossEntropyMultiBranchWithL1'
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
    
    
    
    epoch_stuff = [250,250]
    dataset = 'ucf'
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
    test_post_pend = '_'+test_method+'_diff_viz_multi_only'

    model_nums = [249]
    # ,349,399,449,499]
    retrain = False
    viz_mode = True
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=True
 
    network_params['graph_size'] = 1
    network_params['method'] = 'cos'
    network_params['sparsify'] =sparsify
    # 'percent_0.5'

    # 'static_mid_minmin_maxmax'
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = False

    # network_params['dropout'] = 0.8
    num_similar = 0
    
    post_pend = 'forplot_'+str(num_similar)
    first_thresh= -1
    class_weights = False
    test_after = 10
    all_classes = False
    
    second_thresh = scipy.special.logit(0.1)
    # -0.9
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
                        plot_losses = plot_losses,
                        num_similar = num_similar,
                        save_outfs = save_outfs,
                        det_test = True)



def thumos_bce(loss_weights,sparsify):
    print 'hey girl'
    model_name = 'graph_multi_video_with_L1_retF_tanh'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    # loss_weights = [1,1,0]
    plot_losses = True
    det_test = True

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # # criterion_str = 'BinaryCrossEntropyMultiBranchWithL1'
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
    
    
    
    epoch_stuff = [250,250]
    dataset = 'ucf'
    limit  = None
    save_after = 50
    
    test_mode = True
    save_outfs = False
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method+'_diff_viz_multi_only'

    model_nums = [249]
    # ,349,399,449,499]
    retrain = False
    viz_mode = True
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=True
 
    network_params['graph_size'] = 1
    network_params['method'] = 'cos'
    network_params['sparsify'] =sparsify
    # 'percent_0.5'

    # 'static_mid_minmin_maxmax'
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = True

    # network_params['dropout'] = 0.8
    num_similar = 0
    
    post_pend = 'forplot_'+str(num_similar)
    first_thresh= -1
    class_weights = False
    test_after = 10
    all_classes = False
    
    second_thresh = -0.9
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
                        plot_losses = plot_losses,
                        num_similar = num_similar,
                        save_outfs = save_outfs,
                        det_test = True)

def thumos_gs(gs):
    print 'hey girl'
    model_name = 'graph_multi_video_with_L1_retF_tanh'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1,1]
    plot_losses = True
    det_test = True

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # # criterion_str = 'BinaryCrossEntropyMultiBranchWithL1'
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
    
    
    
    epoch_stuff = [250,250]
    dataset = 'ucf'
    limit  = 750
    save_after = 50
    
    test_mode = True
    save_outfs = False
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = [249]
    # ,349,399,449,499]
    retrain = False
    viz_mode = False
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=True
 
    network_params['graph_size'] = gs
    network_params['method'] = 'cos'
    network_params['sparsify'] ='percent_0.5'

    # 'static_mid_minmin_maxmax'
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = True

    # network_params['dropout'] = 0.8
    num_similar = 0
    
    post_pend = 'ablation_gs_'+str(num_similar)
    first_thresh= -1
    class_weights = False
    test_after = 10
    all_classes = False
    
    second_thresh = 0
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
                        plot_losses = plot_losses,
                        num_similar = num_similar,
                        save_outfs = save_outfs)

def fcasl():
    print 'hey girl'
    model_name = 'fcasl_multi_video_with_L1_retF'
    criterion_str = 'MultiCrossEntropyMultiBranchFakeL1_CASL'
    loss_weights = [1,1,1]
    plot_losses = True
    det_test = True

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # # criterion_str = 'BinaryCrossEntropyMultiBranchWithL1'
    # loss_weights = [1,1.]
    # plot_losses = False

    lr = [0.001,0.001]
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
    save_outfs = False
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = [249]
    # ,349,399,449,499]
    retrain = False
    viz_mode = True
    viz_sim = False

    # post_pend = '_noBiasLastLayer'
    
    network_params = {}
    network_params['deno'] = 8
    # network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,2048]
    network_params['feat_ret']=True
 
    # network_params['graph_size'] = 2
    # network_params['method'] = 'cos'
    # network_params['sparsify'] ='percent_0.5'

    # 'static_mid_minmin_maxmax'
    # network_params['graph_sum'] = attention
    # network_params['non_lin'] = None
    # network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = True

    # network_params['dropout'] = 0.8
    num_similar = 0
    
    post_pend = 'actuallytanh_numSimilar_'+str(num_similar)
    first_thresh= 0
    class_weights = False
    test_after = 10
    all_classes = False
    
    second_thresh = -0.9
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
                        plot_losses = plot_losses,
                        num_similar = num_similar,
                        save_outfs = save_outfs)
def main():
    print 'hello'
    raw_input()
    # fcasl()
    thumos_bce_actual([5,1,1],'percent_0.5')
    # thumos_bce([1,1,1],'percent_0.5')
    # thumos_bce([1,0,0],'percent_0.5')
    # thumos_bce([1,0,0],None)
    # thumos_bce([1,0,1],'percent_0.5')
    # thumos_bce([1,0,1],None)

    # for gs in [1,2,4,8,16,32]:
    #     thumos_gs(gs)

    # activitynet_bce()
    # multithumos_bce()
    # charades_bce()
    # charades_mce()
    # graph_charades_other()
    # graph_charades_32()
    # graph_charades_everything_sim()

if __name__=='__main__':
    main()

