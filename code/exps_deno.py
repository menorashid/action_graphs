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
    det_test = False


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
    save_outfs = False
    
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

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
 
    network_params['graph_size'] = 1
    network_params['method'] = 'cos'
    network_params['sparsify'] = 'percent_0.5'
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = True

    num_similar = 0
    
    post_pend = 'changingSparsityAbs_'+str(num_similar)
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
                        save_outfs = save_outfs,
                        det_test = det_test)

def activitynet(deno, gs = 1):
    print 'hey girl'
    model_name = 'graph_multi_video_with_L1_retF_tanh'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1,1]
    plot_losses = True

    lr = [0.001,0.001, 0.001]
    multibranch = 1

    num_similar = 128

    batch_size = 256
    batch_size_val = 256

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
    limit  = None
    save_after = 50
    
    test_mode = True
    save_outfs = False
    
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

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
    
    post_pend = 'gs1_changingSparsityAbs_'+str(num_similar)
    class_weights = False
    first_thresh=0
    
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
    # criterion_str = 'BinaryCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1,1]

    dataset = 'charades_i3d_charades_both'
    class_weights = False
    num_similar = 128
    det_test = False
    plot_losses = True

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'BinaryCrossEntropyMultiBranchWithL1'
    # loss_weights = [1,1]
    # plot_losses = False

    batch_size = 256
    batch_size_val = 256

    lr = [0.001,0.001, 0.001]
    # lr = [0.01,0.01, 0.01]
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
    limit  = None
    save_after = 50
    test_mode = True
    save_outfs = True
    
    # test_method = 'wtalc'
    # test_method = 'wtalc'
    # test_post_pend = '_'+test_method+'_tp_fp_conf'

    # test_method = 'best_worst_dot'
    # test_post_pend = '_'+test_method
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class_merged'

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
    # network_params['dropout'] = 0.8
    # post_pend = ''
    post_pend = 'changingSparsityAbs_numSim_'+str(num_similar)
    # '_cwNo_justPos_MulNumClasses_numSim_'+str(num_similar)
    # +'_sumnomean_noExclusiveCASL_NEW_noMax'

    first_thresh=0
    # scipy.special.logit(0.1)
    # 
    test_after = 50
    all_classes = False
    
    second_thresh = -0.9
    # scipy.special.logit(0.1)
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
                        save_outfs = save_outfs)





def main():
    # print 'hello'
    # thumos_bce()
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

    # charades('random',gs = 16)
    # for deno in [1,2,4,8]:
    for gs in [2,4,8,16,32]:
        charades('random',gs = gs)
        # activitynet(1, gs =gs)

    # for deno in ['random']:
    #     charades(deno)
    # for deno in [1,2,4]:
    # thumos('random')
    # for deno in [1,2,4]:
    #     thumos(deno)

    # multithumos_bce()
    # charades_bce()
    # charades_mce()
    # graph_charades_other()
    # graph_charades_32()
    # graph_charades_everything_sim()



if __name__=='__main__':
    main()
