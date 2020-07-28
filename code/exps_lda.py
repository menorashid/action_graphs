from exp_mill_bl import train_simple_mill_all_classes
import torch
import random
import numpy as np

def thumos_dt():
    print 'hey girl'
    model_name = 'graph_multi_video_with_L1_retF'
    criterion_str = 'MultiCrossEntropyMultiBranchWithL1_CASL'
    loss_weights = [1,1,1]
    plot_losses = True
    det_test = True

    # model_name = 'graph_multi_video_with_L1'
    # criterion_str = 'MultiCrossEntropyMultiBranchWithL1'
    # criterion_str = 'BinaryCrossEntropyMultiBranchWithL1'
    # loss_weights = [1,1.]
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
    
    test_mode = True
    save_outfs = False
    test_method = 'original'
    test_post_pend = '_'+test_method+'_class'

    model_nums = [99]
    retrain = False
    viz_mode = False
    viz_sim = False

    network_params = {}
    network_params['deno'] = 8
    network_params['in_out'] = [2048,1024]
    network_params['feat_dim'] = [2048,1024]
    network_params['feat_ret']=True
 
    network_params['graph_size'] = 2
    network_params['method'] = 'cos'
    network_params['sparsify'] = 0.5
    # 'percent_0.5'
    network_params['graph_sum'] = attention
    network_params['non_lin'] = None
    network_params['aft_nonlin']='RL_L2'
    network_params['sigmoid'] = False

    # network_params['dropout'] = 0.8
    num_similar = 0
    
    post_pend = 'rechecking_postlda_numSim'+str(num_similar)
    first_thresh=0
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

    thumos_dt()
    

if __name__=='__main__':
    main()
