
from exp_mill_bl import get_criterion, get_data, test_model

def test_ucf(out_dir_train = None):
    if out_dir_train is None:
        out_dir_train = '../experiments/action_graphs_ucf'

    _, _, test_data, _, trim_preds = get_data('ucf', None, False, False, False, None, test_pair = False, num_similar = 0)
    criterion, criterion_str = get_criterion('MultiCrossEntropyMultiBranchWithL1_CASL',True,None,  [1,1,1], 1, num_similar = 0)
    test_params = dict(out_dir_train = out_dir_train,
                model_num = 249,
                test_data = test_data,
                batch_size_val = 32,
                criterion = criterion,
                gpu_id = 0,
                num_workers = 0,
                trim_preds = trim_preds,
                visualize = False,
                det_class = -1,
                second_thresh = -0.9,
                first_thresh = 0,
                post_pend='',
                multibranch = 1,
                branch_to_test =-2,
                dataset = 'ucf', 
                save_outfs = False,
                test_pair = False,
                test_method = 'original')
    test_model(**test_params)
    
def test_anet():
    pass

def test_charades():
    pass

def main():
    test_ucf()

if __name__=='__main__':
    main()