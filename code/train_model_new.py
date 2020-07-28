from train_test_mill import *


def get_sparsity_threshold(model, train_dataloader):
    model.eval()
    min_max_median = []
    
    for num_iter_train,batch in enumerate(train_dataloader):    
        samples = batch['features']
        # labels = batch['label'].cuda()
        print num_iter_train
        min_max_median+=model.get_sparsity_threshold(samples)
        if num_iter_train==100:
            break

    min_max_median = np.array(min_max_median)
    sparsity_threshold = np.mean(min_max_median[:,2])

    # mean_vals = np.mean(min_max_median, axis = 0)
    # sparsity_threshold = (mean_vals[1]+mean_vals[0])*0.5
    

    sparsity_threshold = (np.max(min_max_median[:,1])+np.min(min_max_median[:,0]))*0.5

    print min_max_median.shape
    print np.min(min_max_median, axis = 0)
    print np.max(min_max_median, axis = 0)
    print np.mean(min_max_median, axis = 0)
    print sparsity_threshold



    model.train()
    return float(sparsity_threshold)


def train_model_new(out_dir_train,
                train_data,
                test_data,
                test_args,
                batch_size = None,
                batch_size_val = None,
                num_epochs = 100,
                save_after = 20,
                disp_after = 1,
                plot_after = 10,
                test_after = 1,
                lr = 0.0001,
                dec_after = 100, 
                model_name = 'alexnet',
                criterion = nn.CrossEntropyLoss(),
                gpu_id = 0,
                num_workers = 16,
                model_file = None,
                epoch_start = 0,
                network_params = None,
                weight_decay = 0, 
                multibranch = 1,
                plot_losses = False,
                det_test = False):
    print 'num_workers', num_workers

    util.mkdir(out_dir_train)
    log_file = os.path.join(out_dir_train,'log.txt')
    plot_file = os.path.join(out_dir_train,'loss.jpg')
    
    log_file_writer = open(log_file,'wb')

    plot_file = os.path.join(out_dir_train,'loss.jpg')
    log_arr = []
    plot_arr = [[],[]]
    plot_val_arr =  [[],[]]
    plot_val_acc_arr = [[],[]]
    
    plot_strs_posts = ['Loss']
    plot_acc_file = os.path.join(out_dir_train,'val_accu.jpg')
    plot_det_file = os.path.join(out_dir_train,'val_det.jpg')

    if 'ucf' in out_dir_train:
        plot_det_arr = [([],[]),([],[]),([],[]),([],[]),([],[])]
        lengend_strs_detection = ['0.1','0.2','0.3','0.4','0.5']
    elif 'activitynet' in out_dir_train:
        plot_det_arr = [([],[]),([],[]),([],[])]
        lengend_strs_detection = ['0.5','0.7','0.9']    
    else:
        plot_det_arr = [([],[])]
        # ,([],[]),([],[]),([],[]),([],[])]
        lengend_strs_detection = ['0.1']
    # ,'0.2','0.3','0.4','0.5']


    

    network = models.get(model_name,network_params)

    if model_file is not None:
        network.model = torch.load(model_file)

    model = network.model
    
    if batch_size is None:
        batch_size = len(train_data)

    if batch_size_val is None:
        batch_size_val = len(test_data)

    train_dataloader = torch.utils.data.DataLoader(train_data, 
                        batch_size = batch_size,
                        collate_fn = train_data.collate_fn,
                        shuffle = True, 
                        num_workers = num_workers)
    
    test_dataloader = torch.utils.data.DataLoader(test_data, 
                        batch_size = batch_size_val,
                        collate_fn = test_data.collate_fn,
                        shuffle = False, 
                        num_workers = num_workers)
    
    torch.cuda.device(gpu_id)
    model = model.cuda()
    model.train(True)
    model_str = str(model)
    log_file_writer.write(model_str+'\n')
    # print model_str
    # print 'done printing'
    # out_file = os.path.join(out_dir_train,'model_-1.pt')
    # print 'saving',out_file
    # torch.save(model,out_file)    
    # return
    criterion_str = criterion.__class__.__name__.lower()
    if plot_losses:
        plot_loss_arr = [([],[]) for lw in criterion.loss_weights_all]
        lengend_strs_loss = [loss_str for loss_str in criterion.loss_strs]
        # plot_det_arr = [([],[]),([],[]),([],[]),([],[]),([],[])]
        # plot_strs_posts = ['Train Loss']
        plot_losses_file = os.path.join(out_dir_train,'loss_all.jpg')
    


    optimizer = torch.optim.Adam(network.get_lr_list(lr),weight_decay=weight_decay)

    if dec_after is not None:
        print dec_after
        if dec_after[0] is 'step':
            print dec_after
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=dec_after[1], gamma=dec_after[2])
        elif dec_after[0] is 'exp':
            print 'EXPING',dec_after
            exp_lr_scheduler = Exp_Lr_Scheduler(optimizer,epoch_start*len(train_dataloader),[lr_curr for lr_curr in lr if lr_curr!=0],dec_after[1],dec_after[2],dec_after[3])
    
    criterion = criterion.cuda()

    # print model.sparsify
    # print hasattr(model,'sparsify')
    # print model.sparsify=='static_media_mean'

    if hasattr(model,'sparsify') and type(model.sparsify)==str and model.sparsify.startswith('static'):

        # print 'before',model.sparsify
        model.sparsify = get_sparsity_threshold(model,train_dataloader) 
        # print 'after',model.sparsify
        str_curr = 'Sparsity Threshold '+str(model.sparsify)
        print str_curr
        log_file_writer.write(str_curr+'\n')

    # print 'raw_inputting'
    # raw_input()
    for num_epoch in range(epoch_start,num_epochs):

        plot_arr_epoch = []
        if plot_losses:
            plot_losses_inner_epoch = [[] for i in range(len(plot_loss_arr))]

        # t = time.time()
        # print 'starting'
        for num_iter_train,batch in enumerate(train_dataloader):
            # print 'getting batch',time.time()-t
            # t = time.time()
            # continue

            samples = batch['features']
            labels = batch['label'].cuda()

            
            if 'centerloss' in model_name:
                preds,extra = model.forward(samples, labels)
                labels = [labels]+extra

            else:
                preds = []
                if multibranch>1:
                    preds = [[] for i in range(multibranch)]
                elif 'l1' in criterion_str:
                    preds = [[],[]]
                
                for idx_sample, sample in enumerate(samples):

                    if 'norm_game' in model_name and 'multi_video' in model_name:
                        out,preds = model.forward(samples, epoch_num = num_epoch/float(num_epochs))
                        break
                    elif ('cooc' in model_name or 'perfectG' in model_name) and 'multi_video' in model_name:
                        out,preds = model.forward([samples,batch['gt_vec']])
                        break
                    elif 'alt_train' in model_name:
                        out,pmf = model.forward(sample.cuda(), epoch_num=num_epoch)
                    elif 'perfectG' in model_name:
                        out,pmf = model.forward([sample.cuda(),batch['gt_vec'][idx_sample].cuda()])
                    elif 'multi_video' in model_name:
                        out,preds = model.forward(samples)
                        break
                    else:    
                        out,pmf = model.forward(sample.cuda())

                    if multibranch>1:
                        for idx in range(len(pmf)):
                            preds[idx].append(pmf[idx].unsqueeze(0))
                    elif 'l1' in criterion_str:
                        preds[0].append(pmf[0].unsqueeze(0))
                        preds[1].append(pmf[1])
                    else:
                        preds.append(pmf.unsqueeze(0))
                

                if 'l1' in criterion_str:
                    [preds, att] = preds


                if multibranch>1:
                    preds = [torch.cat(preds_curr,0) for preds_curr in preds]        
                else:
                    preds = torch.cat(preds,0)        

            # print 'forwarding',time.time()-t
            
            # t = time.time()

            if 'casl' in criterion_str:
                loss = criterion(labels, preds,att, out, collate = not plot_losses)
                # print loss
                # raw_input()
            elif 'l1' in criterion_str and 'withplot' in  criterion_str:
                loss = criterion(labels, preds,att, collate = not plot_losses)
            elif 'l1' in criterion_str:
                loss = criterion(labels, preds,att)
            else:
                loss = criterion(labels, preds)
            
            # print 'getting loss',time.time()-t
            # t = time.time()

            # print loss
            if type(loss) == list:

                [loss, loss_broken_down] = loss
                if plot_losses:
                    for idx_loss_curr, loss_curr in enumerate(loss_broken_down):
                        # print loss_curr.data[0]
                        loss_curr = loss_curr.item() if type(loss_curr)!=float else loss_curr
                        plot_losses_inner_epoch[idx_loss_curr].append(loss_curr)
                    # print loss_broken_down,plot_losses_inner_epoch


            loss_iter = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # model.printGraphGrad()
            # grad_rel = model.graph_layers[0].graph_layer.weight.grad
            # print torch.min(grad_rel).data.cpu().numpy(), torch.max(grad_rel).data.cpu().numpy()

            # print criterion.__class__.__name__.lower()
            # if 'centerloss' in criterion.__class__.__name__.lower():
            #     criterion.backward()
            
            num_iter = num_epoch*len(train_dataloader)+num_iter_train
            
            plot_arr_epoch.append(loss_iter)
            str_display = 'lr: %.6f, iter: %d, loss: %.4f' %(optimizer.param_groups[-1]['lr'],num_iter,loss_iter)
            log_arr.append(str_display)
            print str_display
            
            # print 'everything_else',time.time() - t
            # t = time.time()



        plot_arr[0].append(num_epoch)
        plot_arr[1].append(np.mean(plot_arr_epoch))
        if plot_losses:
            for idx_tuple_curr, tuple_curr in enumerate(plot_loss_arr):
                tuple_curr[0].append(num_epoch)
                tuple_curr[1].append(np.mean(plot_losses_inner_epoch[idx_tuple_curr]))


        if num_epoch % plot_after== 0 and num_iter>0:
            
            for string in log_arr:
                log_file_writer.write(string+'\n')
            
            log_arr = []

            if len(plot_val_arr[0])==0:
                visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
            else:
                
                lengend_strs = [pre_str+plot_str_posts for pre_str in ['Train ','Val '] for plot_str_posts in plot_strs_posts]

                # print len(plot_arr),len(plot_val_arr)
                plot_vals = [(arr[0],arr[1]) for arr in [plot_arr]+[plot_val_arr]]
                # print plot_vals
                # print lengend_strs
                visualize.plotSimple(plot_vals,out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=lengend_strs)

                if det_test:
                    visualize.plotSimple(plot_det_arr,out_file = plot_det_file,title = 'Detection',xlabel = 'Iteration',ylabel = 'Accuracy',legend_entries=lengend_strs_detection)
                if plot_losses:                
                    
                    visualize.plotSimple(plot_loss_arr,out_file = plot_losses_file,title = 'Losses',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=lengend_strs_loss)

                visualize.plotSimple([(plot_val_acc_arr[0],plot_val_acc_arr[1])],out_file = plot_acc_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Accuracy',legend_entries=['Val'])
                


        if (num_epoch+1) % test_after == 0 or num_epoch==0:
            model.eval()
            accuracy, loss_iter = test_model_core(model, test_dataloader, criterion, log_arr, multibranch  = multibranch)
            if det_test:
                aps = test_model_overlap(model, test_dataloader, criterion, log_arr,
                    first_thresh = test_args['first_thresh'] ,
                    second_thresh = test_args['second_thresh'] ,
                    bin_trim = test_args['trim_preds'] ,
                    multibranch = test_args['multibranch'],
                    branch_to_test = test_args['branch_to_test'],
                    save_outfs = test_args['save_outfs'],
                    dataset = test_args['dataset'],
                    test_method = test_args['test_method'])
                aps_rel = aps[-1,:]
                [plot_det_arr_curr[0].append(num_epoch) for plot_det_arr_curr in plot_det_arr]
                [plot_det_arr[idx_ap][1].append(ap) for idx_ap,ap in enumerate(aps_rel)]
            
            # print plot_det_arr

            # print aps.shape
            # print aps
            # raw_input()
            plot_val_arr[0].append(num_epoch); plot_val_arr[1].append(loss_iter)
            plot_val_acc_arr[0].append(num_epoch); plot_val_acc_arr[1].append(accuracy)
            # model.train(True)

           

        if (num_epoch+1) % save_after == 0 or num_epoch==0:
            out_file = os.path.join(out_dir_train,'model_'+str(num_epoch)+'.pt')
            print 'saving',out_file
            torch.save(model,out_file)
            # raw_input()

            
        if dec_after is not None and dec_after[0]=='reduce':
            # exp_lr_scheduler
            if accuracy>=best_val:
                best_val = accuracy
                out_file_best = os.path.join(out_dir_train,'model_bestVal.pt')
                print 'saving',out_file_best
                torch.save(model,out_file_best)            
            exp_lr_scheduler.step(loss_iter)

        elif dec_after is not None and dec_after[0]!='exp':
            exp_lr_scheduler.step()
        


    out_file = os.path.join(out_dir_train,'model_'+str(num_epoch)+'.pt')
    print 'saving',out_file
    torch.save(model,out_file)
    
    for string in log_arr:
        log_file_writer.write(string+'\n')
                
    if len(plot_val_arr[0])==0:
        visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=['Train'])
    else:
        
        lengend_strs = [pre_str+plot_str_posts for pre_str in ['Train ','Val '] for plot_str_posts in plot_strs_posts]

        # print len(plot_arr),len(plot_val_arr)
        plot_vals = [(arr[0],arr[1]) for arr in [plot_arr]+[plot_val_arr]]
        # print plot_vals
        # print lengend_strs
        visualize.plotSimple(plot_vals,out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=lengend_strs)

        if det_test:
            visualize.plotSimple(plot_det_arr,out_file = plot_det_file,title = 'Detection',xlabel = 'Iteration',ylabel = 'Accuracy',legend_entries=lengend_strs_detection)
        if plot_losses:                
            visualize.plotSimple(plot_loss_arr,out_file = plot_losses_file,title = 'Losses',xlabel = 'Iteration',ylabel = 'Loss',legend_entries=lengend_strs_loss)


        visualize.plotSimple([(plot_val_acc_arr[0],plot_val_acc_arr[1])],out_file = plot_acc_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Accuracy',legend_entries=['Val'])

    log_file_writer.close()









    