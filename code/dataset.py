import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
from helpers import util, visualize
import models
from criterions import MultiCrossEntropy
import random

class UCF_dataset(Dataset):
    def __init__(self, text_file, feature_limit, select_front = False, num_similar = 0):
        self.anno_file = text_file
        self.files = util.readLinesFromFile(text_file)
        self.feature_limit = feature_limit
        self.select_front = select_front
        self.num_similar = num_similar

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        anno = train_file_curr.split(' ')
        label = anno[1:]
        train_file_curr = anno[0]
        label =np.array([int(label) for label in label]).astype(float)
        label = label/np.sum(label)
        if train_file_curr.endswith('.npz'):
            sample = np.load(train_file_curr)['arr_0']
        else:
            sample = np.load(train_file_curr)
        # print 'in dataset',train_file_curr
        if sample.shape[0]>self.feature_limit and self.feature_limit is not None:
            if self.select_front:
                idx_start = 0
            else:
                idx_start = sample.shape[0] - self.feature_limit
                idx_start = np.random.randint(idx_start+1)
            sample = sample[idx_start:idx_start+self.feature_limit]
            assert sample.shape[0]==self.feature_limit

        # image = Image.open(train_file_curr)
        sample = {'features': sample, 'label': label}
        # if self.transform:
        # sample['image'] = self.transform(sample['image'])
        return sample

    def collate_fn(self,batch):
        data = [torch.FloatTensor(item['features']) for item in batch]
        target = [item['label'] for item in batch]
        target = torch.FloatTensor(target)
        return {'features':data, 'label':target}


class UCF_dataset_withNumSimilar(Dataset):
    def __init__(self, text_file, feature_limit, select_front = False, num_similar = 0, just_one = True):
        self.anno_file = text_file
        self.files = util.readLinesFromFile(text_file)
        self.feature_limit = feature_limit
        self.select_front = select_front
        self.num_similar = num_similar

        self.annos = [[int(val) for val in file_curr.split(' ')[1:]] for file_curr in self.files]
        self.annos = np.array(self.annos).astype(float)

        self.num_classes = self.annos.shape[1]
        ind_idx = np.sum(self.annos, axis = 1)
        # print self.annos[0]


        ind_idx = ind_idx ==1
        # print 'ind_idx.shape, np.sum(ind_idx)',ind_idx.shape, np.sum(ind_idx)

        # raw_input()
        self.idx_per_class = []
        for idx_curr in range(self.num_classes):
            if just_one:
                bin_class = np.logical_and(self.annos[:,idx_curr]>0, ind_idx)
            else:
                bin_class = self.annos[:,idx_curr]>0
                
            self.idx_per_class.append(np.where(bin_class)[0])
        #     print 'idx_curr, self.idx_per_class[-1]',idx_curr, self.idx_per_class[-1]
        # raw_input()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # return None
        train_file_curr = self.files[idx]
        anno = train_file_curr.split(' ')
        label = anno[1:]
        train_file_curr = anno[0]
        label =np.array([int(label) for label in label]).astype(float)
        label = label/np.sum(label)
        
        # print train_file_curr
        if train_file_curr.endswith('.npz'):
            sample = np.load(train_file_curr)['arr_0']
        else:
            sample = np.load(train_file_curr)
        # print 'in dataset',train_file_curr
        if sample.shape[0]>self.feature_limit and self.feature_limit is not None:
            if self.select_front:
                idx_start = 0
            else:
                idx_start = sample.shape[0] - self.feature_limit
                idx_start = np.random.randint(idx_start+1)
            sample = sample[idx_start:idx_start+self.feature_limit]
            assert sample.shape[0]==self.feature_limit

        # image = Image.open(train_file_curr)
        sample = {'features': sample, 'label': label, 'idx': idx}
        # sample = {'features': None, 'label': None, 'idx': idx}
        # if self.transform:
        # sample['image'] = self.transform(sample['image'])
        return sample

    def get_similiar(self, idx_already_selected):
        # print idx_already_selected
        idx_to_pick = []
        while len(idx_to_pick)<self.num_similar*2:
            # print 'iter'
            # pick a rand class
            class_to_pick = np.random.randint(self.num_classes)
            # print 'class_to_pick',class_to_pick
            idx_rel = self.idx_per_class[class_to_pick]
            # print 'idx_rel',idx_rel
            idx_left = idx_rel[np.isin(idx_rel, idx_already_selected, invert = True)]
            # print 'idx_left',idx_left

            if idx_left.size == 0:
                continue

            idx_to_pick.extend(list(np.random.choice(idx_left, 2)))
            # print 'idx_to_pick',idx_to_pick

        samples = [self.__getitem__(idx) for idx in idx_to_pick]
        # print 'len(samples)',len(samples)
        return samples
        # remove already selected options
        # pick n rows that have rand class
        # load and return them
        # pass

    def collate_fn(self,batch):
        num_keep = len(batch) - 2* self.num_similar
        if num_keep<0:
            num_keep = 0

        # print 'original',len(batch)
        batch = batch[:num_keep]
        # print 'kept',len(batch)
        idx_already_selected = [item['idx'] for item in batch]
        # print idx_already_selected
        # idx_already_selected = torch.FloatTensor(idx_already_selected)

        batch = self.get_similiar( idx_already_selected)+ batch
        # print 'new',len(batch)

        data = [torch.FloatTensor(item['features']) for item in batch]
        target = [item['label'] for item in batch]
        target = torch.FloatTensor(target)
        # print target.size()
        # print target[:6]        
        # raw_input()
        return {'features':data, 'label':target}



class UCF_dataset_cooc_graph(UCF_dataset):
    def __init__(self, text_file, feature_limit):
        super(UCF_dataset_cooc_graph, self).__init__(text_file = text_file, feature_limit = feature_limit)

    def __getitem__(self, idx):     
        train_file_curr = self.files[idx]
        anno = train_file_curr.split(' ')
        label = anno[2:]
        train_file_curr = anno[0]
        cooc_file_curr = anno[1]

        label =np.array([int(label) for label in label]).astype(float)
        label = label/np.sum(label)

        sample = np.load(train_file_curr)
        if cooc_file_curr.endswith('.npz'):
            cooc = np.load(cooc_file_curr)['arr_0']
        else:
            cooc = np.load(cooc_file_curr)

        if sample.shape[0]>self.feature_limit and self.feature_limit is not None:
            idx_start = sample.shape[0] - self.feature_limit
            idx_start = np.random.randint(idx_start+1)
            idx_end = idx_start+self.feature_limit
            # print cooc.shape, sample.shape
            # print cooc[idx_start+1,idx_start]

            sample = sample[idx_start:idx_end]
            if len(cooc.shape)==3:
                cooc = cooc[:,idx_start:idx_end,idx_start:idx_end]
                assert cooc.shape[1]==cooc.shape[2]==self.feature_limit
            else:
                assert len(cooc.shape)==2
                cooc = cooc[idx_start:idx_end,idx_start:idx_end]
                assert cooc.shape[0]==cooc.shape[1]==self.feature_limit

            assert sample.shape[0]==self.feature_limit
            # raw_input()

        # image = Image.open(train_file_curr)
        sample = {'features': (sample,cooc), 'label': label}
        # if self.transform:
        # sample['image'] = self.transform(sample['image'])

        return sample

    def collate_fn(self,batch):
        data = [(torch.FloatTensor(item['features'][0]),torch.FloatTensor(item['features'][1])) for item in batch]
        target = [item['label'] for item in batch]
        target = torch.FloatTensor(target)
        return {'features':data, 'label':target}


class UCF_dataset_cooc_per_class_graph(UCF_dataset):
    def __init__(self, text_file, feature_limit):
        super(UCF_dataset_cooc_per_class_graph, self).__init__(text_file = text_file, feature_limit = feature_limit)

    def __getitem__(self, idx):     
        train_file_curr = self.files[idx]

        anno = train_file_curr.split(' ')
        num_classes = (len(anno)-1)/2
        assert num_classes==20

        label = anno[-num_classes:]
        train_file_curr = anno[0]
        cooc_files_curr = anno[1:num_classes+1]
        # cooc_file_curr = anno[1]
        # print train_file_curr, cooc_files_curr
        # print label
        # raw_input()

        label =np.array([int(label) for label in label]).astype(float)
        label = label/np.sum(label)

        sample = np.load(train_file_curr)
        cooc_all = []
        for cooc_file_curr in cooc_files_curr:
            cooc = np.load(cooc_file_curr)['arr_0'][np.newaxis,:,:]
            cooc_all.append(cooc)
        cooc_all = np.concatenate(cooc_all,axis = 0)
        # print cooc_all.shape
        # raw_input()

        if sample.shape[0]>self.feature_limit and self.feature_limit is not None:
            idx_start = sample.shape[0] - self.feature_limit
            idx_start = np.random.randint(idx_start+1)
            idx_end = idx_start+self.feature_limit
            
            sample = sample[idx_start:idx_end]
            cooc_all = cooc_all[:,idx_start:idx_end,idx_start:idx_end]

            # for cooc_idx in range(len(cooc_all)):
            #     cooc_all[cooc_idx] = cooc_all[cooc_idx][idx_start:idx_end,idx_start:idx_end]
            assert cooc_all[cooc_idx].shape[1]==cooc_all[cooc_idx].shape[2]==self.feature_limit
            
            assert sample.shape[0]==self.feature_limit
  
        sample = {'features': (sample,cooc_all), 'label': label}
        # if self.transform:
        # sample['image'] = self.transform(sample['image'])

        return sample

    def collate_fn(self,batch):
        # data = []
        # for item in batch:
        #     tuple_vals = [torch.FloatTensor(item_inner) for item_inner in item['features'][1]]
        #     # print len(tuple_vals), len(item['features'][0]),len(item['features'][1]),len(item['features'])
        #     tuple_vals = (torch.FloatTensor(item['features'][0]),tuple_vals)
        #     data.append(tuple_vals)
        data = [(torch.FloatTensor(item['features'][0]),torch.FloatTensor(item['features'][1])) for item in batch]
        target = [item['label'] for item in batch]
        target = torch.FloatTensor(target)
        return {'features':data, 'label':target}



class UCF_dataset_gt_vec(Dataset):
    def __init__(self, text_file, feature_limit):
        self.anno_file = text_file
        self.files = util.readLinesFromFile(text_file)
        self.feature_limit = feature_limit

        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        train_file_curr = self.files[idx]
        anno = train_file_curr.split(' ')
        label = anno[2:]
        train_file_curr = anno[0]
        gt_file_curr = anno[1]
        # print gt_file_curr,label

        label =np.array([int(label) for label in label]).astype(float)
        label = label/np.sum(label)

        sample = np.load(train_file_curr)
        gt_vec = np.load(gt_file_curr)
        assert sample.shape[0]==gt_vec.shape[0]

        if sample.shape[0]>self.feature_limit and self.feature_limit is not None:
            sample_big = sample
            gt_vec_big = gt_vec

            while True:
                idx_start = sample_big.shape[0] - self.feature_limit
                idx_start = np.random.randint(idx_start+1)
                sample = sample_big[idx_start:idx_start+self.feature_limit]
                gt_vec = gt_vec_big[idx_start:idx_start+self.feature_limit]
                if np.sum(gt_vec)>0:
                    break
                assert sample.shape[0]==self.feature_limit
        


        # image = Image.open(train_file_curr)

        sample = {'features': sample, 'label': label, 'gt_vec': gt_vec}
        # if self.transform:
        # sample['image'] = self.transform(sample['image'])

        return sample

    def collate_fn(self,batch):
        data = [torch.FloatTensor(item['features']) for item in batch]
        gt_vec = [torch.FloatTensor(item['gt_vec']) for item in batch]
        target = [item['label'] for item in batch]
        target = torch.FloatTensor(target)
        return {'features':data, 'label':target, 'gt_vec':gt_vec}



def main():
    print 'hello'

    train_file = '../data/ucf101/train_test_files/train.txt'
    limit = 500
    model_name = 'just_mill'
    network_params = dict(n_classes=20, deno = 8, init=False )

    criterion = MultiCrossEntropy().cuda()
    
    train_data = UCF_dataset(train_file, limit)
    train_dataloader = DataLoader(train_data, collate_fn = train_data.collate_fn,
                        batch_size=10,
                        shuffle=False)
    network = models.get(model_name,network_params)
    model = network.model
    model = model.cuda()
    # net = models.Network(n_classes= 20, deno = 8)
    # print net.model
    # net.model = net.model.cuda()
    # input = np.zeros((32,2048))
    # input = torch.Tensor(input).cuda()
    # input = Variable(input)
    # output, pmf = net.model(input)
    
    optimizer = torch.optim.Adam(network.get_lr_list([1e-6]),weight_decay=0)
    print len(train_dataloader)
    
    # exit = True

    for num_epoch in range(500):

        labels = []
        preds = []

        for num_iter, train_batch in enumerate(train_dataloader):
            # print num_iter
            sample = train_batch['features']
            # [0].cuda()
            label = train_batch['label'].cuda()

            print label.size()
            print len(sample)
            print sample[0].size()

            # print labels.size()
            raw_input()

            out,pmf = model.forward(sample)
            preds.append(pmf.unsqueeze(0))
            labels.append(label)


        preds = torch.cat(preds,0)
        labels = torch.cat(labels,0)
        loss = criterion(labels, preds)
        # raw_input()
            # print pmf.size()
            
        optimizer.zero_grad()

        # loss = model.multi_ce(labels, pmf)

        loss.backward()
        optimizer.step()
            

        loss_val = loss.data[0].cpu().numpy()
        
        labels = labels.data.cpu().numpy()
        preds = torch.nn.functional.softmax(preds).data.cpu().numpy()
        
        # ,np.argmax(preds,axis=1)
        accu =  np.sum(np.argmax(labels,axis=1)==np.argmax(preds,axis=1))/float(labels.shape[0])
        print num_epoch, loss_val, accu
            # print torch.nn.functional.softmax(preds).data.cpu().numpy()




if __name__=='__main__':
    main()