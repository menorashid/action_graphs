import sys
sys.path.append('./')
print sys.path
import os
from helpers import util, visualize
import glob
import numpy as np
import math
import h5py
import scipy.misc


def make_train_test_splits_horse_based():
	data_dir_meta ='../data/horse_51'
	anno_file = os.path.join(data_dir_meta,'Painscore_expert_51images.csv')
	
	im_dir = data_dir_meta

	out_dir_split = os.path.join(data_dir_meta,'train_test_split_horse_based')
	util.mkdir(out_dir_split)
	

	anno = util.readLinesFromFile(anno_file)

	anno = anno[1:]
	anno = [[int(anno_curr.split(',')[val]) for val in [0,1,-2,-1]] for anno_curr in anno]
	anno = np.array(anno)

	exclude_idx = anno[:,-1]!=anno[:,-2]
	print np.sum(exclude_idx)

	kept_files = anno[~exclude_idx,:]
	horse_ids = np.unique(kept_files[:,1])
	num_splits = len(horse_ids)
	
	for split_num in range(num_splits):
		arr_idx = [idx for idx in range(num_splits) if idx!=split_num]
		horse_ids_kept = horse_ids[arr_idx]

		idx_train = np.where(np.isin(kept_files[:,1],horse_ids_kept))[0]

		# np.concatenate([pos_idx_split[val] for val in arr_idx]+[neg_idx_split[val] for val in arr_idx])
		idx_test = np.where(kept_files[:,1]==horse_ids[split_num])[0]
		# np.array(list(pos_idx_split[split_num]) + list(neg_idx_split[split_num]))
		idx_all = [idx_train, idx_test]
		out_files = [os.path.join(out_dir_split,'train_'+str(split_num)+'.txt'),os.path.join(out_dir_split,'test_'+str(split_num)+'.txt')]
		for idx_curr, out_file_curr in zip(idx_all,out_files):
			annos = kept_files[idx_curr,:]
			assert np.all(annos[:,-1]==annos[:,-2])
			lines_all = [os.path.join(im_dir,str(anno_curr[0])+'.jpg')+' '+str(anno_curr[-1]) for anno_curr in annos]
			print len(lines_all),out_file_curr,lines_all[0]
			util.writeFile(out_file_curr,lines_all)

def make_train_test_splits():
	data_dir_meta ='../data/horse_51'
	anno_file = os.path.join(data_dir_meta,'Painscore_expert_51images.csv')
	
	im_dir = data_dir_meta

	out_dir_split = os.path.join(data_dir_meta,'train_test_split')
	util.mkdir(out_dir_split)
	

	anno = util.readLinesFromFile(anno_file)

	anno = anno[1:]
	anno = [[int(anno_curr.split(',')[val]) for val in [0,-2,-1]] for anno_curr in anno]
	anno = np.array(anno)

	exclude_idx = anno[:,1]!=anno[:,2]
	print np.sum(exclude_idx)

	kept_files = anno[~exclude_idx,:]
	pos_idx = np.where(kept_files[:,1]==1)[0]
	neg_idx = np.where(kept_files[:,1]==0)[0]
	
	num_splits = 5

	size_pos = pos_idx.shape[0]
	size_neg = neg_idx.shape[0]
	split_pos_size = int(math.floor(size_pos*0.2))
	split_neg_size = int(math.floor(size_neg*0.2))

	np.random.shuffle(pos_idx)
	np.random.shuffle(neg_idx)

	pos_idx_split = np.array_split(pos_idx,num_splits)
	neg_idx_split = np.array_split(neg_idx,num_splits)

	for split_num in range(num_splits):
		arr_idx = [idx for idx in range(num_splits) if idx!=split_num]

		idx_train = np.concatenate([pos_idx_split[val] for val in arr_idx]+[neg_idx_split[val] for val in arr_idx])
		idx_test = np.array(list(pos_idx_split[split_num]) + list(neg_idx_split[split_num]))
		idx_all = [idx_train, idx_test]
		out_files = [os.path.join(out_dir_split,'train_'+str(split_num)+'.txt'),os.path.join(out_dir_split,'test_'+str(split_num)+'.txt')]
		for idx_curr, out_file_curr in zip(idx_all,out_files):
			annos = kept_files[idx_curr,:]
			assert np.all(annos[:,1]==annos[:,2])
			lines_all = [os.path.join(im_dir,str(anno_curr[0])+'.jpg')+' '+str(anno_curr[1]) for anno_curr in annos]
			print len(lines_all),out_file_curr,lines_all[0]
			util.writeFile(out_file_curr,lines_all)

def save_h5py():
	data_dir_meta ='../data/horse_51'
	im_dir = data_dir_meta
	out_dir_split = os.path.join(data_dir_meta,'train_test_split')
	util.mkdir(out_dir_split)

	split_num = 5
	im_size = 256

	for split_curr in range(split_num):
		for file_curr in ['train','test']:
			text_file = os.path.join(out_dir_split,file_curr+'_'+str(split_curr)+'.txt')
			anno = util.readLinesFromFile(text_file)
			out_file_h5py = os.path.join(out_dir_split,file_curr+'_'+str(split_curr)+'.h5')
			if os.path.exists(out_file_h5py):
				continue;
			
			print out_file_h5py

			with h5py.File(out_file_h5py, "w") as f:
			    
			    data = f.create_dataset('data', (len(anno),im_size,im_size,3))
			    labels = f.create_dataset('labels', (len(anno),))
			    
			    for idx_anno_curr,anno_curr in enumerate(anno):
			    	im_file,label = anno_curr.split(' ')
			    	im = scipy.misc.imread(im_file)
			    	im = scipy.misc.imresize(im,(im_size,im_size))
			    	data[idx_anno_curr,:,:,:] = im
			    	labels[idx_anno_curr] = int(label)


def main():
	# make_train_test_splits_horse_based()
	dir_server = '/disk3'

	im_dir =os.path.join(dir_server,'maheen_data','eccv_18','data/horse_51')
	data_dir_meta = '../data/horse_51'
	out_dir_split = os.path.join(data_dir_meta,'train_test_split_horse_based')
	for split_num in range(6):

		test_file = os.path.join(out_dir_split,'test_'+str(split_num)+'.txt')
		out_file_html = test_file[:test_file.rindex('.')]+'.html'
		ims = [line_curr.split(' ')[0] for line_curr in util.readLinesFromFile(test_file)]
		ims = [im.replace(data_dir_meta,im_dir) for im in ims]
		print ims[0]
		ims = [util.getRelPath(im,dir_server) for im in ims]
		# print ims
		visualize.writeHTML(out_file_html,[ims],[ims],224,224)
		print out_file_html
	
	
	

	
	

if __name__=='__main__':
	main()