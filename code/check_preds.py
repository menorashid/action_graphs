import h5py
import numpy as np
import os
from helpers import util, visualize
import scipy.io
def viz_pascal():

	filename = '../contextlocnet-master/data/with_det_scores_test.h5'
	# out_dir = '../scratch/voc_2007_test_scores/det_viz'
	# out_dir = '../scratch/voc_2007_test_scores/det_viz_smallest'
	out_dir = '../scratch/voc_2007_test_scores/det_viz_biggest'
	det_dir = '../scratch/voc_2007_test_scores/output_softmax'

	util.mkdir(out_dir)

	f = h5py.File(filename, 'r')
	labels = f['labels']

	label_strs = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
	legend_entries= ['608x800',
					 '496x656',
					 '400x544',
					 '720x960',
					 '864x1152',
					 '608x800-h',
					'496x656-h',
					'400x544-h',
					'720x960-h',
					'864x1152-h']

	legend_entries_meta= ['608x800','608x800-h',
					 '496x656', '496x656-h',
					 '400x544', '400x544-h',
					 '720x960', '720x960-h',
					 '864x1152', '864x1152-h']
					 
					# '496x656-h',
					# '400x544-h',
					# '720x960-h',
					# '864x1152-h']
	scales_to_keep = [9]

	for idx_test, label in enumerate(labels):
		gt_classes = np.where(label>0)[0]
		filename = str(idx_test+1)
		det_file = os.path.join(det_dir,filename+'.npy')
		dets_curr = np.load(det_file)

		for gt_class in gt_classes:
			label_str = label_strs[gt_class]
			print label_str
			out_dir_curr = os.path.join(out_dir, label_str)
			util.mkdir(out_dir_curr)
			x = range(dets_curr.shape[2])
			
			xAndYs = [(x,dets_curr[scale_idx,gt_class,:]) for scale_idx in scales_to_keep]
			# range(0,dets_curr.shape[0],2)]
			out_file = os.path.join(out_dir_curr, '0'*(5-len(filename))+filename+'.jpg')

			print out_file

			
			legend_entries = [legend_entries_meta[idx] for idx in scales_to_keep]
			visualize.plotSimple(xAndYs,out_file=out_file,title='Det Branch Output Multiscale',xlabel='ROIs',ylabel='Det Conf',legend_entries = legend_entries)

			# print np.sum(dets_curr[:,gt_class,:],axis=1)
			# print det_curr.shape

			# raw_input()
	for label_str in label_strs:
		out_dir_curr = os.path.join(out_dir,label_str)
		visualize.writeHTMLForFolder(out_dir_curr)




def main():
	mat_file = '../UntrimmedNet-master/matlab/test_set_final.mat'
	data = scipy.io.loadmat(mat_file)
	print data['test_videos']



if __name__=='__main__':
	main()
