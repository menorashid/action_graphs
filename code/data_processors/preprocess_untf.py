# import cv2
import sys
sys.path.append('./')
import os
from helpers import util, visualize
import numpy as np
import glob
import scipy.misc
import scipy.stats
import multiprocessing
import subprocess
import scipy.io

import pickle

dir_meta = '../data/ucf101'
dir_meta_features = '../data/untf'

def save_npy():
	dir_feat = os.path.join(dir_meta_features,'validation')
	feat_files = glob.glob(os.path.join(dir_feat,'*.pickle'))

	out_dir_feat = os.path.join(dir_meta_features,'npy')
	out_dir_time = os.path.join(dir_meta_features,'time')
	util.mkdir(out_dir_feat)
	util.mkdir(out_dir_time)

	print len(feat_files)
	# ['rgb_frame_timestamps', 'video_id', 'rgb_features', 'flow_frame_timestamps', 'labels_name', 'flow_features', 'extracted_fps', 'original_fps', 'labels_indices', 'video_duration']
	for idx_feat_file,feat_file in enumerate(feat_files):
		print idx_feat_file, len(feat_files)

		vid_name = os.path.split(feat_file)[1]
		vid_name = vid_name[:vid_name.rindex('.')]
		print vid_name

		out_file_feat = os.path.join(out_dir_feat, vid_name+'.npy')
		out_file_time = os.path.join(out_dir_time, vid_name+'.npy')

		# if os.path.exists(out_file_feat):
		# 	continue

		try:
			feature_curr = pickle.load(open(feat_file,'r'))
		except:
			print 'ERROR', feat_file
			raw_input()

		
		rgb = feature_curr['rgb_features']
		flow = feature_curr['flow_features']
		
		fps = feature_curr['extracted_fps']
		o_fps = feature_curr['original_fps']
		
		print fps,o_fps
		print feature_curr.keys()
		print flow.shape
		print feature_curr['video_duration']
		raw_input()


		rgb_time = feature_curr['rgb_frame_timestamps']
		flow_time = feature_curr['flow_frame_timestamps']

		assert np.all(rgb_time==flow_time)
		assert rgb.shape[1]==flow.shape[1]

		out_feat = np.concatenate([rgb,flow],axis = 0).T

		# print out_feat.shape
		# print out_file_feat
		np.save(out_file_feat, out_feat)

		out_time = rgb_time
		print out_time.shape
		print out_file_time
		np.save(out_file_time, out_time)


def write_train_test_files():
	post_pend = '_untf.txt'
	data_dir = '../data/ucf101/train_test_files'
	
	new_data_path = '../data/untf/npy'
	

	old_data_path = '../data/i3d_features/Thumos14-I3D-JOINTFeatures_val'
	train_file = os.path.join(data_dir, 'train.txt')
	# old_data_path = '../data/i3d_features/Thumos14-I3D-JOINTFeatures_test'
	# train_file = os.path.join(data_dir, 'test.txt')

	replace_str = [old_data_path, new_data_path]

	out_file = train_file[:train_file.rindex('.')]+post_pend
	lines = util.readLinesFromFile(train_file)
	new_lines = []
	for idx_line, line in enumerate(lines):
		# print line
		line = line.replace(replace_str[0], replace_str[1])
		# print line
		# raw_input()
		npy_file = line.split(' ')[0]
		if not os.path.exists(npy_file):
			print 'skipping', idx_line, npy_file
			continue

		assert os.path.exists(npy_file)
		# raw_input()
		new_lines.append(line)

	print out_file, len(new_lines), len(lines)
	util.writeFile(out_file,new_lines)

	# print len(lines)
	# print lines[0]

	

def main():
	# save_npy()
	write_train_test_files()
	


if __name__=='__main__':
	main()

