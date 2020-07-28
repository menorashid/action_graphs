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
import cv2
import time

def save_flows((in_dir_meta, out_dir_u, out_dir_v, video_name, idx)):
	print idx
	in_dir_curr = os.path.join(in_dir_meta, video_name)
	in_files = glob.glob(os.path.join(in_dir_curr, '*.jpg'))
	in_files.sort()

	optical_flow = cv2.DualTVL1OpticalFlow_create()

	out_dir_u_curr = os.path.join(out_dir_u, video_name)
	out_dir_v_curr = os.path.join(out_dir_v, video_name)
	
	util.mkdir(out_dir_u_curr)
	util.mkdir(out_dir_v_curr)

	for idx_in_file, in_file in enumerate(in_files[:-1]):
		im1 = cv2.imread(in_file, cv2.IMREAD_GRAYSCALE)
		im2 = cv2.imread(in_files[idx_in_file+1], cv2.IMREAD_GRAYSCALE)

		flow = optical_flow.calc(im1, im2, None)
		flow = np.clip(flow, -20, 20)
		flow = (flow+20)/40 * 255
		u = flow[:,:,0]
		v = flow[:,:,1]
		
		out_file_u = os.path.join(out_dir_u_curr, os.path.split(in_file)[1])
		out_file_v = os.path.join(out_dir_v_curr, os.path.split(in_file)[1])

		cv2.imwrite(out_file_u,u)
		cv2.imwrite(out_file_v,v)

def script_save_flows():
	dir_meta = '../data/ucf101'
	fps = 10
	small_dim= 256
	
	in_dir_meta = os.path.join(dir_meta,'val_data','rgb_'+str(fps)+'_fps_'+str(small_dim))
	out_dir_meta = os.path.join(dir_meta,'val_data','flo_'+str(fps)+'_fps_'+str(small_dim))
	
	out_dir_u = os.path.join(out_dir_meta, 'u')
	out_dir_v = os.path.join(out_dir_meta, 'v')

	util.makedirs(out_dir_u)
	util.makedirs(out_dir_v)

	video_names = [os.path.split(dir_curr)[1] for dir_curr in glob.glob(os.path.join(in_dir_meta,'*')) if os.path.isdir(dir_curr)]
	print len(video_names)
	
	args = []
	for idx_video_name, video_name in enumerate(video_names):
		file_check = os.path.join(out_dir_u, video_name, 'frame000001.jpg')
		if os.path.exists(file_check):
			continue

		args.append((in_dir_meta, out_dir_u, out_dir_v, video_name, idx_video_name))

	print len(args)
	# for arg in args:
	# 	print arg
	# 	save_flows(arg)
	# 	break

	pool = multiprocessing.Pool(multiprocessing.cpu_count()-2)
	pool.map(save_flows, args)

	# in_dir_meta = '../data/ucf101/rgb_ziss/jpegs_256'
	# video_name = 'v_CricketShot_g04_c01'
	# out_dir_u = '../scratch/check_u'
	# out_dir_v = '../scratch/check_v'
	# util.mkdir(out_dir_u)
	# util.mkdir(out_dir_v)

	# save_flows((in_dir_meta, out_dir_u, out_dir_v, video_name, 1))


def check_done(in_dir_meta, out_dir_u, out_dir_v, video_name, sample, idx):
	# print idx
	in_dir_curr = os.path.join(in_dir_meta, video_name)
	in_files = glob.glob(os.path.join(in_dir_curr, '*.jpg'))
	in_files.sort()
	# print len(in_files)
	in_files =in_files[::sample]
	# print len(in_files)


	# optical_flow = cv2.DualTVL1OpticalFlow_create()

	out_dir_u_curr = os.path.join(out_dir_u, video_name)
	out_dir_v_curr = os.path.join(out_dir_v, video_name)
	
	util.mkdir(out_dir_u_curr)
	util.mkdir(out_dir_v_curr)

	out_file_input = os.path.join(out_dir_u_curr,'list_input.txt')
	util.writeFile(out_file_input,in_files)

	out_file_final_u = os.path.join(out_dir_u,video_name,os.path.split(in_files[-2])[1])
	out_file_final_v = os.path.join(out_dir_v,video_name,os.path.split(in_files[-2])[1])
	# print out_file_final_u

	if os.path.exists(out_file_final_u) and os.path.exists(out_file_final_v):
		print video_name, 'DONE already'
		return 1
	return 0


def save_flows_gpu((in_dir_meta, out_dir_u, out_dir_v, video_name, sample, idx)):
	print idx
	in_dir_curr = os.path.join(in_dir_meta, video_name)
	in_files = glob.glob(os.path.join(in_dir_curr, '*.jpg'))
	in_files.sort()
	# print len(in_files)
	in_files =in_files[::sample]
	# print len(in_files)


	# optical_flow = cv2.DualTVL1OpticalFlow_create()

	out_dir_u_curr = os.path.join(out_dir_u, video_name)
	out_dir_v_curr = os.path.join(out_dir_v, video_name)
	
	util.mkdir(out_dir_u_curr)
	util.mkdir(out_dir_v_curr)

	out_file_input = os.path.join(out_dir_u_curr,'list_input.txt')
	util.writeFile(out_file_input,in_files)

	out_file_final_u = os.path.join(out_dir_u,video_name,os.path.split(in_files[-2])[1])
	out_file_final_v = os.path.join(out_dir_v,video_name,os.path.split(in_files[-2])[1])
	# print out_file_final_u

	if os.path.exists(out_file_final_u) and os.path.exists(out_file_final_v):
		print video_name, 'DONE already'
		return 1
	# return 0

	

	command = ['./optical_flow_try_2', out_file_input, out_dir_u_curr, out_dir_v_curr]
	command = ' '.join(command)
	# print command
	# t = time.time()
	os.popen(command)
	# print time.time()-t


def script_save_flows_gpu():
	dir_meta = '../data/ucf101'
	fps = 10
	small_dim= 256
	sample = 4
	
	# in_dir_meta = os.path.join(dir_meta,'val_data','rgb_'+str(fps)+'_fps_'+str(small_dim))
	# out_dir_meta = os.path.join(dir_meta,'val_data','flo_'+str(fps/float(sample))+'_fps_'+str(small_dim))
	
	in_dir_meta = os.path.join(dir_meta,'test_data','rgb_'+str(fps)+'_fps_'+str(small_dim))
	out_dir_meta = os.path.join(dir_meta,'test_data','flo_'+str(fps/float(sample))+'_fps_'+str(small_dim))
	

	out_dir_u = os.path.join(out_dir_meta, 'u')
	out_dir_v = os.path.join(out_dir_meta, 'v')

	util.makedirs(out_dir_u)
	util.makedirs(out_dir_v)

	video_names = [os.path.split(dir_curr)[1] for dir_curr in glob.glob(os.path.join(in_dir_meta,'*')) if os.path.isdir(dir_curr)]
	print len(video_names)
	video_names.sort()
	args = []
	to_del = []
	for idx_video_name, video_name in enumerate(video_names):
		# file_check = os.path.join(out_dir_u, video_name, 'frame000001.jpg')
		# if os.path.exists(file_check):
		# 	continue
		if not check_done(in_dir_meta, out_dir_u, out_dir_v, video_name, sample, idx_video_name):
			args.append((in_dir_meta, out_dir_u, out_dir_v, video_name, sample, idx_video_name))

	print len(args)

	# return
	args = args[100:]
	print len(args)
	# for arg in args:
	# 	print arg[-1]
		# save_flows_gpu(arg)
		# raw_input()
		# break

	pool = multiprocessing.Pool(4)
	# rets = 
	pool.map(save_flows_gpu, args)
	# print sum(rets)

	# in_dir_meta = '../data/ucf101/rgb_ziss/jpegs_256'
	# video_name = 'v_CricketShot_g04_c01'
	# out_dir_u = '../scratch/check_u'
	# out_dir_v = '../scratch/check_v'
	# util.mkdir(out_dir_u)
	# util.mkdir(out_dir_v)

	# save_flows((in_dir_meta, out_dir_u, out_dir_v, video_name, 1))



def script_checking_flows():
	in_dir_meta = '../data/ucf101/rgb_ziss/jpegs_256'
	video_name = 'v_CricketShot_g04_c01'
	out_dir_u = '../scratch/check_u_gpu'
	out_dir_v = '../scratch/check_v_gpu'
	util.mkdir(out_dir_u)
	util.mkdir(out_dir_v)

	# save_flows((in_dir_meta, out_dir_u, out_dir_v, video_name, 1))

	old_dir_u = '../data/ucf101/flow_ziss/tvl1_flow/u'
	old_dir_v = '../data/ucf101/flow_ziss/tvl1_flow/v'
	
	out_dir_diff_u = '../scratch/check_u_diff_gpu'
	out_dir_diff_v = '../scratch/check_v_diff_gpu'

	save_flows_gpu((in_dir_meta, out_dir_u, out_dir_v, video_name, 1, 0))

	raw_input()
	dir_pair_u = [os.path.join(dir_curr, video_name) for dir_curr in [old_dir_u, out_dir_u, out_dir_diff_u]]
	dir_pair_v = [os.path.join(dir_curr, video_name) for dir_curr in [old_dir_v, out_dir_v, out_dir_diff_v]]

	for old_dir, new_dir, out_dir_diff in [dir_pair_u, dir_pair_v]:
		util.makedirs(out_dir_diff)
		print old_dir, new_dir
		im_files = glob.glob(os.path.join(old_dir, '*.jpg'))
		im_files.sort()

		for im_file in im_files:
			flo_old = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE).astype(float)
			flo_new = cv2.imread(os.path.join(new_dir, os.path.split(im_file)[1]), cv2.IMREAD_GRAYSCALE)[:,:-1].astype(float)
			print flo_old.shape, flo_new.shape

			
			print np.min(flo_old), np.max(flo_old)
			print np.min(flo_new), np.max(flo_new)


			diff = np.abs(flo_old-flo_new)

			print np.min(diff), np.max(diff)


			cv2.imwrite(os.path.join(out_dir_diff,os.path.split(im_file)[1]), diff)

		visualize.writeHTMLForFolder(out_dir_diff)



def dangerous_script_never_run():
	dir_meta = '../data/ucf101'
	file_curr = os.path.join(dir_meta,'test_data','completed.txt')
	to_del = util.readLinesFromFile(file_curr)
	for folder_curr in to_del:
		command_curr = 'rm -rf '+folder_curr
		print command_curr
		os.popen('rm -rf '+folder_curr)



def main():


	# script_checking_flows()
	script_save_flows_gpu()

	return 
	print 'hello'
	im1 = '../data/ucf101/val_data/rgb_10_fps_256/video_validation_0000001/frame000003.jpg'
	im2 = '../data/ucf101/val_data/rgb_10_fps_256/video_validation_0000001/frame000004.jpg'

	im1 = cv2.imread(im1,cv2.IMREAD_GRAYSCALE)
	im2 = cv2.imread(im2,cv2.IMREAD_GRAYSCALE)
	print im1.shape
	print im2.shape
	optical_flow = cv2.DualTVL1OpticalFlow_create()
	flow = optical_flow.calc(im1, im2, None)
	print flow.shape, np.min(flow), np.max(flow)
	flow = np.clip(flow, -20, 20)
	print flow.shape, np.min(flow), np.max(flow)
	flow = (flow+20)/40 * 255
	print flow.shape, np.min(flow), np.max(flow)
	u = flow[:,:,0]
	v = flow[:,:,1]
	out_file_u = '../scratch/check_u.jpg' 
	out_file_v = '../scratch/check_v.jpg'
	combo_file = '../scratch/combo.jpg'
	combo_file_cv2 = '../scratch/combo_cv2.jpg'
	cv2.imwrite(out_file_u,u)
	cv2.imwrite(out_file_v,v)

	u = cv2.imread(out_file_u,cv2.IMREAD_GRAYSCALE)
	v = cv2.imread(out_file_v,cv2.IMREAD_GRAYSCALE)
	print u.shape, np.min(u), np.max(u)
	print v.shape, np.min(v), np.max(v)
	
	combo = np.concatenate([u[:,:,np.newaxis],v[:,:,np.newaxis],128*np.ones((v.shape[0],v.shape[1],1))],axis = 2)

	print u.shape, np.min(u), np.max(u)
	print v.shape, np.min(v), np.max(v)
	print combo.shape, np.min(combo), np.max(combo)

	scipy.misc.imsave(combo_file,combo.astype(dtype = np.uint32))
	cv2.imwrite(combo_file_cv2,combo)


if __name__=='__main__':
	main()
