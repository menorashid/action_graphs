import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob

def main():
	data_dir_meta = '../data/sheep'
	glob_paths = [os.path.join(data_dir_meta,'set*','*.jpg'),os.path.join(data_dir_meta,'dataset','set*','*.jpg')]
	out_files = ['../data/sheep/sfi_list.txt','../data/sheep/sff_list.txt']
	for glob_path, out_file in zip(glob_paths,out_files):
		im_list = glob.glob(glob_path)
		# print im_list
		im_list = [os.path.split(im_curr)[1] for im_curr in im_list]
		print len(im_list),len(set(im_list))
		print out_file
		util.writeFile(out_file,im_list)


	
	# im_list = [os.path.split(im_curr)[1] for im_curr in im_list if '#' in im_curr]
	# im_list.sort()
	# im_list = list(set(im_list))
	# print len(im_list)
	# picture_nums = [im_curr[im_curr.index('#')+1:im_curr.rindex('.')] for im_curr in im_list]
	# # picture_nums= list(set(picture_nums))
	# print len(picture_nums)
	# picture_nums.sort()
	# for p in list(set(picture_nums)):
	# 	print p, picture_nums.count(p)



if __name__=='__main__':
	main()

