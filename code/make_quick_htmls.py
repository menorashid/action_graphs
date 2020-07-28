import numpy as np
import os
import glob
from helpers import visualize,util
from globals import class_names

def main():
	dir_vids = '../data/ucf101/test_data/rgb_10_fps_256'
	dir_curr = 'video_test_0000273'
	dir_curr = 'video_test_0001324'
	dir_curr = 'video_test_0001484'
	visualize.writeHTMLForFolder(os.path.join(dir_vids,dir_curr))


if __name__=='__main__':
	main()