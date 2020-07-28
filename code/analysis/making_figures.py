import sys
sys.path.append('./')
import os
from helpers import util, visualize
import numpy as np

def 

def main():
	# arr_curr = 0.1*np.ones((11,1))
	# arr_curr[0]=0.9
	# arr_curr[2]

	a = [[0.1,0.1,0.1,1],[0.1,1,0.5,1],[1,0.1,0,1],[1,1,1,1]]
	a = np.array(a)
	print a

	ans = np.linalg.solve(a,np.zeros((4,1)))
	print ans

if __name__=='__main__':
	main()