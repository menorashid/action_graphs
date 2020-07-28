'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math

def plot_em(X, Y, Z):
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	# Make data.
	plt.ion()
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)

	ax.set_xlabel('t')
	ax.set_ylabel('c')
	ax.set_zlabel('W')
	ax.set_aspect('equal')

	# Customize the z axis.
	# ax.set_zlim(-1.01, 1.01)
	# ax.zaxis.set_major_locator(LinearLocator(10))
	# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()
	

def get_rotated_Z(X, Y, Z, angle_deg):
	angle_rad = math.radians(angle_deg)
	X_New = math.cos(angle_rad)*X -math.sin(angle_rad)*Y
	Y_New = math.sin(angle_rad)*X +math.cos(angle_rad)*Y
	return X_New,Y_New,Z

def get_basic():
	X = np.arange(-1, 1, 0.01)
	Y = np.arange(-1, 1, 0.01)
	X, Y = np.meshgrid(X, Y)
	# R = np.sqrt(X**2 + Y**2)
	# Z = np.sin(R)

	Z = X**2 - Y**2
	return X, Y, Z
	

def get_our_one():
	y_00 = 1
	y_10 = 0

	y_01 = 0
	y_11 = 1

	X = np.arange(0, 1, 0.01)
	Y = np.arange(0, 1, 0.01)
	X, Y = np.meshgrid(X, Y)
	Z = np.zeros(X.shape)

	for r in range(X.shape[0]):
		for c in range(X.shape[1]):
			y = Y[r,c]
			x = X[r,c]

			y_s = (y_10 - y_00)*y + y_00
			y_e = (y_11 - y_01)*y + y_01
			z = (y_e - y_s)*x + y_s
			Z[r,c] = z

	return X,Y,Z

def main():
	X, Y, Z = get_basic()
	plot_em(X, Y, Z)

	X, Y, Z = get_rotated_Z(X, Y, Z, 45)
	plot_em(X, Y, Z)

	X, Y, Z = get_our_one()
	plot_em(X, Y, Z)

	raw_input()





if __name__=='__main__':
	main()



