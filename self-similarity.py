import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from math import exp, sqrt


##########################			Leo's little Helpers		##########################

def sub_abs(x0, x1): 
	# built-in func 'abs' somehow returning weird values!!

	if x0 > x1:
		return x0-x1
	return x1-x0

def bgr_to_rgb(img): 
	# opencv color-format: bgr
	# matplotlib: rgb

	b,g,r = cv.split(img)
	return cv.merge([r,g,b])

def show_img_as_heatmap(img, heat_map, y, x, yp, xp, cor_size, patch_size): 
	# show single correlation descriptor
	#
	# ( 0, 0 ) current scaled-up image patch
	# ( 0, 1 ) original image with correlation-square (red), patch-square (blue)
	# (  1   ) correlations as 3D-heatmap


	height, width = heat_map.shape

	fig = plt.figure(figsize=((12, 12)))

	ax1 = plt.subplot2grid((3, 2), (0, 0))
	ax1.imshow(bgr_to_rgb(img[yp:yp+patch_size, xp:xp+patch_size, :]))

	ax2 = plt.subplot2grid((3, 2), (0, 1))
	ax2.imshow(bgr_to_rgb(img))

	ax2.add_patch(patches.Rectangle((x, y), cor_size, cor_size, fill=False, linewidth=4, edgecolor="blue"))
	ax2.add_patch(patches.Rectangle((xp, yp), patch_size, patch_size, fill=False, linewidth=2, edgecolor="red"))

	X, Y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))
	Z = heat_map

	ax3 = plt.subplot2grid((3, 2), (1, 0), projection='3d', colspan=2, rowspan=2)

	surf = ax3.plot_surface(X, Y, Z,
						  rstride = 1,
						  cstride = 1,
						  cmap = cm.coolwarm,
						  linewidth = 0.5,
						  antialiased = True)


	fig.colorbar(surf, 
			 shrink=0.8, 
			 aspect=16,
			 orientation = 'vertical')
	 
	ax3.view_init(elev=50, azim=5)
	ax3.dist=8 

	plt.show()



##########################			SELF-SIMILARITY			##########################


def calc_patch(img, yp, xp, yc, xc, patch_size):
	# calucalte 'sum of squares' 
	# simply subtracting 2 patches from each other
	# squaring the result for each pixel: error --> high error

	assert(len(img.shape) == 3 and img.shape[2] == 3)

	sim = 0
	for y_off in range(patch_size):
		for x_off in range(patch_size):
			color_diff = 0
			for c in range(img.shape[2]):
				color_diff += sub_abs(img[yp+y_off][xp+x_off][c], img[yc+y_off][xc+x_off][c])
			sim += color_diff**2
	return sim


def calc_descriptors(img, y, x, yp, xp, cor_size, patch_size):
	# calculate single self-similarity descriptor for a certain patch_pos (xp, yp) / correlation_sqare(x, y) TODO: correlation_circle?!

	desc_img = np.zeros((cor_size-patch_size+1, cor_size-patch_size+1)) # +1 ?!
	for yc in range(0, cor_size-patch_size, 1):
		for xc in range(0, cor_size-patch_size, 1):
			difference = calc_patch(img, yp, xp, y+yc, x+xc, patch_size)
			desc_img[xc][yc] = exp(-(difference/300000)) # (x, y) ?!
			
			# var_noise = 3000000
			#
			# Constant corresponding to acceptable photometric variations
			# (in colour, illumination or due to noise) which will be
			# suppressed when calculating the bin values. The default value
			# is a good starting point, and used for all experiments in
			# Chatfield et al., but this parameter should be set empirically.
			# A good guideline might be:
			# 	parms.patch_size^2*(image_channel_count)*(estimated_variance)

	# only show heatmap in after certain num of steps in patch_pos --> more significant changes in heatmap
	if x%10 == 0 and y%10 == 0:# and x >= 60 and y >= 50:
		show_img_as_heatmap(img, desc_img, y, x, yp, xp, cor_size, patch_size)


def self_similarity(filename, cor_size=80, patch_size=5, step=10):
	# load a color-image from file 
	# calculate many self-similarity descriptors for all patch_pos in the img
	# 
	# patch_size needs to be odd
	#
	# TODO: Save descriptors in file
	# TODO: Compare with other descriptors
	# TODO: Repeat for different sizes --> Scale-invariant

	# TODO: cor_size --> cor_radius!!

	assert(cor_size%2 == 0 and patch_size%2 == 1)

	img = cv.imread(filename)
	img = cv.resize(img, (200, 200))

	assert(len(img.shape) == 3 and img.shape[2] == 3)

	height, width, c = img.shape

	correlation_img = np.zeros((height-(cor_size/2), width-(cor_size/2))) # +1 ?!

	for y in range(0, height-cor_size, step):
		for x in range(0, width-cor_size, step):
			yp = y+(cor_size/2)-((patch_size-1)/2)
			xp = x+(cor_size/2)-((patch_size-1)/2)
			calc_descriptors(img, y, x, yp, xp, cor_size, patch_size)


if __name__ == "__main__":
	self_similarity(".\\tests\\flower-01.jpg")