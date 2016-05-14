import numpy as np
import cv2 as cv
import struct

import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from math import exp, sqrt, sin, cos, pi


##########################			Leo's little Helpers		##########################

def distance(y0, x0):
	return sqrt(y0**2 + x0**2)

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

def center_to_top_left(x, y, sqr_size):
	# convert center pos to top-left corner pos

	assert(sqr_size%2 == 1)
	
	offset = (sqr_size-1)/2

	return (x-offset, y-offset)

def show_img_as_heatmap(img, heat_map, yp, xp, cor_radius, patch_size): 
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


	correlation_circle = patches.Circle((xp, yp), cor_radius, fill=False, linewidth=4, edgecolor="blue")
	patch_square = patches.Rectangle(center_to_top_left(xp, yp, patch_size), patch_size, patch_size, fill=False, linewidth=2, edgecolor="red")

	ax2.add_patch(correlation_circle)
	ax2.add_patch(patch_square)

	X, Y = np.meshgrid(np.arange(0, height, 1), np.arange(0, width, 1))
	Z = heat_map

	ax3 = plt.subplot2grid((3, 2), (1, 0), projection='3d', colspan=2, rowspan=2)

	ax3.set_ylabel("Hallo")

	surf = ax3.plot_surface(Y, X, Z,
						  rstride = 1,
						  cstride = 1,
						  cmap = cm.coolwarm,
						  linewidth = 0.5,
						  antialiased = True)


	fig.colorbar(surf, 
			 shrink=0.8, 
			 aspect=16,
			 orientation = 'vertical')
	 
	ax3.view_init(elev=70, azim=5)
	ax3.dist=8

	plt.show()



##########################			SELF-SIMILARITY			##########################


def calc_patch(img, yp, xp, yc, xc, patch_size):
	# calucalte 'sum of squares' 
	# simply pixelwise subtracting 2 patches from each other
	# squaring the result for each pixel: error --> high error

	#assert (yp, xp) and (yc, xc) as center positions

	assert(patch_size%2 == 1)
	assert(len(img.shape) == 3 and img.shape[2] == 3)

	# center offset
	offset = (patch_size-1)/2

	diff = 0
	for y_off in range(-offset, offset, 1):
		for x_off in range(-offset, offset, 1):
			color_diff = 0
			for c in range(img.shape[2]):
				color_diff += sub_abs(img[yp+y_off][xp+x_off][c], img[yc+y_off][xc+x_off][c])
			diff += color_diff**2
	return diff

def get_theta(y_off, x_off):
	# returns angle [0 .. 360] from center to pos_offset
	# !!! IV. Quadrant: 0 - 90; III. Quadrant: 91 - 179; IV. Quadrant: 180 - 270; III. Quadrant: 271 - 359 !!!
	#
	# params: 	y_offset [-cor_size .. cor_size]
	#			x_offset [-cor_size .. cor_size]
	# 

	assert(y_off != 0 or x_off != 0)


	# III. Quadrant
	if x_off < 0 and y_off > 0:
		return 90. + get_theta(-x_off, y_off)
	# II. Quadrant
	elif x_off <= 0 and y_off <= 0:
		return 180 + get_theta(-y_off, -x_off)
	# I. Quadrant
	elif x_off > 0 and y_off < 0:
		return 270. + get_theta(x_off, -y_off)

	assert(y_off >= 0 and x_off >= 0)

	if y_off >= x_off:
		# --> [0 .. 45]
		return float(x_off)/y_off * 45. 
	else:
		# --> ]45 .. 90]
		return 90. - (float(y_off)/x_off * 45.)

	# weird: 0.0 and 360.0 both occur!!!!!!!!!!!!!!


def calc_descriptors(img, yp, xp, cor_radius, patch_size, radius=4, perimeter=20):
	# calculate single self-similarity descriptor for a certain patch_center (yp, xp) 
	# and corresponding correlation_square (y_off, x_off) with size (2*cor_radius x 2*cor_radius) -- filter_mask --> correlation_circle
	# save each result in correlation_img (heat_map)
	#
	# generate simple 1D-Descriptor from correlation_img containing char_values [0 .. 255]

	# (y_off, x_off) are offsets relative to the patch_center (yp, xp) 


	var_noise = 200000

	# var_noise = 3000000
	# (colour, illumination or due to noise)
	# parms.patch_size^2*(image_channel_count)*(estimated_variance)

	theta_step = (360./perimeter)
	rho_step = (float(cor_radius)/radius)

	circle_descriptor = []
	for rho_section in range(radius):
		temp = []
		for theta_section in range(perimeter):
			temp.append([])
		circle_descriptor.append(temp)


	correlation_img = np.zeros((cor_radius*2+1, cor_radius*2+1))

	for y_off in range(-cor_radius, cor_radius +1, 1): # +1 necessary!
		for x_off in range(-cor_radius, cor_radius +1, 1): # +1 necessary!
			if distance(y_off, x_off) >= cor_radius or (y_off == 0 and x_off == 0): # mask out /skip values with "radius to patch_center" > cor_radius
				continue

			difference = calc_patch(img, yp, xp, yp+y_off, xp+x_off, patch_size) # calc overall difference in pixels between patch and its offset-patch

			# similarity [0.0 .. 1.0] (with 0.0 --> not similar and 1.0 --> very similar)
			similarity = exp(-(difference/var_noise)) 

			correlation_img[y_off+cor_radius][x_off+cor_radius] = similarity # (x, y) ?! ?!?!?!?!?!
			#correlation_img[y_off+cor_radius][x_off+cor_radius] = 1 # Debugging


			# choose 2D-Descriptor section
			rho = distance(y_off, x_off)
			theta = get_theta(y_off, x_off)

			rho_section = int(rho // rho_step)
			theta_section = int(theta // theta_step)

			circle_descriptor[rho_section][theta_section].append(similarity)

	# create 1D-Descriptor from 2D-Circle Descriptor (circle_descriptor)
	descriptor = []

	for rho_section in range(radius):
		for theta_section in range(perimeter):
			count = len(circle_descriptor[rho_section][theta_section])
			avg = sum(circle_descriptor[rho_section][theta_section]) / count
			
			descriptor.append(int(avg*255))

	# only show heatmap in after certain num of steps in patch_pos --> more significant changes in heatmap
	#if yp%10 == 0 and xp%10 == 0:
	show_img_as_heatmap(img, correlation_img, yp, xp, cor_radius, patch_size)

	return descriptor


def self_similarity(img, cor_radius=40, patch_size=5, step=10): # usually step should be 3!

	# calculate many self-similarity descriptors for all patch_pos in the img
	# 
	# patch_size needs to be odd
	#

	# (yp, xp) == coord of patch (center)
	# offset: corner and edge (yp, xp)-values, which don't fit a complete correlation-circle into the image, have to be skipped
	
	## (ys, xs) == coord of correlation_square (center)

		
	# TODO: Save descriptors TO file

	# TODO: create descriptors with missing data (edge/corner descriptors)

	# TODO: only save interesting descriptors 50% -> 0.0 and 50% -> 1.0

	assert(patch_size%2 == 1)
	assert(len(img.shape) == 3 and img.shape[2] == 3)

	height, width, c = img.shape

	offset = cor_radius+((patch_size-1)/2)

	descriptors = []

	for yp in range(offset, height-offset +1, step): # +1 necessary --> otherwise yp = height-cor_radius would be skipped
		for xp in range(offset, width-offset +1, step): # same as yp
			desc = calc_descriptors(img, yp, xp, cor_radius, patch_size)
			descriptors.append(desc)

	return descriptors


def save_descriptors(filename, descriptors):
	# descriptors_count
	# radius & perimeter vs len(descriptor) ?!

	assert(len(descriptors) > 0)

	with open(filename, "wb") as f:
		f.write(struct.pack('>I', len(descriptors)))
		f.write(struct.pack('>I', len(descriptors[0])))

		for desc in descriptors:
			for section in desc:
				f.write(struct.pack('B', section))

	pass

def load_descriptors(filename):

	descriptors = []
	with open(filename, "rb") as f:

		count = struct.unpack('>I', f.read(4))[0]
		descriptor_size = struct.unpack('>I', f.read(4))[0]

		for i in range(count):
			desc = []
			for j in range(descriptor_size):
				section = struct.unpack('B', f.read(1))[0]
				desc.append(section)
			descriptors.append(desc)

	return descriptors

def compare_descriptors(desc1, desc2):
	# compare 2 descriptors and return overall difference [0 .. (255**2)*len(desc1)]

	assert(len(desc1) == len(desc2))

	diff = 0
	for i in range(len(descriptors[i])):
		diff += (desc2 - desc1)**2

	return diff

def main(filename):
	# load a color-image from file and create its self-similarity descriptors
	#

	# TODO: Repeat for different sizes --> Scale-invariant
	# TODO: Compare with other descriptors

	img = cv.imread(filename)
	img = cv.resize(img, (150, 100))

	#a = time.time()
	descriptors = self_similarity(img)
	#b = time.time()
	#print b-a

	save_descriptors("test.dsc", descriptors)
	#c = time.time()
	#print c-b

	reconstuction = load_descriptors("test.dsc")
	#d = time.time()
	#print d-c


	# check if saving/loading descriptors work correctly
	assert(len(descriptors) == len(reconstuction))
	for i in range(len(descriptors)):
		assert(len(descriptors[i]) == len(reconstuction[i]))
		for j in range(len(descriptors[i])):
			assert(descriptors[i][j] == reconstuction[i][j])




if __name__ == "__main__":
	main(".\\tests\\flower-01.jpg")