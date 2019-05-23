# for reading DICOM files obtained from ultrasound machines,
# letting user draw ROI, and outputting intensity-time graph
import time

import sys
import matplotlib.pyplot as plt
import pydicom
import numpy as np
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import matplotlib.patches as patches

class ROIPolygon(object):

	def __init__(self, ax, row, col):
		self.canvas = ax.figure.canvas
		self.poly = PolygonSelector(ax,
									self.onselect,
									lineprops = dict(color = 'g', alpha = 1),
									markerprops = dict(mec = 'g', mfc = 'g', alpha = 1))
		self.path = None

	def onselect(self, verts):
		path = Path(verts)
		self.canvas.draw_idle()
		self.path = path

def draw_roi(img, row, col):
	q = 'n'
	while q == 'n' or q == 'N':
		fig, ax = plt.subplots()
		plt.imshow(img)
		plt.show(block = False)
		print('Draw desired ROI')

		roi = ROIPolygon(ax, row, col)

		plt.imshow(img)
		plt.show()
		# Overlaying ROI onto image
		patch = patches.PathPatch(roi.path,
								facecolor = 'green',
								alpha = 0.5)
		fig, ax = plt.subplots()
		plt.imshow(img)
		ax.add_patch(patch)
		plt.show(block = False)

		q = input("Is this ROI correct? y/n: ")
		while True:
			if q == 'y' or q == 'Y' or q == 'n' or q == 'N':
				break
			else:
				print('Please enter y or n')
	return roi

# To initialize mask
def get_mask(drawn_roi, row, col):
	# matplotlib.Path.contains_points returns True if the point is inside the ROI
	# Path vertices are considered in different order than numpy arrays
	mask = np.zeros([row, col], dtype = int)
	for i in np.arange(row):
		for j in np.arange(col):
			 if drawn_roi.path.contains_points([(j,i)]) == [True]:
				 mask[i][j] = 1
	return mask

# Extract image info based on mask
# Returns average of
def mask_extract(pydcm, mask, frame_all, row, col):
	masked_dcm = np.zeros([row, col, 3])
	gray_all_frames = np.zeros([frame_all, row, col]) # grayscale images for all frames
	gray_frame_avg = np.zeros([frame_all, 1]) # for avg brightness per frame
	for n in np.arange(frame_all):
		for i in np.arange(row):
			for j in np.arange(col):
				if mask[i][j] == 1:
					masked_dcm[i][j] = pydcm[n][i][j]
		g_frame = rgb2gray(masked_dcm)
		g_frame_avg = np.mean(g_frame)
		gray_all_frames[n] = g_frame
		gray_frame_avg[n] = g_frame_avg
	return gray_all_frames, gray_frame_avg

# Converting rgb image to grayscale.
def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Reading the .dcm file. Pixel data accessed via pixel_array and is stored as a
# 4D numpy array - (frame number, rows, columns, rgb channel)
if __name__ == "__main__":
	frame = int(input("Enter desired frame to view: "))

	d_pix_arr = pydicom.dcmread(sys.argv[1]).pixel_array
	frame_all = d_pix_arr.shape[0] # total number of frames in the dicom file
	r = d_pix_arr.shape[1] # number of rows of pixels in image
	c = d_pix_arr.shape[2] # number of columns of pixels in image

	# Draw ROI
	img = d_pix_arr[frame]
	roi = draw_roi(img, r, c)

	# Initialize mask
	start_time = time.time()
	mask = get_mask(roi, r, c)

	# Use mask to extract correct info
	grayed, grayed_avg = mask_extract(d_pix_arr, mask, frame_all, r, c)
	print("--- %s seconds to run get_mask in a loop---" % (time.time() - start_time))
	# print(grayed.shape)
	# print(grayed_avg.shape)
	plt.plot(np.arange(frame_all),grayed_avg)
	plt.show()

	# Test plots and info
	plt.figure(1)
	plt.subplot(121)
	plt.imshow(img)

	plt.subplot(122)
	plt.imshow(grayed[1], cmap = 'gray')
	plt.show()

	mean_masked_dcm = np.mean(grayed[1])
	print(mean_masked_dcm)
