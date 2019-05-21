# for reading DICOM files obtained from ultrasound machines,
# letting user draw ROI, and outputting intensity-time graph

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
		self.mask = np.zeros([row, col], dtype = int)
		self.masked_dcm = np.zeros([row, col, 3])

	def onselect(self, verts):
		path = Path(verts)
		self.canvas.draw_idle()
		self.path = path

	def get_mask(self, row, col, dcm, frame_n):
		for i in range(row):
			for j in range(col):
				# matplotlib.Path.contains_points returns True if the point is inside the ROI
				# Path vertices are considered in different order than numpy arrays
				 if self.path.contains_points([(j,i)]) == [True]:
					 self.mask[i][j] = 1
				 else:
					 self.mask[i][j] = 0
		# Extracting pixel information from .dcm file based on ROI
		for i in range(row):
			for j in range(col):
				if self.mask[i][j] == 1:
					self.masked_dcm[i][j] = dcm.pixel_array[frame_n][i][j]
				else:
					self.masked_dcm[i][j] = 0

# Converting rgb image to grayscale.
def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Reading the .dcm file. Pixel data accessed via pixel_array and is stored as a
# 4D numpy array - (frame number, row, cols, rgb channel)
dcm = pydicom.dcmread(sys.argv[1])
frame_n = int(input("Enter desired frame to view: "))
row = dcm.pixel_array.shape[1] # number of rows of pixels in image
col = dcm.pixel_array.shape[2] # number of columns of pixels in image

# Drawing ROI on selected frame. If ROI is decided by user to be incorrect,
# they have the option to redraw.
img = dcm.pixel_array[frame_n]
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

# The previously drawn ROI is used to create a mask to extract the ROI from the
# original .dcm frame
roi.get_mask(row, col, dcm, frame_n)
gray_masked_dcm = rgb2gray(roi.masked_dcm)

# Test plots and info
plt.figure(1)
plt.subplot(131)
plt.imshow(img)

plt.subplot(132)
plt.imshow(gray_masked_dcm, cmap = 'gray')

plt.subplot(133)
plt.imshow(roi.mask)
plt.show()

mean_masked_dcm = np.mean(gray_masked_dcm)
print(mean_masked_dcm)
