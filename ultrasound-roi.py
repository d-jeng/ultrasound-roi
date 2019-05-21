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
		# self.masked_dcm = np.zeros([row, col, 3])

	def onselect(self, verts):
		path = Path(verts)
		self.canvas.draw_idle()
		self.path = path

	def get_mask(self, row, col, dcm_vid, frame_n):
		masked_dcm = np.zeros([row, col, 3])
		for i in np.arange(row):
			for j in np.arange(col):
				# matplotlib.Path.contains_points returns True if the point is inside the ROI
				# Path vertices are considered in different order than numpy arrays
				 if self.path.contains_points([(j,i)]) == [True]:
					 # self.masked_dcm[i][j] = dcm_vid.pixel_array[frame_n][i][j]
					 masked_dcm[i][j] = dcm_vid.pixel_array[frame_n][i][j]
				 else:
					 # self.masked_dcm[i][j] = 0
					 masked_dcm[i][j] = 0
		return masked_dcm

# Converting rgb image to grayscale.
def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Reading the .dcm file. Pixel data accessed via pixel_array and is stored as a
# 4D numpy array - (frame number, rows, columns, rgb channel)
dcm = pydicom.dcmread(sys.argv[1])
frame = int(input("Enter desired frame to view: "))
r = dcm.pixel_array.shape[1] # number of rows of pixels in image
c = dcm.pixel_array.shape[2] # number of columns of pixels in image

# Drawing ROI on selected frame. If ROI is decided by user to be incorrect,
# they have the option to redraw.
img = dcm.pixel_array[frame]
q = 'n'
while q == 'n' or q == 'N':
	fig, ax = plt.subplots()
	plt.imshow(img)
	plt.show(block = False)

	print('Draw desired ROI')
	roi = ROIPolygon(ax, r, c)

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
# roi.get_mask(r, c, dcm, frame)
# gray_masked_dcm = rgb2gray(roi.masked_dcm)
gray_masked_dcm = rgb2gray(roi.get_mask(r, c, dcm, frame))

# Test plots and info
plt.figure(1)
plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(gray_masked_dcm, cmap = 'gray')

# plt.subplot(133)
# plt.imshow(roi.masked_dcm)
plt.show()

mean_masked_dcm = np.mean(gray_masked_dcm)
print(mean_masked_dcm)

# Loop through all frames in video
gray_dcm_mean = []
number_frames = dcm.pixel_array.shape[0]
index = 0
for i in np.arange(number_frames):
	# roi.get_mask(r, c, dcm, i)
	g = rgb2gray(roi.get_mask(r, c, dcm, i))
	avg = np.mean(g)
	gray_dcm_mean.append(avg)
	index += 1
	print(index)

# Plot the mean vs frames
# plt.plot(np.arange(number_frames), gray_dcm_mean)
# plt.show()
