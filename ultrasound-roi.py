# for reading DICOM files obtained from ultrasound machines, letting user draw ROI, and outputting intensity-time graph

import sys
import matplotlib.pyplot as plt
import pydicom
import numpy as np
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import matplotlib.patches as patches

class ROIPolygon(object):

	def __init__(self, ax, rows, col):
		self.canvas = ax.figure.canvas
		self.poly = PolygonSelector(ax,
									self.onselect,
									lineprops = dict(color = 'g', alpha = 1),
									markerprops = dict(mec = 'g', mfc = 'g', alpha = 1))
		self.path = None
		self.mask = np.zeros([rows, col], dtype = int)

	def onselect(self, verts):
		path = Path(verts)
		self.canvas.draw_idle()
		self.path = path

	def get_mask(self, rows, col):
		for i in range(rows):
			for j in range(col):
				# matplotlib.Path.contains_points returns True if the point is inside the ROI
				# Path vertices are considered in different order than numpy arrays
				 if self.path.contains_points([(j,i)]) == [True]:
					 self.mask[i][j] = 1
				 else:
					 self.mask[i][j] = 0

def rgb2gray(rgb):
	# from stackoverflow
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

ds = pydicom.dcmread(sys.argv[1])

# pixel data stored as 4D numpy array
# (frame number, rows, cols, rgb channel)
# pixel_array[int] slices one frame where int is frame number
frame_n = int(input("Enter desired frame to view: "))
img = ds.pixel_array[frame_n]

rows = ds.pixel_array.shape[1] # number of rows of pixels in image
col = ds.pixel_array.shape[2] # number of columns of pixels in image


# Drawing ROI on selected frame
q = 'n'
while q == 'n' or q == 'N':
	fig, ax = plt.subplots()
	plt.imshow(img)
	plt.show(block = False)

	print('Draw desired ROI')
	roi1 = ROIPolygon(ax, rows, col)

	plt.imshow(img)
	plt.show()

	# Overlaying ROI onto image
	patch = patches.PathPatch(roi1.path,
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

# Drawing mask
mask_test = roi1.get_mask(rows, col)
masked_ds = np.zeros([rows, col, 3]) # empty matrix to fill in

for i in range(rows):
	for j in range(col):
		if roi1.mask[i][j] == 1:
			masked_ds[i][j] = ds.pixel_array[frame_n][i][j]
		else:
			masked_ds[i][j] = 0

gray_masked_ds = rgb2gray(masked_ds)

plt.figure(1)

plt.subplot(131)
plt.imshow(img)

plt.subplot(132)
plt.imshow(gray_masked_ds, cmap = 'gray')

plt.subplot(133)
plt.imshow(roi1.mask)
plt.show()

print(masked_ds.shape)
mean_masked_ds = np.mean(gray_masked_ds)
print(mean_masked_ds)
