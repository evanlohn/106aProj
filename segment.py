import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import argparse

#finds the (potentially rotated) rectangular corners in a black/white image
def find_corners(img):
	contours,_ = cv.findContours(np.uint8(img), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

	coords = []
	assert len(contours) == 1
	outline = contours[0]
	ul = max(outline, key=lambda pt: -sum(pt[0]))[0]
	ur = max(outline, key=lambda pt: pt[0][0] - pt[0][1])[0]
	ll = max(outline, key=lambda pt: pt[0][1] - pt[0][0])[0]
	lr = max(outline, key=lambda pt: sum(pt[0]))[0]
	    #cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

	coords = [(x[1], x[0]) for x in [ul, ur, lr, ll]]
	return coords


# see https://stackoverflow.com/questions/10316057/filling-holes-inside-a-binary-object
# basically takes a binary image and fills in the black smudges inside white blobs with white
def fill_holes(img):
	des = img
	contour,hier = cv.findContours(des,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
	for cnt in contour:
	    cv.drawContours(des,[cnt],0,255,-1)

	return des

#NOTE: img should be grayscale. see: 
#https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=adaptivethreshold
adap_types = {'mean': cv.ADAPTIVE_THRESH_MEAN_C, 'gaussian': cv.ADAPTIVE_THRESH_GAUSSIAN_C}
def segment_adaptive(img, adap_type='mean', blockSize=11, C=2):

	#simple adaptive binary thresholding on a (blurred) grayscale image
	first_thresh = cv.adaptiveThreshold(img, 255, adap_types[adap_type], cv.THRESH_BINARY, blockSize, C)

	# the puzzle blob isn't homogenous, so some pieces will have some black in them. So, we fill those holes with white.
	first_thresh = fill_holes(first_thresh)

	# see https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
	# num_labels is how many connected components there are
	# labels     is a new version of the image where each connected blob has had its values replaced with a "label".
	# 		 	 for example, one connected blob of 1's from first_thresh might have all its 1's replaced by 3's; another
	#		     blob of 1's might have its 1's replaced by 25's. blobs of 0's also count, so the huge black "background"
	#			 of the image will have its 0's replaced by some label as well.
	connectivity = 8
	num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(first_thresh , connectivity , cv.CV_32S)
	
	# sorting connected components by area
	areas = sorted([(i, stat[cv.CC_STAT_AREA], centroids[i]) for i, stat in enumerate(stats)], key = lambda x:x[1])
	#print(areas[-5:])
	#for area in areas[-5:]:
	#	plt.imshow(labels == area[0], 'gray')
	#	plt.show()

	#TODO: write a more sophisticated way to pick out the rectangle
	puzzle_box = areas[-2] # the biggest area is the background, typically.
	#print(stats)
	#print(centroids[:5])
	puzzle_mask = (labels == puzzle_box[0])

	# see: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
	# the idea is to smooth the boundary of the rectangle
	kernel = np.ones((blockSize,blockSize),np.uint8)
	puzzle_mask = cv.morphologyEx(np.float32(puzzle_mask), cv.MORPH_CLOSE, kernel)

	# one last dilation; we'd rather get a bit extra than miss parts of the puzzle
	puzzle_mask = cv.dilate(puzzle_mask, np.ones((10,10)))
	return puzzle_mask

def segment_reference(ref_img, method):
	print('segmenting reference img')

	gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)

	if method == 'evan':
		#calc mask
		#print(stats(gray))
		blockSize= 101
		C = -1

		gray2 = cv.medianBlur(gray,41) # TODO: change 41 to a parameter... or maybe half of blocksize?

		adap_mean = segment_adaptive(gray2, adap_type='mean', blockSize=blockSize, C=C)
		adap_gauss = segment_adaptive(gray2, adap_type='gaussian', blockSize=blockSize, C=C)



		images = [adap_mean, adap_gauss, gray, gray * adap_mean, gray * adap_gauss]
		titles = ['adap mean', 'adap gauss', 'orig', 'adap mean seg', 'adap gauss seg']
		for i in range(len(images)):
		    plt.subplot(2,len(images)//2 + 1,i+1),plt.imshow(images[i],'gray')
		    plt.title(titles[i])
		    plt.xticks([]),plt.yticks([])
		plt.show()

		corners = find_corners(adap_mean)

		corner_mat = np.zeros_like(adap_mean)
		for coords in corners:
			print(np.int32(coords))
			print(corner_mat.shape)
			corner_mat[tuple(np.int32(coords))] = 1
		corner_mat = np.uint8(cv.dilate(corner_mat, np.ones((10,10)))) # just for visual effect
		corners_found = np.copy(gray)
		print(corner_mat.max())
		print(stats(corners_found))
		corners_found[corner_mat == 1] = 0
		print(stats(corners_found))


		#plt.imshow(corner_mat, 'gray')
		#plt.show()
		plt.imshow(corners_found, 'gray')
		plt.show()

	else:
	
		img = cv.imread('./raw_img_data/puzzle_pieces.png',0)
		img = cv.medianBlur(img,5)

		ret,th1 = cv.threshold(gray,127,255,cv.THRESH_BINARY)
		th2 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
		th3 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

		titles = ['Original Image', 'Global Thresholding (v = 127)',
	            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
		images = [img, th1, th2, th3]

		contours, hierarchy = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		cv.drawContours(img, contours, -1, (0,255,0), 3)

		for i in range(4):
		    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
		    plt.title(titles[i])
		    plt.xticks([]),plt.yticks([])
		plt.show()


def stats(img):
	return {'mean': np.mean(img),
			'std': np.std(img),
			'max': np.max(img),
			'min': np.min(img),
			'size': img.shape,
			}


def main():
	parser = argparse.ArgumentParser(description='specify which file(s) to segment')
	parser.add_argument('file', type=str, nargs='?', default='./raw_img_data/puzzle_pieces.png')

	args = parser.parse_args()
	ref_img = cv.imread(args.file)
	segment_reference(ref_img, 'evan')


if __name__ == '__main__':
	main()