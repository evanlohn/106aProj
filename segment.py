import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse
import math
from contrast import increase_contrast


#finds the (potentially rotated) rectangular corners in a black/white image
def find_corners(img):
	#print(stats(img))
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
def calc_reference_mask(img, adap_type='mean', blockSize=11, C=2, invert=False, debug=False, dilation_fac=15):

	threshold_type = cv.THRESH_BINARY_INV if invert else cv.THRESH_BINARY
	#simple adaptive binary thresholding on a (blurred) grayscale image
	first_thresh = cv.adaptiveThreshold(img, 255, adap_types[adap_type], threshold_type, blockSize, C)

	if debug:
		plt.imshow(first_thresh, 'gray')
		plt.show()

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
	# NOTE: increase this if a bit of the puzzle is getting cut off
	#dilation_fac = 15
	puzzle_mask = cv.dilate(puzzle_mask, np.ones((dilation_fac, dilation_fac)))
	return puzzle_mask

def preproc(img):
	return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def reference_mask(gray, C=-10, blockSize= 101, invert=False, blur=41, debug=False, dilation_fac=15):
	#calc mask
	#print(stats(gray))



	gray = np.uint8(gray)
	if debug:
		print(stats(np.uint8(gray)))
	gray2 = cv.medianBlur(gray,blur) # TODO: change 41 to a parameter... or maybe half of blocksize?

	if debug:
		plt.imshow(gray2, 'gray')
		plt.show()

	adap_mean = calc_reference_mask(gray2, adap_type='mean', invert=invert, blockSize=blockSize, C=C, debug=debug, dilation_fac=dilation_fac)
	#adap_gauss = calc_reference_mask(gray2, adap_type='gaussian', blockSize=blockSize, C=C)
	return adap_mean

def segment_reference(ref_img, method):
	print('segmenting reference img')

	gray = preproc(ref_img)

	if method == 'evan':

		adap_mean = reference_mask(gray)

		#images = [adap_mean, adap_gauss, gray, gray * adap_mean, gray * adap_gauss]
		#titles = ['adap mean', 'adap gauss', 'orig', 'adap mean seg', 'adap gauss seg']
		#for i in range(len(images)):
		#    plt.subplot(2,len(images)//2 + 1,i+1),plt.imshow(images[i],'gray')
		#    plt.title(titles[i])
		#    plt.xticks([]),plt.yticks([])
		#plt.show()

		corners = find_corners(adap_mean)

		#corner_mat = np.zeros_like(adap_mean)
		#for coords in corners:
			#print(np.int32(coords))
			#print(corner_mat.shape)
		#	corner_mat[tuple(np.int32(coords))] = 1
		#corner_mat = np.uint8(cv.dilate(corner_mat, np.ones((10,10)))) # just for visual effect
		#corners_found = np.copy(gray)
		#print(corner_mat.max())
		#print(stats(corners_found))
		#corners_found[corner_mat == 1] = 0
		#print(stats(corners_found))


		#plt.imshow(corner_mat, 'gray')
		#plt.show()
		#plt.imshow(corners_found, 'gray')
		#plt.show()

		corners = np.float32([list(corn)[::-1] for corn in corners])
		from deskew import calculate_deskew, deskew_transform
		transform = calculate_deskew(corners)
		tmp = deskew_transform(gray, transform)
		#plt.imshow(tmp, 'gray')
		#plt.show()

		dst = deskew_transform(gray * adap_mean, transform)
		mask = deskew_transform(adap_mean, transform)
		#images = [dst, mask]
		#titles = ['deskewed image', 'deskewed mask']
		#for i in range(len(images)):
		#    plt.subplot(2,len(images)//2 + 1,i+1),plt.imshow(images[i],'gray')
		#    plt.title(titles[i])
		#    plt.xticks([]),plt.yticks([])
		#plt.show()
		ul, ur, lr, ll = find_corners(mask)
		new_img = dst[ul[0]:lr[0], ul[1]:lr[1]]
		#plt.imshow(new_img, 'gray')
		#plt.show()
		return new_img, transform
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


#Returns the origin pixel position and ppm (pixels per meter) from paper image
def paper_calibration(img):
	transform, dsk_mask = transform_from_paper(img)
	corners = find_corners(dsk_mask)
	origin = corners[0]
	corners = np.float32([list(corn)[::-1] for corn in corners])
	ppm = calculate_ppm(corners)
	return origin, ppm


#Calculates ppm from corners of DESKEWED paper mask
def calculate_ppm(corners):
	#dimensions of paper in meters, assumes landscape orientation
	dimensions = [.2794, .2159]
	side_ppm = [(corners[1][0] - corners[0][0]) / dimensions[0],
				 (corners[2][0] - corners[3][0]) / dimensions[0],
				 (corners[3][1] - corners[0][1]) / dimensions[1],
				 (corners[2][1] - corners[1][1]) / dimensions[1]]
	return np.mean(side_ppm)


#Obtains the deskew transform and deskewed paper mask from the paper image
def transform_from_paper(img, aspect_ratio=1.294):
	gray = preproc(img)
	adap_mean = reference_mask(gray) # TODO: might need to change this a bit to segment the paper

	corners = find_corners(adap_mean)
	corners = np.float32([list(corn)[::-1] for corn in corners])
	from deskew import calculate_deskew, deskew_transform
	transform = calculate_deskew(corners, ratio=aspect_ratio)
	tmp = deskew_transform(gray, transform)
	dsk_mask = deskew_transform(adap_mean, transform)

	return transform, dsk_mask

# input is any image of scattered puzzle pieces on a table,
# output is a list of binary masks the same size as the image, one for each puzzle piece
def segment_pieces(img, transform=None):
	from deskew import deskew_transform
	#print(stats(img))
	#plt.imshow(img)
	#plt.show()
	if transform is not None:
		img = deskew_transform(img, transform)
	#plt.imshow(img)
	#plt.show()

def stats(img):
	return {'mean': np.mean(img),
			'std': np.std(img),
			'max': np.max(img),
			'min': np.min(img),
			'size': img.shape,
			'dtype': img.dtype
			}


def main_reference():
	parser = argparse.ArgumentParser(description='specify which file(s) to segment')
	parser.add_argument('file', type=str, nargs='?', default='./raw_img_data/full_puzzle.png')
	args = parser.parse_args()
	ref_img = cv.imread(args.file)
	#plt.imshow(ref_img)
	#plt.show()
	new_img, transform = segment_reference(ref_img, 'evan')
	plt.imshow(new_img)
	plt.show()

def main_pieces():
	parser = argparse.ArgumentParser(description='specify which file(s) to segment')
	parser.add_argument('file', type=str, nargs='?', default='./individual_pieces/img0.png')
	parser.add_argument('--ref', type=str, nargs='?', default='./raw_img_data/full_puzzle.png')
	args = parser.parse_args()
	ref_img = cv.imread(args.ref)
	transform, dsk_mask = transform_from_paper(ref_img)

	p_img = cv.imread(args.file)
	segment_pieces(p_img, transform=transform)


def main_test():
	from piece import Piece
	parser = argparse.ArgumentParser(description='specify which file(s) to used for testing')
	parser.add_argument('file', type=str, nargs='?', default='./individual_pieces/img0.png')
	parser.add_argument('--cut_img', type=str, nargs='?', default='./individual_pieces/extra_cropped_img00.png')
	parser.add_argument('--ref', type=str, nargs='?', default='./raw_img_data/full_puzzle.png')
	args = parser.parse_args()
	ref_img = increase_contrast(cv.imread(args.ref))
	cut_img = cv.imread(args.cut_img)

	print(stats(cut_img))

	other = cv.imread('./individual_pieces/cropped_img0.png')

	#print(stats(other - cut_img))
	#print(args.cut_img)
	#print(cut_img[720:730, 520:530])

	new_img, transform = segment_reference(ref_img, 'evan')

	pre = preproc(cut_img)
	from deskew import deskew_transform
	dsk_cut_img = deskew_transform(pre, transform)

	first_thresh = cv.adaptiveThreshold(dsk_cut_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, 0)
	plt.imshow(first_thresh, 'gray')
	plt.show()
	blockSize = 10
	kernel = np.ones((blockSize,blockSize),np.uint8)
	closed = cv.morphologyEx(np.float32(first_thresh), cv.MORPH_CLOSE, kernel)
	dsk_cut_mask = fill_holes(np.uint8(closed))
	plt.imshow(dsk_cut_mask, 'gray')
	plt.show()

	connectivity = 8
	num_labels, labels, statistics, centroids = cv.connectedComponentsWithStats(dsk_cut_mask, connectivity, cv.CV_32S)

	# sorting connected components by area
	areas = sorted([(i, stat[cv.CC_STAT_AREA], centroids[i]) for i, stat in enumerate(statistics)], key = lambda x:x[1])

	piece = areas[-2] # the biggest area is the background, typically.
	#print(stats)
	#print(centroids[:5])
	dsk_cut_mask = (labels == piece[0])
	#kernel = np.ones((30,30))
	#dsk_cut_mask =  dsk_cut_mask * cv.morphologyEx(np.float32(dsk_cut_mask > 0), cv.MORPH_CLOSE, kernel)
	#plt.imshow(dsk_cut_mask)
	#plt.show()

	#print(stats(np.uint8(dsk_cut_mask)))
	contours, _ = cv.findContours(np.uint8(dsk_cut_mask), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

	assert len(contours) == 1
	ct = contours[0]
	x,y,w,h = cv.boundingRect(ct)

	final_cut = dsk_cut_img[y:y+h, x:x+w]

	plt.imshow(final_cut, 'gray')
	plt.show()
	p = Piece(args.file, cut_img=final_cut/(np.sum(final_cut) + 1))

	from deskew import deskew_transform
	ref = new_img  # new_img is the segmented reference
	print(stats(ref))
	p.solve_piece(ref) # TODO: replace with ref_img

def main_calibration():
	parser = argparse.ArgumentParser(description='specify which file(s) to segment')
	parser.add_argument('file', type=str, nargs='?', default='./raw_img_data/img0.png')
	args = parser.parse_args()
	paper_img = cv.imread(args.file)

if __name__ == '__main__':
	main_test()
