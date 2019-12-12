import sys
if sys.version[0] != '2':
	print('please run all code with python2!')
	exit(0)

cant_plot= True

import cv2 as cv
import numpy as np
from capture import mkFileNamer

import argparse
import math
from contrast import increase_contrast



expected_dims = (2560, 1440)

def standard_resize(img):
	return cv.resize(img, expected_dims)


def imread(img_path, typ=None, should_resize=True):
	if typ is not None:
		img = cv.imread(img_path, typ)
	else:
		img = cv.imread(img_path)

	resized = img
	if should_resize:
		resized = standard_resize(resized)
	return resized

def imshow(img, title='', cmap='gray', inds=None, just_write=False):
	if just_write:
		ret = cv.imwrite('./tmp_images/tmp_{}.png'.format(title), img)
		assert ret
		return
	else:
		plt.imshow(img, cmap)
		if title:
			plt.title(title)
		if inds:
			plt.plot(inds[1], inds[0], 'r+')

		if cant_plot:
			plt.savefig(fnamer())
		else:
			plt.show()

def imshow_mult(images, titles, inds=[0,0]):
	plt.figure()
	for i in range(len(images)):
	    plt.subplot(2,len(images)//2 + 1,i+1),plt.imshow(images[i],'gray')
	    if i < 2:
	    	plt.subplot(2,len(images)//2 + 1,i+1),plt.plot(inds[1], inds[0], 'r+')
	    plt.title(titles[i])
	if cant_plot:
		plt.savefig(fnamer())
		plt.close()
	else:
		plt.show()



#finds the (potentially rotated) rectangular corners in a black/white image
def find_corners(img):
	#print(stats(img))
	_, contours,_ = cv.findContours(np.uint8(img), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

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
	_, contour,hier = cv.findContours(des,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
	for cnt in contour:
	    cv.drawContours(des,[cnt],0,255,-1)

	return des

def fill_between_corners(img, corners):
	fillval = np.max(img)
	#print(stats(img))
	#print(fillval)
	#print(np.array(corners, np.int32))
	#print(np.array(corners))
	cv.fillConvexPoly(img, np.array(corners), int(fillval))
	#imshow(img)

#NOTE: img should be grayscale. see:
#https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=adaptivethreshold
adap_types = {'mean': cv.ADAPTIVE_THRESH_MEAN_C, 'gaussian': cv.ADAPTIVE_THRESH_GAUSSIAN_C}
def calc_reference_mask(img, adap_type='mean', blockSize=11, C=2, invert=False, debug=False, dilation_fac=15, thresh_type='adap', manual_thresh=200):

	threshold_type = cv.THRESH_BINARY_INV if invert else cv.THRESH_BINARY
	if debug:
		imshow(img)


	if thresh_type == 'adap':
		#simple adaptive binary thresholding on a (blurred) grayscale image
		first_thresh = cv.adaptiveThreshold(img, 255, adap_types[adap_type], threshold_type, blockSize, C)
	else:
		ret, first_thresh = cv.threshold(img,manual_thresh,255,cv.THRESH_BINARY)

	if debug:
		imshow(first_thresh, 'gray')

	#imshow(first_thresh, title='adaptive_threshold', just_write=True)

	# the puzzle blob isn't homogenous, so some pieces will have some black in them. So, we fill those holes with white.
	first_thresh = fill_holes(first_thresh)

	#imshow(first_thresh, title='morphological', just_write=True)

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

def reference_mask(gray, C=-10, blockSize= 101, invert=False, blur=41, debug=False, dilation_fac=15, thresh_type='adap', manual_thresh=200):
	#calc mask
	#print(stats(gray))

	gray = np.uint8(gray)
	if debug:
		print(stats(np.uint8(gray)))
	gray2 = cv.medianBlur(gray,blur) # TODO: change 41 to a parameter... or maybe half of blocksize?

	if debug:
		imshow(gray2, 'gray')

	adap_mean = calc_reference_mask(gray2, adap_type='mean', invert=invert, blockSize=blockSize, C=C, debug=debug, dilation_fac=dilation_fac, thresh_type=thresh_type, manual_thresh=manual_thresh)
	#adap_gauss = calc_reference_mask(gray2, adap_type='gaussian', blockSize=blockSize, C=C)
	return adap_mean

def segment_reference(ref_img):
	print('segmenting reference img')

	gray = preproc(ref_img)
	adap_mean = reference_mask(gray, debug=False)
	#print(stats(np.uint8(adap_mean)))
	#imshow(np.uint8(adap_mean)*255, title='contour', just_write=True)

	#images = [adap_mean, adap_gauss, gray, gray * adap_mean, gray * adap_gauss]
	#titles = ['adap mean', 'adap gauss', 'orig', 'adap mean seg', 'adap gauss seg']
	#for i in range(len(images)):
	#    plt.subplot(2,len(images)//2 + 1,i+1),plt.imshow(images[i],'gray')
	#    plt.title(titles[i])
	#    plt.xticks([]),plt.yticks([])
	#plt.show()

	corners = find_corners(adap_mean)

	adap_mean = np.uint8(adap_mean)
	corners = np.int32([list(corn)[::-1] for corn in corners])
	fill_between_corners(adap_mean, corners)
	corners = np.float32(corners)
	
	from deskew import calculate_deskew, deskew_transform
	transform = calculate_deskew(corners) # NOTE: uses default aspect ratio (which corresponds to the Toy Story puzzle)
	tmp = deskew_transform(gray, transform)


	dst = deskew_transform(ref_img * adap_mean[:,:,None], transform)
	mask = deskew_transform(adap_mean, transform)
	#images = [dst, mask]
	#titles = ['deskewed image', 'deskewed mask']
	#for i in range(len(images)):
	#    plt.subplot(2,len(images)//2 + 1,i+1),plt.imshow(images[i],'gray')
	#    plt.title(titles[i])
	#    plt.xticks([]),plt.yticks([])
	#plt.show()
	ul, ur, lr, ll = find_corners(mask)
	new_img = dst[ul[0]:lr[0], ul[1]:lr[1], :]
	#imshow(new_img, title='deskewed', just_write=True)
	#plt.imshow(new_img, 'gray')
	#plt.show()
	imshow(new_img, title='reference', just_write=True)
	return new_img, transform



#Returns the origin pixel position and ppm (pixels per meter) from paper image
def paper_calibration(img):
	transform, dsk_mask = transform_from_paper(img)
	corners = find_corners(dsk_mask)
	origin = corners[0]
	corners = np.float32([list(corn)[::-1] for corn in corners])
	ppm = calculate_ppm(corners)
	return origin, ppm, transform


#Calculates ppm from corners of DESKEWED paper mask
def calculate_ppm(corners):
	#dimensions of paper in meters, assumes landscape orientation
	#dimensions = [.2794, .2159]
	#dimensions of paper in meters, assumes portrait orientation
	dimensions = [.2159, .2794]
	side_ppm = [np.float32(corners[1][0] - corners[0][0]) / dimensions[0],
				 np.float32(corners[2][0] - corners[3][0]) / dimensions[0],
				 np.float32(corners[3][1] - corners[0][1]) / dimensions[1],
				 np.float32(corners[2][1] - corners[1][1]) / dimensions[1]]
	return np.mean(side_ppm)


#Obtains the deskew transform and deskewed paper mask from the paper image
a1 = np.float32(8.5)/np.float32(11)
a2 = np.float32(11)/np.float32(8.5)
def transform_from_paper(img, aspect_ratio=a1):
	gray = preproc(img)
	#imshow(img)
	adap_mean = reference_mask(gray, blur=5, debug=False, thresh_type='manual', manual_thresh=200) # TODO: might need to change this a bit to segment the paper

	imshow(adap_mean, title='calibration_seg', just_write=True)

	corners = find_corners(adap_mean)
	corners = np.float32([list(corn)[::-1] for corn in corners])
	from deskew import calculate_deskew, deskew_transform
	transform = calculate_deskew(corners, ratio=aspect_ratio)
	tmp = deskew_transform(gray, transform)
	imshow(tmp, title='calibration_seg', just_write=True)
	dsk_mask = deskew_transform(adap_mean, transform)

	#imshow(dsk_mask)

	return transform, dsk_mask

# input is an image of a single puzzle piece on the table and the deskewing transform
def segment_pieces(img, background, transform=None):
	from deskew import deskew_transform

	#print(stats(img))
	#imshow(img, title='curr', just_write=True)
	#imshow(background, title='background', just_write=True)
	img = cv.absdiff(img, background)
	img = increase_contrast(img)  #TODO: keep? delete?
	#imshow(img)
	adaptive_th = preproc(img)
	#imshow(adaptive_th, title='diff', just_write=True)
	#th2 = cv.adaptiveThreshold(adaptive_th,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,101,10)
	#TODO: put the rest of this block of code in a for loop over different absolute thresholds, picking the first one
	#      that satisfies some reasonable conditions like a single piece gets extracted and it's not at the top/bottom of the image
	#20
	_, th2 = cv.threshold(adaptive_th, 23,1,cv.THRESH_BINARY)
	#print(stats(th2))
	imshow(th2, title='piece_seg_adaptive', just_write=True)

	#imshow(np.uint8(th2)*255, title='adap', just_write=True)
	sz = 5
	#k1 = np.ones((sz,sz), np.uint8)
	#th2 = cv.dilate(th2, k1)
	#th2 = cv.erode(th2, k1)
	kernel = np.ones((sz,sz),np.uint8)
	opening = cv.morphologyEx(th2, cv.MORPH_OPEN, kernel)
	#imshow(opening)

	connectivity = 8
	num_labels, labels, statistics, centroids = cv.connectedComponentsWithStats(opening, connectivity, cv.CV_32S)

	# sorting connected components by area
	areas = sorted([(i, stat[cv.CC_STAT_AREA], centroids[i]) for i, stat in enumerate(statistics)], key = lambda x:x[1])

	piece = areas[-2] # the biggest area is the background, typically.
	opening = np.uint8(labels == piece[0])
	#opening = cv.morphologyEx(opening, cv.MORPH_OPEN, kernel)
	opening = cv.dilate(opening, kernel)
	opening = fill_holes(opening)
	opening = cv.erode(opening, kernel)
	imshow(opening, title='piece_seg_postmorph', just_write=True)

	#contours, hierarchy = cv.findContours(np.uint8(opening), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
	#contours.sort(reverse=True, key=lambda x: cv.contourArea(x))
	#print(contours)
	#piece = contours[0]
	#black = np.zeros_like(deskew_img)
	#cs = cv.drawContours(black, [piece], -1, 255, 3)
	#opening=cv.cvtColor(opening,cv.COLOR_GRAY2BGR)
	#cs = cv.drawContours(opening, [piece], -1, (0,255,0), 3)
	#imshow(cs)
	dsk_mask = deskew_transform(standard_resize(opening), transform)
	dsk_img = deskew_transform(standard_resize(img), transform)

	ct = cv.findNonZero(dsk_mask)
	x,y,w,h = cv.boundingRect(ct)

	dsk_mask[dsk_mask > 0] = 1
	#print(stats(dsk_mask))
	masked_img = dsk_mask[:,:,None] * dsk_img
	#imshow(adaptive_th)
	#imshow(masked_img)

	final_cut = masked_img[y:y+h, x:x+w, :]
	imshow(final_cut, title='piece_seg_final', just_write=True)
	#final_cut = increase_contrast(final_cut)
	#imshow(final_cut, title='final_cut', just_write=True)
	return final_cut, np.array([y + h//2, x + w//2])
	#opening=cv.cvtColor(opening,cv.COLOR_GRAY2BGR)
	#cs = cv.drawContours(opening, [contours[2]], -1, (0,255,0), 3)
	#images = [img, deskew_img, th2, cs]
	#titles = ["Original", "Deskewed", "Adaptive Threshold", "contours"]
	#for i in range(4):
	#    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
	#    plt.title(titles[i])
	#    plt.xticks([]),plt.yticks([])
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
	ref_img = increase_contrast(imread(args.file))
	#imshow(ref_img, title='ref', just_write=True)
	#plt.show()
	new_img, transform = segment_reference(ref_img)
	#print(stats(new_img))

	imshow(new_img)

def main_pieces():
	parser = argparse.ArgumentParser(description='specify which file(s) to segment')
	parser.add_argument('file', type=str, nargs='?', default='./individual_pieces/img8.png')
	parser.add_argument('--ref', type=str, nargs='?', default='./raw_img_data/full_puzzle.png')
	args = parser.parse_args()
	ref_img = imread(args.ref)
	transform, dsk_mask = transform_from_paper(ref_img)

	p_img = imread(args.file)
	background = imread('./individual_pieces/img1.png')
	segment_pieces(p_img, background, transform=transform)


def main_test():
	from piece import Piece
	parser = argparse.ArgumentParser(description='specify which file(s) to used for testing')
	parser.add_argument('file', type=str, nargs='?', default='./individual_pieces/img3.png')
	parser.add_argument('--prev_state', type=str, nargs='?', default='./raw_img_data/empty_table.png')

	#parser.add_argument('--cut_img', type=str, nargs='?', default='./individual_pieces/extra_cropped_img00.png')
	parser.add_argument('--ref', type=str, nargs='?', default='./raw_img_data/full_puzzle.png')
	parser.add_argument('--cal', type=str, nargs='?', default='./raw_img_data/calib.png')

	args = parser.parse_args()
	ref_img = increase_contrast(imread(args.ref))
	#ref_img = imread(args.ref)

	calib = imread(args.cal)
	origin, ppm, transform = paper_calibration(calib)

	prev_state = imread(args.prev_state, should_resize=False)
	curr_state = imread(args.file, should_resize=False)

	#imshow(prev_state)
	#imshow(curr_state)
	new_img, ref_transform = segment_reference(ref_img)
	cv.imwrite("tmp_images/go.png", new_img)
	dsk_cut_img, init_pos = segment_pieces(curr_state, prev_state, transform)

	#print(stats(cut_img))
	cv.imwrite("tmp_images/bears.png", dsk_cut_img)

	#other = imread('./individual_pieces/cropped_img0.png')


	#print(stats(other - cut_img))
	#print(args.cut_img)
	#print(cut_img[720:730, 520:530])



	#pre = preproc(cut_img)
	#imshow(pre)
	#from deskew import deskew_transform
	#dsk_cut_img = deskew_transform(pre, transform)

	#print(stats(dsk_cut_img[:,:,0]))
	#print(stats(dsk_cut_img[:,:,1]))
	#print(stats(dsk_cut_img[:,:,2]))
	#dsk_cut_img = increase_contrast(dsk_cut_img)

	#dsk_cut_img = dsk_cut_img

	#print(stats(dsk_cut_img[:,:,0]))
	#print(stats(dsk_cut_img[:,:,1]))
	#print(stats(dsk_cut_img[:,:,2]))

	#imshow(dsk_cut_img[:,:,0])
	#imshow(dsk_cut_img[:,:,1])
	#imshow(dsk_cut_img[:,:,2])

	p = Piece(dsk_cut_img, init_pos)

	ref = new_img  # new_img is the segmented reference
	#print('ref stats: {}'.format(stats(ref)))
	p.solve_piece(ref) # TODO: replace with ref_img

def main_calibration():
	parser = argparse.ArgumentParser(description='specify which file(s) to segment')
	parser.add_argument('file', type=str, nargs='?', default='./raw_img_data/img0.png')
	args = parser.parse_args()
	paper_img = imread(args.file)

if __name__ == '__main__':
	if cant_plot:
		import os
		import matplotlib
		matplotlib.use("Agg")
		loc = './tmp_images'
		print('yeeting')
		fnamer = mkFileNamer(loc, 'tmp')
		files = [os.path.join(loc, file) for file in os.listdir(loc) if file[-4:] == '.png']
		for file in files:
			if (os.path.exists(file)):
				os.remove(file)
	from matplotlib import pyplot as plt
	main_test()
