import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse
import math
from segment import preproc, reference_mask, find_corners

#Returns the deskew transformation. Requires the corners and aspect ratio of the reference image.
#corners is an np.float32([top_left, top_right, bottom_right, bottom_left]) of pixel positions
def calculate_deskew(corners, ratio=1.403):
	height = math.sqrt((corners[2][0] - corners[1][0])**2 + (corners[2][1] - corners[1][1])**2)
	width = ratio * height
	pts = np.float32([[corners[0][0],corners[0][1]], [corners[0][0] + width, corners[0][1]], [corners[0][0] + width, corners[0][1] + height], [corners[0][0], corners[0][1] + height]])
	return cv.getPerspectiveTransform(corners, pts)


#Plots and returns the deskewed version of an image given the image and transform
def deskew_transform(img, transform, new_width=2560, new_height=1440):
    rows, cols = img.shape[0], img.shape[1]
    transformed = np.zeros((new_width, new_height), dtype=np.uint8)
    deskewed = cv.warpPerspective(img, transform, transformed.shape)
    plt.imshow(deskewed,'gray')
    plt.title("Deskewed")
    plt.xticks([]),plt.yticks([])
    plt.show()
    return deskewed


#Assumes A4 paper is in landscape position once deskewed, calculates pixels per meter
def calibrate_ppm(deskewed_mask):
	corners = find_corners(deskewed_mask)
	corners = np.float32([list(corn)[::-1] for corn in corners])
	#dimensions of paper in meters
	x_length = .2794
	y_length = .2159
	side_ppi = [(corners[1][0] - corners[0][0]) / x_length,
				 (corners[2][0] - corners[3][0]) / x_length,
				 (corners[0][1] - corners[3][1]) / y_length,
				 (corners[1][1] - corners[2][1]) / y_length]
	return np.mean(side_ppi)


def main():
	parser = argparse.ArgumentParser(description='specify which file(s) to segment')
	parser.add_argument('file', type=str, nargs='?', default='./reference_data/img0.png')
	args = parser.parse_args()
	ref_img = cv.imread(args.file)
	ref_img = preproc(ref_img) # for now, just grayscales
	adap_mean = reference_mask(ref_img)
	corners = find_corners(adap_mean)
	corners = np.float32([list(corn)[::-1] for corn in corners])
	print(corners)
	transform = calculate_deskew(corners)
	dst = deskew_transform(ref_img * adap_mean, transform)


if __name__ == '__main__':
	main()
