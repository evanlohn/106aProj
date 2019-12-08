import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse
import math

#Returns the deskew transformation. Requires the corners and aspect ratio of the reference image.
#corners is an np.float32([top_left, top_right, bottom_right, bottom_left]) of pixel positions
def calculate_deskew(corners, ratio=1.421):
	height = math.sqrt((corners[2][0] - corners[1][0])**2 + (corners[2][1] - corners[1][1])**2)
	width = ratio * height
	pts = np.float32([[corners[0][0],corners[0][1]], [corners[0][0] + width, corners[0][1]], [corners[0][0] + width, corners[0][1] + height], [corners[0][0], corners[0][1] + height]])
	return cv.getPerspectiveTransform(corners, pts)


#Plots and returns the deskerwed version of an image given the image and transform
def deskew_transform(img, transform, new_width=2560, new_height=1440):
    rows, cols, ch = img.shape
    transformed = np.zeros((new_width, new_height), dtype=np.uint8)
    deskewed = cv.warpPerspective(img, transform, transformed.shape)
    deskewed = cv.cvtColor(deskewed, cv.COLOR_BGR2GRAY)
    plt.imshow(deskewed,'gray')
    plt.title("Deskewed")
    plt.xticks([]),plt.yticks([])
    plt.show()
    return deskewed


def main():
	parser = argparse.ArgumentParser(description='specify which file(s) to segment')
	parser.add_argument('file', type=str, nargs='?', default='./reference_data/img0.png')
	args = parser.parse_args()
	ref_img = cv.imread(args.file)
	#TODO: Get corners programmatically
	corners = np.float32([[484, 205], [1311, 227], [1374, 793], [314, 749]])
	transform = calculate_deskew(corners)
	dst = deskew_transform(ref_img, transform)


if __name__ == '__main__':
	main()
