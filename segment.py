import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse
import math

#NOTE: img should be grayscale. see:
#https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=adaptivethreshold
adap_types = {'mean': cv.ADAPTIVE_THRESH_MEAN_C, 'gaussian': cv.ADAPTIVE_THRESH_GAUSSIAN_C}
def segment_adaptive(img, adap_type='mean', blockSize=11, C=2):

	first_thresh = cv.adaptiveThreshold(img, 255, adap_types[adap_type], cv.THRESH_BINARY, blockSize, C)

	connectivity = 8
	num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(first_thresh , connectivity , cv.CV_32S)

	print(num_labels)
	print(labels[:5])
	print(stats)
	print(centroids[:5])
	return first_thresh

def segment_reference(ref_img, method):
	print('segmenting reference img')

	gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)

	if method == 'evan':
		#calc mask
		#print(stats(gray))
		blockSize= 801
		C = 2

		gray = cv.medianBlur(gray,41)

		adap_mean = segment_adaptive(gray, adap_type='mean', blockSize=blockSize, C=C)
		adap_gauss = segment_adaptive(gray, adap_type='gaussian', blockSize=blockSize, C=C)


		images = [adap_mean, adap_gauss, gray]
		titles = ['adap mean', 'adap gauss', 'orig']
		for i in range(len(images)):
		    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
		    plt.title(titles[i])
		    plt.xticks([]),plt.yticks([])
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
