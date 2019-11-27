import os
import cv2
import numpy as np

#NOTE: img should be grayscale. see: 
#https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=adaptivethreshold
adap_types = {'mean': cv2.ADAPTIVE_THRESH_MEAN_C, 'gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C}
def segment_adaptive(img, adap_type='mean', blockSize=11, C=2):
	first_thresh = cv2.adaptiveThreshold(img, 255, adap_types[adap_type], cv2.THRESH_BINARY, blockSize, C)

	connectivity = 8
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(first_thresh , connectivity , cv2.CV_32S)

	print(num_labels)
	print(labels[:5])
	print(stats)
	print(centroids[:5])
	return first_thresh

def segment_reference(ref_img):
	print('segmenting reference img')

	gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
	#calc mask
	#print(stats(gray))
	blockSize= 501
	C = -5

	adap_mean = segment_adaptive(gray, adap_type='mean', blockSize=blockSize, C=C)
	adap_gauss = segment_adaptive(gray, adap_type='gaussian', blockSize=blockSize, C=C)

	cv2.imshow('image', adap_mean)
	cv2.waitKey()
	cv2.imshow('image', adap_gauss)
	cv2.waitKey()


def stats(img):
	return {'mean': np.mean(img),
			'std': np.std(img),
			'max': np.max(img),
			'min': np.min(img),
			'size': img.shape,
			}


def main():
	ref_img = cv2.imread('./reference_data/img0.png')
	segment_reference(ref_img)


if __name__ == '__main__':
	main()