import cv2
import numpy as np


def stats(img):
	return {'mean': np.mean(img),
			'std': np.std(img),
			'max': np.max(img),
			'min': np.min(img),
			'size': img.shape,
			'dtype': img.dtype
			}

def imshow(title, image):
	cv2.imshow(title, image)
	cv2.waitKey()

# see https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
def increase_contrast(img):
	#print(stats(img))
	#-----Converting image to LAB Color model----------------------------------- 
	lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	#imshow("lab",lab)

	#-----Splitting the LAB image to different channels-------------------------
	l, a, b = cv2.split(lab)
	#imshow('l_channel', l)
	#imshow('a_channel', a)
	#imshow('b_channel', b)

	#-----Applying CLAHE to L-channel-------------------------------------------
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	cl = clahe.apply(l)
	#imshow('CLAHE output', cl)

	#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
	limg = cv2.merge((cl,a,b))
	#imshow('limg', limg)

	#-----Converting image from LAB Color model to RGB model--------------------
	final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	return final

def main():
	#-----Reading the image-----------------------------------------------------
	img = cv2.imread('./individual_pieces/extra_cropped_img00.png', 1)
	imshow("img", img) 
	
	final = increase_contrast(img)

	imshow('final', final)

if __name__ == '__main__':
	main()