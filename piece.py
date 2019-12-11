import cv2 as cv
import imutils
from matplotlib import pyplot as plt
import numpy as np
from contrast import increase_contrast

class Piece:

    #Initializes a piece with the deskewed, segmented image of the piece,
    # as well as the piece's initial pixel location in image coordinates (row, col)
    def __init__(self, img, init_pos):
        self.img = img
        self.init_pos = init_pos

    #Solves and sets the final pixel position and rotation delta
    def solve_piece(self, ref_img):
        # find the pixel location of the puzzle piece in the reference image
        # assume we have an instance variable named self.cut_img that has a small image of just
        # the puzzle.
        piece = self.cut_img
        n = 12
        max_ang = 360 # imutils uses degrees
        rotation_angles = [max_ang * i / n for i in range(n)]

        for i, rot in enumerate(rotation_angles):
            rot_piece = imutils.rotate_bound(piece, rot)
            #plt.imshow(rot_piece)
            #plt.title('Piece {} of {}'.format(i+1, n))
            #plt.show()
            position, confidence = argmax_convolve(rot_piece, ref_img)

        #self.final_pos = (x, y)
        #self.rot_delta = 0
        pass

    #Places the piece using Baxter/Sawyer
    def place(self):
        # self.init_pos should have the initial pixel position
        # self.final_pos should have the final pixel position
        # self.rot_delta has the amount of rotation about the z axis necessary
        pass

def argmax_convolve(rot_piece, ref_img):


    conv_res = cv.filter2D(ref_img, -1, rot_piece)
    assert conv_res.shape == ref_img.shape, str('shapes differ: conv is {}, ref is {}'.format(conv_res.shape, ref.shape))

    inds = np.unravel_index(np.argmax(conv_res), conv_res.shape)
    confidence = conv_res[inds[0]][inds[1]]

    images = [conv_res, ref_img, rot_piece]
    titles = ['convolution: max conf {}'.format(confidence), 'reference', 'piece']
    for i in range(len(images)):
        plt.subplot(2,len(images)//2 + 1,i+1),plt.imshow(images[i],'gray')
        if i < 2:
            plt.subplot(2,len(images)//2 + 1,i+1),plt.plot(inds[1], inds[0], 'r+')
        plt.title(titles[i])
        #plt.xticks([]),plt.yticks([])

    plt.show()

    #TODO: only look at the argmax over small sections corresponding to piece centers



    return inds, confidence
    #return position, confidence

# origin is the pixel coordinates (x, y) of the origin of the table frame (extracted by calibrate_ppm)
# pixel_loc is the pixel coordinates of the pixel we want to determine
# ppm is pixels per meter, found in calibrate_ppm
def pixel_to_table_frame(origin, pixel_loc, ppm):
    pixel_diff = np.array(pixel_loc) - np.array(origin)
    # note that this ^^ implicitly assumes that the "vertical" of the image is the x axis of
    # the table frame
    return pixel_diff/ppm