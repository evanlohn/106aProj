import cv2 as cv
import imutils
from matplotlib import pyplot as plt
import numpy as np
from contrast import increase_contrast

class Piece:

    puzzle_dims = [4,6] # a 4 by 6 = 24 piece puzzle being solved
    #TODO: initialize this ^^ via command line args or something

    #Initializes a piece with the deskewed, segmented image of the piece,
    # as well as the piece's initial pixel location in image coordinates (row, col)
    def __init__(self, img, init_pos):
        self.img = img
        self.init_pos = init_pos

    #based on the puzzle_dims and the shape of the reference image, pick the ranges of indices
    # to consider out of the full convolution result
    def calc_possible_locs(self, ref_img_shape, box_size=11):
        #divide the grid up as evenly as possible
        num_rows, num_cols = Piece.puzzle_dims
        big_box_height = ref_img_shape[0]//num_rows
        big_box_width = ref_img_shape[1]//num_cols

        possible_locs = [] # list of (row_start, col_start, box_size) tuples
        center_offset = np.array([big_box_height//2, big_box_width//2])
        for row in range(num_rows):
            for col in range(num_cols):
                upper_left = np.array([row*big_box_height, col * big_box_width])
                center = upper_left + center_offset
                box_upper_left = center - box_size//2
                possible_locs.append((box_upper_left[0], box_upper_left[1], box_size))

        return possible_locs

    #Solves and sets the final pixel position and rotation delta
    def solve_piece(self, ref_img):
        # find the pixel location of the puzzle piece in the reference image
        # assume we have an instance variable named self.cut_img that has a small image of just
        # the puzzle.
        piece = self.img
        n = 12
        box_size = 11
        max_ang = 360 # imutils uses degrees
        rotation_angles = [max_ang * i / n for i in range(n)]

        possible_locs = self.calc_possible_locs(ref_img.shape, box_size=box_size)


        best_position = (-1, -1)
        best_confidence = -1
        best_rot = 0
        for i, rot in enumerate(rotation_angles):
            rot_piece = imutils.rotate_bound(piece, rot)
            #plt.imshow(rot_piece)
            #plt.title('Piece {} of {}'.format(i+1, n))
            #plt.show()
            position, confidence = argmax_convolve(rot_piece, ref_img, possible_locs)

            if confidence > best_confidence:
                best_confidence = confidence
                best_position = position
                best_rot = rot
        self.final_pos = best_position
        self.rot_delta = best_rot

    #Places the piece using Baxter/Sawyer
    def place(self, pixel_origin, ppm):
        # self.init_pos should have the initial pixel position
        # self.final_pos should have the final pixel position
        # self.rot_delta has the amount of rotation about the z axis necessary
        
        #calculate current coordinates of the piece in the table frame
        start_table_coords = pixel_to_table_frame(pixel_origin, self.init_pos, ppm)
        end_table_coords = pixel_to_table_frame(pixel_origin, self.final_pos, ppm)

        # move end effector to start_table_coords
        # engage gripper; record current orientation
        # move to end_table_coords with orientation prev_orientation + self.rot_delta
        # release gripper
        # move back to neutral (i.e. not in view of camera) position

def argmax_convolve(rot_piece, ref_img, possible_locs):


    conv_res = cv.filter2D(ref_img, -1, rot_piece)
    assert conv_res.shape == ref_img.shape, str('shapes differ: conv is {}, ref is {}'.format(conv_res.shape, ref.shape))

    if possible_locs is None:
        inds = np.unravel_index(np.argmax(conv_res), conv_res.shape)
        confidence = conv_res[inds[0]][inds[1]]
    else:
        inds = (-1, -1)
        confidence = -1
        for loc in possible_locs:
            section = conv_res[loc[0]:loc[0] + loc[2], loc[1]:loc[1]+loc[2]]
            box_inds = np.unravel_index(np.argmax(section), conv_res.shape) 
            global_inds = np.array(box_inds) + np.array([loc[0], loc[1]])
            tmp_confidence = conv_res[global_inds[0]][global_inds[1]]
            if tmp_confidence > confidence:
                confidence = tmp_confidence
                inds = global_inds
    

    images = [conv_res, ref_img, rot_piece]
    titles = ['convolution: max conf {}'.format(confidence), 'reference', 'piece']
    for i in range(len(images)):
        plt.subplot(2,len(images)//2 + 1,i+1),plt.imshow(images[i],'gray')
        if i < 2:
            plt.subplot(2,len(images)//2 + 1,i+1),plt.plot(inds[1], inds[0], 'r+')
        plt.title(titles[i])
        #plt.xticks([]),plt.yticks([])

    plt.show()

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