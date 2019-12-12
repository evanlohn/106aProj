import sys
if sys.version[0] != '2':
    print('please run all code with python2!')
    exit(0)

import cv2 as cv
import imutils
from matplotlib import pyplot as plt
import numpy as np
from contrast import increase_contrast

from segment import imshow, imshow_mult, preproc

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
                #box_upper_left = center - box_size//2
                #possible_locs.append((box_upper_left[0], box_upper_left[1], box_size))

        print(possible_locs)
        return possible_locs

    #Solves and sets the final pixel position and rotation delta
    def solve_piece(self, ref_img):
        # find the pixel location of the puzzle piece in the reference image
        # assume we have an instance variable named self.cut_img that has a small image of just
        # the puzzle.
        piece = self.img

        possible_locs = self.calc_possible_locs(ref_img.shape, box_size=box_size)
        position, rotation = SURF_detect(piece, ref_img, possible_locs)

        self.final_pos = best_position
        self.rot_delta = best_rot

def SURF_detect(piece, ref_img, possible_locs):
    img_object = preproc(piece)
    img_scene = preproc(ref_img)

    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)

    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    #-- Draw matches
    img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

    H, _ =  cv.findHomography(obj, scene, cv.RANSAC)

    #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0l
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img_object.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img_object.shape[1]
    obj_corners[2,0,1] = img_object.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img_object.shape[0]

    obj_centroid = np.int32(np.mean(obj_corners[:,0,:], axis=0))

    scene_corners = cv.perspectiveTransform(obj_corners, H)
    scene_centroid = np.int32(np.mean(scene_corners[:,0,:], axis=0))


    #-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
        (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
        (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
        (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
        (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)


# origin is the pixel coordinates (x, y) of the origin of the table frame (extracted by calibrate_ppm)
# pixel_loc is the pixel coordinates of the pixel we want to determine
# ppm is pixels per meter, found in calibrate_ppm
def pixel_to_table_frame(origin, pixel_loc, ppm):
    pixel_diff = np.array(pixel_loc) - np.array(origin)
    # note that this ^^ implicitly assumes that the "vertical" of the image is the x axis of
    # the table frame
    return float(pixel_diff)/ppm
