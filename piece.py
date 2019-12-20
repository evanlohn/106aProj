import sys
if sys.version[0] != '2':
    print('please run all code with python2!')
    exit(0)

import cv2 as cv
import imutils
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as la
from contrast import increase_contrast
import math

from segment import imshow, imshow_mult, preproc, stats

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
    def calc_possible_locs(self, ref_img_shape):
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
                possible_locs.append(center)
                #box_upper_left = center - box_size//2
                #possible_locs.append((box_upper_left[0], box_upper_left[1], box_size))

        possible_locs = np.array(possible_locs)
        return possible_locs

    #Solves and sets the final pixel position and rotation delta
    def solve_piece(self, ref_img):
        # find the pixel location of the puzzle piece in the reference image
        # assume we have an instance variable named self.cut_img that has a small image of just
        # the puzzle.
        piece = self.img

        possible_locs = self.calc_possible_locs(ref_img.shape)
        position, rotation = SURF_detect(preproc(piece), preproc(ref_img))

        if position is None:
            print "SURF FAILED"
            position, rotation = SURF_detect(piece[:,:,0], ref_img[:,:,0])
        if position is None:
            position, rotation = SURF_detect(piece[:,:,0], ref_img[:,:,1])
        if position is None:
            position, rotation = SURF_detect(piece[:,:,0], ref_img[:,:,2])
        if position is None:
            position, rotation = SIFT_detect(piece, ref_img)
        if position is None:
            position = np.random.choice(possible_locs)
            rotation = 0
        print position
        position = pick_closest(position, possible_locs)
        imshow(ref_img, title='position chosen for piece', inds=position)
        print('position was:', position)
        self.final_pos = position
        self.rot_delta = rotation


def pick_closest(point, choices):
    closeness = np.sum(np.square(choices-point), axis=1)
    closest = np.argmin(closeness)
    return choices[closest]

def get_centroid_and_rot(obj_corners, scene_corners):
    scene_centroid = np.int32(np.mean(scene_corners[:,0,:], axis=0))[::-1]
    v1 =  np.array([obj_corners[1, 0, 0], obj_corners[1, 0, 1]]) - np.array([obj_corners[0, 0, 0], obj_corners[0, 0, 1]])
    v2 = np.array([scene_corners[1, 0, 0], scene_corners[1, 0, 1]]) - np.array([scene_corners[0, 0, 0], scene_corners[0, 0, 1]])
    
    nv2 = (v2 / la.norm(v2))

    print('vec used for rotation', nv2)
    print('obj corners', obj_corners[:, 0, :])
    print('scene corners', scene_corners[:, 0, :])
    scene_rotation = np.arccos(nv2[0])
    if nv2[1] < 0:
        scene_rotation = scene_rotation * -1

    # cosang = np.dot(v1, v2)
    # sinang = la.norm(np.cross(v1, v2))

    # scene_rotation = np.arctan2(sinang, cosang)


    return scene_centroid, -1 * scene_rotation

def SURF_detect(piece, ref_img):

    #img_object = preproc(piece)
    #img_scene = preproc(ref_img)
    #print('ex1',stats(img_object))
    img_object = piece
    img_scene = ref_img

    #img_object = cv.imread("tmp_images/bears.png", cv.IMREAD_GRAYSCALE)
    #img_scene = cv.imread("tmp_images/go.png", cv.IMREAD_GRAYSCALE)
    #print('ex2', stats(img_object))

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
    if H is None:
        return None, None
    #print('H')
    #print(H)

    #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img_object.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img_object.shape[1]
    obj_corners[2,0,1] = img_object.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img_object.shape[0]


    obj_centroid = np.int32(np.mean(obj_corners[:,0,:], axis=0))
    #print(stats(obj_corners))


    scene_corners = cv.perspectiveTransform(obj_corners, H)
    scene_centroid, scene_rotation = get_centroid_and_rot(obj_corners, scene_corners)
    print('print scene_centroid')
    print(scene_rotation * 180 / math.pi)


    #print('CENTROID: ', scene_centroid)
    #print('ROTATION: ', scene_rotation)
    #-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
       (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
       (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
       (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
       (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)

    #print(scene_rotation)

    #-- Show detected matches
    imshow(img_matches, title="matches", cmap=None, just_write=True)

    return scene_centroid, scene_rotation

# see : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
def SIFT_detect(piece, ref_img):
    good = []
    MIN_MATCH_COUNT = 6
    while len(good) < MIN_MATCH_COUNT:
        MIN_MATCH_COUNT -= 1
        good = []
        img1 = piece         # queryImage
        img2 = ref_img # trainImage

        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

    if len(good) < 2:
        print('uh oh... random it is!')
        return None
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,4.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)

    #img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    scene_centroid, scene_rotation = get_centroid_and_rot(pts, dst)

    return scene_centroid, scene_rotation

