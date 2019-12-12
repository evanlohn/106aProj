#!/usr/bin/env python
import sys
if sys.version[0] != '2':
	print('please run all code with python2!')
	exit(0)

import numpy as np
#import segment
from piece import Piece
from capture import single_capture

#import rospy
#import tf2_ros
#from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
#from tf.transformations import quaternion_from_euler


def request_calibration():
	print('entering calibration mode\n')
	print('Ensure tf_echo is running\n')

	print('position the robot arm so it is not blocking the view of the table\n')
	#standby currently hardcoded

	print('place the standard calibration paper square on the table, near the top left corner relative to the camera\n')
	print('Once the window pops up, press c to capture.')
	calibration_pic = single_capture()

	print('move end effector to the origin; should be a corner of the paper used for calibration\n')
	print('once the arm is there, record the position:')

	val = str(raw_input())
	origin = np.array([float(value) for value in val.split(',')])

	print('\nnow move the end effector to anywhere along the edge of the paper that faces away from the robot.\n')
	print('again record the position once the end effector is in place')

	val = str(raw_input())
	along_x_axis = np.array([float(value) for value in val.split(',')])

	print('\nremove the calibration paper and robot arm from the view of the camera. Take a picture of the empty table\n')
	print('Once the window pops up, press c to capture.')
	empty_table = single_capture()

	print('done! Calibration results will now be used to create the table frame.\n')
	print('do not use the p command quite yet; wait for confirmation that the frame was created\n')

	return calibration_pic, origin, along_x_axis, empty_table

# good tutorial on adding frames in tf:
# http://wiki.ros.org/tf/Tutorials/Adding%20a%20frame%20%28Python%29
def calibrate(origin, along_x_axis):
	#goal is the translation and the rotation about the z axis
	# translation is already given by "origin", modulo a small correction factor to z
	trans = tuple(origin)
	x_axis = along_x_axis - origin
	x_axis = float(x_axis)/np.linalg.norm(x_axis)
	assert np.allclose(np.dot(x_axis, x_axis), 1)
	# the angle between two unit-length vectors (in this case, [1, 0, 0] and x_axis)
	# is the arccos of the dot product. dot product of x_axis and 0 is the first element of x_axis.
	theta = np.arccos(x_axis[0])

	print "translation: {}    rotation: {}".format(trans, theta)

	# this is supposed to broadcast the transform from base frame to a new "table" frame.
	#br = tf.TransformBroadcaster()
    #br.sendTransform(trans,
#                     tf.transformations.quaternion_from_euler(0, 0, theta),
#                     rospy.Time.now(),
#                     "table",
#                     "base")

# origin is the pixel coordinates (x, y) of the origin of the table frame (extracted by calibrate_ppm)
# pixel_loc is the pixel coordinates of the pixel we want to determine
# ppm is pixels per meter, found in calibrate_ppm
def pixel_to_table_frame(origin, pixel_loc, ppm):
    pixel_diff = np.array(pixel_loc) - np.array(origin)
    # note that this ^^ implicitly assumes that the "vertical" of the image is the x axis of
    # the table frame
    return pixel_diff/ppm

#converts coords and theta to Pose
def coords_to_pose(coords, theta):
	pose = Pose()
	pose.position.x = coords[0]
	pose.position.y = coords[1]
	pose.position.z = coords[2]
	q = quaternion_from_euler(0, -1 * math.pi, theta)
	pose.orientation.x = q[0]
	pose.orientation.y = q[1]
	pose.orientation.z = q[2]
	pose.orientation.w = q[3]
	return pose

# Places a piece using Baxter
def place(piece, pixel_origin, ppm):
	# piece.init_pos should have the initial pixel position
	# piece.final_pos should have the final pixel position
	# piece.rot_delta has the amount of rotation about the z axis necessary

	# calculate current coordinates of the piece in the table frame and convert to poses
	start_coords = pixel_to_table_frame(pixel_origin, piece.init_pos, ppm)
	end_coords = pixel_to_table_frame(pixel_origin, piece.final_pos, ppm)
	start_pose = (start_coords, 0)
	end_pose = (end_coords, piece.rot_delta)

	pose_arr = PoseArray()
	pose_arr.header.frame_id = "table"
	pose_arr.poses = [start_pose, end_pose]

	# Send message to path planner
	return pick_and_place_client(poses)

# Client for communicating with pick and place service
# Send 2 poses for pick and place operation
def pick_and_place_client(poses):
	#wait for service to start
	rospy.wait_for_service('pick_and_place')
	try:
		pick_and_place = rospy.ServiceProxy('pick_and_place', PlacePieceRequest)
		response = pick_and_place(poses)
		return response.success
	except rospy.ServiceException, e:
		print 'pick_and_place service failed'

def wait_for(str):
	count = 0
	while True:
		letter = raw_input()
		if letter == str:
			break
		count += 1
		if count % 69 == 0:
			print('remember: you just need to type {} to move to the next step. smh Beccy'.format(str))

print('requesting calibration! delete me when you use main()')
request_calibration()

def help():
	print('commands are: (press enter after typing in the letter')
	print('c: calibrates robot. Place a standard sheet of paper at wherever the origin of the table frame will be')
	print('p: pick and place a new piece')
	print('otherwise: display this help message')

def main():
	help()
	last_img = None
	ref_path = './raw_img_data/full_puzzle.png'
	ref_raw = increase_contrast(imread(ref_path))
	ref_img = segment_reference(ref_raw)
	while True:
		command = input()
		if command == 'c':
			cal_pic, o, a, empty_table = request_calibration()
			pixel_origin, ppm, deskew_transform = segment.paper_calibration(cal_pic)
			last_img = empty_table
			#TODO: store the above somewhere useful
			#TODO: send rest position to pick_and_place service
			calibrate(o, a)
		elif command == 'p':
			if last_img is None:
				print('you must use the c command to calibrate at least once before running p')
				continue
			print('<<< beginning pick and place >>>')
			print('segmenting piece... click the window and press c to start')
			new_img = single_capture()
			p_img, init_pos = segment_pieces(new_img, last_img,deskew_transform)
			print('piece segmented from rest of image')
			piece = Piece(p_img, init_pos)
			print('picking best position and orientation for piece in final puzzle')
			piece.solve_piece(ref_img)
			print('found final piece location. Starting pick and place...')
			place(piece, pixel_origin, ppm)
			print('piece has been placed. Place next piece and run ')
			#TODO: Take picture of end result for next segmentation
		else:
			help()
