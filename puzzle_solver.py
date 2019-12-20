#!/usr/bin/env python
import sys
if sys.version[0] != '2':
	print('please run all code with python2!')
	exit(0)

import numpy as np
from piece import Piece
from capture import single_capture

import rospy
import tf2_ros as tf
import tf2_msgs.msg
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion, TransformStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply

from planning.srv import PlacePiece, PlacePieceResponse

from contrast import increase_contrast
from segment import imread, segment_reference, paper_calibration, segment_pieces

import math

import cv2 as cv

def request_calibration(debug=False):
	if (debug):
		origin = np.array([.27, -.78, -.2])
		along = np.array([.44, -.79, -.2])
		calib = imread('./raw_img_data/calib.png')
		empty = imread('./raw_img_data/empty_table.png')
		return calib, origin, along, empty

	tfBuffer = tf.Buffer()
	listener = tf.TransformListener(tfBuffer)
	print('entering calibration mode\n')
	#print('Ensure tf_echo is running\n')

	print('Position the AR tag at the origin of the table frame\n')

	#print('position the robot arm so it is not blocking the view of the table\n')
	#standby currently hardcoded

	#print('place the standard calibration paper square on the table, near the top left corner relative to the camera\n')
	print('Once the window pops up, press c to capture.\n')
	calibration_pic = single_capture()


	print('Now, position the left arm such that the AR tag is in view.\n')
	print('Once the arm is positioned correctly, press enter, and avoid touching the arm or AR tag until the next step appears.\n')
	enter = raw_input()
	#TODO: Calculate table frame transform using base to camera and camera to base transforms from TF, and publish it to TF
	#Wait for camera to AR tag
	rate = rospy.Rate(10.0)
	#grabbing the ar tag transform
	
	while (True):
		print('trying to get ar transform')
		while not rospy.is_shutdown():
			try:
				ar_trans = tfBuffer.lookup_transform('left_hand_camera_axis', 'ar_marker_3', rospy.Time())
				break
			except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
				rate.sleep()
				continue
		#grabbing the camera transform once the ar tag transform is retrieved.
		print('trying to get hand transform')
		while not rospy.is_shutdown():
			try:
				camera_trans = tfBuffer.lookup_transform('base', 'right_hand_camera_axis', rospy.Time())
				break
			except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
				rate.sleep()
				continue
		#Finally compose to get table frame, and publish to TF
		calibrate(camera_trans, ar_trans)
		print('type c to exit loop\n')

		if (str(raw_input()) == 'c'):
			break

	#print('move end effector to the origin; should be a corner of the paper used for calibration\n')
	#print('once the arm is there, record the position:')

	#val = str(raw_input())
	#origin = np.array([float(value) for value in val.split(',')])

	#print('\nnow move the end effector to anywhere along the edge of the paper that faces away from the robot.\n')
	#print('again record the position once the end effector is in place')

	#val = str(raw_input())
	#along_x_axis = np.array([float(value) for value in val.split(',')])

	print('\nremove the AR tag and robot arm from the view of the camera. Take a picture of the empty table\n')
	print('Once the window pops up, press c to capture.')
	empty_table = single_capture()

	#print('done! Calibration results will now be used to create the table frame.\n')
	print('do not use the p command quite yet; wait for confirmation that the frame was created\n')

	return calibration_pic, empty_table#along_x_axis, empty_table

# good tutorial on adding frames in tf:
# http://wiki.ros.org/tf/Tutorials/Adding%20a%20frame%20%28Python%29
def calibrate(c_trans, ar_trans):#(origin, along_x_axis):
	#goal is the translation and the rotation about the z axis
	# translation is already given by "origin", modulo a small correction factor to z

	#TODO: calculate composition of two frames, publish to TF

	c_q = np.array([c_trans.transform.rotation.x, c_trans.transform.rotation.y, c_trans.transform.rotation.z, c_trans.transform.rotation.w])
	ar_q = np.array([ar_trans.transform.rotation.x, ar_trans.transform.rotation.y, ar_trans.transform.rotation.z, ar_trans.transform.rotation.w])
	composed_q = quaternion_multiply(c_q, ar_q)
	(roll, pitch, yaw) = euler_from_quaternion(composed_q)
	composed_q = quaternion_from_euler(0, 0, yaw)

	#x_axis = along_x_axis - origin
	#x_axis = np.float32(x_axis)/np.linalg.norm(x_axis)
	#assert np.allclose(np.dot(x_axis, x_axis), 1)
	# the angle between two unit-length vectors (in this case, [1, 0, 0] and x_axis)
	# is the arccos of the dot product. dot product of x_axis and 0 is the first element of x_axis.
	#cosang = np.dot(x_axis, np.array([1, 0, 0]))
	#sinang = np.linalg.norm(np.cross(x_axis, np.array([1, 0, 0])))

	#theta = -1 * np.arctan2(sinang, cosang)

	#print "translation: {}    rotation: {}".format(trans, theta)

	#pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)

	# this is supposed to broadcast the transform from base frame to a new "table" frame.
	br = tf.StaticTransformBroadcaster()
	t = TransformStamped()
	t.header.stamp = rospy.Time.now()
	t.header.frame_id = "base"
	t.child_frame_id = "table"
	t.transform.translation.x = c_trans.transform.translation.x - ar_trans.transform.translation.x
 	t.transform.translation.y = c_trans.transform.translation.y - ar_trans.transform.translation.y
	t.transform.translation.z = c_trans.transform.translation.z - ar_trans.transform.translation.z
	#q = quaternion_from_euler(0, 0, theta)
	t.transform.rotation.x = composed_q[0]
	t.transform.rotation.y = composed_q[1]
	t.transform.rotation.z = composed_q[2]
	t.transform.rotation.w = composed_q[3]
	br.sendTransform(t)#(trans, quaternion_from_euler(0, 0, theta), rospy.Time.now(), "table", "base")
	#tfm = tf2_msgs.msg.TFMessage([t])
	#pub_tf.publish(tfm)

# origin is the pixel coordinates (x, y) of the origin of the table frame (extracted by calibrate_ppm)
# pixel_loc is the pixel coordinates of the pixel we want to determine
# ppm is pixels per meter, found in calibrate_ppm
def pixel_to_table_frame(origin, pixel_loc, ppm):
    print "PIXEL TO TABLE FRAME---------------------"
    print origin
    print pixel_loc
    pixel_diff = np.array(pixel_loc) - np.array(origin)
    print pixel_diff 
    print ppm
    # note that this ^^ implicitly assumes that the "vertical" of the image is the x axis of
    # the table frame
    return np.float32(pixel_diff)/ppm

#converts coords and theta to Pose
def coords_to_pose(coords, theta):
	pose = Pose()
	pose.position.x = coords[0]
	pose.position.y = coords[1]
	pose.position.z = 0.01
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
	start_coords = [0,0,0]#pixel_to_table_frame(pixel_origin, piece.init_pos, ppm)#[.607, -.454, -.226]#
	end_coords = pixel_to_table_frame(np.array([0,0]), piece.final_pos, ppm)#[.546, .045, -.226]#
	print(start_coords)
	print(end_coords)
	#start_coords = [0,0]
	start_pose = coords_to_pose(start_coords, 0)
	start_pose.position.x += .02
	end_pose = coords_to_pose(end_coords, piece.rot_delta)#math.pi)
	end_pose.position.x -= .1
	pose_arr = PoseArray()
	pose_arr.header.frame_id = "table"
	pose_arr.poses = [start_pose, end_pose]
	print('theta: ')
	print(piece.rot_delta)

	# Send message to path planner
	return pick_and_place_client(pose_arr)

# Client for communicating with pick and place service
# Send 2 poses for pick and place operation
def pick_and_place_client(poses):
	#wait for servcvice to start
	rospy.wait_for_service('pick_and_place')
	try:
		pick_and_place = rospy.ServiceProxy('pick_and_place', PlacePiece)
		response = pick_and_place(poses)
		return response.success
	except rospy.ServiceException, e:
		print 'pick_and_place service failed'

def wait_for(s):
	count = 0
	while True:
		letter = raw_input()
		if letter == s:
			break
		count += 1
		if count % 69 == 0:
			print('remember: you just need to type {} to move to the next step. smh Beccy'.format(s))
def scale_to_ppm(ref_img, ppm, phy_size):
	new_size = int(ppm * phy_size[0]), int(ppm *phy_size[1])
	return cv.resize(ref_img, new_size)


def help():
	print('commands are: (press enter after typing in the letter')
	print('c: calibrates robot. Place a standard sheet of paper at wherever the origin of the table frame will be')
	print('p: pick and place a new piece')
	print('otherwise: display this help message')

def main(debug=False):
	rospy.init_node('puzzle_solver_node')
	help()
	last_img = None
	ref_path = './raw_img_data/full_puzzle.png'
	ref_raw = increase_contrast(imread(ref_path))
	ref_img, _ = segment_reference(ref_raw)
	while True:
		command = str(raw_input())
		if command == 'c':
			#cal_pic, o, a, empty_table = request_calibration(debug)
			cal_pic, empty_table = request_calibration(debug)
			pixel_origin, ppm, deskew_transform = paper_calibration(cal_pic)
			last_img = empty_table
			ref_img = scale_to_ppm(ref_img, ppm, (0.69, 0.49))
			#calibrate(o, a)
		elif command == 'p':
			if last_img is None:
				print('you must use the c command to calibrate at least once before running p')
				continue
			if (debug):
				last_img = imread('./raw_img_data/img12.png')
				new_img = imread('./raw_img_data/img13.png')
				p_img, init_pos = segment_pieces(new_img, last_img, deskew_transform)
				piece = Piece(p_img, init_pos)
				piece.solve_piece(ref_img)
				last_img = new_img
			else:
				print('<<< beginning pick and place >>>')
				print('segmenting piece... click the window and press c to start')
				new_img = single_capture()
				p_img, init_pos = segment_pieces(new_img, last_img, deskew_transform)
				print('piece segmented from rest of image')
				piece = Piece(p_img, init_pos)
				print('picking best position and orientation for piece in final puzzle')
				piece.solve_piece(ref_img)
				print('found final piece location. Starting pick and place...')
				place(piece, pixel_origin, ppm)
				print('piece has been placed. Capture progress')
				last_img = single_capture()
				print('progress captured. Place next piece and run')

		else:
			help()


if __name__ == '__main__':
    main()#debug=True)
    print 'done'
