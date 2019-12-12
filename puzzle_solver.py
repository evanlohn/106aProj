import sys
if sys.version[0] != '2':
	print('please run all code with python2!')
	exit(0)

import numpy as np
from piece import Piece

def request_calibration():
	print('entering calibration mode')

	print('position the robot arm so it is not blocking the view of the table')
	print('place the standard calibration paper on the table; press r and hit enter when this is done')
	wait_for('r')
	# TODO: code to record robot arm position and take a picture of the calibration paper on the table
	calibration_pic = np.zeros((1080, 1920))
	calibration_pic[400:600, 700:1300] = 128
	calibration_pic[420:444, 720:737] = 200
	rest_position = None

	print('move end effector to the origin; should be a corner of the paper used for calibration')
	print('once the arm is there, type \'r\' to record the position.')
	wait_for('r')
	#TODO: code to record the end effector position
	origin = np.array([0.3, -0.1, 0.2])

	print('now move the end effector to anywhere along the edge of the paper that faces away from the robot.')
	print('again press r to record the position once the end effector is in place')
	wait_for('r')
	#TODO: code to record the end effector position
	along_x_axis = np.array([0.2, 0.1, 0.21])
	print('remove the calibration paper and robot arm from the view of the camera')
	print('again press r once you have done so.')
	wait_for('r')

	#TODO: code to take a picture of the empty table
	empty_table = np.zeros((1080, 1920))
	empty_table[400:600, 700:1300] = 128
	print('done! Calibration results will now be used to create the table frame.')
	print('do not use the p command quite yet; wait for confirmation that the frame was created')

	return calibration_pic, rest_positions, origin, along_x_axis, empty_table


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

	print'translation: {}    rotation: {}'.format(trans, theta)
	"""
	# this is supposed to broadcast the transform from base frame to a new "table" frame.
	br = tf.TransformBroadcaster()
    br.sendTransform(trans,
                     tf.transformations.quaternion_from_euler(0, 0, theta),
                     rospy.Time.now(),
                     "table",
                     "base")
    print()
    """



def wait_for(str):
	count = 0
	while True:
		letter = input()
		if letter == str:
			break
		count += 1
		if count % 69 == 0:
			print('remember: you just need to type {} to move to the next step. smh Beccy'.format(str))

def help():
	print('commands are: (press enter after typing in the letter')
	print('c: calibrates robot. Place a standard sheet of paper at wherever the origin of the table frame will be')
	print('p: pick and place a new piece')
	print('otherwise: display this help message')

def main():
	help()
	while True:
		command = input()
		if command == 'c':
			cal_pic, rest_pos, o, a, empty_table = request_calibration()
			#TODO: store the above somewhere useful
			calibrate(o, a)
		elif command == 'p':
			print('<<< beginning pick and place >>>')
			print('segmenting piece...')
			p_img, init_pos = segment_piece()
			print('piece segmented from rest of image')
			piece = Piece(p_img, init_pos)
			print('picking best position and orientation for piece in final puzzle')
			piece.solve_piece(ref_img)
			print('found final piece location. Starting pick and place...')
			piece.place()
			# TODO: wait for message from baxter saying "done"
			print('piece has been placed. Place next piece and run ')
		else:
			help()






