#!/usr/bin/env python
"""
Path Planning Script for Lab 7, adapted for Group 37's final project
Author: Valmik Prabhu, Jason Huynh, Rebecca Abraham, Evan Lohn
"""
import sys
assert sys.argv[1] in ("sawyer", "baxter")
ROBOT = sys.argv[1]

if ROBOT == "baxter":
    from baxter_interface import Limb
else:
    from intera_interface import Limb

import rospy
import numpy as np
import traceback

from moveit_msgs.msg import OrientationConstraint
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

from path_planner import PathPlanner
from controller import Controller

from baxter_interface import gripper as robot_gripper
from tf.transformations import quaternion_from_euler
import math

from planning.srv import PlacePiece, PlacePieceResponse

def pick_and_place(pt1, pt2, q1, q2):
    """
    Pick and place script for puzzle solving.
    Receives two puzzle piece positions (initial and final) and a rotation in
    the table frame coordinates and moves the piece accordingly.
    """
    right_gripper = robot_gripper.Gripper('right')
    right_gripper.calibrate()
    x1, y1, z1 = pt1.x, pt1.y, pt1.z
    x2, y2, z2 = pt2.x, pt2.y, pt2.z
    q_1 = [q1.x, q1.y, q1.z, q1.w]
    q_2 = [q2.x, q2.y, q2.z, q2.w]

    standby = PoseStamped()
    standby.header.frame_id = "base"
    standby.pose.position.x = 0.306
    standby.pose.position.y = -0.813
    standby.pose.position.z = 0.128
    standby.pose.orientation.x = 0.0
    standby.pose.orientation.y = -1.0
    standby.pose.orientation.z = 0.0
    standby.pose.orientation.w = 0.0

    # Make sure that you've looked at and understand path_planner.py before starting

    planner = PathPlanner("right_arm")

    if ROBOT == "sawyer":
        Kp = 0.2 * np.array([0.4, 2, 1.7, 1.5, 2, 2, 3])
        Kd = 0.01 * np.array([2, 1, 2, 0.5, 0.8, 0.8, 0.8])
        Ki = 0.01 * np.array([1.4, 1.4, 1.4, 1, 0.6, 0.6, 0.6])
        Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    else:
        Kp = 0.45 * np.array([0.8, 2.5, 1.7, 2.2, 2.4, 3, 4])
        Kd = 0.015 * np.array([2, 1, 2, 0.5, 0.8, 0.8, 0.8])
        Ki = 0.01 * np.array([1.4, 1.4, 1.4, 1, 0.6, 0.6, 0.6])
        Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

    closed_controller = Controller(Kp, Kd, Ki, Kw, Limb("right"))

    # #Create a path constraint for the arm
    # #UNCOMMENT FOR THE ORIENTATION CONSTRAINTS PART
    orien_const = OrientationConstraint()
    orien_const.link_name = "right_gripper";
    orien_const.header.frame_id = "base";
    orien_const.orientation.y = -1.0;
    orien_const.absolute_x_axis_tolerance = 0.1;
    orien_const.absolute_y_axis_tolerance = 0.1;
    orien_const.absolute_z_axis_tolerance = 0.1;
    orien_const.weight = 1.0;

    #Obstacle
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = "table"
    pose_stamped.pose = Pose()
    #pose_stamped.pose.position = Point()
    pose_stamped.pose.position.x = 0
    pose_stamped.pose.position.y = 0
    pose_stamped.pose.position.z = -0.01
    #pose_stamped.pose.orientation = Quaternion()
    pose_stamped.pose.orientation.x = 0
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 1

    planner.add_box_obstacle(np.array([2, 4, .01]), "table0", pose_stamped)

    #Move to above initial puzzle piece position
    while not rospy.is_shutdown():
        try:
            goal_1 = PoseStamped()
            goal_1.header.frame_id = "table"

            #x, y, and z position
            goal_1.pose
            goal_1.pose.position.x = x1
            goal_1.pose.position.y = y1
            goal_1.pose.position.z = z1 + .1

            #Orientation as a quaternion
            goal_1.pose.orientation.x = q_1[0]
            goal_1.pose.orientation.y = q_1[1]
            goal_1.pose.orientation.z = q_1[2]
            goal_1.pose.orientation.w = q_1[3]

            plan = planner.plan_to_pose(goal_1, [])
            if not closed_controller.execute_path(plan):
                raise Exception("Execution failed")

            rospy.sleep(0.5)

        except Exception as e:
            print e
            traceback.print_exc()
        else:
            break

    #Lower and grip puzzle piece
    while not rospy.is_shutdown():
        try:
            goal_1b = PoseStamped()
            goal_1b.header.frame_id = "table"

            #x, y, and z position
            goal_1b.pose.position.x = x1
            goal_1b.pose.position.y = y1
            goal_1b.pose.position.z = z1 + .03

            #Orientation as a quaternion
            goal_1b.pose.orientation.x = q_1[0]
            goal_1b.pose.orientation.y = q_1[1]
            goal_1b.pose.orientation.z = q_1[2]
            goal_1b.pose.orientation.w = q_1[3]

            plan = planner.plan_to_pose(goal_1b, [])
            print(plan)
            if not closed_controller.execute_path(plan):
                raise Exception("Execution failed")

            rospy.sleep(0.5)
            right_gripper.close()
            rospy.sleep(0.5)

        except Exception as e:
            print e
            traceback.print_exc()
        else:
            break

    #Move up from initial puzzle piece position
    while not rospy.is_shutdown():
        try:
            goal_1c = PoseStamped()
            goal_1c.header.frame_id = "table"

            #x, y, and z position
            goal_1c.pose.position.x = x1
            goal_1c.pose.position.y = y1
            goal_1c.pose.position.z = z1 + .1

            #Orientation as a quaternion
            goal_1c.pose.orientation.x = q_1[0]
            goal_1c.pose.orientation.y = q_1[1]
            goal_1c.pose.orientation.z = q_1[2]
            goal_1c.pose.orientation.w = q_1[3]

            plan = planner.plan_to_pose(goal_1c, [])
            if not closed_controller.execute_path(plan):
                raise Exception("Execution failed")

            rospy.sleep(0.25)

        except Exception as e:
            print e
            traceback.print_exc()
        else:
            break

    #Move to final puzzle piece position, performing necessary rotation
    while not rospy.is_shutdown():
        try:
            goal_2 = PoseStamped()
            goal_2.header.frame_id = "table"

            #x, y, and z position
            goal_2.pose.position.x = x2
            goal_2.pose.position.y = y2
            goal_2.pose.position.z = z2 + .1

            #Orientation as a quaternion
            goal_2.pose.orientation.x = q_2[0]
            goal_2.pose.orientation.y = q_2[1]
            goal_2.pose.orientation.z = q_2[2]
            goal_2.pose.orientation.w = q_2[3]

            plan = planner.plan_to_pose(goal_2, [])
            if not closed_controller.execute_path(plan):
                raise Exception("Execution failed")

            rospy.sleep(0.25)

        except Exception as e:
            print e
        else:
            break

    #Lower, place piece at final position, ungrip
    while not rospy.is_shutdown():
        try:
            goal_2b = PoseStamped()
            goal_2b.header.frame_id = "table"

            #x, y, and z position
            goal_2b.pose.position.x = x2
            goal_2b.pose.position.y = y2
            goal_2b.pose.position.z = z2 + .03

            #Orientation as a quaternion
            goal_2b.pose.orientation.x = q_2[0]
            goal_2b.pose.orientation.y = q_2[1]
            goal_2b.pose.orientation.z = q_2[2]
            goal_2b.pose.orientation.w = q_2[3]

            plan = planner.plan_to_pose(goal_2b, [])
            if not closed_controller.execute_path(plan):
                raise Exception("Execution failed")

            rospy.sleep(0.5)
            right_gripper.open()
            rospy.sleep(0.5)

        except Exception as e:
            print e
        else:
            break

    #Rise back above final position
    while not rospy.is_shutdown():
        try:
            goal_2c = PoseStamped()
            goal_2c.header.frame_id = "table"

            #x, y, and z position
            goal_2c.pose.position.x = x2
            goal_2c.pose.position.y = y2
            goal_2c.pose.position.z = z2 + .1

            #Orientation as a quaternion
            goal_2c.pose.orientation.x = q_2[0]
            goal_2c.pose.orientation.y = q_2[1]
            goal_2c.pose.orientation.z = q_2[2]
            goal_2c.pose.orientation.w = q_2[3]

            plan = planner.plan_to_pose(goal_2c, [])
            if not closed_controller.execute_path(plan):
                raise Exception("Execution failed")

            rospy.sleep(0.25)

        except Exception as e:
            print e
            traceback.print_exc()
        else:
            break

    #Move to standby position
    while not rospy.is_shutdown():
        try:
            goal_3 = PoseStamped()
            goal_3.header.frame_id = "base"

            """goal_3.pose.position.x = 0.306
            goal_3.pose.position.y = -0.813
            goal_3.pose.position.z = 0.128

            #Orientation as a quaternion
            goal_3.pose.orientation.x = 0.0
            goal_3.pose.orientation.y = -1.0
            goal_3.pose.orientation.z = 0.0
            goal_3.pose.orientation.w = 0.0"""

            plan = planner.plan_to_pose(standby, [])#goal_3, [])
            if not closed_controller.execute_path(plan):
                raise Exception("Execution failed")

            rospy.sleep(0.25)

        except Exception as e:
            print e
        else:
            break

def handle_pick_and_place(req):
    pose_arr = req.poses
    if (len(pose_arr.poses) == 2):
        q1 = pose_arr.poses[0].orientation
        q2 = pose_arr.poses[1].orientation
        pick_and_place(pose_arr.poses[0].position, pose_arr.poses[1].position, q1, q2)
        return PlacePieceResponse(True)
    return PlacePieceResponse(False)

def pick_and_place_server():
    rospy.init_node('pick_and_place_node')
    s = rospy.Service('pick_and_place', PlacePiece, handle_pick_and_place)
    print("service started")
    rospy.spin()

if __name__ == '__main__':
    pick_and_place_server()
