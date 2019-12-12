#!/usr/bin/env python
"""
Path Planning Script for Lab 7
Author: Valmik Prabhu
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

def main(x1, y1, x2, y2, z, theta):
    """
    Main Script
    """
    right_gripper = robot_gripper.Gripper('right')
    q = [0, -1, 0, 0]
    rotated_q = quaternion_from_euler(0, -1 * math.pi, theta)

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

    ##
    ## Add the obstacle to the planning scene here
    ##

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



    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = "base"
    pose_stamped.pose = Pose()
    pose_stamped.pose.position = Point()
    pose_stamped.pose.position.x = .5
    pose_stamped.pose.position.y = 0
    pose_stamped.pose.position.z = -.244
    pose_stamped.pose.orientation = Quaternion()
    pose_stamped.pose.orientation.x = 0
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0   
    pose_stamped.pose.orientation.w = 1


    planner.add_box_obstacle(np.array([2, 1, .1]), "table0", pose_stamped)
    #planner.add_box_obstacle(np.array([.4, 0.1, .4]), "wall", wall)

    while not rospy.is_shutdown():

        while not rospy.is_shutdown():
            try:
                #if ROBOT == "baxter":
                #    x, y, z = 0.47, -0.85, 0.07
                #else:
                #    x, y, z = 0.8, 0.05, -0.23
                goal_1 = PoseStamped()
                goal_1.header.frame_id = "base"

                #x, y, and z position
                goal_1.pose.position.x = x1
                goal_1.pose.position.y = y1
                goal_1.pose.position.z = z + .1

                #Orientation as a quaternion
                goal_1.pose.orientation.x = q[0]
                goal_1.pose.orientation.y = q[1]
                goal_1.pose.orientation.z = q[2]
                goal_1.pose.orientation.w = q[3]

                # Might have to edit this . . . 
                plan = planner.plan_to_pose(goal_1, [])

                if not closed_controller.execute_path(plan):#planner.execute_plan(plan):
                    raise Exception("Execution failed")
            except Exception as e:
                print e
                traceback.print_exc()
            else:
                break
        
        rospy.sleep(0.5)

        while not rospy.is_shutdown():
            try:
                #if ROBOT == "baxter":
                #    x, y, z = 0.47, -0.85, 0.07
                #else:
                #    x, y, z = 0.8, 0.05, -0.23
                goal_1b = PoseStamped()
                goal_1b.header.frame_id = "base"

                #x, y, and z position
                goal_1b.pose.position.x = x1
                goal_1b.pose.position.y = y1
                goal_1b.pose.position.z = z

                #Orientation as a quaternion
                goal_1b.pose.orientation.x = q[0]
                goal_1b.pose.orientation.y = q[1]
                goal_1b.pose.orientation.z = q[2]
                goal_1b.pose.orientation.w = q[3]

                # Might have to edit this . . . 
                plan = planner.plan_to_pose(goal_1b, [])


                if not closed_controller.execute_path(plan):#planner.execute_plan(plan):
                    raise Exception("Execution failed")
            except Exception as e:
                print e
                traceback.print_exc()
            else:
                break

        rospy.sleep(0.5)
        right_gripper.close()
        rospy.sleep(0.5)

        while not rospy.is_shutdown():
            try:
                #if ROBOT == "baxter":
                #    x, y, z = 0.47, -0.85, 0.07
                #else:
                #    x, y, z = 0.8, 0.05, -0.23
                goal_1c = PoseStamped()
                goal_1c.header.frame_id = "base"

                #x, y, and z position
                goal_1c.pose.position.x = x1
                goal_1c.pose.position.y = y1
                goal_1c.pose.position.z = z + .1

                #Orientation as a quaternion
                goal_1c.pose.orientation.x = q[0]
                goal_1c.pose.orientation.y = q[1]
                goal_1c.pose.orientation.z = q[2]
                goal_1c.pose.orientation.w = q[3]

                # Might have to edit this . . . 
                plan = planner.plan_to_pose(goal_1c, [])


                if not closed_controller.execute_path(plan):#planner.execute_plan(plan):
                    raise Exception("Execution failed")
            except Exception as e:
                print e
                traceback.print_exc()
            else:
                break

        while not rospy.is_shutdown():
            try:
                goal_2 = PoseStamped()
                goal_2.header.frame_id = "base"

                #x, y, and z position
                goal_2.pose.position.x = x2
                goal_2.pose.position.y = y2
                goal_2.pose.position.z = z + .1

                #Orientation as a quaternion
                goal_2.pose.orientation.x = rotated_q[0]
                goal_2.pose.orientation.y = rotated_q[1]
                goal_2.pose.orientation.z = rotated_q[2]
                goal_2.pose.orientation.w = rotated_q[3]

                plan = planner.plan_to_pose(goal_2, [])

                if not closed_controller.execute_path(plan):#planner.execute_plan(plan):
                    raise Exception("Execution failed")
            except Exception as e:
                print e
            else:
                break

        while not rospy.is_shutdown():
            try:
                goal_2b = PoseStamped()
                goal_2b.header.frame_id = "base"

                #x, y, and z position
                goal_2b.pose.position.x = x2
                goal_2b.pose.position.y = y2
                goal_2b.pose.position.z = z

                #Orientation as a quaternion
                goal_2b.pose.orientation.x = rotated_q[0]
                goal_2b.pose.orientation.y = rotated_q[1]
                goal_2b.pose.orientation.z = rotated_q[2]
                goal_2b.pose.orientation.w = rotated_q[3]

                plan = planner.plan_to_pose(goal_2b, [])


                if not closed_controller.execute_path(plan):#planner.execute_plan(plan):
                    raise Exception("Execution failed")
            except Exception as e:
                print e
            else:
                break

        rospy.sleep(0.5)
        right_gripper.open()
        rospy.sleep(0.5)

        while not rospy.is_shutdown():
            try:
                #if ROBOT == "baxter":
                #    x, y, z = 0.47, -0.85, 0.07
                #else:
                #    x, y, z = 0.8, 0.05, -0.23
                goal_2c = PoseStamped()
                goal_2c.header.frame_id = "base"

                #x, y, and z position
                goal_2c.pose.position.x = x2
                goal_2c.pose.position.y = y2
                goal_2c.pose.position.z = z + .1

                #Orientation as a quaternion
                goal_2c.pose.orientation.x = rotated_q[0]
                goal_2c.pose.orientation.y = rotated_q[1]
                goal_2c.pose.orientation.z = rotated_q[2]
                goal_2c.pose.orientation.w = rotated_q[3]

                # Might have to edit this . . . 
                plan = planner.plan_to_pose(goal_2c, [])

                if not closed_controller.execute_path(plan):#planner.execute_plan(plan):
                    raise Exception("Execution failed")
            except Exception as e:
                print e
                traceback.print_exc()
            else:
                break

        while not rospy.is_shutdown():
            try:
                goal_3 = PoseStamped()
                goal_3.header.frame_id = "base"

                #x, y, and z position
                goal_3.pose.position.x = 0.306
                goal_3.pose.position.y = -0.813
                goal_3.pose.position.z = 0.128

                #Orientation as a quaternion
                goal_3.pose.orientation.x = 0.0
                goal_3.pose.orientation.y = -1.0
                goal_3.pose.orientation.z = 0.0
                goal_3.pose.orientation.w = 0.0

                plan = planner.plan_to_pose(goal_3, [])

                if not closed_controller.execute_path(plan):#planner.execute_plan(plan):
                    raise Exception("Execution failed")
            except Exception as e:
                print e
            else:
                break

if __name__ == '__main__':
    rospy.init_node('moveit_node')
    main(0.644, -0.531, .777, -.042, -.135, 3.14159)
