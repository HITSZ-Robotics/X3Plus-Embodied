#!/usr/bin/env python
# coding: utf-8
import argparse
import rospy
from math import pi
from time import sleep
import moveit_commander
from geometry_msgs.msg import Pose,PoseStamped
from moveit_commander.move_group import MoveGroupCommander
from tf.transformations import quaternion_from_euler
import sys
#sys.path.append(r'/home/yahboom/yahboomcar_ws/src/arm_ctrl/scripts/arm_test/'s)
from upload_download_cilent import *
from sensor_msgs.msg import JointState
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, PlanningScene, PlannerInterfaceDescription
from yahboomcar_msgs.msg import *

#path=r"/home/yahboom/yahboomcar_ws/src/robot/data/"
path=r"/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/data/"
# 角度转弧度
DE2RA = pi / 180

def arm_ctrl(joints):
    # 初始化节点
    rospy.init_node("set_joint_py", anonymous=True,log_level=rospy.INFO, disable_signals=True) 
    # 初始化机械臂
    yahboomcar_gripper = MoveGroupCommander("gripper_group")
    yahboomcar = MoveGroupCommander("arm_group")
    # 当运动规划失败后，允许重新规划
    yahboomcar.allow_replanning(True)
    yahboomcar.set_planning_time(5)
    yahboomcar_gripper.allow_replanning(True)
    yahboomcar_gripper.set_planning_time(5)
    # 尝试规划的次数
    yahboomcar.set_num_planning_attempts(10)
    yahboomcar_gripper.set_num_planning_attempts(10)
    # 设置允许目标角度误差
    yahboomcar.set_goal_joint_tolerance(0.001)
    yahboomcar_gripper.set_goal_joint_tolerance(0.001)
    # 设置允许的最大速度和加速度
    yahboomcar.set_max_velocity_scaling_factor(1.0)
    yahboomcar.set_max_acceleration_scaling_factor(1.0)
    yahboomcar_gripper.set_max_velocity_scaling_factor(1.0)
    yahboomcar_gripper.set_max_acceleration_scaling_factor(1.0)
    # 设置目标点 弧度
    joint = joints[0:5]
    gripper_angle = joints[5]
    yahboomcar.set_joint_value_target(joint)
    yahboomcar_gripper.set_joint_value_target([gripper_angle,-gripper_angle,-gripper_angle,gripper_angle,-gripper_angle,gripper_angle])
    # 多次执行,提高成功率
    for i in range(5):
        # 运动规划
        plan = yahboomcar.plan()
        if len(plan.joint_trajectory.points) != 0:
            print ("plan success")
            # 规划成功后运行
            yahboomcar.execute(plan)
            break
        else:
            print ("plan error")
    for i in range(5):
        # 运动规划
        plan_gripper = yahboomcar_gripper.plan()
        if len(plan_gripper.joint_trajectory.points) != 0:
            print ("plan_gripper success")
            # 规划成功后运行
            yahboomcar.execute(plan_gripper)
            break
        else:
            print ("plan_gripper error")        
joint1 = [-0.33,-1.45,-0.65,-1.07,-0.0,0]
joint2 = [-0.33,-1.45,-0.65,-1.07,-0.0,-0.5]
joint3 = [0.0,-0.74,-0.83,-0.12,-0.0,-0.5]
joint4 = [0.7,-1.39,-0.93,-0.81,-0.0,-1.54]           
arm_ctrl(joint2)
print("plan1 is okkkkkkkkk")
arm_ctrl(joint3)  
print("plan2 is okkkkkkkkkkk")
arm_ctrl(joint4)
