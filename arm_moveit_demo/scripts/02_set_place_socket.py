#!/usr/bin/env python
# coding: utf-8
import rospy
from math import pi
import math
from time import sleep
import moveit_commander
from geometry_msgs.msg import Pose
from moveit_commander.move_group import MoveGroupCommander
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Point
import numpy as np
# 角度转弧度
DE2RA = pi / 180    
    
def arm_ctrl():
        # 初始化机械臂
        import argparse

# 创建解析器对象
        parser = argparse.ArgumentParser(description='Set position for the robot arm')

# 定义三个命令行参数
        parser.add_argument('--x', type=float, required=True, help='X coordinate')
        parser.add_argument('--y', type=float, required=True, help='Y coordinate')
        parser.add_argument('--z', type=float, required=True, help='Z coordinate')

# 解析命令行参数
        args = parser.parse_args()

# 打印解析后的参数
        print(args.x, args.y,args.z) 
        yahboomcar = MoveGroupCommander("arm_group")
        yahboomcar_gripper = MoveGroupCommander("gripper_group")
        print(yahboomcar.get_active_joints())
        print(yahboomcar.get_current_joint_values())
        print(yahboomcar.get_current_pose)
        # 当运动规划失败后，允许重新规划
        yahboomcar.allow_replanning(True)
        yahboomcar.set_planning_time(5)
        yahboomcar_gripper.allow_replanning(True)
        yahboomcar_gripper.set_planning_time(5)
        # 尝试规划的次数
        yahboomcar.set_num_planning_attempts(10)
        yahboomcar_gripper.set_num_planning_attempts(10)
        # 设置位置(单位：米)和姿态（单位：弧度）的允许误差
        yahboomcar.set_goal_position_tolerance(0.01)
        yahboomcar.set_goal_orientation_tolerance(1)
        # 设置允许目标误差
        yahboomcar.set_goal_tolerance(0.01)
        # 设置允许的最大速度和加速度
        yahboomcar_gripper.set_max_velocity_scaling_factor(1.0)
        yahboomcar_gripper.set_max_acceleration_scaling_factor(1.0)
        yahboomcar.set_max_velocity_scaling_factor(1.0)
        yahboomcar.set_max_acceleration_scaling_factor(1.0)
        # 设置"down"为目标点
        gripper_angle = -0.4
    #  yahboomcar.set_named_target("init")
    # yahboomcar.go()
    # sleep(0.5)
        # 创建位姿实例
        pos = Pose()
        # 设置具体的位置
        pos.position.x = args.x
        pos.position.y = args.y
        pos.position.z = args.z
        if(0.03>args.y and -0.03<args.y):
              roll = -180.0 * np.pi / 180.0
              pitch_range = [i * np.pi / 180.0 for i in range(20, 51, 10)]
              yaw_range=[]
              for i in range(180, 140, -10):
                  yaw_range.append(i * np.pi / 180.0)
                  yaw_range.append(-i * np.pi / 180.0)

        elif(args.y>=0.03):
              roll = -180.0 * np.pi / 180.0  # -180度，转换为弧度
              pitch_range = [i * np.pi / 180.0 for i in range(20, 51, 10)] 
              #bate = math.atan2(args.x,args.y)
              yaw_range = [i * np.pi / 180.0 for i in range(-90, -180,-10)] 
        elif(args.y<=-0.03):
              roll = -180.0 * np.pi / 180.0  # -180度，转换为弧度
              pitch_range = [i * np.pi / 180.0 for i in range(20, 51, 10)] 
              #bate = math.atan2(args.x,-args.y)
              yaw_range = [i * np.pi / 180.0 for i in range(int(90), int(180), 10)]       

        # 枚举 RPY 角度
        success = False
        for pitch in pitch_range:
                if success:
                        break
                for yaw in yaw_range:
                        # 将 RPY 转换为四元数
                        q = quaternion_from_euler(roll, pitch, yaw)

                        # 设置目标姿态
                        pos.orientation.x = q[0]
                        pos.orientation.y = q[1]
                        pos.orientation.z = q[2]
                        pos.orientation.w = q[3]
                        
                        # 设置目标点
                        yahboomcar.set_pose_target(pos)
                        
                        gripper_angle=-0.4
                        yahboomcar_gripper.set_joint_value_target([gripper_angle,-gripper_angle,-gripper_angle,gripper_angle,-gripper_angle,gripper_angle])
                        for i in range(1):
                                        plan_gripper = yahboomcar_gripper.plan()
                                        if len(plan_gripper.joint_trajectory.points) != 0:
                                        
                                           print("Gripper plan success")
                                           yahboomcar.execute(plan_gripper)
                                           break
                                        else:
                                           print("Gripper plan error")
                                

                        # 尝试运动规划
                        for i in range(1):
                           plan = yahboomcar.plan()
                           if len(plan.joint_trajectory.points) != 0:
                                print("Plan success with RPY: roll=-180, pitch={}, yaw={}".format(pitch * 180.0 / np.pi,yaw * 180.0 / np.pi))
                               # a=int(input())
                                #print("33333333333333333333333333333")
                                #b=int(input())
                                yahboomcar.execute(plan)
                                success = True
                                #print("44444444444444444444444444444444")
                                break
                           else:
                                print("Plan error with RPY: roll=-180, pitch={}, yaw={}".format(pitch * 180.0 / np.pi,yaw * 180.0 / np.pi))
                        #sleep(6)
                       # print("sleeping")
                        #sleep(6)
                        gripper_angle=-1.5

                        yahboomcar_gripper.set_joint_value_target([gripper_angle,-gripper_angle,-gripper_angle,gripper_angle,-gripper_angle,gripper_angle])

                        if success:
                                for i in range(1):
                                        plan_gripper = yahboomcar_gripper.plan()
                                        if len(plan_gripper.joint_trajectory.points) != 0:
                                           sleep(6)
                                          # print("Gripper plan success")
                                           yahboomcar.execute(plan_gripper)
                                           break
                                        else:
                                           print("Gripper plan error")
                                break

                if not success:
                   print("Failed to find a valid RPY combination for planning")
        #yahboomcar.set_named_target("init")
        #yahboomcar.go()      
        moveit_commander.roscpp_shutdown()
        #moveit_commander.os._exit(0)                                                                                                                           65,32         63%
    
arm_ctrl()
