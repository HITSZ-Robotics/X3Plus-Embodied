#!/usr/bin/env python
# coding: utf-8
import rospy
from math import pi
from time import sleep
from geometry_msgs.msg import Pose
from moveit_commander.move_group import MoveGroupCommander
from tf.transformations import quaternion_from_euler

import math

# 角度转弧度
DE2RA = pi / 180

if __name__ == '__main__':
    # 初始化节点
    rospy.init_node("set_joint_py", anonymous=True)
    # 初始化机械臂
    yahboomcar = MoveGroupCommander("arm_group")
    # 当运动规划失败后，允许重新规划
    yahboomcar.allow_replanning(True)
    yahboomcar.set_planning_time(5)
    # 尝试规划的次数
    yahboomcar.set_num_planning_attempts(10)
    # 设置允许目标角度误差
    yahboomcar.set_goal_joint_tolerance(0.001)
    # 设置允许的最大速度和加速度
    yahboomcar.set_max_velocity_scaling_factor(1.0)
    yahboomcar.set_max_acceleration_scaling_factor(1.0)
    # 设置"down"为目标点
    #yahboomcar.set_named_target("down")
    #yahboomcar.go()
    sleep(0.5)
    # 设置目标点 弧度
    #joints = [0, 0.79, -1.57, -1.57, 0]


    
    print("Input x:")
    x_base=float(input())
    print("Input y:")
    y_base=float(input())
    print("Input z:")
    z_base=float(input())
    print("Input mode(1/2):")
    mode=int(input())
    

    joints=[0,0,0,0,0]

    if mode==1:
        # 法一：把远端两个连杆合为一个连杆，控制夹爪到达目标点
        L1 = 0.083
        L2 = 0.255
        # x_base = 0.39
        # y_base = 0.1
        # z_base = -0.08
        x_raw = x_base - 0.098
        y_raw = y_base
        z_raw = z_base - 0.142
        theta0 = math.atan(-y_raw / x_raw)
        x = math.sqrt(x_raw * x_raw + y_raw * y_raw)
        y = 0
        z = z_raw
        alpha = math.acos((L1 * L1 + L2 * L2 - x * x - y * y) / (2 * L1 * L2))
        theta2 = pi - alpha
        theta1 = math.atan(z / x) + math.atan((L2 * math.sin(theta2)) / (L1 + L2 * math.cos(theta2)))
        joints = [theta0, -pi / 2 + theta1, -theta2, 0, 0]
        # 到此结束


    elif mode==2:
        # 法二：保证最远端的连杆垂直地面抓取，规划最远端两个连杆的交点到达目标点上方L3处
        L1 = 0.083
        L2 = 0.083
        L3 = 0.172
        # x_base = 0.39;
        # y_base = 0.1;
        # z_base = -0.08;
        x_raw = x_base - 0.098
        y_raw = y_base
        z_raw = z_base - 0.142
        theta0 = math.atan(-y_raw / x_raw)
        x = math.sqrt(x_raw * x_raw + y_raw * y_raw)
        y = 0
        z = z_raw + L3
        alpha = math.acos((L1 * L1 + L2 * L2 - x * x - y * y) / (2 * L1 * L2))
        theta2 = pi - alpha
        theta1 = math.atan(z / x) + math.atan((L2 * math.sin(theta2)) / (L1 + L2 * math.cos(theta2)))
        joints = [theta0, -pi / 2 + theta1, -theta2, -theta1+theta2-pi/2, 0]
        # 到此结束
    
    

    print(joints)

    yahboomcar.set_joint_value_target(joints)
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
