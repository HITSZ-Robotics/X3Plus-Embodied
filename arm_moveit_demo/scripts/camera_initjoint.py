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
    #yahboomcar.set_named_target("init")
    #yahboomcar.go()
    #sleep(0.5)
    # 设置目标点 弧度
    #joints = [0, 0.79, -1.57, -1.57, 0
    
    joints=[-1.57,-1.57,0,1.57,0]
  
    

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
