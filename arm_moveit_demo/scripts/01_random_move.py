#!/usr/bin/env python
# coding: utf-8
import rospy
from time import sleep
from moveit_commander.move_group import MoveGroupCommander

if __name__ == '__main__':
    # 初始化节点
    rospy.init_node("yahboomcar_random_move")
    # 初始化机械臂规划组
    yahboomcar = MoveGroupCommander("arm_group")
    # 当运动规划失败后，允许重新规划
    yahboomcar.allow_replanning(True)
    # 设置规划时间
    yahboomcar.set_planning_time(5)
    # 尝试规划的次数
    yahboomcar.set_num_planning_attempts(10)
    # 设置允许目标位置误差
    yahboomcar.set_goal_position_tolerance(0.01)
    # 设置允许目标姿态误差
    yahboomcar.set_goal_orientation_tolerance(0.01)
    # 设置允许目标误差
    yahboomcar.set_goal_tolerance(0.01)
    # 设置最大速度
    yahboomcar.set_max_velocity_scaling_factor(1.0)
    # 设置最大加速度
    yahboomcar.set_max_acceleration_scaling_factor(1.0)
    while not rospy.is_shutdown():
        # 设置随机目标点
        yahboomcar.set_random_target()
        # 开始运动
        yahboomcar.go()
        sleep(0.5)
        # 设置"up"为目标点
        #yahboomcar.set_named_target("up")
        #yahboomcar.go()
        #sleep(0.5)
        # 设置"down"为目标点
        #yahboomcar.set_named_target("down")
        #yahboomcar.go()
        #sleep(0.5)
