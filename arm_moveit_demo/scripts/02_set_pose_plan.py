#!/usr/bin/env python
# coding: utf-8
import rospy
from math import pi
from time import sleep
import moveit_commander
from geometry_msgs.msg import Pose
from moveit_commander.move_group import MoveGroupCommander
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Point
import numpy as np 

# 角度转弧度
DE2RA = pi / 180
point = []
class GraspNode:
    def __init__(self):
        self.grasp_point_sub = rospy.Subscriber('/3d_grasp_point', Point, self.point_callback)
    
    def point_callback(self,data):
        point.append(data.x)
        point.append(data.y)
        point.append(data.z)
        self.arm_ctrl(point)

    def arm_ctrl(self,point):
        # 初始化机械臂
        #print("i come innnnnnnnnnnnnnnnnnn")
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
        
    #  yahboomcar.set_named_target("init")
    # yahboomcar.go()
    # sleep(0.5)
        # 创建位姿实例
        pos = Pose()
        # 设置具体的位置
        pos.position.x = point[0]
        pos.position.y = point[1]
        pos.position.z = point[2]
        
        if(0.03>point[1] and -0.03<point[1]):
              roll = -180.0 * np.pi / 180.0
              pitch_range = [i * np.pi / 180.0 for i in range(20, 51, 10)]
              yaw_range = [-np.pi,np.pi]
        elif(point[1]>=0.03):
              roll = -180.0 * np.pi / 180.0  # -180度，转换为弧度
              pitch_range = [i * np.pi / 180.0 for i in range(20, 51, 10)] 
              #bate = math.atan2(args.x,args.y)
              yaw_range = [i * np.pi / 180.0 for i in range(-90, -180,-10)] 
        elif(point[1]<=-0.03):
              roll = -180.0 * np.pi / 180.0  # -180度，转换为弧度
              pitch_range = [i * np.pi / 180.0 for i in range(20, 51, 10)] 
              #bate = math.atan2(args.x,-args.y)
              yaw_range = [i * np.pi / 180.0 for i in range(int(90), int(180), 10)]       
        
        gripper_angle=-1.5
        yahboomcar_gripper.set_joint_value_target([gripper_angle,-gripper_angle,-gripper_angle,gripper_angle,-gripper_angle,gripper_angle])           
        for i in range(5):
            plan_gripper = yahboomcar_gripper.plan()
            if len(plan_gripper.joint_trajectory.points) != 0:
                sleep(6)
                print("Gripper plan success")
                yahboomcar.execute(plan_gripper)
                break
            else:
                print("Gripper plan error") 
        
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
                # 尝试运动规划
                for i in range(5):
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

            if not success:
                print("Failed to find a valid RPY combination for planning")
        
        sleep(4)
        gripper_angle=-0.5
        yahboomcar_gripper.set_joint_value_target([gripper_angle,-gripper_angle,-gripper_angle,gripper_angle,-gripper_angle,gripper_angle])           
        for i in range(5):
            plan_gripper = yahboomcar_gripper.plan()
            if len(plan_gripper.joint_trajectory.points) != 0:
                print("Gripper plan success")
                yahboomcar.execute(plan_gripper)
                break
            else:
                print("Gripper plan error")  
        sleep(3)                 
        yahboomcar.set_named_target("init")
        yahboomcar.go()      
        moveit_commander.roscpp_shutdown() 

if __name__ == '__main__':
     # 初始化节点
    rospy.init_node("set_joint_py", anonymous=True,log_level=rospy.INFO, disable_signals=True) 
    grasp_node = GraspNode()
    rospy.spin()



