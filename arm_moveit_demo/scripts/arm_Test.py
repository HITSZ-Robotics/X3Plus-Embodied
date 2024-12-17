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

def arm_ctrl():
    #接受命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default = None)
    args = parser.parse_args()
    print(args.file_name)
    file_name=args.file_name
    #下载文件并安装
   # file_name='joint1.json'
    download(file_name)
    with open (path+file_name,'r')as f:
        joint=json.load(f)
    
    print (joint)




    # 初始化节点
  #  rospy.init_node("set_joint_py", anonymous=True,log_level=rospy.INFO, disable_signals=True) 
    # 初始化机械臂
   # pub_joint = rospy.Publisher("/move_group/fake_controller_joint_states", JointState, queue_size=1000)
    # 真机
   # pub_Arm = rospy.Publisher("TargetAngle", ArmJoint, queue_size=1000)
   # arm_joint = ArmJoint()
   # arm_joint.id = 6
   # arm_joint.angle = joint[5]*180/pi
   # joint_state = JointState()
   # joint_state.name = ["grip_joint"]
   # joint_state.position = [joint[5]]
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
    yahboomcar.set_named_target("down")
    yahboomcar.go()
    sleep(0.5)
    # 设置目标点 弧度
    joints = joint
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
   # for i in range(10):
    #    pub_joint.publish(joint_state)
     #   pub_Arm.publish(arm_joint)
      #  sleep(0.1)        
arm_ctrl()  
