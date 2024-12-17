#!/usr/bin/env python3
# coding: utf-8
import argparse
import rospy
import math
from math import pi
from time import sleep
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
    download(file_name)
    with open (path+file_name,'r')as f:
        point=json.load(f)
    
    print (point)


    #这里就替换为从终端读取输入的坐标(base_link系下)
    #开头那些读入的可以删掉
    x_base = point[0]
    y_base = point[1]
    z_base = point[2]
    

    
    # 法一：把远端两个连杆合为一个连杆，控制夹爪到达目标点
    L1 = 0.1159107954
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
    

    '''
    # 法二：保证最远端的连杆垂直地面抓取，规划最远端两个连杆的交点到达目标点上方L3处
    L1 = 0.083
    L2 = 0.083
    L3 = 0.172
   #x_base = 0.39;
   #y_base = 0.1;
   #z_base=-0.08;
    x_raw = x_base - 0.098
    y_raw = y_base
    z_raw = z_base - 0.142
    theta0 = math.atan(-y_raw / x_raw)
    x = sqrt(x_raw * x_raw + y_raw * y_raw)
    y = 0
    z = z_raw + L3
    alpha = math.acos((L1 * L1 + L2 * L2 - x * x - y * y) / (2 * L1 * L2))
    theta2 = pi - alpha
    theta1 = math.atan(z / x) + math.atan((L2 * math.sin(theta2)) / (L1 + L2 * math.cos(theta2)))
    joints = [theta0, -pi / 2 + theta1, -theta2, -theta1+theta2-pi/2, 0]
    # 到此结束
    '''

    arm_ctrl1(joints)











def arm_ctrl1(joints):
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


arm_ctrl()
