#!/usr/bin/env python3
# encoding: utf-8
import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from geometry_msgs.msg import Point
import subprocess
import signal
import os
import time

class test_sub:
    def  __init__(self):
        self.grasp_point_sub = rospy.Subscriber('/2d_grasp_point', Point, self.point_callback)
        self.pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
        self.config = rs.config() 
        self.align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
        self.align = rs.align(self.align_to)  
    def point_callback(self,data):
        # subprocess.run(['sudo', 'sh', '-c', f'echo "0" > /sys/bus/usb/devices/2-1/2-1.4/2-1.4.4/2-1.4.4.3/authorized'], check=True)
        # time.sleep(3)
        # subprocess.run(['sudo', 'sh', '-c', f'echo "1" > /sys/bus/usb/devices/2-1/2-1.4/2-1.4.4/2-1.4.4.3/authorized'], check=True)
        # context = rs.context()
        # if len(context.devices) > 0:
        #     print("Device connected11111.")
        # else:
        #     print("No device connected1111.")
        # self.stop_rosnode('/camera/realsense2_camera_manager')
        # self.stop_rosnode('/camera/realsense2_camera')
        # time.sleep(4)
        # subprocess.run(['sudo', 'sh', '-c', f'echo "0" > /sys/bus/usb/devices/2-1/2-1.4/2-1.4.4/2-1.4.4.3/authorized'], check=True)
        # time.sleep(3)
        # subprocess.run(['sudo', 'sh', '-c', f'echo "1" > /sys/bus/usb/devices/2-1/2-1.4/2-1.4.4/2-1.4.4.3/authorized'], check=True)
        # context = rs.context()
        # if len(context.devices) > 0:
        #     print("Device connected.")
        # else:
        #     print("No device connected.")
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # 配置depth流
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # 配置color流
        
        pipe_profile = self.pipeline.start(self.config)  # streaming流开始
        
        # 创建对齐对象与color流对齐
   
        point = []
        color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = self.get_aligned_images()  # 获取对齐图像与相机参数
        point.append(int(data.x))
        point.append(int(data.y))
        print(point)
        dis, camera_coordinate = self.get_3d_camera_coordinate(point, aligned_depth_frame, depth_intrin)
        print('depth: ', dis)  # 深度单位是mm
        print('camera_coordinate: ', camera_coordinate)
        # realsense_process = self.start_realsense()
        # print("恢复realsense2_camera...")

    def get_3d_camera_coordinate(self,depth_pixel, aligned_depth_frame, depth_intrin):
        x = depth_pixel[0]
        y = depth_pixel[1]
        dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
        # print ('depth: ',dis)       # 深度单位是m
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
        # print ('camera_coordinate: ',camera_coordinate)
        return dis, camera_coordinate  

    def get_aligned_images(self):
        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
        aligned_frames = self.align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐
    
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
        aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧
    
        #### 获取相机参数 ####
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    
        #### 将images转为numpy arrays ####
        img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB图
        img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    
        return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame  
    
    def start_realsense(self):
        process = subprocess.Popen(['roslaunch', 'realsense2_camera', 'rs_camera.launch'])
        return process

    def get_rosnode_processes(self,node_name):
        # 获取所有rosnode的信息
        output = subprocess.check_output(['rosnode', 'info', node_name], universal_newlines=True)
        processes = []
        for line in output.split('\n'):
            if 'Pid' in line:
                pid = int(line.split()[-1])
                processes.append(pid)
        return processes

    def stop_rosnode(self,node_name):
        processes = self.get_rosnode_processes(node_name)
        for pid in processes:
            os.kill(pid, signal.SIGTERM)
        # 确保进程已终止
    

if __name__ == "__main__":
    rospy.init_node("set_joint_py", anonymous=True,log_level=rospy.INFO, disable_signals=True) 
    grasp_node = test_sub()
    rospy.spin()
 
      