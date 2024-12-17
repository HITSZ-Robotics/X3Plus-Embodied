# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from geometry_msgs.msg import Point
 
''' 
设置
'''
pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # 配置depth流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # 配置color流
 
pipe_profile = pipeline.start(config)  # streaming流开始
 
# 创建对齐对象与color流对齐
align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
align = rs.align(align_to)  
point = []
 
''' 
获取对齐图像帧与相机参数
'''
 
 
def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐
 
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧
 
    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
 
    #### 将images转为numpy arrays ####
    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
 
    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame
 
 
''' 
获取随机点三维坐标
'''
 
def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
    # print ('depth: ',dis)       # 深度单位是m
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate

class test_sub:
    def  __init__(self):
        self.grasp_point_sub = rospy.Subscriber('/grasp_point', Point, self.point_callback)
    def point_callback(self,data):
        point[0] = data.x
        point[1] = data.y
        dis, camera_coordinate = get_3d_camera_coordinate(point, aligned_depth_frame, depth_intrin)
        print('depth: ', dis)  # 深度单位是mm
        print('camera_coordinate: ', camera_coordinate)
     
if __name__ == "__main__":
    color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()  # 获取对齐图像与相机参数
    rospy.init_node("set_joint_py", anonymous=True,log_level=rospy.INFO, disable_signals=True) 
    grasp_node = test_sub()
    rospy.spin()
 
      