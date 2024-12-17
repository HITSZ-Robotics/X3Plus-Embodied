#!/usr/bin/env python
# coding: utf-8
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import image_geometry
import json
import os
import message_filters
import shutil

script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(script_path)
base_path = parent_directory

bridge = CvBridge()

depth_image = None
color_image = None
camera_model = image_geometry.PinholeCameraModel()

depth_scale = 0.001  # 根据您的相机的深度比例调整

def create_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, mode=0o777)

def save_camera_parameters(camera_info):
    camera_model.fromCameraInfo(camera_info)

def color_image_callback(msg):
    global color_image
    try:
        color_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)

def depth_image_callback(msg):
    global depth_image
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
        print(e)

def camera_info_callback(msg):
    save_camera_parameters(msg)

def aligned_callback(color_msg, depth_msg, camera_info_msg):
    global color_image, depth_image
    color_image_callback(color_msg)
    depth_image_callback(depth_msg)
    camera_info_callback(camera_info_msg)

def get_nearest_valid_depth(x, y, search_radius=5):
    global depth_image, depth_scale
    rows, cols = depth_image.shape
    for radius in range(1, search_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    dis = depth_image[ny, nx] * depth_scale
                    if dis > 0:
                        return dis, (nx, ny)
    return None, None

def get_camera_coordinates(x, y):
    global depth_image, camera_model, depth_scale

    if depth_image is None or not camera_model.distortion_coefficients:
        print("Depth image or camera intrinsics not available")
        return None

    dis = depth_image[y, x] * depth_scale
    if dis == 0:
        print("Distance is zero at the provided pixel, searching nearby points")
        dis, (nx, ny) = get_nearest_valid_depth(x, y)
        if dis is None:
            print("No valid depth found nearby")
            return None
        print("Using nearby point with valid depth: ({}, {})".format(nx, ny))
        x, y = nx, ny

    # 将像素坐标转换为相机坐标
    camera_coordinate = camera_model.projectPixelTo3dRay((x, y))
    camera_coordinate = [coord * dis for coord in camera_coordinate]
    return camera_coordinate

def pixel_to_camera_coordinates(x, y):
    camera_coordinate = get_camera_coordinates(x, y)
    return camera_coordinate

if __name__ == '__main__':
    rospy.init_node('realsense_listener', anonymous=True)

    color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
    info_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)

    ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, info_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(aligned_callback)

    while not rospy.is_shutdown():
        # 获取用户输入的像素点
        try:
            x = int(input("Enter x coordinate: "))
            y = int(input("Enter y coordinate: "))

            camera_coordinate = pixel_to_camera_coordinates(x, y)
            if camera_coordinate:
                print("Camera coordinates at pixel ({}, {}): {}".format(x, y, camera_coordinate))
        except Exception as e:
            print("Error: {}".format(e))

