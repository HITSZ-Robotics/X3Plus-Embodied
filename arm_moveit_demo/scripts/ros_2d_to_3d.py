#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import cv2
import rospy
from geometry_msgs.msg import Point
import subprocess
import signal
import json
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

base_path=r"/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/"
with open(base_path+"calibration_tool/data/camera_parameters.json", 'r') as f:
    data=json.load(f)
camera_matrix = np.array([
    [data['fx'], 0, data["ppx"]],
    [0, data['fy'],  data["ppy"]],
    [0, 0, 1]
], dtype=np.float64)
depth_scale = data["depth_scale"]

with open('/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/calibration_tool/data/p_matrix.json', 'r') as file:
    p = json.load(file)
p = np.array(p)
class _2d_to_3d_node:
    def __init__(self):
        self.grasp_point_sub = rospy.Subscriber('/2d_grasp_point', Point, self.point_callback)
        self.align_depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.distance_callback)
        self.pub_3d_point = rospy.Publisher("/3d_grasp_point", Point, queue_size=100)
        self.dis = None
        self.bridge = CvBridge()

    def point_callback(self,data):
        if self.dis is not None:  # Check if a depth image is available
            point = [int(data.x), int(data.y)]
            cv_image = self.bridge.imgmsg_to_cv2(self.dis, desired_encoding="passthrough")
            distance = cv_image[point[1],point[0]]
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            
            # Convert pixel coordinates to normalized camera coordinates
            x_normalized = (point[0] - cx) / fx
            y_normalized = (point[1] - cy) / fy
            
            # Convert depth value to meters
            z = distance * depth_scale
            
            # Scale normalized coordinates by depth to get camera coordinates in meters
            x = x_normalized * z
            y = y_normalized * z
            point = Point()
            point.x,point.y,point.z=self.camera2arm(x,y,z,p)
            print(point)
            self.pub_3d_point.publish(point)
    
    
    def camera2arm (self,x,y,z,mastrix):
        camera_place=np.array([[x],[y],[z],[1]],dtype=np.float32)
        world_place=np.dot(mastrix,camera_place)
        #print("world=={}".format(world_place))

        #  print("world2=={}".format(world_place))
        x_w=world_place[0][0]
        y_w=world_place[1][0]
        z_w=world_place[2][0]
        return x_w,y_w,z_w
    
    def distance_callback(self,data):
        self.dis = data

if __name__ == '__main__':
    # 初始化节点 || Initialize node
    rospy.init_node('point_publisher')
    z=_2d_to_3d_node()
    rospy.spin()        