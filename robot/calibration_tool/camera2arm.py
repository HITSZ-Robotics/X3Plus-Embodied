#!/usr/bin/env python
# encoding: utf-8
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from read_json import get_3dpoint,get_imagepoint

base_path=r"/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/"

def camera_ins_ans():
    ###############
        #3D
    # 3D object points
    object_points=get_3dpoint()

    
    # 对应的2D图像点（单位：像素）
    image_points=get_imagepoint()

    
    # 相机内参矩阵（已知）
    with open(base_path+"calibration_tool/data/camera_parameters.json", 'r') as f:
     data=json.load(f)
    camera_matrix = np.array([
        [data['fx'], 0, data["ppx"]],
        [0, data['fy'],  data["ppy"]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 畸变系数（假设无畸变）
    dist_coeffs = np.zeros(5)
    
    # 使用RANSAC算法进行PnP求解
    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs)
      
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    print("Success:", success)
    #print("Rotation Vector:\n", rotation_vector)
    #print("Translation Vector:\n", translation_vector)
    #print("Rotation Matrix:\n", rotation_matrix)
    rotation_matrix,a=cv2.Rodrigues(rotation_vector)
    print(translation_vector)
    print(rotation_matrix)
    R=rotation_matrix
    tvec=translation_vector
    T = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    T[0, 0] = R[0, 0]
    T[0, 1] = R[0, 1]
    T[0, 2] = R[0, 2]
    T[1, 0] = R[1, 0]
    T[1, 1] = R[1, 1]
    T[1, 2] = R[1, 2]
    T[2, 0] = R[2, 0]
    T[2, 1] = R[2, 1]
    T[2, 2] = R[2, 2]
    T[0, 3] = tvec[0, 0]
    T[1, 3] = tvec[1, 0]
    T[2, 3] = tvec[2, 0]
    T[3, 0] = 0.0
    T[3, 1] = 0.0
    T[3, 2] = 0.0
    T[3, 3] = 1.0

    p=np.linalg.inv(T)
   # transfrom_mastrix={'masrix':p,'Rotation Vector':rotation_vector,'Translation Vector':rotation_vector}
   # with open(base_path+"calibration_tool/data/transfrom_mastrix.json", 'w') as f :
    #  json.dump(transfrom_mastrix,f)
    print("sucess!!  p={}".format(p))
    pa = np.array(p)
    paa = pa.tolist()
    with open('/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/calibration_tool/data/p_matrix.json', 'w') as file:
        json.dump(paa, file)
    return p
    #worlk_place=np.array([[x],[y],[z],[1]],dtype=np.float32)
    #camera_place=np.dot(T,worlk_place)
    #return camera_place
#x,y,z=-0.12876421213150024, -0.15158987045288086, 0.47600001096725464
#print(x,y,z)



## how to use it 

""
#camera_place=np.array([[x],[y],[z],[1]],dtype=np.float32)
#world_place=np.dot(p,camera_place)
""
def camera2arm (x,y,z,mastrix):
   
   camera_place=np.array([[x],[y],[z],[1]],dtype=np.float32)
   world_place=np.dot(mastrix,camera_place)
   print("world=={}".format(world_place))

 #  print("world2=={}".format(world_place))
   x_w=world_place[0][0]
   y_w=world_place[1][0]
   z_w=world_place[2][0]
   return x_w,y_w,z_w
