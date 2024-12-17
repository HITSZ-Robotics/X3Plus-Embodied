import os
from tqdm import tqdm
import cv2
import numpy as np

import matplotlib.pyplot as plt
#from pnp  import *


def depth_prepare(image):
  sigma = 1.5
  kernel_size = int(sigma*3)*2+1
  kernel = cv2.getGaussianKernel(kernel_size,sigma)
  
  image_smooth  = cv2.filter2D(image, cv2.CV_16UC1, kernel)
  return image_smooth

def undistort(img):

 K = [[605.68896484375, 0, 329.8467712402344],
      [0, 605.5736694335930, 254.8546905517578],
      [0, 0, 1]]
 D = [0, 0, 0, 0, 0]
 MK = np.array(K, dtype=np.float64)
 MD = np.array(D, dtype=np.float64)
 width = 640  # 图像宽度
 height = 480  # 图像高度
 mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, (width, height), 5)
 return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

#def distort_point(xCorrected,yCorrected):
#   
#    fx=550.996861583249
#    fy=553.080904240857
#    ux=319.764490195525
#    uy=239.089058158080
#    k1=0.1058
#    k2=-0.1554
#    p1=0
#    p2=0
#    for i in range(0,9):
#      r2=x

def camera_coordinate(i,j,filename):
   # Depth camera parameters:
  FX_DEPTH = 605.68896484375
  FY_DEPTH = 605.5736694335930
  CX_DEPTH = 329.8467712402344
  CY_DEPTH = 254.8546905517578
  path= "depth_"+filename+".raw"
 
## 使用numpy从RAW文件读取数据


# 假设你的.raw文件是8位的，并且尺寸是宽度x高度
  width = 640
  height = 480

# 读取.raw文件为Numpy数组，指定dtype=np.uint8
  depth_data = np.fromfile('depth_image6.raw', dtype=np.uint16).reshape((height, width))

#
## 根据图像尺寸重塑数组


  #image=array_data.astype("uint8")
  #cv2.imshow("window",image)
  #cv2.waitKey(0)
   
#rgb=img.postprocess()

      #!!!!!!!!!!
  #array_data=img.raw_image_visible##//读取RAW图像数据信息
  #print(array_data.shape)
  #gray = cv2.cvtColor(ar, cv2.COLOR_BGR2GRAY)
#width = array_data.shape[0]
#height = array_data.shape[1]
 ##change the channel
  #print(bit_depth.dtype)
  #print(f"Max value: {np.max(bit_depth)}")
#print(bit_depth.shape)####//打印长和宽
#print(bit_depth[3,4])##//打印矩阵信息
  
   #width = 640  # 图像宽度 j
 #height = 480  # 图像高度 i
  z = depth_data[j,i]
  
  x = (j - CX_DEPTH) * z / FX_DEPTH
  y = (i - CY_DEPTH) * z / FY_DEPTH
  return 0.001*x,0.001*y,0.001*z


   
#def covert(point=[[0.0, 0.0]], z=1):
#    point = np.array(point, dtype=np.float64)
#    pts_uv = cv2.undistortPoints(point, MK, MD) * z
#    return pts_uv[0][0]

#print(covert([0,0],1))

#p1=camera_coordinate(385,247,'object1')
#p2=camera_coordinate(374,218,'object1')
#print(p1)
#print(p2)
#p5=camera_coordinate(8,451,"clib1")
#print(p5)

def camera2arm(x,y,z):
    ###############
        #3D
    # 3D object points
    object_points = np.array([
        [0.415 , 0.011 , 0.142],
        [0.252 ,-0.277 , 0.142],
        [0.296 , -0.228,  0.240],
        [0.310 , -0.243,  0.129],
        [0.414 , 0 , 0.103],
        [0.347, -0.194  ,0.180],
       
    ], dtype=np.float32)
    
    # 对应的2D图像点（单位：像素）
    image_points = np.array([
        [187, 64],
        [609, 202],
        [595, 40],
        [531, 157],
        [213, 107],
        [481, 77],
      
    ], dtype=np.float32)
    
    # 相机内参矩阵（已知）
    camera_matrix = np.array([
        [605.68896484375, 0, 329.8467712402344],
        [0, 605.5736694335938,  254.8546905517578],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 畸变系数（假设无畸变）
    dist_coeffs = np.zeros(5)
    
    # 使用RANSAC算法进行PnP求解
    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs)
      
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    print("Success:", success)
    print("Rotation Vector:\n", rotation_vector)
    print("Translation Vector:\n", translation_vector)
    print("Rotation Matrix:\n", rotation_matrix)
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
    ##########
    camera_place=np.array([[x],[y],[z],[1]],dtype=np.float32)
    world_place=np.dot(p,camera_place)
    return world_place
    #worlk_place=np.array([[x],[y],[z],[1]],dtype=np.float32)
    #camera_place=np.dot(T,worlk_place)
    #return camera_place
x,y,z=-0.12876421213150024, -0.15158987045288086, 0.47600001096725464
#print(x,y,z)

p=camera2arm(x,y,z)
print(p)