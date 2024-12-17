import cv2
import numpy as np
#3D
object_points = np.array([[154, 222, -26.5], [312, 19, -26.5], [267, 123, -26.5], [438, 6, -26.5],[383,185,-26.5]], dtype=np.float32)
#2D
image_points = np.array([[85.0,451.5], [322, 310], [466.5, 341.5], [331.0, 209.0],[497.0,242.5]], dtype=np.float32)
#camera inside
camera_matrix = np.array([[605.68896484375, 0, 329.8467712402344],
       [0, 605.5736694335930, 254.8546905517578],
      [0, 0, 1]],dtype=np.float32)

#
#rotation_matrix=np.array([[-0.98773901,  0.11561566,  0.10490317],
       [-0.11832788, -0.11612288, -0.98616124],
       [-0.10183402, -0.98648289,  0.12837965]],dtype=np.float32)

#rotation_matrix_inv=np.array([[-0.98773897 ,-0.11832788, -0.10183402],
 [ 0.11561565 ,-0.11612286, -0.98648286],
 [ 0.10490317 ,-0.98616123 , 0.12837964]],dtype=np.float32)

#translation_matrix=np.array( [[255.103502  ],
 [ 21.27252889],
 [159.98015119]],dtype=np.float32)
# 畸变系数
dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)
#rotation_vector,translation_vector= cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
sucess,rotation_vector,translation_vector= cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
p=np.linalg.inv(rotation_matrix)
print(p)
print("Rotation Vector:\n", rotation_vector)
print("Translation Vector:\n", translation_vector)

