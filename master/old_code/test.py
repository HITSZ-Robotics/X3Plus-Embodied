#import cv2
#import numpy as np
#import itertools
#import matplotlib.pyplot as plt
#
## Load the image
#image_path = 'image5.png'  # 替换为您图像的路径
#image = cv2.imread(image_path)
#image_height, image_width, _ = image.shape
#
## Calculate the center of the image
#image_center = np.array([image_width / 2, image_height / 2])
#
## 重新定义对象点和图像点
#object_points_full = np.array([
#    [0.294 , 0.083 , 0.067],
#    [0.266 ,-0.136 , 0.066],
#    [0.272 , 0.180,  0.069],
#    [0.298 , 0.247,  0.122],
#    [0.349 , -0.026 , 0.119],
#    [0.261 , -0.194  ,0.113],
#    [0.290, 0.205,  0.087],
#    [0.319 ,-0.160 , 0.087],
#    [0.282, -0.212 ,0.031],
#], dtype=np.float32)
#
#image_points_full = np.array([
#    [258, 249],
#    [527, 291],
#    [120, 267],
#    [16, 192],
#    [385, 167],
#    [634, 262],
#    [83, 232],
#    [552, 224],
#    [597, 308],
#  
#], dtype=np.float32)
#
## Camera matrix (intrinsic parameters)
#camera_matrix = np.array([
#    [605.68896484375, 0, 329.8467712402344],
#    [0, 605.5736694335930, 254.8546905517578],
#    [0, 0, 1]
#], dtype=np.float64)
#
## Distortion coefficients (assuming no distortion)
#dist_coeffs = np.zeros(5)
#
## Define the function to perform PnP and project the coordinates
#def pnp_and_project(points_indices):
#    # Select the specific points
#    obj_points = object_points_full[np.array(points_indices)]
#    img_points = image_points_full[np.array(points_indices)]
#
#    # Solve for the rotation and translation vectors using PnP
#    success, rvec, tvec = cv2.solvePnP(
#        obj_points, img_points, camera_matrix, dist_coeffs
#    )
#
#    if not success:
#        print("PnP solution failed.")
#        return image
#
#    # Project the 3D points to the image plane
#    image_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
#
#    # Translate image points to center the origin
#    image_points += image_center - image_points[0]
#
#    # Convert image points to integer tuples
#    image_points = np.int32(image_points).reshape(-1, 2)
#
#    # Draw the axes on the image
#    img_copy = image.copy()
#    origin = tuple(image_points[0])
#    img_copy = cv2.line(img_copy, origin, tuple(image_points[1]), (0, 0, 255), 5)  # X axis in red
#    img_copy = cv2.line(img_copy, origin, tuple(image_points[2]), (0, 255, 0), 5)  # Y axis in green
#    img_copy = cv2.line(img_copy, origin, tuple(image_points[3]), (255, 0, 0), 5)  # Z axis in blue
#
#    # Return the modified image
#    return img_copy
#
## 使用一个示例组合（6 个点）
#points_indices = (0,1,2,3,7,6)  # 示例组合
#img = pnp_and_project(points_indices)
#title = f'Using Points {points_indices[0]+1}, {points_indices[1]+1}, {points_indices[2]+1}, {points_indices[3]+1}, {points_indices[4]+1}, {points_indices[5]+1}'
#
## Display the result
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.axis('off')  # Hide axes
#plt.title(title)
#plt.show()
#import cv2
#import numpy as np
#import matplotlib.pyplot as plt
#
## 加载图像
#image_path = 'image2.png'  # 替换为您图像的路径
#image = cv2.imread(image_path)
#image_height, image_width, _ = image.shape
#
## 计算图像中心
#image_center = np.array([image_width / 2, image_height / 2])
#
## 定义表示 X、Y、Z 轴的3D点
#axis_length = 0.1  # 轴长度（单位：米）
#object_points = np.array([
#    [0, 0, 0],  # 原点
#    [axis_length, 0, 0],  # X 轴
#    [0, axis_length, 0],  # Y 轴
#    [0, 0, axis_length]   # Z 轴
#], dtype=np.float32)
#
## 相机内参矩阵（已知）
#camera_matrix = np.array([
#    [605.68896484375, 0, 329.8467712402344],
#    [0, 605.5736694335930, 254.8546905517578],
#    [0, 0, 1]
#], dtype=np.float64)
#
## 畸变系数（假设无畸变）
#dist_coeffs = np.zeros(5)
#
## 示例旋转向量和平移向量（假设值）
#rotation_vector = np.array([0.0, 0.0, 0.0], dtype=np.float64)
#translation_vector = np.array([0.0, 0.0, 0.0], dtype=np.float64)
#
## 将3D点投影到图像平面
#image_points, _ = cv2.projectPoints(object_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
#
## 平移图像点以将原点放在图像中心
#image_points += image_center - image_points[0]
#
## 将图像点转换为整数元组
#image_points = np.int32(image_points).reshape(-1, 2)
#
## 在图像上绘制轴线
#origin = tuple(image_points[0])
#image = cv2.line(image, origin, tuple(image_points[1]), (0, 0, 255), 5)  # X 轴（红色）
#image = cv2.line(image, origin, tuple(image_points[2]), (0, 255, 0), 5)  # Y 轴（绿色）
#image = cv2.line(image, origin, tuple(image_points[3]), (255, 0, 0), 5)  # Z 轴（蓝色）
#
## 显示图像
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.axis('off')  # 隐藏坐标轴
#plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image_path = 'image2.png'  # 替换为您图像的路径
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# 计算图像中心
image_center = np.array([image_width / 2, image_height / 2])

# 定义已知的3D点（单位：米）
object_points = np.array([
    [0.294 , 0.083 , 0.067],
    [0.266 ,-0.136 , 0.066],
    [0.272 , 0.180,  0.069],
    [0.298 , 0.247,  0.122],
    [0.349 , -0.026 , 0.119],
    [0.261 , -0.194  ,0.113],
   # [0.290, 0.205,  0.087],
    [0.319 ,-0.160 , 0.087],
    [0.282, -0.212 ,0.031],
    [0.294 , 0.083 , -0.076],
   # [0.266 ,-0.136 , -0.076],
    [0.298 , 0.247,  -0.076]
], dtype=np.float32)

# 对应的2D图像点（单位：像素）
image_points = np.array([
    [258, 249],
    [527, 291],
    [120, 267],
    [16, 192],
    [385, 167],
    [634, 262],
   # [83, 232],
    [552, 224],
    [597, 308],
    [265,331],
    #[483,366],
    [155,349]
], dtype=np.float32)

# 相机内参矩阵（已知）
camera_matrix = np.array([
    [605.68896484375, 0, 329.8467712402344],
    [0, 605.5736694335930, 254.8546905517578],
    [0, 0, 1]
], dtype=np.float64)

# 畸变系数（假设无畸变）
dist_coeffs = np.zeros(5)

# 使用RANSAC算法进行PnP求解
success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs)

if success:
    print("Rotation Vector:\n", rotation_vector)
    print("Translation Vector:\n", translation_vector)
else:
    print("PnP RANSAC solution failed.")

# 将3D坐标轴的点定义出来，用于投影到图像平面
axis_length = 0.1
axis_points = np.array([
    [0, 0, 0],  # 原点
    [axis_length, 0, 0],  # X轴
    [0, axis_length, 0],  # Y轴
    [0, 0, axis_length]   # Z轴
], dtype=np.float32)

# 将3D点投影到图像平面
image_points, _ = cv2.projectPoints(axis_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

# 平移图像点以将原点放在图像中心
image_points += image_center - image_points[0]

# 将图像点转换为整数元组
image_points = np.int32(image_points).reshape(-1, 2)

# 在图像上绘制轴线
origin = tuple(image_points[0])
image = cv2.line(image, origin, tuple(image_points[1]), (0, 0, 255), 5)  # X轴（红色）
image = cv2.line(image, origin, tuple(image_points[2]), (0, 255, 0), 5)  # Y轴（绿色）
image = cv2.line(image, origin, tuple(image_points[3]), (255, 0, 0), 5)  # Z轴（蓝色）

# 显示图像
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # 隐藏坐标轴
plt.show()
