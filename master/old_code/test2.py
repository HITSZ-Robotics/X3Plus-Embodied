import cv2
import numpy as np
import scipy.optimize
import scipy.spatial.transform
import matplotlib.pyplot as plt

# 加载图像
image_path = 'image1.png'  # 替换为您图像的路径
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
    [0.319 ,-0.160 , 0.087],
    [0.282, -0.212 ,0.031],
    [0.294 , 0.083 , -0.076],
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
    [552, 224],
    [597, 308],
    [265, 331],
    [155, 349]
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

# 将旋转向量转换为四元数
def rotation_vector_to_quaternion(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    return scipy.spatial.transform.Rotation.from_matrix(rotation_matrix).as_quat()

# 将四元数转换为旋转向量
def quaternion_to_rotation_vector(quat):
    rotation_matrix = scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    return rvec

# 定义目标函数，使得Z轴尽可能垂直于地面
def objective_function(quat):
    rotation_matrix = scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()
    z_axis = rotation_matrix[:, 2]
    return np.linalg.norm(z_axis[:2])  # 最小化Z轴的X和Y分量

# 当前估计的四元数
q_current = rotation_vector_to_quaternion(rotation_vector)

# 使用L-BFGS-B进行优化
result = scipy.optimize.minimize(objective_function, q_current, method='L-BFGS-B', bounds=[(-1, 1), (-1, 1), (-1, 1), (-1, 1)])
q_optimized = result.x

# 优化后的旋转向量
rotation_vector_optimized = quaternion_to_rotation_vector(q_optimized)

# 将3D坐标轴的点定义出来，用于投影到图像平面
axis_length = 0.1
axis_points = np.array([
    [0, 0, 0],  # 原点
    [axis_length, 0, 0],  # X轴
    [0, axis_length, 0],  # Y轴
    [0, 0, axis_length]   # Z轴
], dtype=np.float32)

# 将3D点投影到图像平面
image_points, _ = cv2.projectPoints(axis_points, rotation_vector_optimized, translation_vector, camera_matrix, dist_coeffs)

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

