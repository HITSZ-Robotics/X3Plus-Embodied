import open3d as o3d
import numpy as np
import cv2

def depth_to_point_cloud(depth_image, color_image, intrinsics):
    height, width = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 生成网格
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xv, yv = np.meshgrid(x, y)

    # 深度值转换为点云
    z = depth_image / 1000.0  # 假设深度值单位为毫米
    x = (xv - cx) * z / fx
    y = (yv - cy) * z / fy
    xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # 获取颜色信息
    colors = color_image.reshape(-1, 3) / 255.0  # 将颜色值归一化到0-1

    # 过滤掉无效点
    valid = z.flatten() > 0
    valid_points = xyz[valid]
    valid_colors = colors[valid]

    return valid_points, valid_colors

def compute_rotation_matrix_from_vectors(src, dst):
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)
    axis = np.cross(src, dst)
    angle = np.arccos(np.dot(src, dst))
    if np.linalg.norm(axis) < 1e-6:  # Prevent division by zero
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def rotation_matrix_to_vector(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    r = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(theta))
    return theta * r

def main():
    # 读取深度图像和颜色图像
    width = 640
    height = 480

    # 读取.raw文件为Numpy数组，指定dtype=np.uint16
    depth_image = np.fromfile('depth_image6.raw', dtype=np.uint16).reshape((height, width))

    # 假设颜色图像与深度图像的尺寸相同
    color_image = cv2.imread('image6.png')

    # 相机内参矩阵（根据实际相机参数调整）
    intrinsics = np.array([[605.68896484375, 0, 329.8467712402344],
                           [0, 605.5736694335930, 254.8546905517578],
                           [0, 0, 1]])

    # 将深度图和颜色图转换为点云
    points, colors = depth_to_point_cloud(depth_image, color_image, intrinsics)

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 平面拟合
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)

    [a, b, c, d] = plane_model
    print(f"平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    # 计算法向量
    normal_vector = np.array([a, b, c])
    print(f"法向量: {normal_vector}")

    # 确定平面上的一个点
    if c != 0:
        plane_origin = np.array([0, 0, -d / c])
    else:
        plane_origin = np.array([0, -d / b, 0])

    print(f"平面坐标系原点（相机坐标系下）: {plane_origin}")

    # 生成平面坐标系的X轴和Y轴
    manual_x_axis = np.array([1, 0, -a/c]) if c != 0 else np.array([0, 1, -b/c])
    manual_x_axis = manual_x_axis / np.linalg.norm(manual_x_axis)
    y_axis = np.cross(normal_vector, manual_x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # 定义平面坐标系的基向量
    plane_axes = np.vstack([manual_x_axis, y_axis, normal_vector]).T

    # 定义相机坐标系的基向量
    camera_axes = np.eye(3)

    # 计算旋转矩阵
    R = np.linalg.inv(camera_axes).dot(plane_axes)
    print("相机坐标系到平面坐标系的旋转矩阵:")
    print(R)

    # 计算旋转向量
    rotation_vector = rotation_matrix_to_vector(R)
    print("旋转向量:")
    print(rotation_vector)

    # 可视化点云和坐标轴
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    pcd.paint_uniform_color([1, 0, 0])  # 将点云涂成红色
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0, 1, 0])  # 将内点涂成绿色
    o3d.visualization.draw_geometries([pcd, axis, inlier_cloud])

    # 保存点云（可选）
    o3d.io.write_point_cloud("point_cloud.ply", pcd)

if __name__ == "__main__":
    main()
