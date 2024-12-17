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

    # 计算法向量与Z轴的夹角（俯仰角）
    normal_vector = np.array([-a, -b, -c])
    z_axis = np.array([0, 0, 1])
    
    cosine_angle = np.dot(normal_vector, z_axis) / (np.linalg.norm(normal_vector) * np.linalg.norm(z_axis))
    print(cosine_angle)
    angle = np.arccos(cosine_angle)
    pitch_angle = np.degrees(angle)

    print(f"俯仰角: {pitch_angle:.2f}度")

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
