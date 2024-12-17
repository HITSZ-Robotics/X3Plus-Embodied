import open3d as o3d
import numpy as np
import cv2

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
  depth_data = np.fromfile(path, dtype=np.uint16).reshape((height, width))

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

def rotation_matrix_to_vector(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    r = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(theta))
    return theta * r

def depth_to_point_cloud(depth_image, color_image,intrinsics):
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

def create_arrow(origin, vector, color=[1, 0, 0], length=1.0, radius=0.02):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius, cone_radius=radius * 1.5,
        cylinder_height=length * 0.8, cone_height=length * 0.2)
    arrow.paint_uniform_color(color)

    rotation_matrix = calculate_rotation_matrix_from_vector(vector)
    arrow.rotate(rotation_matrix, center=(0, 0, 0))
    arrow.translate(origin)

    return arrow

def calculate_rotation_matrix_from_vector(vector):
    vector = vector / np.linalg.norm(vector)
    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, vector)
    angle = np.arccos(np.dot(z_axis, vector))

    if np.isclose(angle, 0):
        return np.eye(3)
    if np.isclose(angle, np.pi):
        return -np.eye(3)

    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    rotation_matrix = (np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K))
    return rotation_matrix

def main():
    # 读取深度图像
    width = 640
    height = 480

    # 读取.raw文件为Numpy数组，指定dtype=np.uint16
    depth_image = np.fromfile('depth_image1.raw', dtype=np.uint16).reshape((height, width))
    color_image =cv2.imread("image1.png")
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
    normal_vector = np.array([-a, -b, -c])
    print(f"法向量: {normal_vector}")

    # 确定平面上的一个点
    origin = [0, 0, -d/c] if c != 0 else [0, 0, 0]

    # 生成平面坐标系的X轴和Y轴
    manual_x_axis = np.array([1, 0,-a/c]) if c != 0 else np.array([0, 1, -b/c])
    manual_x_axis = manual_x_axis / np.linalg.norm(manual_x_axis)
    y_axis = np.cross(normal_vector, manual_x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # 创建平面坐标系的箭头
    length = 0.5  # 箭头的长度
    normal_arrow = create_arrow(origin, normal_vector, color=[0, 0, 1], length=length)  # blue色
    x_axis_arrow = create_arrow(origin, manual_x_axis, color=[1, 0,0 ], length=length)  # red色
    y_axis_arrow = create_arrow(origin, y_axis, color=[0, 1, 0], length=length)  # green色

    # 创建相机坐标系
    camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # 可视化点云、相机坐标系和平面坐标系
    #pcd.paint_uniform_color([1, 0, 0])  # 将点云涂成红色
    #inlier_cloud = pcd.select_by_index(inliers)
    #inlier_cloud.paint_uniform_color([0, 1, 0])  # 将内点涂成绿色
    plane_axes = np.vstack([manual_x_axis, y_axis, normal_vector]).T
    camera_axes = np.eye(3)
    R = np.linalg.inv(camera_axes).dot(plane_axes)
    print("相机坐标系到平面坐标系的旋转矩阵:")
    print(R)
    rotation_vector = rotation_matrix_to_vector(R)
    print("旋转向量:")
    print(rotation_vector)

    o3d.visualization.draw_geometries([pcd, camera_axis,  normal_arrow, x_axis_arrow, y_axis_arrow])

    # 保存点云（可选）
    o3d.io.write_point_cloud("point_cloud.ply", pcd)
    

    return R

    

if __name__ == "__main__":
    #rotation_vector=main()
    #rotation_matrix,a=cv2.Rodrigues(rotation_vector)
    translation_vector=np.array([[0.0],[0],[-0.39]])
    print(translation_vector)
    #print(rotation_matrix)
    R=main()
    #R=rotation_matrix
    tvec=translation_vector
   

    #R=np.linalg.inv(R)
    x,y,z=camera_coordinate(327, 265,'image1')
    print (x,y,z)
    camera_place=np.array([[x],[y],[z]],dtype=np.float32)
    #world_place=camera_place+tvec
    world_place=np.dot(R,camera_place)-tvec
    print(world_place)
    
   


