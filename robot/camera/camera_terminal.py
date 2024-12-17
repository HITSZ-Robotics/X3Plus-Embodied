import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_directory,"calibration_tool"))
sys.path.append(os.path.join(os.path.dirname(__file__), 'calibration_tool'))
import camera_data  # 导入共享信号模块
from camera2arm import camera2arm, camera_ins_ans
# 初始化 RealSense 相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
p=camera_ins_ans()

#init
camera_data.set_capture_signal(False)
camera_data.set_process_signal(False)



def capture_and_save_image():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        print("Failed to capture frames")
        return

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    
    camera_data.set_shared_depth_image(depth_image)
    camera_data.set_shared_color_image(color_image)

def get_nearest_valid_depth(depth_image,depth_scale,x, y, search_radius=5):
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

def process_2dto3d():

    intr = color_frame.profile.as_video_stream_profile().intrinsics
    depth_scale= profile.get_device().first_depth_sensor().get_depth_scale()
    x,y=camera_data.get_shared_point()
    depth_image=camera_data.get_shared_depth_image()
    rows, cols = depth_image.shape

    dis = depth_image[y, x] * depth_scale
    print(dis)
    if dis == 0:
       print("Distance is zero at the provided pixel, searching nearby points")
       dis, (nx, ny) = get_nearest_valid_depth(depth_image,depth_scale,x, y)
       if dis is None:
           print("No valid depth found nearby")
           return None
       print(f"Using nearby point with valid depth: ({nx}, {ny})")
       x, y = nx, ny

    pixel = [float(x), float(y)]
    camera_coordinate = rs.rs2_deproject_pixel_to_point(intr, pixel, dis)
    x_c=camera_coordinate[0]
    y_c=camera_coordinate[1]
    z_c=camera_coordinate[2]
    x_w,y_w,z_w=camera2arm(x_c,y_c,z_c,p)
    print(x_w,y_w,z_w)
    camera_data.set_shared_3dpoint(x_w, y_w,z_w)


try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # 显示彩色图像
        cv2.imshow('Color Frame', color_image)

        key = cv2.waitKey(1)
        if(camera_data.get_capture_signal==True):
           print("01")       
        if key & 0xFF == ord('q') or key == 27:
            break
        elif key & 0xFF == ord('c'):  # 按 'c' 键请求拍照
            camera_data.capture_signal = True

        if camera_data.get_capture_signal():
            capture_and_save_image()
            print("get_capture_signal")
            camera_data.set_capture_signal( False)

        if camera_data.get_process_signal():
            process_2dto3d()
            camera_data.set_process_signal( False)
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
