import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import subprocess
import shutil
from camera2arm import camera_ins_ans


script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(script_path)
base_path=parent_directory

click_number=0



def create_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, mode=0o777)

def run_ros_command():
    try:
        result = subprocess.run(['rosrun', 'arm_moveit_demo', 'cb_tf_echo.py'], check=True)
        if result.returncode == 0:
            print("Command executed successfully.")
        else:
            print(f"Command failed with return code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")


def get_aligned_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    return aligned_depth_frame, color_frame

def save_camera_parameters(color_frame, aligned_depth_frame):
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    camera_parameters = {
        'fx': intr.fx, 'fy': intr.fy,
        'ppx': intr.ppx, 'ppy': intr.ppy,
        'height': intr.height, 'width': intr.width,
        'depth_scale': depth_scale
    }

    with open(base_path+'/data/camera_parameters.json', 'w') as fp:
        json.dump(camera_parameters, fp)

def save_click_data(x, y, dis, camera_coordinate):
    data = {
        'x': x,
        'y': y,
        'distance': dis,
        'camera_coordinate': camera_coordinate
    }

    with open(base_path+'/data/click_data.json', 'a') as fp:
        json.dump(data, fp)
        fp.write('\n')  # 每次保存数据后换行

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    img, aligned_depth_frame, intr, depth_scale = param
    global click_number 

    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 5, (0, 0, 255), thickness=-1)  # 添加红色标记，半径为 5
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)

        dis = aligned_depth_frame.get_distance(x, y)
        print("distance:", dis)

        # Convert pixel coordinates to list of floats
        pixel = [float(x), float(y)]

        # Perform deprojection
        try:
            if dis!=0 : 
              click_number+=1
              camera_coordinate = rs.rs2_deproject_pixel_to_point(intr, pixel, dis)
              print("camera_coordinate =", camera_coordinate)
              print("x, y =", x, y)
  
              # Save click data
              save_click_data(x, y, dis, camera_coordinate)
              run_ros_command()
              # Save RGB image with marker
              img_with_marker = img.copy()
              cv2.circle(img_with_marker, (x, y), 5, (0, 0, 255), thickness=-1)  # 添加红色标记，半径为 5
              cv2.imwrite( base_path+f'/picture/rgb_with_marker_at_click_{x}_{y}.jpg', img_with_marker)
              
              # Convert aligned depth frame to numpy array
              depth_image = np.asanyarray(aligned_depth_frame.get_data())
  
              # Save depth image
              cv2.imwrite(base_path+f'/picture/depth_at_click_{x}_{y}.png', depth_image)

              #put the ros_key on
              key_data={'key':1}
              with open(parent_directory+f'/data/key.json', 'w') as f :
                 json.dump(key_data,f)

              print('ckick {} is sucessful'.format(click_number))
            else:
                print("distance=0,click again")
        except Exception as e:
            print(f"Error in deprojection: {e}")

#mkdir data picture
create_directory(parent_directory+f"/data")
create_directory(parent_directory+f"/picture")            

#init key
key_data={'key':1}
with open(parent_directory+f'/data/key.json', 'w') as f :
    json.dump(key_data,f)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

printed_once = False

try:
    while True:

        ##juge ros capture the 
        #with open(parent_directory+f'/data/key.json', 'r') as f :
        #   key_data=json.load(f)
        #if key_data["key"]==1:
        #     
        #     aligned_depth_frame, color_frame = get_aligned_frames(pipeline, align)
        #     rgb = np.asanyarray(color_frame.get_data())
        #     cv2.namedWindow("image")
        #     cv2.imshow("image", rgb)
#
        #     if printed_once == False:
        #       print("wait\n")
        #       printed_once == True
        #       continue
        #     else:                 
        #      continue
        #
        #printed_once= False 
        #print("continue clicking")
        aligned_depth_frame, color_frame = get_aligned_frames(pipeline, align)
        save_camera_parameters(color_frame, aligned_depth_frame)
        
        
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        
        rgb = np.asanyarray(color_frame.get_data())
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, (rgb, aligned_depth_frame, intr, depth_scale))
        cv2.imshow("image", rgb)

        key = cv2.waitKey(1)
        with open(parent_directory+f'/data/key.json', 'r') as f :
           key_data=json.load(f)
        if (key & 0xFF == ord('q') or key == 27 ) :
            if click_number <=4 :
                print("click more")
            else :
                  camera_ins_ans()
                  break
 

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
