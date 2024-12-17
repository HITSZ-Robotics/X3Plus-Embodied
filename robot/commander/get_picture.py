import sys
import os
import time
import cv2
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_dir)

sys.path.append(parent_directory)
sys.path.append(os.path.join(current_dir,"camera"))
from camera.camera_data import set_capture_signal,get_shared_color_image
from upload_download_cilent import upload
data_path=r"/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/data/" 

def get_picture(file_name):
    os.system("rosrun arm_moveit_demo camera_initjoint.py")
    set_capture_signal(True)
    time.sleep(1)
    image=get_shared_color_image()
    print("get color picture")
    cv2.imwrite(data_path+file_name,image)
    upload(file_name)

