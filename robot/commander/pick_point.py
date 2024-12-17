import sys
import os
import time
import cv2
import json
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_dir)

sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory,"camera"))
from camera.camera_data import set_process_signal,set_shared_point,get_shared_3dpoint
from upload_download_cilent import download
data_path=r"/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/data/" 

def pick_point(file_name):
    download(file_name)
    with open(data_path+file_name, 'r') as f :
           image_point=json.load(f)
    x=image_point["x"]
    y=image_point["y"]
    set_shared_point(x,y)
    set_process_signal(True)
    time.sleep(1)
    x,y,z=get_shared_3dpoint()
    print(x,y,z)
    with open(data_path+'pick_3d_point', 'w') as f :
           json.dump([x,y,z],f)
    os.system("rosrun arm_moveit_demo 02_set_pos_plan_socket.py --x=%f --y=%f --z=%f" %(x,y-0.02,z+0.070))
#os.system("rosrun arm_moveit_demo 02_set_pos_plan_socket.py --x=%f --y=%f --z=%f" %(0.218,0.081,0.007))

#pick_point("pick_point.json")


