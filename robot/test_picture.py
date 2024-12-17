import cv2 
import threading
from time import sleep
#from dofbot_config import *
from upload_download_cilent import download
base_path=r"/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/data/" #客户端文件位置

def get_picture(file_name):
    global msg
    # 打开摄像头 Open camera
    capture = cv2.VideoCapture("/dev/usb_cam")
    # Be executed in loop when the camera is opened normally 
    # 当摄像头正常打开的情况下循环执行
    while capture.isOpened():
        try:
            _, img = capture.read()
            img = cv2.resize(img, (640, 480))
            #xy=[joint1_slider.value,joint2_slider.value]
            realpath=base_path+file_name
            cv2.imwrite(realpath,img)
            upload(file_name)
            print("ok")
        except KeyboardInterrupt:capture.release()

get_picture('test.jpg')

