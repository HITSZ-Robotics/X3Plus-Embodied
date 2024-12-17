import sys
import os
import time
import cv2
import json
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_dir)

def forward_onestep():
    os.system("rosrun yahboomcar_ctrl forward_mykeyboard.py")


def backward_onestep():
    os.system("rosrun yahboomcar_ctrl backward_mykeyboard.py")

def turnleft():
    os.system("rosrun yahboomcar_ctrl left_mykeyboard.py")

def turnright():
    os.system("rosrun yahboomcar_ctrl right_mykeyboard.py")

