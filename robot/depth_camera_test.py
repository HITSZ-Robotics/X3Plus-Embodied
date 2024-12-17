# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************

##to import pyorbbecsdk 
import os
import sys
new_path ="/home/jetson/pyorbbecsdk/install/lib/"
sys.path.append(new_path)
str2 ="sudo bash /home/jetson/pyorbbecsdk/scripts/install_udev_rules.sh"
os.system(str2)
str3 ="sudo udevadm control --reload-rules && sudo udevadm trigger"
os.system(str3)
##
base_path=r"/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/data/"
##!!set_filename 无后缀
from pyorbbecsdk import *
import cv2
import numpy as np
from utils import frame_to_bgr_image
from upload_download_cilent import *


def save_depth_frame(frame: DepthFrame,set_filename):
    if frame is None:
        return
    width = frame.get_width()
    height = frame.get_height()
    timestamp = frame.get_timestamp()
    scale = frame.get_depth_scale()
    data = np.frombuffer(frame.get_data(), dtype=np.uint16)
    data = data.reshape((height, width))
    data = data.astype(np.float32) * scale
    data = data.astype(np.uint16)
    ##save_image_dir = os.path.join(base_path, "depth_images")
    save_image_dir =base_path
    #if not os.path.exists(save_image_dir):
    #    os.mkdir(save_image_dir)
    raw_filename = save_image_dir + "/depth_{}.raw".format(set_filename)
    data.tofile(raw_filename)


def save_color_frame(frame: ColorFrame, set_filename):
    if frame is None:
        return
    width = frame.get_width()
    height = frame.get_height()
    timestamp = frame.get_timestamp()
    save_image_dir = base_path
    #if not os.path.exists(save_image_dir):
    #    os.mkdir(save_image_dir)
    filename = save_image_dir + "/color_{}.png".format(set_filename)
    image = frame_to_bgr_image(frame)
    if image is None:
        print("failed to convert frame to image")
        return
    cv2.imwrite(filename, image)


def camera_action(set_filename):
    pipeline = Pipeline()
    config = Config()
    saved_color_cnt: int = 0
    saved_depth_cnt: int = 0
    has_color_sensor = False
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is not None:
            color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            has_color_sensor = True
    except OBError as e:
        print(e)
    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    if depth_profile_list is not None:
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
    pipeline.start(config)
    
    #try:
    #    frames = pipeline.wait_for_frames(100)
    #    color_frame = frames.get_color_frame()
    #    if color_frame is not None :
    #        save_color_frame(color_frame, set_filename)
    #        print("saved color frame")
    #    else:
    #        print("color_frame has problem")
    #        
    #    depth_frame = frames.get_depth_frame()
    #    if depth_frame is not None :
    #        save_depth_frame(depth_frame, set_filename)
    #        print("save depth frame")
    #    else:
    #        print("depth_frame has problem")
    #except Exception as e:
    # print(e)
    while True:
        try:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            if has_color_sensor:
                if saved_color_cnt >= 5 and saved_depth_cnt >= 5:
                    break
            elif saved_depth_cnt >= 5:
                break
            color_frame = frames.get_color_frame()
            if color_frame is not None and saved_color_cnt < 5:
                save_color_frame(color_frame, set_filename)
                saved_color_cnt += 1
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None and saved_depth_cnt < 5:
                save_depth_frame(depth_frame, set_filename)
                saved_depth_cnt += 1
        except KeyboardInterrupt:
            break


def get_depth_camera(set_filename):
    camera_action(set_filename)
    color_filename="color_"+set_filename+".png"
    upload(color_filename)
    depth_filename="depth_"+set_filename+".raw"
    upload(depth_filename)

#get_depth_camera("test30")
#camera_action('clib8')
