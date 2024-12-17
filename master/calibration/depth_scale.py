#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# Start streaming
profile = pipeline.start(config)
while True:
    frames = pipeline.wait_for_frames()
    depth_frames = frames.get_depth_frame()
    depth_image = np.asarray(depth_frames.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    distance = depth_frames.get_distance(100, 200)
    print(distance)
    cv2.imshow("depth image:", depth_colormap)
    key = cv2.waitKey(1)
    if key == 27:
        break
pipeline.stop()


cfg = pipeline.start(config)
device1 = cfg.get_device()
for s in device1.sensors:
    print(s.get_info(rs.camera_info.name))

