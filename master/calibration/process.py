import cv2
import numpy as np
import pyrealsense2 as rs
import os

# 配置
pipe1 = rs.pipeline()
cfg1 = rs.config()
cfg1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

i = 0
profile1 = pipe1.start(cfg1)

while True:
    # 获取图片帧
    frameset = pipe1.wait_for_frames()
    color_frame = frameset.get_color_frame()
    color_img = np.asanyarray(color_frame.get_data())

    # 更改通道的顺序为RGB
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_img)
    k = cv2.waitKey(1)
    # Esc退出，
    if k == 27:
        cv2.destroyAllWindows()
        break
    # 输入空格保存图片
    elif k == ord(' '):
        i = i + 1
        cv2.imwrite(os.path.join("D:\\Realsense\\pic_capture", str(i) + '.jpg'), color_img)
        print("Frames{} Captured".format(i))

pipe1.stop()
