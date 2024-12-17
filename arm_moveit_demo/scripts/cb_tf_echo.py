#!/usr/bin/env python
# -*- coding: utf-8 -*-

#标定时调用tf，echo base_link与arm_link5之间的坐标变换,分两个文件
# coordinate_listener.py

import rospy
import tf
import json
import os
from tf import TransformListener

class CoordinateListener:
    def __init__(self):
        self.listener = TransformListener()
        self.last_trans = None  # 记录最近一次成功获取的坐标
        self.control_file = '/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/calibration_tool/data/key.json'  # 控制文件

    def get_transform(self):
        try:
            now = rospy.Time()
            self.listener.waitForTransform('/base_link', '/arm_link5', now, rospy.Duration(5.0))
            (trans, _) = self.listener.lookupTransform('/base_link', '/arm_link5', rospy.Time(0))
            return trans
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None

    def check_and_write_to_json(self, trans):
        if not os.path.exists(self.control_file):
            rospy.logwarn("Control file does not exist")
            return

        with open(self.control_file, 'r') as f:
            control_data = json.load(f)

        if control_data.get("key") == 1:
            self.write_to_json(trans)
            control_data={"key":0}
            with open(self.control_file, 'w') as f:
                json.dump(control_data, f)

    def write_to_json(self, trans):
        coordinate_dict = {"x": trans[0], "y": trans[1], "z": trans[2]}
        with open('/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/calibration_tool/data/coordinates.json', 'a') as f:
            json.dump(coordinate_dict, f)
            f.write('\n')  # 换行以分隔不同的记录

    def start_listening(self):
        rate = rospy.Rate(10)  # 10Hz 检查频率
        timeout = rospy.Duration(10.0)  # 超时时间为10秒
        start_time = rospy.Time.now()

        while not rospy.is_shutdown() and (rospy.Time.now() - start_time < timeout):
            trans = self.get_transform()
            if trans:
                self.last_trans = trans  # 更新最近一次成功获取的坐标
                self.check_and_write_to_json(trans)
                break
            rate.sleep()

        # 如果超时，且未获取到新变换，使用最后一次成功获取的变换
        if not self.last_trans:
            rospy.logwarn("Timeout: No TF message received within the time limit.")
        else:
            self.check_and_write_to_json(self.last_trans)

if __name__ == '__main__':
    rospy.init_node('coordinate_listener', anonymous=True)
    listener = CoordinateListener()  # 创建 CoordinateListener 实例
    listener.start_listening()       # 开始监听坐标


