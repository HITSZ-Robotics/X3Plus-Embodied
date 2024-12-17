#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#标定时调用tf，echo base_link与arm_link5之间的坐标变换，融合

import rospy
import tf
import tkinter as tk
from geometry_msgs.msg import TransformStamped
from tf import TransformListener

class CoordinateListener:
    def __init__(self):
        self.listener = TransformListener()
        self.coordinates = []
        
    def get_coordinates(self):
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform('/base_link', '/arm_link5', now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform('/base_link', '/arm_link5', now)
            self.coordinates.append(trans)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("TF Exception: {}".format(e))