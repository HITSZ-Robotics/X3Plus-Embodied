#!/usr/bin/env python
# encoding: utf-8


import rospy
import math
from std_msgs.msg import Bool
from move_base_msgs.msg import *
from geometry_msgs.msg import Twist
from actionlib_msgs.msg import GoalID
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped


class Goal_publisher:
    def __init__(self):
        self.pub_goal = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=1)

    def MOVE_TO(self, x, y):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = rospy.Time.now()
        # The location of the target point
        pose.pose.position.x = x
        pose.pose.position.y = y
        # The posture of the target point.
       # pose.pose.orientation.z=math.sin(45) 
       # pose.pose.orientation.w=math.cos(45)
        pose.pose.orientation.z = 1
        pose.pose.orientation.w = 0
        self.pub_goal.publish(pose)    

if __name__ == '__main__':
    # 初始化节点 || Initialize node
    rospy.init_node('Goal_publisher')
    z=Goal_publisher()
    x = input('input-x:')   
    y = input('input-y:')
    #Z = input('input-angle/2:')
    z.MOVE_TO(x,y)
    rospy.spin()
   
