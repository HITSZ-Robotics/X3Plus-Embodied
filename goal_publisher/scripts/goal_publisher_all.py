#!/usr/bin/env python
# encoding: utf-8


import rospy
from std_msgs.msg import Bool
from move_base_msgs.msg import *
from geometry_msgs.msg import Twist
from actionlib_msgs.msg import GoalID
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
import json


class Goal_publisher:
    def __init__(self,pose = [0,0]):
        self.pub_goal = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=1)
        self.pose_x_y = pose

    def MOVE_TO(self):
        pose_x = self.pose_x_y[0]
        pose_y = self.pose_x_y[1]
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = rospy.Time.now()
        # The location of the target point
        pose.pose.position.x = pose_x
        pose.pose.position.y = pose_y
        # The posture of the target point. z=sin(angle/2) w=cos(angle/2)
        pose.pose.orientation.z = 0
        pose.pose.orientation.w = 1
        self.pub_goal.publish(pose)    

if __name__ == '__main__':
    filename = 'www'
#    with open(filename,'r') as file_object:
#        goal = json.load(file_object)
    pose = [2.25,-1.42] 
    # if goal == 'A':
    #     pose = [1,1]#预设位置，可修改
    # elif goal == 'B':
    #     pose = [2,2]        
    # 初始化节点 || Initialize node
    rospy.init_node('Goal_publisher')
    z=Goal_publisher(pose)
    z.MOVE_TO()
    rospy.spin()
   
