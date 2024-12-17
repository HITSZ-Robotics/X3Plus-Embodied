import socketserver
import os
import json
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'commander'))
from get_picture import get_picture
from pick_point import pick_point
from place_point import place_point 
from test_arm import *
from move import *
from ip import get_local_ip 
from ip import ip_json
#ip_json()
robot_ip=get_local_ip()
ip_port =( robot_ip,9001)     ##robot_ip
print("robot IP 地址: {}".format(robot_ip))
print("robot_terminal begin")


   
class MyServer(socketserver.BaseRequestHandler):
    
     def handle(self):
        base_path = ''
        conn = self.request
        while True:
          try:
             op = conn.recv(1024).decode()
             
             if not op:
                  break
             
             if(op=='get_picture'):
               conn.send("get order get_picture ".encode())
               file_name=conn.recv(1024).decode()
               get_picture(file_name)
               conn.send("order get_picture is finish".encode())
 
 
             elif op =='pick':#请求下载
               conn.send("get order pick ".encode())
               file_name=conn.recv(1024).decode()
               pick_point(file_name)       
               conn.send("order pick is finish".encode())
             
             elif op =='place':
               conn.send("get order place ".encode())
               file_name=conn.recv(1024).decode()
               place_point(file_name)
               conn.send("order place is finish".encode())

             elif op =='navigation':
               conn.send("get order get_picture ".encode())
               file_name=conn.recv(1024).decode()
               navigation(file_name)
               conn.send("order is navigation finish".encode())
             
                      
             elif op=="listen":
               conn.send("get order get_picture ".encode())
               file_name=conn.recv(1024).decode()
               listen(file_name) 
               conn.send("order is listen finish".encode())
             
             elif op=="speak":
               conn.send("get order get_picture ".encode())
               file_name=conn.recv(1024).decode()
               speak(file_name)
               conn.send("order is speak finish".encode())
               
             elif op=="test_arm":
               conn.send("get order test_arm ".encode())
               file_name=conn.recv(1024).decode()
             
               test_arm(file_name)
           
               conn.send("order is test_arm finish".encode())
 
 
             elif op=="depth_camera":
               conn.send("get order depth_camera ".encode())
               file_name=conn.recv(1024).decode() 
               get_depth_camera(file_name) 
               conn.send("order is depth_camera finish".encode())
             
             elif op=="forward_onestep":
               conn.send("get order forward_onestep ".encode())
               file_name=conn.recv(1024).decode()
               forward_onestep()
               conn.send("order is forward_onestep finish".encode())

             elif op=="backward_onestep":
               conn.send("get order backward_onestep ".encode())
               file_name=conn.recv(1024).decode()
               backward_onestep()
               conn.send("order is backward_onestep  finish".encode())
             
             elif op=="turnleft":
               conn.send("get order turnleft ".encode())
               file_name=conn.recv(1024).decode()
               turnleft()
               conn.send("order is turnleft  finish".encode())

             elif op=="turnright":
               conn.send("get order turnright ".encode())
               file_name=conn.recv(1024).decode()
               turnright()
               conn.send("order is turnright finish".encode())


             else:
                  conn.send("Invalid command".encode())

          except Exception as e:
                conn.send(f"Error: {str(e)}".encode())
                break


if __name__ == '__main__':
    instance = socketserver.ThreadingTCPServer(ip_port, MyServer)
    instance.serve_forever()
