import socket
import os
import sys
import json
from ip_get import get_manual_ip
from write_json import *
from visual_system import object_detection
from speaker import speak_str
from robotic_exec_generation import exec_steps
robot_ip=get_manual_ip()
ip_port = ('{}'.format(robot_ip),9001)       #### robot_terminal address
sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sk.connect(ip_port)
script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(script_path)
base_path=parent_directory+"/data/"

def get_picture(file_name):    #file_name depend on you
 
    sk.send("get_picture".encode())      #change the order name
    print(sk.recv(1024).decode())
    sk.send(file_name.encode())
    print(sk.recv(1024).decode())

def pick(file_name):    #file_name depend on you

    sk.send("pick".encode())      #change the order name
    print(sk.recv(1024).decode())
    sk.send(file_name.encode())
    print(sk.recv(1024).decode())

def place(file_name):    #file_name depend on you

    sk.send("place".encode())      #change the order name
    print(sk.recv(1024).decode())
    sk.send(file_name.encode())
    print(sk.recv(1024).decode())

def navigate(location):    #file_name depend on you
     navigate_filename="pick_photo.jpg"
     navigate_jsonname="navigatepoint.json"
     #get_picture(imagefilename)
     #x,y=object_detection(imagefilename,object)
     write_navigate(1.86, -0.203, -0.3963,navigate_jsonname)
     send_command("navigate","navigatepoint.json")

def speak (content):
    speak_str(content)


def send_command(command, file_name):
    try:
        sk.send(command.encode())
        print(sk.recv(1024).decode())
        sk.send(file_name.encode())
        print(sk.recv(1024).decode())
    except Exception as e:
        print(f"Error: {str(e)}")

#def pick_object(imagefile_name,object,point_filename):
#    x,y=object_detection(imagefile_name,object)
#    write_point(x,y,point_filename)
#    send_command("pick",point_filename)

def pick_object(object):
     imagefilename="pick_photo.jpg"
     pick_jsonname="pick.json"
     get_picture(imagefilename)
     x,y=object_detection(imagefilename,object)
     write_point(x,y,pick_jsonname)
     send_command("pick",pick_jsonname)

def place_object(object):
     imagefilename="place_photo.jpg"
     pick_jsonname="place.json"
     get_picture(imagefilename)
     x,y=object_detection(imagefilename,object)
     write_point(x,y,pick_jsonname)
     send_command("place",pick_jsonname)

def turnleft():
     send_command("turnleft","nullfile.txt")

def turnright():
     send_command("turnright","nullfile.txt")

def backward_onestep():
     send_command("backward_onestep","nullfile.txt")

def forward_onestep():
     send_command("forward_onestep","nullfile.txt")


     

#if __name__ == '__main__':
#    get_picture('photo.png')
#    pick_object('photo.png',"the red square object","pick_point.json")
def exec_process(content):
   exec_codes=exec_steps(content)
   for code in exec_codes.splitlines():
                try:
                    exec(code)
                except Exception as e:
                    print(e)


#write_point(320,240,"pick_point.json")
#send_command("pick","pick.json")

#place_object("black square box")
#place_object("white circle object") 
                    
#pick_object("white circle object") 
#pick_object("green smaller cube") 
#place_object("green circle box")  
#forward_onestep()
#turnright()
#forward_onestep()
#turnright()
#forward_onestep()
#turnright()
#forward_onestep()
#turnright()
#backward_onestep()               
#turnleft()
#backward_onestep()               
#turnleft()
#backward_onestep()               
#turnleft()
#backward_onestep()               
#turnleft()