from old_code.transformer import *
from old_code.clip import *
import cv2

x1,y1=object_detection("clib1","the red square object")
p1=camera_coordinate(x1,y1,"clib1")
print(p1)
x3,y3=object_detection("clib3","the red square object")
p3=camera_coordinate(x3,y3,"clib1")
print(p3)
x4,y4=object_detection("clib4","the red square object")
p4=camera_coordinate(x4,y4,"clib4")
print(p4)
x5,y5=object_detection("clib5","the red square object")
p5=camera_coordinate(x5,y5,"clib5")
print(p5)
x6,y6=object_detection("clib6","the red square object")
p6=camera_coordinate(x6,y6,"clib6")
print(p6)