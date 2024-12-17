import subprocess
import cv2
import numpy as np 
world_place=np.array([[2],[1],[3],[1]],dtype=np.float32)

x_w=world_place[0,0]
y_w=world_place[1,0]
z_w=world_place[2,0]
print(x_w)