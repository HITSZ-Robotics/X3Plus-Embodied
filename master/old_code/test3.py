import os
from tqdm import tqdm
import cv2
import numpy as np
base_path = "/home/zzl/Desktop/data/"

def get_rgb(set_filename):
    color_filename =base_path+set_filename
    image = cv2.imread(color_filename,3)
    image=image.astype("uint8")
    return image
image=get_rgb("photo.png")
print(image)