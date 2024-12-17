import os
import sys
import json
import numpy as np
current_dir = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_dir)
sys.path.append(parent_directory)
# 定义文件路径
capture_signal_file = 'capture_signal.json'
process_signal_file = 'process_signal.json'
depth_image_file = 'depth_image.npy'
color_image_file = 'color_image.npy'
image_point_file = 'image_point.json'
point3d_file = '3d_point.json'
capture_signal_file=os.path.join(parent_directory,capture_signal_file)
process_signal_file=os.path.join(parent_directory,process_signal_file)
depth_image_file =os.path.join(parent_directory,depth_image_file)
color_image_file = os.path.join(parent_directory, color_image_file)
image_point_file = os.path.join(parent_directory, image_point_file)
point3d_file = os.path.join(parent_directory, point3d_file)
def write_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def read_json(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

def write_npy(file_path, data):
    np.save(file_path, data)

def read_npy(file_path):
    if not os.path.exists(file_path):
        return None
    return np.load(file_path)

def set_capture_signal(value):
    write_json(capture_signal_file, {'value': value})

def get_capture_signal():
    data = read_json(capture_signal_file)
    return data['value'] if data else False

def set_process_signal(value):
    write_json(process_signal_file, {'value': value})

def get_process_signal():
    data = read_json(process_signal_file)
    return data['value'] if data else False

def set_shared_depth_image(image):
    write_npy(depth_image_file, image)

def get_shared_depth_image():
    return read_npy(depth_image_file)

def set_shared_color_image(image):
    write_npy(color_image_file, image)

def get_shared_color_image():
    return read_npy(color_image_file)

def set_shared_point(x, y):
    write_json(image_point_file, {'x': x, 'y': y})

def get_shared_point():
    data = read_json(image_point_file)
    return (data['x'], data['y']) if data else (0, 0)

def set_shared_3dpoint(x, y, z):
    write_json(point3d_file, {'x': x, 'y': y, 'z': z})

def get_shared_3dpoint():
    data = read_json(point3d_file)
    return (data['x'], data['y'], data['z']) if data else (0.0, 0.0, 0.0)

