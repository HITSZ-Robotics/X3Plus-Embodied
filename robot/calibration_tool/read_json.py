import json
import os
import numpy as np
script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(script_path)
image_points_list=[]

def get_imagepoint():
    image_points_list=[]
    with open(parent_directory+"/data/click_data.json", 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                print(f"x: {data['x']}, y: {data['y']}, distance: {data['distance']}, camera_coordinate: {data['camera_coordinate']}")
                
                data_arry=[data['x'],data['y']]
                image_points_list.append(data_arry)
    
                
            except json.JSONDecodeError as e:
                 print(f"Error parsing JSON: {e}")
    
    image_points=np.array(image_points_list, dtype=np.float32)
    return image_points

    
def get_3dpoint():
    robot_points_list=[]
    with open(parent_directory+"/data/coordinates.json", 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                print(f"x: {data['x']}, y: {data['y']}, Z: {data['z']}")
                
                data_arry=[data['x'],data['y'],data['z']]
                robot_points_list.append(data_arry)
    
                
            except json.JSONDecodeError as e:
                 print(f"Error parsing JSON: {e}")
    
    robot_points=np.array(robot_points_list, dtype=np.float32)
    return robot_points
#with open(parent_directory+f'/data/key.json', 'r') as f :
#           key_data=json.load(f)

#print(key_data['key'])

#key_data={'key':1}
#with open(parent_directory+f'/data/key.json', 'w') as f :
#           json.dump(key_data,f)
#robot_point=get_3dpoint()
#print(robot_point)