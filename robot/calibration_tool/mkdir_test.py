import os
import subprocess


script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(script_path)
base_path=parent_directory



def create_directory(path):
    if os.path.exists(path):
        os.removedirs(path)
    os.makedirs(path, mode=0o777)

create_directory(parent_directory+"/test")