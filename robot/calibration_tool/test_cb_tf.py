import os
import subprocess

def run_ros_command():
    try:
        result = subprocess.run(['rosrun', 'arm_moveit_demo', 'cb_tf_echo.py'], check=True)
        if result.returncode == 0:
            print("Command executed successfully.")
        else:
            print(f"Command failed with return code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")

run_ros_command()        