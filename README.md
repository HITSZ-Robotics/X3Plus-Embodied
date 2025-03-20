# X3Plus-Embodied
HIT (Shenzhen) 2023 Computer Science Project: An Embodied Intelligence Path Planning System Based on YahboomCar X3 Plus.
## Structure
The following is the code structure of the project:
![Project Screenshot](images/structure.png)

## Overview
Watch the demo video to understand how it works:  
![Demo](images/output.gif)

## Installation
1. Clone the repository:
   ```bash
   cd ~/yahboomcar_ws/src
   git clone https://github.com/HITSZ-Robotics/X3Plus-Embodied.git
   
2. Download models:
   -[GroundingDINO link](https://github.com/IDEA-Research/GroundingDINO)
   -[whisper link](https://github.com/openai/whisper)
   -[TTS](https://github.com/coqui-ai/TTS)

3. Create a Conda environment
   ```bash
   conda create -n embodied_X3 python=3.9
   conda activate embodied_X3
   pip install -r requirement.txt
   
4. Recompiling IKFast:
   We have tried both KDL and Trace-IK,neither of these methods could provide fast and accurate solutions.As a result, we chose to use IKFast for better performance and reliability.
   ```bash
   sudo apt-get install ros-melodic-moveit-kinematics
   source ~/yahboomcar_ws/devel/setup.bash
   cd ~/yahboomcar_ws
   catkin_make
## Usages
  1. Change the OpenAI API-key in master/speaker.py  master/robotic_exec_generation.py
  2. Change the ips of the robot and master in get_ip.py
  3. run the robot/upload_download_server.py
  4. run the master/upload_download_cilent.py
  5. run the master/listener.py
