# X3Plus-Embodied
HIT (Shenzhen) 2023 Computer Science Freshman Project: An Embodied Intelligence Path Planning System Based on YahboomCar X3 Plus, open-sourcing the practical achievements of the freshman team.\n
## Team Members
The project is developed by the following team members (in alphabetical order):
- Ma Yujie  
- Wei Jie  
- Yang Yanyan  
- Zhou Zhiling  
## Table of Contents
- [Overview](#Overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
## Overview
The following is the code structure of the project:
![Project Screenshot](images/screenshot.png)
Watch the demo video to understand how it works:  
![Demo](images/demo.gif)
## Installation
1. Clone the repository:
   ```bash
   cd ~/yahboomcar_ws/src
   git clone https://github.com/zzl410/X3Plus-Embodied.git
   ```bash
2. Download GroundingDINO:
   [GroundingDINO link]([https://www.google.com](https://github.com/IDEA-Research/GroundingDINO))
3. Recompiling IKFast:
   We have tried both KDL and Trace-IK, but due to the limited degrees of freedom (DOF) of the robotic arm, neither of these methods could provide fast and accurate solutions.
   -  As a result, we chose to use IKFast for better performance and reliability.
   ```bash
   sudo apt-get install ros-melodic-moveit-kinematics
   source ~/yahboomcar_ws/devel/setup.bash
   cd ~/yahboomcar_ws
   catkin_make
   
   
