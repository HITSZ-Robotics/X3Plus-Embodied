#include <iostream>
#include "ros/ros.h"
#include <moveit/move_group_interface/move_group_interface.h>

using namespace std;

int main(int argc, char **argv) {
    ros::init(argc, argv, "yahboomcar_random_move_cpp");
    ros::NodeHandle n;
	ros::AsyncSpinner spinner(1);
	spinner.start();
    moveit::planning_interface::MoveGroupInterface yahboomcar("arm_group");
	// 设置最大速度
    yahboomcar.setMaxVelocityScalingFactor(1.0);
    // 设置最大加速度
    yahboomcar.setMaxAccelerationScalingFactor(1.0);
    //设置目标点
    yahboomcar.setNamedTarget("down");
    //开始移动
    yahboomcar.move();
    sleep(0.1);
    while (!ros::isShuttingDown()){
    	//设置随机目标点
    	yahboomcar.setRandomTarget();
    	yahboomcar.move();
    	sleep(0.5);
    }
    return 0;
}
