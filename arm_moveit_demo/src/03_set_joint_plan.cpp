#include <iostream>
#include "ros/ros.h"
#include <tf/LinearMath/Quaternion.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>

using namespace std;

int main(int argc, char **argv) {
    ros::init(argc, argv, "set_joint_plan_cpp");
    ros::NodeHandle n;
    ros::AsyncSpinner spinner(1);
    spinner.start();
    moveit::planning_interface::MoveGroupInterface yahboomcar("arm_group");
    yahboomcar.allowReplanning(true);
    // 规划的时间(单位：秒)
    yahboomcar.setPlanningTime(5);
    yahboomcar.setNumPlanningAttempts(10);
    // 设置允许目标角度误差
    yahboomcar.setGoalJointTolerance(0.01);
    // 设置允许的最大速度和加速度
    yahboomcar.setMaxVelocityScalingFactor(1.0);
    yahboomcar.setMaxAccelerationScalingFactor(1.0);
    yahboomcar.setNamedTarget("up");
    yahboomcar.move();
//    sleep(0.1);
    //设置具体位置
    vector<double> pose{0, 0.79, -1.57, -1.57, 0};
    yahboomcar.setJointValueTarget(pose);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const moveit::planning_interface::MoveItErrorCode &code = yahboomcar.plan(plan);
    if (code == code.SUCCESS) {
        ROS_INFO_STREAM("plan success");
        // 显示轨迹
        string frame = yahboomcar.getPlanningFrame();
        moveit_visual_tools::MoveItVisualTools tool(frame);
        tool.deleteAllMarkers();
        tool.publishTrajectoryLine(plan.trajectory_, yahboomcar.getCurrentState()->getJointModelGroup("arm_group"));
        tool.trigger();
        yahboomcar.execute(plan);
    } else {
        ROS_INFO_STREAM("plan error");
    }
    return 0;
}

