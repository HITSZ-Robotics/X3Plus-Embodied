#include <iostream>
#include "ros/ros.h"
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf/LinearMath/Quaternion.h>

using namespace std;
// 角度转弧度
const float DE2RA = M_PI / 180.0f;

int main(int argc, char **argv) {
    ros::init(argc, argv, "set_pose_plan_cpp");
    ros::NodeHandle n;
    ros::AsyncSpinner spinner(1);
    spinner.start();
    moveit::planning_interface::MoveGroupInterface yahboomcar("arm_group");
    yahboomcar.allowReplanning(true);
    // 规划的时间(单位：秒)
    yahboomcar.setPlanningTime(10);
    yahboomcar.setNumPlanningAttempts(10);
    // 设置位置(单位：米)和姿态（单位：弧度）的允许误差
    yahboomcar.setGoalPositionTolerance(0.01);
    yahboomcar.setGoalOrientationTolerance(0.01);
    // 设置允许的最大速度和加速度
    yahboomcar.setMaxVelocityScalingFactor(1.0);
    yahboomcar.setMaxAccelerationScalingFactor(1.0);
//    sleep(0.1);
    //设置具体位置
    geometry_msgs::Pose pose;
    pose.position.x = 0.20;
    pose.position.y = 0.5;
    pose.position.z = 0.20;
    // 设置目标姿态
    tf::Quaternion quaternion;
    // RPY的单位是角度值
    double Roll = 0;
    double Pitch = 0;
    double Yaw = 0;
    // RPY转四元数
    quaternion.setRPY(Roll * DE2RA, Pitch * DE2RA, Yaw * DE2RA);
    pose.orientation.x = quaternion.x();
    pose.orientation.y = quaternion.y();
    pose.orientation.z = quaternion.z();
    pose.orientation.w = quaternion.w();
    yahboomcar.setJointValueTarget(pose);
    int index = 0;
    // 多次执行,提高成功率
    while (index <= 10) {
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        // 运动规划
        const moveit::planning_interface::MoveItErrorCode &code = yahboomcar.plan(plan);
        if (code == code.SUCCESS) {
            ROS_INFO_STREAM("plan success");
            yahboomcar.execute(plan);
            break;
        } else {
            ROS_INFO_STREAM("plan error");
        }
        index++;
    }
    return 0;
}

