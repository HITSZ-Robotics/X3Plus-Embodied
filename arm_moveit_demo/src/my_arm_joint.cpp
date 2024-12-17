#include <iostream>
#include <cmath>
#include "ros/ros.h"
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf/LinearMath/Quaternion.h>
#include <moveit_visual_tools/moveit_visual_tools.h>

using namespace std;
// 角度转弧度
const float DE2RA = M_PI / 180.0f;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "my_arm_joint_cpp");
    ros::NodeHandle n;
    ros::AsyncSpinner spinner(1);
    spinner.start();
    moveit::planning_interface::MoveGroupInterface yahboomcar("arm_group");
    moveit::planning_interface::MoveGroupInterface yahboomcar_gripper("gripper_group");
    yahboomcar.allowReplanning(true);
    // 规划的时间(单位：秒)
    yahboomcar.setPlanningTime(5);
    yahboomcar.setNumPlanningAttempts(10);
    // 设置位置(单位：米)和姿态（单位：弧度）的允许误差
    yahboomcar.setGoalPositionTolerance(0.01);
    yahboomcar.setGoalOrientationTolerance(0.01);
    // 设置允许的最大速度和加速度
    yahboomcar.setMaxVelocityScalingFactor(1.0);
    yahboomcar.setMaxAccelerationScalingFactor(1.0);
    yahboomcar.setNamedTarget("up");
    yahboomcar.move();
    //    sleep(0.1);


    double x_base,y_base,z_base;
    cout<<"Input x_base:";
    cin>>x_base;
    cout<<"Input y_base:";
    cin>>y_base;
    cout<<"Input z_base:";
    cin>>z_base;




    //法一：把远端两个连杆合为一个连杆，控制夹爪到达目标点
    const double pi = 3.14159265359;
    const double L1 = 0.0829, L2 = 0.0829+0.17456;
    //x_base = 0.39;
    //y_base = 0.1;
    //z_base = -0.08;
    double x_raw = x_base - 0.098;
    double y_raw = y_base;
    double z_raw = z_base - 0.142;
    double theta0 = atan(-y_raw / x_raw);
    double x = sqrt(x_raw * x_raw + y_raw * y_raw);
    double y = 0;
    double z = z_raw;
    double theta1, theta2;
    double alpha = acos((L1 * L1 + L2 * L2 - x * x - y * y) / (2 * L1 * L2));
    theta2 = pi - alpha;
    theta1 = atan(z / x) + atan((L2 * sin(theta2)) / (L1 + L2 * cos(theta2)));
    vector<double> pose = {theta0, -pi / 2 + theta1, -theta2, 0, 0};
   // vector<double> pose = {0,0, 0, 0, 0};
    //到此结束
    for(int i=0;i<5;i++){
	cout<<pose[i]<<" ";
    }
    cout<<endl;


/*
    //法二：保证最远端的连杆垂直地面抓取，规划最远端两个连杆的交点到达目标点上方L3处
    const double pi = 3.14159265359;
    const double L1 = 0.0829, L2 = 0.0829, L3 = 0.17456;
    // x_base = 0.39;
    // y_base = 0.1;
    // z_base = -0.08;
    double x_raw = x_base - 0.098;
    double y_raw = y_base;
    double z_raw = z_base - 0.142;
    double theta0 = atan(-y_raw / x_raw);
    double x = sqrt(x_raw * x_raw + y_raw * y_raw);
    double y = 0;
    double z = z_raw + L3;
    double theta1, theta2;
    double alpha = acos((L1 * L1 + L2 * L2 - x * x - y * y) / (2 * L1 * L2));
    theta2 = pi - alpha;
    theta1 = atan(z / x) + atan((L2 * sin(theta2)) / (L1 + L2 * cos(theta2)));
    vector<double> pose = {theta0, -pi / 2 + theta1, -theta2, -theta1+theta2-pi/2, 0};
    //到此结束
*/

    double gripper_angle=0;
    vector<double> angles={gripper_angle,-gripper_angle,-gripper_angle,gripper_angle,-gripper_angle,gripper_angle};
    yahboomcar.setJointValueTarget(pose);
    //yahboomcar_gripper.setJointValueTarget(angles);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const moveit::planning_interface::MoveItErrorCode &code = yahboomcar.plan(plan);
    if (code == code.SUCCESS)
    {
        ROS_INFO_STREAM("plan success");
        // 显示轨迹
        string frame = yahboomcar.getPlanningFrame();
        moveit_visual_tools::MoveItVisualTools tool(frame);
        tool.deleteAllMarkers();
        tool.publishTrajectoryLine(plan.trajectory_, yahboomcar.getCurrentState()->getJointModelGroup("arm_group"));
        tool.trigger();
        yahboomcar.execute(plan);
    }
    else
    {
        ROS_INFO_STREAM("plan error");
    }
    return 0;
}
