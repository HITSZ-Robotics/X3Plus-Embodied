#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit_msgs/OrientationConstraint.h>
#include <moveit_visual_tools/moveit_visual_tools.h>

using namespace std;


void multi_trajectory(
        moveit::planning_interface::MoveGroupInterface &yahboomcar,
        const vector<double> &pose,
        moveit_msgs::RobotTrajectory &trajectory) {
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const robot_state::JointModelGroup *joint_model_group;
    // 获取机器人的起始位置
    moveit::core::RobotStatePtr start_state(yahboomcar.getCurrentState());
    joint_model_group = start_state->getJointModelGroup(yahboomcar.getName());
    yahboomcar.setJointValueTarget(pose);
    yahboomcar.plan(plan);
    start_state->setJointGroupPositions(joint_model_group, pose);
    yahboomcar.setStartState(*start_state);
    trajectory.joint_trajectory.joint_names = plan.trajectory_.joint_trajectory.joint_names;
    for (size_t j = 0; j < plan.trajectory_.joint_trajectory.points.size(); j++) {
        trajectory.joint_trajectory.points.push_back(plan.trajectory_.joint_trajectory.points[j]);
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "moveit_revise_trajectory_demo");
    ros::NodeHandle node_handle;
    ros::AsyncSpinner spinner(1);
    spinner.start();

    moveit_msgs::RobotTrajectory trajectory;
    moveit::planning_interface::MoveGroupInterface yahboomcar("arm_group");
    moveit_visual_tools::MoveItVisualTools tool(yahboomcar.getPlanningFrame());
    tool.deleteAllMarkers();

    yahboomcar.allowReplanning(true);
    // 规划的时间(单位：秒)
    yahboomcar.setPlanningTime(5);
    yahboomcar.setNumPlanningAttempts(10);
    // 设置允许目标角度误差
    yahboomcar.setGoalJointTolerance(0.01);
    yahboomcar.setGoalPositionTolerance(0.01);
    yahboomcar.setGoalOrientationTolerance(0.01);
    yahboomcar.setGoalTolerance(0.01);
    // 设置允许的最大速度和加速度
    yahboomcar.setMaxVelocityScalingFactor(1.0);
    yahboomcar.setMaxAccelerationScalingFactor(1.0);

    // 控制机械臂先回到初始化位置
    yahboomcar.setNamedTarget("down");
    yahboomcar.move();

    vector<vector<double>> poses{
            {1.34,  -1.0,  -0.61, 0.2,   0},
            {0,     0,     0,     0,     0},
            {-1.16, -0.97, -0.81, -0.79, 3.14}
    };
    for (int i = 0; i < poses.size(); ++i) {
        multi_trajectory(yahboomcar, poses.at(i), trajectory);
    }

    moveit::planning_interface::MoveGroupInterface::Plan joinedPlan;
    robot_trajectory::RobotTrajectory rt(yahboomcar.getCurrentState()->getRobotModel(), "arm_group");
    rt.setRobotTrajectoryMsg(*yahboomcar.getCurrentState(), trajectory);
    trajectory_processing::IterativeParabolicTimeParameterization iptp;
    iptp.computeTimeStamps(rt, 1, 1);
    rt.getRobotTrajectoryMsg(trajectory);
    joinedPlan.trajectory_ = trajectory;

    // 显示轨迹
    tool.publishTrajectoryLine(joinedPlan.trajectory_, yahboomcar.getCurrentState()->getJointModelGroup("arm_group"));
    tool.trigger();

    if (!yahboomcar.execute(joinedPlan)) {
        ROS_ERROR("Failed to execute plan");
        return false;
    }
    sleep(1);
    ROS_INFO("Finished");
    ros::shutdown();
    return 0;
}

