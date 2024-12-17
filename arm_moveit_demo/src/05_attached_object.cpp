#include <iostream>
#include "ros/ros.h"
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf/LinearMath/Quaternion.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

using namespace std;

int main(int argc, char **argv) {
    ros::init(argc, argv, "attached_object_cpp");
    ros::NodeHandle n;
    ros::AsyncSpinner spinner(1);
    spinner.start();
    moveit::planning_interface::MoveGroupInterface yahboomcar("arm_group");
    yahboomcar.allowReplanning(true);
    // 规划的时间(单位：秒)
    yahboomcar.setPlanningTime(5);
    // 设置规划尝试次数
    yahboomcar.setNumPlanningAttempts(10);
    // 设置允许目标姿态误差(单位：米)
    yahboomcar.setGoalPositionTolerance(0.01);
    // 设置允许目标位置误差(单位：弧度)
    yahboomcar.setGoalOrientationTolerance(0.01);
    yahboomcar.setMaxVelocityScalingFactor(1.0);
    yahboomcar.setMaxAccelerationScalingFactor(1.0);
    yahboomcar.setNamedTarget("up");
    yahboomcar.move();
    sleep(0.1);
    string frame = yahboomcar.getPlanningFrame();
    moveit::planning_interface::PlanningSceneInterface scene;
    /////////////////////////添加障碍物///////////////////////
    vector<string> object_ids;
    scene.removeCollisionObjects(object_ids);
    vector<moveit_msgs::CollisionObject> objects;
    moveit_msgs::CollisionObject obj;
    // 设置障碍物的id
    obj.id = "obj";
    object_ids.push_back(obj.id);
    // 障碍物的状态
    obj.operation = obj.ADD;
    obj.header.frame_id = frame;
    shape_msgs::SolidPrimitive primitive;
    // 设置障碍物类型
    primitive.type = primitive.BOX;
    // 设置障碍物维度
    primitive.dimensions.resize(3);
    // 设置障碍物的长宽高
    primitive.dimensions[0] = 0.2;
    primitive.dimensions[1] = 0.1;
    primitive.dimensions[2] = 0.02;
    obj.primitives.push_back(primitive);
    geometry_msgs::Pose pose;
    // 设置障碍物的位置信息[x,y,z]
    pose.position.x = 0.2;
    pose.position.y = 0;
    pose.position.z = 0.26;
    tf::Quaternion quaternion;
    // R,P,Y的单位是角度
    double Roll = 0.0;
    double Pitch = 0.0;
    double Yaw = 90.0;
    quaternion.setRPY(Roll * M_PI / 180, Pitch * M_PI / 180, Yaw * M_PI / 180);
    pose.orientation.x = quaternion.x();
    pose.orientation.y = quaternion.y();
    pose.orientation.z = quaternion.z();
    pose.orientation.w = quaternion.w();
    // 设置障碍物的位姿信息
    obj.primitive_poses.push_back(pose);
    objects.push_back(obj);
    /////////////////////////设置障碍物的颜色///////////////////////
    // 创建检测对象的颜色容器
    std::vector<moveit_msgs::ObjectColor> colors;
    moveit_msgs::ObjectColor color;
    // 添加需要设置颜色的id
    color.id = "obj";
    // 设置RGBA值,范围[0~1]
    color.color.r = 1.0;
    color.color.g = 0;
    color.color.b = 0;
    color.color.a = 0.5;
    colors.push_back(color);
    // 将设置的信息添加到场景中
    scene.applyCollisionObjects(objects, colors);
    yahboomcar.setNamedTarget("down");
    yahboomcar.move();
    sleep(0.1);
    int index = 0;
    while (index < 10) {
        yahboomcar.setRandomTarget();
        yahboomcar.move();
        sleep(0.5);
        index++;
        cout << "第 " << index << " 次规划!!!" << endl;
    }
    vector<string> ids{"obj"};
    scene.removeCollisionObjects(ids);
    return 0;
}

