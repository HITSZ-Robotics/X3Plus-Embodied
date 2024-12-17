search_mode=OPTIMIZE_MAX_JOINT
srdf_filename=yahboomcar_X3plus.srdf
robot_name_in_srdf=yahboomcar_X3plus
moveit_config_pkg=x3plus_moveit_config
robot_name=yahboomcar_x3plus
planning_group_name=arm_group
ikfast_plugin_pkg=x3plus_ikfast_plugin
base_link_name=base_link
eef_link_name=arm_link5
ikfast_output_path=/home/jetson/yahboomcar_ws/src/x3plus_ikfast_plugin/src/yahboomcar_x3plus_arm_group_ikfast_solver.cpp

rosrun moveit_kinematics create_ikfast_moveit_plugin.py\
  --search_mode=$search_mode\
  --srdf_filename=$srdf_filename\
  --robot_name_in_srdf=$robot_name_in_srdf\
  --moveit_config_pkg=$moveit_config_pkg\
  $robot_name\
  $planning_group_name\
  $ikfast_plugin_pkg\
  $base_link_name\
  $eef_link_name\
  $ikfast_output_path
