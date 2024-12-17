import os
#命令行执行rosrun


def test_arm(file_name):
    
   # str="rosrun arm_moveit_demo 03_set_joint_plan"
    #os.system(str)

    #print('rosrun ok')

    os.system("rosrun arm_moveit_demo arm_test2.py --file_name=%s" %(file_name))
    #os.system("rosrun arm_moveit_demo arm_Test.py")
    print('arm is finish')


