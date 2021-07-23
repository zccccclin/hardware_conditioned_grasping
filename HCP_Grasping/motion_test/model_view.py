import pybullet as p
import os
import numpy as np
import time
pc = p.connect(p.GUI)
act_joint_indices = [0,1,2,3,4,5,9,11,14,16,19,21]

def reset_robot_pose(robot_id, act_joint_indices):
        desired_joint_positions = [-0.184, -1.09, 1.497, -1.98, -1.5715, 1.38]
        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=act_joint_indices[:6],
            controlMode=p.POSITION_CONTROL,
            targetPositions=desired_joint_positions
            #forces=torque,
        )
        for i in range(20):
            p.stepSimulation()
        return desired_joint_positions
file_list = os.listdir('../../assets/gen_gripper/3f_2j/')
pos = np.array([0.0,0.0,0.0])
list = [i for i in range(len(file_list))]
for i in list:
    p.loadURDF(f'../../assets/gen_gripper/3f_2j/robot_{i}/model.urdf', pos, useFixedBase=True)
    pos += np.array([0,0.5,0])
    reset_robot_pose(i, act_joint_indices)
    
time.sleep(10000)