import time

import numpy as np
import pybullet as p
import pybullet_data
from collections import OrderedDict

import keyboard

def main():
    p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF('plane.urdf')
    p.setGravity(0,0,-9.81)
    hand_2f1j = p.loadURDF("../../assets/grippers/2f_1j.urdf", [0, 0 , .3],[1,1,0,0],useFixedBase=True )
    hand_2f2j = p.loadURDF("../../assets/grippers/2f_2j.urdf", [.3, 0 , .3],[1,1,0,0],useFixedBase=True )
    hand_3f1j = p.loadURDF("../../assets/grippers/3f_1j.urdf", [.6, 0 , .3],[1,1,0,0],useFixedBase=True )
    hand_3f2j = p.loadURDF("../../assets/grippers/3f_2j.urdf", [.9, 0 , .3],[1,1,0,0],useFixedBase=True )
    gen1_3f_1j = p.loadURDF('../../assets/gen_gripper/3f_1j/robot_0/model.urdf', [0, .3 , .3],[1,1,0,0],useFixedBase=True)
    gen2_3f_1j = p.loadURDF('../../assets/gen_gripper/3f_1j/robot_1/model.urdf', [.3, .3 , .3],[1,1,0,0],useFixedBase=True)
    gen1_3f_2j = p.loadURDF('../../assets/gen_gripper/3f_2j/robot_0/model.urdf', [.6, .3 , .3],[1,1,0,0],useFixedBase=True)
    gen2_3f_2j = p.loadURDF('../../assets/gen_gripper/3f_2j/robot_3/model.urdf', [.9, .3 , .3],[1,1,0,0],useFixedBase=True)

    barrett = p.loadURDF("../assets/barrett_hand/model.urdf", [1.2,0, .3],[1,1,0,0],useFixedBase=True)

    cube1 = p.loadURDF('cube_small.urdf',[0,0,.275])
    cube2 = p.loadURDF('cube_small.urdf',[.3,0,.275])
    cube3 = p.loadURDF('cube_small.urdf',[.6,0,.275])
    cube4 = p.loadURDF('cube_small.urdf',[.9,0,.275])
    cube5 = p.loadURDF('cube_small.urdf',[.0,0.3,.275])
    cube6 = p.loadURDF('cube_small.urdf',[.3,.3,.275])
    cube7 = p.loadURDF('cube_small.urdf',[.6,.3,.275])
    cube8 = p.loadURDF('cube_small.urdf',[.9,.3,.275])
    
    gripper_list = [hand_2f1j, hand_2f2j, hand_3f1j, hand_3f2j, gen1_3f_1j, gen2_3f_1j, gen1_3f_2j, gen2_3f_2j]

    joint_indices_list = []
    for gripper in gripper_list:  
        joint_indices = range(1, p.getNumJoints(gripper), 2)  
        p.setJointMotorControlArray(
            bodyIndex=gripper,
            jointIndices=joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            forces=np.zeros(len(joint_indices)),
        )
        joint_indices_list.append([i for i in joint_indices])

    while True:
        time.sleep(.1)
        angle = -.6
        if keyboard.is_pressed('r'):
                angle *= -0.41666666666
        for gripper, idx in zip (gripper_list,joint_indices_list):
            joint_states = p.getJointStates(gripper, idx)
            joint_positions = np.array([j[0] for j in joint_states])
            joint_velocity = np.array([j[1] for j in joint_states])
            #error = desired_joint_positions - joint_positions
            #torque = error * P_GAIN 
            torque = np.ones(len(idx))*angle
            p.setJointMotorControlArray(
                bodyIndex=gripper,
                jointIndices=idx,
                controlMode=p.POSITION_CONTROL,
                targetPositions=torque
            )
            
        

        p.stepSimulation()
if __name__=="__main__":
    main()