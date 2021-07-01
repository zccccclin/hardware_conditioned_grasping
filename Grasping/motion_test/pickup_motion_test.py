import time

import numpy as np
import pybullet as p
import pybullet_data
import keyboard

def reset(robot_id, act_joint_indices):
    desired_joint_positions = [-0.184, -1.09, 1.497, -1.98, -1.5715, 1.38]
    p.setJointMotorControlArray(
        bodyIndex=robot_id,
        jointIndices=act_joint_indices[:6],
        controlMode=p.POSITION_CONTROL,
        targetPositions=desired_joint_positions
        #forces=torque,
    )
    p.stepSimulation()
    return desired_joint_positions

def main(robot):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0,0,-9.81)
    robot_pos = [0,0,0]
    cube = p.loadURDF('cube_small.urdf', [.65,0,0])
    ll = [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14]
    #upper limits for null space (todo: set them to proper range)
    ul = [3.14, 0, 3.14, 3.14, 3.14, 3.14]
    end_factor = 7
    simulate = False

    if robot == 1:
        robot_id = p.loadURDF("../../assets/ur5_w_gripper/2f_1j.urdf", robot_pos, useFixedBase=True)
        act_joint_indices = [0,1,2,3,4,5,9,11]
        simulate=True
    elif robot == 2:
        robot_id = p.loadURDF("../../assets/ur5_w_gripper/2f_2j.urdf", robot_pos, useFixedBase=True)
        act_joint_indices = [0,1,2,3,4,5,9,11,13,15]
        simulate=True
    elif robot == 3:
        robot_id = p.loadURDF("../../assets/ur5_w_gripper/3f_1j.urdf", robot_pos, useFixedBase=True)
        act_joint_indices = [0,1,2,3,4,5,9,11,13]
        simulate=True
    elif robot == 4:
        robot_id = p.loadURDF("../../assets/ur5_w_gripper/3f_2j.urdf", robot_pos, useFixedBase=True)
        act_joint_indices = [0,1,2,3,4,5,9,11,13,15,17,19]
        simulate=True
    desired_joint_positions = reset(robot_id,act_joint_indices)
    gripper_act = np.zeros(len(act_joint_indices[6:]))
    desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])



    while simulate:
        time.sleep(0.01)
        move_factor = 0.01
        joint_states = p.getJointStates(robot_id, act_joint_indices)
        hand_pose = list(p.getLinkState(robot_id, 7)[4])
        if keyboard.is_pressed('i'):
            hand_pose[0] += move_factor
            desired_joint_positions = p.calculateInverseKinematics(robot_id,end_factor,hand_pose,[1,1,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])

        elif keyboard.is_pressed('k'):
            hand_pose[0] -= move_factor
            desired_joint_positions = p.calculateInverseKinematics(robot_id,end_factor,hand_pose,[1,1,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])

        elif keyboard.is_pressed('j'):
            hand_pose[1] -= move_factor
            desired_joint_positions = p.calculateInverseKinematics(robot_id,end_factor,hand_pose,[1,1,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])

        elif keyboard.is_pressed('l'):
            hand_pose[1] += move_factor
            desired_joint_positions = p.calculateInverseKinematics(robot_id,end_factor,hand_pose,[1,1,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])

        elif keyboard.is_pressed('8'):
            hand_pose[2] += move_factor
            desired_joint_positions = p.calculateInverseKinematics(robot_id,end_factor,hand_pose,[1,1,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])

        elif keyboard.is_pressed('m'):
            hand_pose[2] -= move_factor
            desired_joint_positions = p.calculateInverseKinematics(robot_id,7,hand_pose,[1,1,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])

        elif keyboard.is_pressed('f'):
            desired_joint_positions = p.calculateInverseKinematics(robot_id,7,hand_pose,[1,1,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            gripper_act = np.ones(len(gripper_act))*-.6
            desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])
        elif keyboard.is_pressed('r'):
            desired_joint_positions = p.calculateInverseKinematics(robot_id,7,hand_pose,[1,1,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            gripper_act = np.ones(len(gripper_act))*.6
            desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])

        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=act_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=desired_joint_positions
            #forces=torque,
        )
        #print(p.getBasePositionAndOrientation(goal)[0], p.getLinkState(robot_id,6)[0])
        #print(desired_joint_positions, joint_positions)
        p.stepSimulation()

if __name__ == "__main__":
    print('enter robot num: ')
    print('1 - 2f_1j')
    print('2 - 2f_2j')
    print('3 - 3f_1j')
    print('4 - 3f_2j')
    robot = int(input())
    main(robot)