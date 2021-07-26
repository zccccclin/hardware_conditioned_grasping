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
    for i in range(20):
        p.stepSimulation()
    return desired_joint_positions

def main(robot):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    robot_pos = [0,0,0]
    ll = [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14]
    #upper limits for null space (todo: set them to proper range)
    ul = [3.14, 0, 3.14, 3.14, 3.14, 3.14]
    end_factor = 7
    simulate = False

    if robot == 1:
        robot_id = p.loadURDF("../../assets/multi_robot_gen_gripper/robot_2f_1j_0/model.urdf", robot_pos, useFixedBase=True)
        act_joint_indices = [0,1,2,3,4,5,9,12]
        for i in [9,12]:
            p.changeDynamics(robot_id, i, lateralFriction=5)
        simulate=True
    elif robot == 2:
        robot_id = p.loadURDF("../../assets/multi_robot_gen_gripper/robot_2f_2j_0/model.urdf", robot_pos, useFixedBase=True)
        act_joint_indices = [0,1,2,3,4,5,9,11,14,16]
        for i in act_joint_indices[6:]:
            p.changeDynamics(robot_id, i, lateralFriction=5)
        simulate=True
    elif robot == 3:
        robot_id = p.loadURDF("../../assets/multi_robot_gen_gripper/robot_3f_1j_0/model.urdf", robot_pos, useFixedBase=True)
        act_joint_indices = [0,1,2,3,4,5,9,12,15]
        for i in [9,12,15]:
            p.changeDynamics(robot_id, i, lateralFriction=5)
        simulate=True
    elif robot == 4:
        robot_id = p.loadURDF("../../assets/multi_robot_gen_gripper/robot_3f_2j_0/model.urdf", robot_pos, useFixedBase=True)
        act_joint_indices = [0,1,2,3,4,5,9,11,14,16,19,21]
        for i in act_joint_indices[6:]:
            p.changeDynamics(robot_id, i, lateralFriction=5)
        simulate=True

    desired_joint_positions = reset(robot_id,act_joint_indices)
    gripper_act = np.zeros(len(act_joint_indices[6:]))
    plane = p.loadURDF("plane.urdf")

    cube = p.loadURDF('cube_small.urdf', [.65,0,0], globalScaling=1.2)

    desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])

    #for i in range(p.getNumJoints(robot_id)):
        #print(p.getJointInfo(robot_id, i))
    _link_name_to_index = {p.getBodyInfo(robot_id)[0].decode('UTF-8'):-1,}
        
    for _id in range(p.getNumJoints(robot_id)):
        _name = p.getJointInfo(robot_id, _id)[12].decode('UTF-8')
        _link_name_to_index[_name] = _id
    print(_link_name_to_index)
    height = .1-0.03 -0.0
    while simulate:
        time.sleep(0.01)
        move_factor = 0.01
        joint_states = p.getJointStates(robot_id, act_joint_indices)
        hand_pose = list(p.getLinkState(robot_id, 7)[4])
        if keyboard.is_pressed('i') and hand_pose[0] <= .725:
            hand_pose[0] += move_factor
            desired_joint_positions = p.calculateInverseKinematics(robot_id,end_factor,hand_pose,[1,1,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])

        elif keyboard.is_pressed('k') and hand_pose[0] >= .575:
            hand_pose[0] -= move_factor
            desired_joint_positions = p.calculateInverseKinematics(robot_id,end_factor,hand_pose,[1,1,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])

        elif keyboard.is_pressed('j') and hand_pose[1] >= -.15:
            hand_pose[1] -= move_factor
            desired_joint_positions = p.calculateInverseKinematics(robot_id,end_factor,hand_pose,[1,1,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])

        elif keyboard.is_pressed('l') and hand_pose[1] <= .15:
            hand_pose[1] += move_factor
            desired_joint_positions = p.calculateInverseKinematics(robot_id,end_factor,hand_pose,[1,1,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])

        elif keyboard.is_pressed('8') and hand_pose[2] <= (.5 - height):
            hand_pose[2] += move_factor
            desired_joint_positions = p.calculateInverseKinematics(robot_id,end_factor,hand_pose,[1,1,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            desired_joint_positions = np.concatenate([desired_joint_positions,gripper_act])

        elif keyboard.is_pressed('m') and hand_pose[2] >= (.1 - height):
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
            targetPositions=desired_joint_positions,
            #forces=0
        )
        end_factor_pos = np.array(list(p.getLinkState(robot_id,6)[4]))
        goal = np.array(list(p.getBasePositionAndOrientation(cube)[0]))
        dist = np.linalg.norm(end_factor_pos - goal)
        #print(dist)
        #print(np.linalg.norm(hand_pose[:2]-goal[:2]))
        contact_pt = len(p.getContactPoints(robot_id, cube))
        set1 = set({})
        if  contact_pt != 0 :
            for i in range(contact_pt):
                print(f'contact {i}: ', p.getContactPoints(robot_id, cube)[i])
        #print(p.getContactPoints(robot_id, plane))
        #print(desired_joint_positions)
        p.stepSimulation()

if __name__ == "__main__":
    print('enter robot num: ')
    print('1 - 2f_1j')
    print('2 - 2f_2j')
    print('3 - 3f_1j')
    print('4 - 3f_2j')
    robot = int(input())
    main(robot)
