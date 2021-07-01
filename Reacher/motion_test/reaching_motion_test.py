import time

import numpy as np
import pybullet as p
import pybullet_data

P_GAIN = 50


def main():
    p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0,0,-9.81)
    robot_id = p.loadURDF("../../assets/ur5/model.urdf", useFixedBase=True)
    num_dofs = 6
    joint_indices = [0,1,2,3,4,5]

    # The magic that enables torque control

    goal = p.loadURDF('../../assets/goal.urdf',[1,1,1])
    ll = [-3.142,-3.14,-3.142,-3.142,-3.142,-3.142]
    #upper limits for null space (todo: set them to proper range)
    ul = [3.142,0,3.142,3.142,3.142,3.142,3.142]
    #joint ranges for null space (todo: set them to proper range)
    while True:
        x = np.random.uniform(0.3,0.8)
        y = np.random.uniform(-0.3,0.3)
        z = np.random.uniform(0.3,0.8)
        goal_pose = np.array([x,y,z])
        p.resetBasePositionAndOrientation(goal,goal_pose,[0,0,0,1])
        for i in range(100):
            time.sleep(0.01)

            joint_states = p.getJointStates(robot_id, joint_indices)
            joint_positions = np.array([j[0] for j in joint_states])
            desired_joint_positions = p.calculateInverseKinematics(robot_id,6,p.getBasePositionAndOrientation(goal)[0],[1,0,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            
            #error = desired_joint_positions - joint_positions
            #torque = error * P_GAIN
            for idx, pos in zip(joint_indices,desired_joint_positions):
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=pos
                    #forces=torque,
                )
            #print(p.getBasePositionAndOrientation(goal)[0], p.getLinkState(robot_id,6)[0])
            #print(desired_joint_positions, joint_positions)
            p.stepSimulation()

if __name__ == "__main__":
    main()
