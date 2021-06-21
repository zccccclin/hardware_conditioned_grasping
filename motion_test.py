import time

import numpy as np
import pybullet as p
import pybullet_data

P_GAIN = 50
desired_joint_positions = np.array([0, 0, 0, 0, 0, 0])

def main():
    p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0,0,-9.81)
    robot_id = p.loadURDF("assets/ur5_w_gripper/model.urdf", useFixedBase=True)
    x = np.random.uniform(0.3,0.8)
    y = np.random.uniform(-0.3,0.3)
    z = np.random.uniform(0.2,0.6)
    goal_pose = np.array([x,y,z])
    goal = p.loadURDF('assets/goal.urdf',goal_pose)
    num_dofs = 6
    joint_indices = range(num_dofs)

    # The magic that enables torque control
    p.setJointMotorControlArray(
        bodyIndex=robot_id,
        jointIndices=joint_indices,
        controlMode=p.VELOCITY_CONTROL,
        forces=np.zeros(num_dofs),
    )

    while True:
        time.sleep(0.01)

        joint_states = p.getJointStates(robot_id, joint_indices)
        joint_positions = np.array([j[0] for j in joint_states])
        joint_velocity = np.array([j[1] for j in joint_states])
        error = desired_joint_positions - joint_positions
        torque = error * P_GAIN 
        print(torque)

        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=[0,-100,0,0,0,0]
        )

        p.stepSimulation()

if __name__ == "__main__":
    main()
