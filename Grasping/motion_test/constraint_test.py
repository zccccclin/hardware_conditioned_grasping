import pybullet as p
import numpy as np
import time

pc = p.connect(p.GUI)
p.setGravity(0,0,-10)

arm = p.loadURDF('../../assets/ur5/model.urdf',useFixedBase=True)
pos = p.getLinkState(arm, 6)[4]
quat = [ -0.7071081, 0, 0, 0.7071055 ]
gripper = p.loadURDF('../../assets/grippers/3f_2j.urdf', pos, quat)
connection = p.createConstraint(arm, 6, gripper, -1, p.JOINT_FIXED, [0,0,0],[0,0,0], [0,0,0],quat)
p.changeConstraint(connection,maxForce=10000)
while True:
    time.sleep(0.01)
    p.stepSimulation()