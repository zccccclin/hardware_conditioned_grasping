import pybullet as p
import numpy as np
import time
import pybullet_data

pc = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

arm = p.loadURDF('../ur5/model.urdf', useFixedBase=1, )
linkpos = p.getLinkState(arm,6)[4]
p.loadURDF("plane.urdf")

hand = p.loadURDF('one_direction.urdf',linkpos)
cube = p.loadURDF('cube_small.urdf', [.65,0,0])

p.createConstraint(arm,6,hand,-1,p.JOINT_FIXED,[0,1,0],[0,0,0],[0,0,.1])
joint_indices = [0,1,2,3,4,5]
p.setJointMotorControlArray(
    bodyIndex=arm,
    jointIndices=joint_indices,
    controlMode=p.VELOCITY_CONTROL,
    forces=np.zeros(len(joint_indices)),
)
count = 0
while True:
    p.setGravity(0,0,-10)  
    count+= 1
    if count >800:
        time.sleep(10000)
    p.stepSimulation()
    time.sleep(0.1)