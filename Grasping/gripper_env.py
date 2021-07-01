import json
import os

import numpy as np
import pybullet as p
import time

from env_base import BaseEnv

class GripperEnv(BaseEnv):
    def __init__(self,
                 robot_folders,
                 robot_dir,
                 render,
                 tol=0.02,
                 train=True,
                 with_kin=None):
        super().__init__(robot_folders=robot_folders,
                         robot_dir=robot_dir,
                         render=render,
                         tol=tol,
                         train=train,
                         with_kin=with_kin)

    def reset(self, robot_id=0): #change 0 to None for multi
        if robot_id is None:
            self.robot_id = np.random.randint(0, self.train_robot_num, 1)[0]
        else:
            self.robot_id = robot_id
        self.reset_robot(self.robot_id)

        ob = self.get_obs()
        self.ep_reward = 0
        self.ep_len =  0
        return ob
        
    def step(self, action):
        scaled_action = self.scale_action(action)
        hand_pose = np.array(p.getLinkState(self.sim, 7)[4])
        hand_pose += action[:3]
        desired_joint_positions = p.calculateInverseKinematics(
                                            self.sim, self.end_factor,
                                            hand_pose,[1,1,0,0],
                                            lowerLimits=self.ll, upperLimits=self.ul, residualThreshold=1e-5 )[:6]
        gripper1_pos = p.getJointState(self.sim,6)[0]
        gripper2_pos = p.getJointState(self.sim,7)[0]
        gripper_act = np.array([gripper1_pos + action[3]])*50
        print(gripper_act)
        desired_joint_positions = np.concatenate([desired_joint_positions, gripper_act, gripper_act])
        
        p.setJointMotorControlArray(
            bodyIndex=self.sim,
            jointIndices=self.act_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=desired_joint_positions
            #forces=torque,
        )
        p.stepSimulation()
        time.sleep(0.01)
        if self.testing:
            time.sleep(0.05)

        ob = self.get_obs()
        re_target = np.array([0,0,.3])
        reward, dist, done = self.cal_reward(ob[1],
                                             re_target,
                                             action)

        self.ep_reward += reward
        self.ep_len += 1
        info = {'reward_so_far': self.ep_reward, 'steps_so_far': self.ep_len,
                'reward': reward, 'step': 1, 'dist': dist}
        return ob, reward, done, info
            

