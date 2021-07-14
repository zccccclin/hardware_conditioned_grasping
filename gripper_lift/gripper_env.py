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
        hand_height = np.array(p.getJointState(self.sim, 0)[0])
        #print(hand_height, scaled_action)

        if -.3 <= (hand_height + scaled_action[0]) <= .01:
            hand_height += scaled_action[0]
        #print(scaled_action)
        p.setJointMotorControl2(self.sim, 0, p.POSITION_CONTROL, targetPosition=hand_height)
        gripper_pos = p.getJointState(self.sim,2)[0]
        gripper_act = np.array([gripper_pos + scaled_action[1]*10])* np.ones(len(self.act_joint_indices))
        p.setJointMotorControlArray(
            bodyIndex=self.sim,
            jointIndices=self.act_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=gripper_act
            #forces=torque,
        )
        p.stepSimulation()
        time.sleep(0.01)
        if self.testing:
            time.sleep(0.05)

        ob = self.get_obs()
        re_target = np.array([0,0,.25])
        re_target = np.concatenate([ob[1][3:], re_target])
        reward, dist, done = self.cal_reward(ob[1],
                                             re_target,
                                             action)

        self.ep_reward += reward
        self.ep_len += 1
        info = {'reward_so_far': self.ep_reward, 'steps_so_far': self.ep_len,
                'reward': reward, 'step': 1, 'dist': dist}
        return ob, reward, done, info
            

