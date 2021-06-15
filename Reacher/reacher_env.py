import json
import os

import numpy as np
import pybullet as p
import time

from env_base import BaseEnv

class ReacherEnv(BaseEnv):
    def __init__(self,
                 robot_folders,
                 robot_dir,
                 render,
                 tol=0.02,
                 train=True,
                 with_dyn=None,
                 multi_goal=False):
        super().__init__(robot_folders=robot_folders,
                         robot_dir=robot_dir,
                         render=render,
                         tol=tol,
                         train=train,
                         with_dyn=with_dyn,
                         multi_goal=multi_goal)
    def reset(self, goal_pose=None, robot_id=None): #):
        
        if  goal_pose is None:
            goal_pose = self.gen_random_goal()
        if robot_id is None:
            self.robot_id = np.random.randint(0, self.train_robot_num, 1)[0]
        else:
            self.robot_id = robot_id
        self.reset_robot(self.robot_id, goal_pose)

        
        #self.reset_robot()
        ob = self.get_obs()
        self.ep_reward = 0
        self.ep_len =  0
        return ob
        
    def step(self, action):
        scaled_action = self.scale_action(action[:6])
        joint_indices =  range(6)
        p.setJointMotorControlArray(
            bodyIndex=self.sim,
            jointIndices=joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            forces=np.zeros(6),
        )
        p.setJointMotorControlArray(
            bodyIndex=self.sim,
            jointIndices=joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=scaled_action,
        )
        p.stepSimulation()
        if self.testing:
            time.sleep(0.1)


        ob = self.get_obs()
        re_target = np.array(p.getBasePositionAndOrientation(self.goal)[0])
        reward, dist, done = self.cal_reward(ob[1],
                                             re_target,
                                             action)

        self.ep_reward += reward
        self.ep_len += 1
        info = {'reward_so_far': self.ep_reward, 'steps_so_far': self.ep_len,
                'reward': reward, 'step': 1, 'dist': dist}
        return ob, reward, done, info
            
    def gen_random_goal(self):
        radius = np.random.uniform(0,0.75)
        theta = np.random.uniform(0,2)*np.pi
        phi = np.random.uniform(0,1)*np.pi
        x = radius*np.cos(theta)*np.sin(phi)
        y = radius*np.sin(theta)*np.cos(phi)
        z = radius*np.cos(phi)
        xyz = np.array([x,y,z])
        return xyz

            
