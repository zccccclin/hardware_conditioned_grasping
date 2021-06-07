import json
import os

import numpy as np

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
        def reset(self,  goal=None, robot_id=None):
            if  goal is None:
                goal = self.gen_random_goal()
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
            scaled_action = self.scale_action(action[:6])
            self.stepSimulation()
            for idx in range(1,p.getNumJoints(self.sim)):
                p.setJointMotorControl2(self.sim, idx, p.POSITION_CONTROL,
                                        targetPosition=action[idx])
            

        def gen_random_goal(self):
            radius = np.random.uniform(0,1.5)
            theta = np.random.uniform(0,2)*np.pi
            phi = np.random.uniform(0,1)*np.pi
            x = radius*np.cos(theta)*np.sin(phi)
            y = radius*np.sin(theta)*np.cos(phi)
            z = np.abs(radius*np.cos(phi))
            xyz = np.array([x,y,z])
            return xyz

            
