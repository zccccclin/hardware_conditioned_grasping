import json
import os
from typing import Sized

import numpy as np
import gym
from gym import spaces
import pybullet as p
from pybullet_utils import bullet_client
from urdfpy import  URDF
from util import rotations

class BaseEnv:

    def __init__(self,
                 robot_folders,
                 robot_dir,
                 render,
                 tol=0.02,
                 train=True,
                 with_dyn=None,
                 multi_goal=False,
                 ):
        self.with_dyn = with_dyn
        self.multi_goal = multi_goal
        self.goal_dim = 3

        if self.with_dyn:
            normal_file = os.path.join(robot_dir, 'stats/dyn_stats.json')
            with open(norm_file, 'r') as f:
                stats = json.load(f)
            self.dyn_mu = np.array(stats['mu'].reshape(-1))
            self.dyn_signma = np.array(stats['sigma']).reshape(-1)
            self.dyn_min = np.array(stats['min']).reshape(-1)
            self.dyn_max = np.array(stats['max']).reshape(-1)
        
        self.reward_range = (-np.inf, np.inf)
        self.spec = None
        self.dist_tol = tol
        self.pc = bullet_client.BulletClient(p.GUI if render else p.DIRECT)

        self.robots = []
        for folder in robot_folders:
            self.robots.append(os.path.join(robot_dir, folder))

        self.dir2id = {folder: idx for idx, folder in enumerate(self.robots)}
        self.robot_num = 1#len(self.robots)
        
        if train:
            self.test_robot_num = 1 #min(100, self.robot_num)
            self.train_robot_num = self.robot_num - self.test_robot_num
            self.test_robot_ids = list(range(self.train_robot_num,
                                             self.robot_num))
            self.train_test_robot_num = 1 #min(100, self.train_robot_num)
            self.train_test_robot_ids = list(range(self.train_test_robot_num))
            self.train_test_conditions = self.train_test_robot_num
        else:
            self.test_robot_num = self.robot_num
            self.train_robot_num = self.robot_num - self.test_robot_num
            self.test_robot_ids = list(range(self.robot_num))
            self.testing = True

        self.test_conditions = self.test_robot_num

        print('Train robots: ', self.train_robot_num)
        print('Test robots: ', self.test_robot_num)
        print('Multi goal:', self.multi_goal)
        self.reset_robot()#0, None)

        self.ob_dim = self.get_obs()[0].size
        print('Ob dim: ', self.ob_dim)

        high = np.inf * np.ones(self.ob_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.ep_reward = 0
        self.ep_len = 0

    def reset(self, robot_id=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def update_action_space(self):
        #for 6 axis arm joints only
        valid_joints = 6
        #for getting num of actuator joint including gripper
        #valid_joints = len(self.model_urdf.actuated_joints)
        '''
        bounds = self.sim_urdf.joint_limits[:valid_joints]
        self.ctrl_low = np.copy(bounds[:,0])
        self.ctrl_high = np.copy(bounds[:,1])
        self.action_space = spaces.Box(self.ctrl_low, self.ctrl_high, dtype=np.float32)
        '''
        self.ctrl_low = np.array([-80,-80,-40,-40,-9,-9])
        self.ctrl_high = np.array([80,80,40,40,9,9])
        self.action_space = spaces.Box(self.ctrl_low, self.ctrl_high, dtype=np.float32)


    def scale_action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/2.
        act_b = (self.action_space.high + self.action_space.low)/2.
        return act_k * action + act_b

    def reset_robot(self): #, robot_id, goal_pose):
        '''
        self.robot_folder_id = self.dir2id[self.robots[robot_id]]
        robot_file = os.path.join(self.robots[robot_id], 'model.urdf')
        p.resetSimulation()
        arm_base_pose = [0,0,0]
        plane_pose = [0,0,-.01]
        self.sim = p.loadURDF(robot_file, basePosition=arm_base_pose, useFixedBase=1, physicsClientId=self.pc._client)
        self.sim_urdf = URDF.load(robot_file)
        self.plane = p.loadURDF('../assets/plane.urdf', basePosition=plane_pose, physicsClientId=self.pc._client)
        if goal_pose is not None:
            self.goal = p.loadURDF('../assets/goal.urdf', basePosition=goal_pose, physicsClientId=self.pc._client)
        self.update_action_space()
        '''

        robot_file = '../assets/generated/ur5_w_gripper/robot_0/model.urdf'
        p.resetSimulation()
        arm_base_pose = [0,0,0]
        plane_pose = [0,0,-.01]
        self.sim = p.loadURDF(robot_file, basePosition=arm_base_pose, useFixedBase=1, physicsClientId=self.pc._client)
        self.sim_urdf = URDF.load(robot_file)
        self.plane = p.loadURDF('../assets/plane.urdf', basePosition=plane_pose, physicsClientId=self.pc._client)
        self.goal = p.loadURDF('../assets/goal.urdf', basePosition=[.5,.5,.5], physicsClientId=self.pc._client)
        self.update_action_space()

    def test_reset(self, cond):
        robot_id = self.test_robot_ids[cond]
        return self.reset() #robot_id=robot_id)

    def train_test_reset(self, cond):
        robot_id = self.train_test_robot_ids[cond]
        return self.reset() #robot_id=robot_id)

    def cal_reward(self, s, goal, a):
        dist = np.linalg.norm(s - goal) 
        if dist < self.dist_tol:
            done = True
            reward_dist = 1
        else:
            done = False
            reward_dist = -1
        reward = reward_dist
        reward -= 0.1 * np.square(a).sum()
        return reward, dist, done

    def get_obs(self):
        qpos = self.get_qpos(self.sim)
        qvel = self.get_qvel(self.sim)

        ob = np.concatenate([qpos,qvel])
        if self.with_dyn:
            dyn_vec = self.get_dyn(self.sim_urdf)
            dyn_vec = np.divide((dyn_vec - self.dyn_min),
                                self.dyn_max - self.dyn_min)
            ob = np.concatenate([ob, dyn_vec])

        ref_point = np.array(p.getLinkState(self.sim,6)[0])
        return ob, ref_point

    def get_qpos(self, sim):
        qpos = p.getBasePositionAndOrientation(sim)
        qpos = np.hstack(qpos)
        return qpos

    def get_qvel(self, sim):
        qvel = p.getBaseVelocity(sim)
        qvel = np.hstack(qvel)
        return qvel


    def get_dyn(self, sim):
        body_mass = []
        for link in sim.links:
            body_mass.append(link.inertial.mass)
        friction = []
        damping = []
        for joint in sim.joints:
            friction.append(joint.dynamics.friction)
            damping.append(joint.dynamics.damping)
        dyn_vec = np.concatenate((np.asarray(body_mass), 
                        np.asarray(friction), 
                        np.asarray(damping)))
        return dyn_vec

    def relative_rotation(self, mat1, mat2):
        # return the euler x,y,z of the relative rotation
        # (w.r.t site1 coordinate system) from site2 to site1
        rela_mat = np.dot(np.linalg.inv(mat1), mat2)
        return rotations.mat2euler(rela_mat)

    def close(self):
        pass