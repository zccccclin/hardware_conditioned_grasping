import json
import os
from typing import Sized

import numpy as np
import gym
from gym import spaces
from numpy.random import f
import pybullet as p
from urdfpy import URDF
from pybullet_utils import bullet_client
from util import rotations
import pybullet_data
from collections import OrderedDict


class BaseEnv:

    def __init__(self,
                 robot_folders,
                 robot_dir,
                 render,
                 tol=0.02,
                 train=True,
                 with_kin=None,
                 ):
        self.with_kin = with_kin
        self.goal_dim = 6


        self.reward_range = (-np.inf, np.inf)
        self.spec = None
        self.dist_tol = tol
        self.pc = bullet_client.BulletClient(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ll = [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14]
        self.ul = [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
        self.end_factor = 7
        self.finger_height_offset = 0
        self.robots = []
        for folder in robot_folders:
            self.robots.append(os.path.join(robot_dir, folder))

        self.dir2id = {folder: idx for idx, folder in enumerate(self.robots)}
        self.robot_num = len(self.robots)
        
        if train:
            self.test_robot_num = min(10, self.robot_num)
            self.train_robot_num = self.robot_num - self.test_robot_num
            self.test_robot_ids = list(range(self.train_robot_num,
                                             self.robot_num))
            self.train_test_robot_num = min(10, self.train_robot_num)
            self.train_test_robot_ids = list(range(self.train_test_robot_num))
            self.train_test_conditions = self.train_test_robot_num
            self.testing = False
        else:
            self.test_robot_num = self.robot_num
            self.train_robot_num = self.robot_num - self.test_robot_num
            self.test_robot_ids = list(range(self.robot_num))
            self.testing = True

        self.test_conditions = self.test_robot_num

        print('Train robots: ', self.train_robot_num)
        print('Test robots: ', self.test_robot_num)
        self.reset_robot(0)

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
    
    def reset_robot_pose(self, robot_id, act_joint_indices):
        print(act_joint_indices)
        desired_joint_positions = [-0.197, -1.17, 1.661, -2.06, -1.5715, 1.37]
        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=act_joint_indices[:6],
            controlMode=p.POSITION_CONTROL,
            targetPositions=desired_joint_positions
            #forces=torque,
        )
        for i in range(10):
            p.stepSimulation()
        desired_joint_positions = p.calculateInverseKinematics(
                                            self.sim, self.end_factor,
                                            [.65, 0, .2-self.finger_height_offset],[1,1,0,0],
                                            lowerLimits=self.ll, upperLimits=self.ul,  residualThreshold=1e-5)[:6]
        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=act_joint_indices[:6],
            controlMode=p.POSITION_CONTROL,
            targetPositions=desired_joint_positions
            #forces=torque,
        )
        for i in range(10):
            p.stepSimulation()
        return desired_joint_positions

    
    def update_action_space(self):
        self.ctrl_high = np.ones(4) * .05
        self.ctrl_low = -self.ctrl_high
        self.action_space = spaces.Box(self.ctrl_low, self.ctrl_high, dtype=np.float32)


    def scale_action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/2.
        act_b = (self.action_space.high + self.action_space.low)/2.
        #print(act_k * action + act_b)
        return act_k * action + act_b
        #return [.01 if b>0 else -.01 for b in action]

    def reset_robot(self, robot_id):
        
        self.robot_folder_id = self.dir2id[self.robots[robot_id]]
        robot_file = os.path.join(self.robots[robot_id], 'model.urdf')
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        robot_pose = [0,0,0]
        cube_pose = [.65, 0, 0.025]        
        self.sim = p.loadURDF(robot_file, basePosition=robot_pose,
                              useFixedBase=1, physicsClientId=self.pc._client)
        self.sim_urdf = URDF.load(robot_file)
        self.robot_name = self.sim_urdf.name
        if '2f1j' in self.robot_name:
            self.act_joint_indices = [0,1,2,3,4,5,9,12]
        elif '2f2j' in self.robot_name:
            self.act_joint_indices = [0,1,2,3,4,5,9,11,14,16]
        elif '3f1j' in self.robot_name:
            self.act_joint_indices = [0,1,2,3,4,5,9,12,15]
        elif '3f2j' in self.robot_name:
            self.act_joint_indices = [0,1,2,3,4,5,9,11,14,16,19,21]
        #pasd = []
        #for i in range(p.getNumJoints(self.sim)):
            #pasd.append((p.getJointInfo(self.sim,i)[0],p.getJointInfo(self.sim,i)[1]))
        #print(pasd)
        self.reset_robot_pose(self.sim, self.act_joint_indices)
        self.plane = p.loadURDF("plane.urdf")
        #self.tray = p.loadURDF('tray/tray.urdf', [.7, 0, 0],[0,0,1,1],useFixedBase=True,)

        self.cube = p.loadURDF('cube_small.urdf', cube_pose, globalScaling=1.3)
        if self.with_kin:
            links_l, links_t = self.get_link_properties()
            self.link_prop = np.concatenate([links_l, links_t])
            self.finger_height_offset = .1 - links_l[0] - links_l[1]
        for i in self.act_joint_indices[6:]:
            p.changeDynamics(robot_id, i, lateralFriction=5)
        self.update_action_space()

    def test_reset(self, cond):
        robot_id = self.test_robot_ids[cond]
        return self.reset(robot_id=robot_id)

    def train_test_reset(self, cond):
        robot_id = self.train_test_robot_ids[cond]
        return self.reset(robot_id=robot_id)

    def cal_reward(self, s, goal, a ):
        reached = np.linalg.norm(s[:3] - goal[:3])
        dist = np.linalg.norm(s[3:] - goal[3:])
        flooring = len(p.getContactPoints(self.sim, self.plane))
        contact_pts = len(p.getContactPoints(self.sim, self.cube))
        link_set = set({})
        if  contact_pts!= 0:
            for i in range(contact_pts):
                link_set.add(p.getContactPoints(self.sim, self.cube)[i][3])

        
        if dist < self.dist_tol:
            done = True
            reward_dist = 10
        elif flooring != 0:
            done = False
            reward_dist = -10
        elif dist < .1:
            done = False
            reward_dist = 5-dist
        elif contact_pts!=0:
            done = False
            if '1j' in self.robot_name:
                if 7 in link_set:
                    reward_dist = 1-dist
                elif (9 in link_set) or (12 in link_set) or (15 in link_set):
                    reward_dist = .5-dist
                else:
                    reward_dist = .25-dist
            elif '2j' in self.robot_name:
                if (9 in link_set) or (14 in link_set) or (19 in link_set):
                    reward_dist = 1-dist
                elif (11 in link_set) or (16 in link_set) or (21 in link_set):
                    reward_dist = .5-dist
                else:
                    reward_dist = .25-dist

        elif np.linalg.norm(s[:2] - goal[:2]) < .02:
            done = False
            reward_dist = -(s[3]-goal[3])-dist
        else:
            done = False
            reward_dist = -reached -dist
        reward = reward_dist
        final_dist = [reached,dist]
        #print(reward)
        #reward -= 0.1 * np.square(a).sum()
        #print(a)
        

        
        return reward, final_dist, done

    def get_obs(self):

        endfactor_pos  = np.array((p.getLinkState(self.sim, self.end_factor)[4]))
        endfactor_pos = endfactor_pos - np.array([0,0,self.finger_height_offset])
        joint_states = p.getJointStates(self.sim, self.act_joint_indices[6:])
        g = [j[0] for j in joint_states]
        if '2f1j' in self.robot_name:
            self.act_joint_indices = [0,1,2,3,4,5,9,12]
            gripper_qpos = np.array([g[0], 0.0, g[1], 0.0, 0.0, 0.0])
        elif '2f2j' in self.robot_name:
            gripper_qpos = np.array([g[0], g[1], g[2], g[3], 0.0, 0.0])
        elif '3f1j' in self.robot_name:
            gripper_qpos = np.array([g[0], 0.0, g[1], 0.0, g[2], 0.0])
        elif '3f2j' in self.robot_name:
            gripper_qpos = np.array([g[0], g[1], g[2], g[3], g[4], g[5]])
        print(gripper_qpos)

        height_target = np.array([.65, 0,.2])
        if self.with_kin:
            ob = self.link_prop
        ref_point = np.array(p.getBasePositionAndOrientation(self.cube)[0])
        ob = np.concatenate([ob, gripper_qpos, endfactor_pos, ref_point, endfactor_pos, height_target])
        ref_point = np.concatenate([endfactor_pos,ref_point])
        return ob, ref_point

    def get_link_properties(self):
            size = [link.visuals[0].geometry.box.size for link in self.sim_urdf.links if len(link.name)==3]
            if '2f1j' in self.robot_name:
                links_t = [size[0][0], size[0][1], 0.0, 0.0, size[1][0], size[1][1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                links_l = [size[0][2], 0.0, size[1][2], 0.0, 0.0, 0.0]
            elif '2f2j' in self.robot_name:
                links_t = [size[0][0], size[0][1], size[1][0], size[1][1], size[2][0], size[2][1], size[3][0], size[3][1], 0.0, 0.0, 0.0, 0.0]
                links_l = [size[0][2], size[1][2], size[2][2], size[3][2], 0.0, 0.0]
            elif '3f1j' in self.robot_name:
                links_t = [size[0][0], size[0][1], 0.0, 0.0, size[1][0], size[1][1], 0.0, 0.0, size[2][0], size[2][1], 0.0, 0.0]
                links_l = [size[0][2], 0.0, size[1][2], 0.0, size[2][2], 0.0]
            elif '3f2j' in self.robot_name:
                links_t = [size[0][0], size[0][1], size[1][0], size[1][1], size[2][0], size[2][1], size[3][0], size[3][1], size[4][0], size[4][1], size[5][0], size[5][1]]
                links_l = [size[0][2], size[1][2], size[2][2], size[3][2], size[4][2], size[5][2]]

            links_t = np.array(links_t)
            links_l = np.array(links_l)
            return links_l, links_t

    def close(self):
        pass
