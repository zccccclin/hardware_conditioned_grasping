import json
import os
from typing import Sized

import numpy as np
import gym
from gym import spaces
from numpy.random import f
import pybullet as p
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

        self.robots = []
        for folder in robot_folders:
            self.robots.append(os.path.join(robot_dir, folder))

        self.dir2id = {folder: idx for idx, folder in enumerate(self.robots)}
        self.robot_num = len(self.robots)
        self.act_joint_indices=[2,4,7,9,12,14]
        if train:
            self.test_robot_num = 1 #min(10, self.robot_num)
            self.train_robot_num = self.robot_num - self.test_robot_num
            self.test_robot_ids = list(range(self.train_robot_num,
                                             self.robot_num))
            self.train_test_robot_num = 1 #min(10, self.train_robot_num)
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
    
    '''
    def update_link_data(self):
        self.robot_name = p.getBodyInfo(self.sim)[1].decode('UTF-8')
        self.link_name_dict = OrderedDict()
        for id in range(p.getNumJoints(self.sim)):
            name = p.getJointInfo(self.sim, id)[12].decode('UTF-8')
            if len(name) < 4:
                self.link_name_dict[name] = id
        f11_v_idx = self.link_name_dict['f11'] + 1
        self.fingers_total_height = p.getVisualShapeData(self.sim)[f11_v_idx][3][1]
        if '2j' in self.robot_name:
            f12_v_idx = self.link_name_dict['f12'] + 1
            self.fingers_total_height += p.getVisualShapeData(self.sim)[f12_v_idx][3][1]
    '''
    
    def update_action_space(self):
        self.ctrl_high = np.ones(2) 
        self.ctrl_low = -self.ctrl_high
        self.action_space = spaces.Box(self.ctrl_low, self.ctrl_high, dtype=np.float32)


    def scale_action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/2.
        act_b = (self.action_space.high + self.action_space.low)/2.
        #print(act_k * action + act_b)
        #return act_k * action + act_b
        return [.01 if b>0 else -.01 for b in action]

    def reset_robot(self, robot_id):
        
        #self.robot_folder_id = self.dir2id[self.robots[robot_id]]
        #robot_file = os.path.join(self.robots[robot_id], 'model.urdf')
        robot_file = '../assets/grippers/one_direction.urdf'
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        robot_pose = [0,0,0.1]
        cube_pose = [0, 0, 0.03]

        self.sim = p.loadURDF(robot_file, basePosition=robot_pose, baseOrientation=[1, 1, 0, 0],
                              useFixedBase=1, physicsClientId=self.pc._client)
        p.setJointMotorControl2(self.sim,0,p.POSITION_CONTROL,targetPosition=-.1)
        p.stepSimulation()
        self.plane = p.loadURDF("plane.urdf")
        self.cube = p.loadURDF('cube_small.urdf', cube_pose,) #globalScaling=2)

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
        if dist < self.dist_tol and reached < 0.065:
            done = True
            reward_dist = 10 #1
        elif reached < 0.075:
            done = False
            reward_dist = 1#-dist
        elif reached > 0.2:
            done=False
            reward_dist= -10
        else:
            done = False
            reward_dist = -1 #-reached + -dist
        reward = reward_dist
        final_dist = [reached,dist]
        #print(reward)
        #reward -= 0.1 * np.square(a).sum()

        return reward, final_dist, done

    def get_obs(self):
        joint_states = p.getJointStates(self.sim, self.act_joint_indices)
        gripper_qpos = np.array([j[0] for j in joint_states])

        height_target = np.array([0, 0, .25])
        end_factor = np.array(p.getLinkState(self.sim,0)[4])

        
        ref_point = np.array(p.getBasePositionAndOrientation(self.cube)[0])
        ob = np.concatenate([gripper_qpos, height_target])
        ref_point = np.concatenate([end_factor,ref_point])
        return ob, ref_point

    def get_qpos_qvel(self, sim):
        joint_states = p.getJointStates(sim, self.joint_indices_list)
        qpos = np.array([j[0] for j in joint_states])
        angle_noise_range = 0.02
        qpos += np.random.uniform(-angle_noise_range,
                                  angle_noise_range,
                                  6)
        qvel = np.array([j[1] for j in joint_states])
        velocity_noise_range = 0.02
        qvel += np.random.uniform(-velocity_noise_range,
                                  velocity_noise_range,
                                  6)
        return qpos, qvel




    def relative_rotation(self, mat1, mat2):
        # return the euler x,y,z of the relative rotation
        # (w.r.t site1 coordinate system) from site2 to site1
        rela_mat = np.dot(np.linalg.inv(mat1), mat2)
        return rotations.mat2euler(rela_mat)

    def close(self):
        pass
