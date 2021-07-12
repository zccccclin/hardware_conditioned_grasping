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
        self.ll = [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14]
        self.ul = [3.14, 0, 3.14, 3.14, 3.14, 3.14]
        self.end_factor = 7

        self.robots = []
        for folder in robot_folders:
            self.robots.append(os.path.join(robot_dir, folder))

        self.dir2id = {folder: idx for idx, folder in enumerate(self.robots)}
        self.robot_num = len(self.robots)
        
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
    
    def reset_robot_pose(self, robot_id, act_joint_indices):
        desired_joint_positions = [-0.184, -1.09, 1.497, -1.98, -1.5715, 1.38]
        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=act_joint_indices[:6],
            controlMode=p.POSITION_CONTROL,
            targetPositions=desired_joint_positions
            #forces=torque,
        )
        for i in range(20):
            p.stepSimulation()
        return desired_joint_positions
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
        self.ctrl_high = np.ones(4)
        self.ctrl_low = -self.ctrl_high
        self.action_space = spaces.Box(self.ctrl_low, self.ctrl_high, dtype=np.float32)


    def scale_action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/2.
        act_b = (self.action_space.high + self.action_space.low)/2.
        #print(act_k * action + act_b)
        return act_k * action + act_b

    def reset_robot(self, robot_id):
        
        #self.robot_folder_id = self.dir2id[self.robots[robot_id]]
        #robot_file = os.path.join(self.robots[robot_id], 'model.urdf')
        robot_file = '../assets/ur5_w_gripper/3f_2j.urdf'
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        robot_pose = [0,0,0]
        cube_pose = [.65, 0, 0.03]
        self.act_joint_indices = [0,1,2,3,4,5,9,11,14,16,19,21]

        self.sim = p.loadURDF(robot_file, basePosition=robot_pose,
                              useFixedBase=1, physicsClientId=self.pc._client)
        self.reset_robot_pose(self.sim, self.act_joint_indices)
        self.plane = p.loadURDF("plane.urdf")
        #self.tray = p.loadURDF('tray/tray.urdf', [.7, 0, 0],[0,0,1,1],useFixedBase=True,)

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
        if reached < self.dist_tol and dist < self.dist_tol:
            done = True
            reward_dist = 1
        else:
            done = False
            reward_dist = -1
        reward = reward_dist

        reward -= 0.1 * np.square(a).sum()
        return reward, dist, done

    def get_obs(self):
        endfactor_pos  = np.array((p.getLinkState(self.sim, self.end_factor)[4]))
        joint_states = p.getJointStates(self.sim, self.act_joint_indices[6:])
        gripper_qpos = np.array([j[0] for j in joint_states])

        height_target = np.array([0,0,.25])
        ob = np.concatenate([gripper_qpos, endfactor_pos, height_target])

        
        ref_point = np.array(p.getBasePositionAndOrientation(self.cube)[0])
        ref_point = np.concatenate([endfactor_pos,ref_point])
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

    def get_xpos_xrot(self, sim):
        xpos = []
        xrot = []
        for joint_id in range(self.act_dim):
            joint = sim.model._actuator_id2name[joint_id]
            if joint == 'j0':   
                pos1 = sim.data.get_body_xpos('base_link')
                mat1 = sim.data.get_body_xmat('base_link')
            else:
                prev_id = joint_id - 1
                prev_joint = sim.model._actuator_id2name[prev_id]
                pos1 = sim.data.get_site_xpos(prev_joint)
                mat1 = sim.data.get_site_xmat(prev_joint)
            pos2 = sim.data.get_site_xpos(joint)
            mat2 = sim.data.get_site_xmat(joint)
            relative_pos = pos2 - pos1
            rot_euler = self.relative_rotation(mat1, mat2)
            xpos.append(relative_pos)
            xrot.append(rot_euler)
        xpos = np.array(xpos).flatten()
        xrot = np.array(xrot).flatten()
        xpos = np.pad(xpos, (0, (7 - self.act_dim) * 3),
                      mode='constant', constant_values=0)
        xrot = np.pad(xrot, (0, (7 - self.act_dim) * 3),
                      mode='constant', constant_values=0)
        ref_pt_xpos = self.sim.data.get_site_xpos('ref_pt')
        ref_pt_xmat = self.sim.data.get_site_xmat('ref_pt')
        relative_pos = ref_pt_xpos - pos2
        rot_euler = self.relative_rotation(mat2, ref_pt_xmat)
        xpos = np.concatenate((xpos, relative_pos.flatten()))
        xrot = np.concatenate((xrot, rot_euler.flatten()))
        pos_rot = np.concatenate((xpos, xrot))
        return pos_rot


    def relative_rotation(self, mat1, mat2):
        # return the euler x,y,z of the relative rotation
        # (w.r.t site1 coordinate system) from site2 to site1
        rela_mat = np.dot(np.linalg.inv(mat1), mat2)
        return rotations.mat2euler(rela_mat)

    def close(self):
        pass
