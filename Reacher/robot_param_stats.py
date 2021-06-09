import argparse
import json
import os
from datetime import datetime
from networkx.algorithms.smallworld import sigma

import numpy as np
from urdfpy import URDF
from tqdm import tqdm
from urdfpy.utils import parse_origin

from util import rotations


def relative_rotation(mat1, mat2):
    # return the euler x,y,z of the relative rotation
    # (w.r.t site1 coordinate system) from site2 to site1
    rela_mat = np.dot(np.linalg.inv(mat1), mat2)
    return rotations.mat2euler(rela_mat)

def get_obs(model, act_dim, type='dyn'):
    if type == 'dyn':
        return get_dyn(model, act_dim)


def get_dyn(sim, act_dim):
    body_mass = []
    for link in sim.links:
        body_mass.append(link.inertial.mass)
    friction = []
    damping = []
    for joint_num in range(act_dim):
        friction.append(sim.joints[joint_num].dynamics.friction)
        damping.append(sim.joints[joint_num].dynamics.damping)
    dyn_vec = np.concatenate((np.asarray(body_mass), 
                        np.asarray(friction), 
                        np.asarray(damping)))
    return dyn_vec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot Statistics')
    parser.add_argument('--robot_dir', 
                        default='../assets/generated/ur5_w_gripper',
                         type=str, help='path to robot configs')
    parser.add_argument('--save_dir', '-sd', type=str,
                         default='./stats',
                         help='directory to save triplet training data')
    parser.add_argument('--type', type=str, default='dyn', help='params type')
    print('Program starts at: \033[92m %s \033[0m' %
          datetime.now().strftime("%Y-%m-%d %H:%M"))
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 

    robot_folders = os.listdir(args.robot_dir)
    vecs = []
    for robot_folder in tqdm(robot_folders):
        robot_file = os.path.join(args.robot_dir,
                                  robot_folder,
                                  'model.urdf')
        model = URDF.load(robot_file)
        act_dim = 6
        vecs.append(get_obs(model, act_dim, type=args.type))
    vecs = np.array(vecs)
    mins = np.min(vecs, axis=0)
    maxs = np.max(vecs, axis=0)
    mus = np.mean(vecs, axis=0)
    sigmas = np.std(vecs, axis=0)
    print('Min: ', mins)
    print('Max: ', maxs)
    print('Mu: ', mus)
    print('Sigma: ', sigmas)
    stats = {'min': mins.tolist(),
             'max': maxs.tolist(),
             'mu': mus.tolist(),
             'sigma': sigmas.tolist()}
    with open(os.path.join(args.save_dir,
                           '%s_stats.json' % args.type),
              'w') as outfile:
        json.dump(stats, outfile, indent=4)
    print('Program ends at: \033[92m %s \033[0m' %
          datetime.now().strftime("%Y-%m-%d %H:%M"))
