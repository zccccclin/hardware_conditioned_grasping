import urdfpy
from urdfpy import URDF

import argparse
import copy
import os
import time

import numpy as np
from tqdm import tqdm
from urdfpy.urdf import Link


def robot_param_random_sample(robot, pre_gen_params, vars_to_change):
    link_idx = [robot.links.index(link) for link in robot.links if len(link.name)==3]
    joint_idx = [robot.joints.index(joint) for joint in robot.joints if '_slot_joint' in joint.name]
    length_id = np.random.randint(0, len(pre_gen_params['length']),1)[0]
    radius_id = np.random.randint(0, len(pre_gen_params['radius']),1)[0]
    length = pre_gen_params['length'][length_id]
    radius = pre_gen_params['radius'][radius_id]
    if '1j' in robot.name:
        #diff layer finger thickness:
        for idx in link_idx:
            if vars_to_change['layer_rad']:
                radius_id =np.random.randint(0, len(pre_gen_params['radius']),1)[0]
                radius = pre_gen_params['radius'][radius_id]
            geo_change(robot, idx, radius, length)
    elif '2j' in robot.name:
        #1st layer:
        for f_idx , j_idx in zip(link_idx[::2],joint_idx):
            if vars_to_change['layer_rad']:
                radius_id =np.random.randint(0, len(pre_gen_params['radius']),1)[0]
                radius = pre_gen_params['radius'][radius_id]
            geo_change(robot, f_idx, radius, length)
            robot.joints[j_idx].origin[2][3] = length

        #2nd layer:
        length_id = np.random.randint(0, len(pre_gen_params['length']),1)[0]
        length = pre_gen_params['length'][length_id]
        half = int(len(joint_idx)/2)
        for f_idx, j_idx in zip(link_idx[1::2],joint_idx[half:]):
            if vars_to_change['layer_rad']:
                radius_id =np.random.randint(0, len(pre_gen_params['radius']),1)[0]
                radius = pre_gen_params['radius'][radius_id]
            geo_change(robot, f_idx, radius, length)
            robot.joints[j_idx].origin[2][3] = length

            
        
def geo_change(robot, idx, r, l):
    midpoint = l/2
    robot.links[idx].collisions[0].geometry.cylinder.length = l
    robot.links[idx].collisions[0].geometry.cylinder.radius = r
    robot.links[idx].visuals[0].geometry.cylinder.length = l
    robot.links[idx].visuals[0].geometry.cylinder.radius = r
    robot.links[idx].visuals[0].origin[2][3] = midpoint
    #print(robot.links[idx].name)

def get_variable_params():
    vars_dict = {'length_range': [0.03,0.1],
                 'radius_range': [0.005 ,0.015]}
    return vars_dict

def pre_gen_robot_params(vars_to_change, param_var_num):
    vars_dict = get_variable_params()
    pre_gen_params = {}

    if vars_to_change.get('length', False):
        length_range = vars_dict['length_range']
        pre_gen_params['length'] = np.random.uniform(length_range[0],
                                                          length_range[1],
                                                          param_var_num)
    
    if vars_to_change.get('radius', False):
        radius_range = vars_dict['radius_range']
        pre_gen_params['radius'] = np.random.uniform(radius_range[0],
                                                radius_range[1],
                                                param_var_num)
    return pre_gen_params

         
def gen_robot_configs(ref_file, robot_num,
                      vars_to_change, param_var_num,
                      root_save_dir):
    pre_gen_params = pre_gen_robot_params(vars_to_change=vars_to_change,
                                          param_var_num=param_var_num)
    for idx in tqdm(range(robot_num)):
        robot = URDF.load(ref_file)
        robot_param_random_sample(robot, pre_gen_params, vars_to_change)
        sub_save_dir = os.path.join(root_save_dir, 'robot_%d' % idx)
        os.makedirs(sub_save_dir, exist_ok=True)
        urdf_name = os.path.join(sub_save_dir, 'model.urdf')
        robot.save(urdf_name)



if __name__ == "__main__":
    desp = 'Generate new robot configuration files with different kinematic properties randomly'
    parser = argparse.ArgumentParser(description=desp)
    parser.add_argument('--robot_num', '-rn', type=int, default=100, 
                        help='number of robots to generate')
    parser.add_argument('--param_var_num', '-pvn', type=int, default=100, 
                        help='size of the pool for each dynamic parameter')
    parser.add_argument('--random_seed', '-rs', type=int, default=1,
                        help='seed for random number')
    parser.add_argument('--model', '-gm', type=str,
                        default='2f_1j',
                        help='opts= 2f_1j, 2f_2j, 3f_1j, 3f_2j')
    args = parser.parse_args()
    np.random.seed(seed=args.random_seed)
    ref_urdf = f'../assets/{args.model}.urdf'
    save_dir = f'../assets/gen_gripper/{args.model}'
    print('Reference file: ', os.path.abspath(ref_urdf))
    assert os.path.exists(ref_urdf)
    print('Generating files to: ', os.path.abspath(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    vars_to_change = {'radius': True,
                      'length': True,
                      'layer_rad': False}

    start = time.time()
    gen_robot_configs(ref_file=ref_urdf,
                      robot_num=args.robot_num,
                      vars_to_change=vars_to_change,
                      param_var_num=args.param_var_num,
                      root_save_dir=save_dir)

    end = time.time()
    print("Generating %d robots took %.3f seconds"
            "" % (args.robot_num, end - start))
