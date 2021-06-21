import urdfpy
from urdfpy import URDF

import argparse
import copy
import os
import time

import numpy as np
from tqdm import tqdm


def robot_param_random_sample(robot, pre_gen_params, vars_to_change):
    if vars_to_change.get('damping', False):
        for joint in robot.joints:
            if joint.dynamics != None:
                pre_param_id = np.random.randint(0, len(pre_gen_params['damping']),1)[0]
                vdamp = pre_gen_params['damping'][pre_param_id]
                joint.dynamics.damping = vdamp

    if vars_to_change.get('friction', False):
        for joint in robot.joints:
            if joint.dynamics != None:
                pre_param_id = np.random.randint(0, len(pre_gen_params['friction']),1)[0]
                vfriction = pre_gen_params['friction'][pre_param_id]
                joint.dynamics.friction = vfriction
    
    if vars_to_change.get('body_mass', False):
        for link in robot.links:
            pre_param_id = np.random.randint(0, len(pre_gen_params['body_mass']), 1)[0]
            vbmass = pre_gen_params['body_mass'][pre_param_id]
            link.inertial.mass *= vbmass

def get_variable_params():
    vars_dict = {'damping_range': [0.01,30],
                 'friction_range': [0,10],
                 'mass_ratio': [0.25, 2]}
    return vars_dict

def pre_gen_robot_params(vars_to_change, param_var_num):
    vars_dict = get_variable_params()
    pre_gen_params = {}

    if vars_to_change.get('damping', False):
        damping_range = vars_dict['damping_range']
        if damping_range[1] <= 1 or damping_range[0] >= 1:
            pre_gen_params['damping'] = np.random.uniform(damping_range[0],
                                                          damping_range[1],
                                                          param_var_num)
        else:
            half_num = int(param_var_num/2)
            under_damping = np.random.uniform(damping_range[0], 1, half_num)
            over_damping = np.random.uniform(1, damping_range[1], half_num)
            pre_gen_params['damping'] = np.concatenate((under_damping,over_damping))
    
    if vars_to_change.get('friction', False):
        friction_range = vars_dict['friction_range']
        pre_gen_params['friction'] = np.random.uniform(friction_range[0],
                                                friction_range[1],
                                                param_var_num)

    if vars_to_change.get('body_mass', False):
        mass_ratio = vars_dict["mass_ratio"]
        if mass_ratio[1] <= 1 or mass_ratio[0] >= 1:
            pre_gen_params['body_mass'] = np.random.uniform(mass_ratio[0],
                                                            mass_ratio[1],
                                                            param_var_num)
        else:
            half_num = int(param_var_num / 2)
            under_one = np.random.uniform(mass_ratio[0], 1, half_num)
            over_one = np.random.uniform(1, mass_ratio[1], half_num)
            pre_gen_params['body_mass'] = np.concatenate((under_one, over_one))

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
    desp = 'Generate new robot configuration files with different dynamic properties randomly'
    parser = argparse.ArgumentParser(description=desp)
    parser.add_argument('--robot_num', '-rn', type=int, default=100, 
                        help='number of robots to generate')
    parser.add_argument('--param_var_num', '-pvn', type=int, default=10000, 
                        help='size of the pool for each dynamic parameter')
    parser.add_argument('--random_seed', '-rs', type=int, default=1,
                        help='seed for random number')
    parser.add_argument('--ref_urdf', '-ru', type=str,
                        default='assets/ur5_w_gripper/model.urdf',
                        help='reference robot urdf file')
    parser.add_argument('--save_dir', '-sd', type=str,
                        default='assets/generated/ur5_w_gripper',
                        help='save directory for generated models')
    args = parser.parse_args()
    np.random.seed(seed=args.random_seed)
    print('Reference file: ', os.path.abspath(args.ref_urdf))
    assert os.path.exists(args.ref_urdf)
    print('Generating files to: ', os.path.abspath(args.save_dir))
    os.makedirs(args.save_dir, exist_ok=True)

    vars_to_change = {'damping': True,
                      'friction': True,
                      'body_mass': True}

    start = time.time()
    gen_robot_configs(ref_file=args.ref_urdf,
                      robot_num=args.robot_num,
                      vars_to_change=vars_to_change,
                      param_var_num=args.param_var_num,
                      root_save_dir=args.save_dir)

    end = time.time()
    print("Generating %d robots took %.3f seconds"
            "" % (args.robot_num, end - start))
