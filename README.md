# hardware_conditioned_grasping

This repo explores hardware conditioned grasping and reaching on a UR5 arm with a custom gripper.
The dynamics properties are varied in the reaching tasks, while the kinematic properties are varied for the grasping task.

The reaching task:
The DRL alogrithm used in reaching task is DDPG+HER, HER is used since the reward function is sparse. with +1 for target reached, -1 for otherwise. 
To train the model, run the follows files with desired arguments in the Reacher folder:
1. python generator.py  (generate robot with different dynamics properties)
--robot_num=(number of robot to generate)
2. python robot_param_stats.py (generates json file to keep track of dynamics properties)
3. python main.py (training file)
details of input arguments can be found within the file


The Grasping task:
The DRL alogrithm used in reaching task is DDPG, HER is turned off since the reward function is dense. Detail of the reward function can be found in the env_base.py file under cal_reward.
To train the model, run:
python main.py
Use Ctrl + C to interrupt training, 
To resume:
python main.py --resume
To see training in GUI:
python main.py --render
hyperparameters can be changed by input relevant arguments, more details can be found in the main.py file.

To test the trained model, run:
python main.py --test --render
The final dist to goal will be printed out.

