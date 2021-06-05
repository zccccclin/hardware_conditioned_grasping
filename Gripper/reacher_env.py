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
        def reset