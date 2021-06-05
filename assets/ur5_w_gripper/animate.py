import urdfpy
import numpy as np
ct = {'shoulder_pan_joint' : [-np.pi / 4, np.pi / 4], 
        'shoulder_lift_joint' : [0.0, -np.pi / 2.0], 
        'elbow_joint' : [0.0, np.pi / 2.0], 
        'bh_j32_joint': [0.0 ,np.pi/2],
        'bh_j22_joint': [0.0 ,np.pi/2],
        'bh_j12_joint': [0.0 ,np.pi/2]}

model = urdfpy.URDF.load('model.urdf')
model.animate(cfg_trajectory=ct)