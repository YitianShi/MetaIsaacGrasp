# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path

import omni.isaac.lab.sim as sim_utils
import torch
import numpy as np

# from omni.isaac.lab.sensors.camera import Camera, PinholeCameraCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.utils.math import *
from dataclasses import MISSING

# Hyperparameters
# Parameters frequently changed
use_sb3 = False # use stable baselines3 for training
on_hand = True # whether the camera is on the hand 
num_objs = 12 # number of objects
use_urdf_converter =  False # use urdf converter to convert usd to urdf
random_drop_obj = True # drop object randomly under the table

# Task parameters 
collect_data = False # collect data for training
save_data = True # save data during training 
read_from_hdf5 = False # read data from hdf5 file to get grasp pose

# Learning Environment parameters
successive_grasp_failure_limit = 12 # number of successive grasp failure before reset the environment
step_total = 30 # total number of steps for each episode

# Robot arm parameters
robot_name = "ur10e_hand_e" # Choose from "ur10", "ur10e", "ur10e_2f85", "ur10e_hand_e"
continuous_control = False # continuous control for robot
robot_pos = (0.1, 0.6, 0.925) # robot position
bow_angle = 0.1 # bow angle of the robot
approach_distance = 0.1 # distance to approach the object before grasp
lift_height = 0.5 # height to lift the object
grasp_distance = 0.1 # apprach distance before grasp the object
distance_limit = 2e-2 # distance limit for robot to reach the goal before state transition
ee_vel_limit = 5e-2 # velocity limit for robot to reach the goal before state transition
ee_grasp_quat_default = (.5, -0.5, 0.5, -0.5) # default quaternion after grasping
CONTROLLER = "NOT_IMPLEMENTED" # Not used, in future version, choose from "RMPFLOW"
remote_agent = False # use remote agent for training

# Object parameters
object_name = "obj"
obj_vel_limit = 5e-1 # velocity limit for object to be regarded as stable
obj_height_limit = 0.1 # height limit for object to be regarded as stable
ee_obj_default = [
    (0.6, 0.9),
    (-0., 0.4),
    (0.05, 0.15),
    (-1, 1),
    (-1, 1),
    (-1, 1),
    (-1, 1),
]
obj_drop_pose = (0.3, 0.7, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0) # object drop pose

# Camera parameters
n_multiple_cam = 1 # number of multiple cameras, at least 1 for on_hand camera otherwise disable_camera = True
fix_rand_camera = True
camera_name = "VMS3D_Femto_Mega_S_0/" if on_hand else "" 
cam_height, cam_width = 480, 640 # camera resolution
view_pos_w = torch.tensor((0.7, 0, 1.)) # viewpoint wrt. world
# view_pos_cam = (24.46*0.001, 15.9234*0.001, -.33675*0.001) # viewpoint wrt. camera
# view_quat_cam = (0.0, 0, 1, .0) # quaternion of viewpoint wrt. camera
view_pos_cam = (-2, 57.55, 1.5) # viewpoint wrt. camera
view_quat_cam = (-0.6691306, 0.7431448, 0, 0,) # quaternion of viewpoint wrt. camera
depth_max = 2. # maximum depth for the camera
sphere_radius = 0.6 

STATE_MACHINE = {
    "init_env": 0,
    "init": 1,
    "start": 2,
    "choose_object": 3,
    "reach": 4,
    "approach": 5,
    "grasp": 6,
    "lift": 7,
    "execute": 8,
}

obj_exclude = [ '003', '033', '030', '055', '074', '078', '018',
               #'017', '059', '003', '069', '068', '055', '065', '070',
                #'022', '074', '077', '044', '056', '027', '060', '008',
                #'026', '019', '043', '061', '076', '025', '018', '079',
                #'028', '067', '010', '012', '004', '035', '020', '023'
                ]
################# PATH CONFIG #################
HOME_PATH = Path(os.getcwd()) # "omniverse://nucleus.ifl.kit.edu/Users/yitian/" 

nums = [f"{i:03}" for i in range(0, 83)]
for i in obj_exclude:
    nums = [j for j in nums if i not in j]
    
MODEL_PATH = os.path.join(HOME_PATH, "models", "models_ifl") # path to the models
MGN_PATHS = [os.path.join(MODEL_PATH, a, "orbit_obj.usd") for a in nums] # path to the objects
 
CORNER_PATHS = [os.path.join(HOME_PATH, "models/obj.usd")] # path to the corner marker squares
MGN_PATHS_URDF = [os.path.join(MODEL_PATH, a, "orbit_obj.urdf") for a in nums] # path to the objects in urdf format


USD_PATH = os.path.join(HOME_PATH, "models") # path to the usd files
IMG_PATH = os.path.join("./", "data") # path to the data storage

# Create directories if not exist
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)
    print(f"Directory {IMG_PATH} created since it does not exist")

# use urdf converter to convert usd to urdf or not    
OBJ_PATH = MGN_PATHS_URDF if use_urdf_converter else MGN_PATHS
    
random.shuffle(OBJ_PATH)
OBJ_LABLE = [i.split("/")[-2] for i in OBJ_PATH][:num_objs]


################# ROBOT ARM CONFIG #################


ROBOTIQ_2F85_JOINT_CFG = {
    "default": {
        "left_inner_knuckle_joint": 0.0,
        "right_inner_knuckle_joint": 0.0,
        "left_inner_finger_joint": 0.0,
        "right_inner_finger_joint": 0.0,
        # "left_outer_knuckle_joint":0.,
        "right_outer_knuckle_joint": 0.0,
        "left_outer_finger_joint": 0.0,
        "right_outer_finger_joint": 0.0,
        "finger_joint": 0.0,
    },
    "open": {
        "left_inner_knuckle_joint": 0.0,
        "right_inner_knuckle_joint": 0.0,
        "left_inner_finger_joint": 0.0,
        "right_inner_finger_joint": 0.0,
    },
    "close": {
        "left_inner_knuckle_joint": 0.7,
        "right_inner_knuckle_joint": 0.7,
        "left_inner_finger_joint": -0.55,
        "right_inner_finger_joint": -0.55,
    },
}

ROBOTIQ_HAND_E_JOINT_CFG = {
    "default": {
        "hande_left_finger_joint": 0.0425,
        "hande_right_finger_joint": 0.0425,
    },
    "open": {
        "hande_left_finger_joint": 0.0425,
        "hande_right_finger_joint": 0.0425,
    },
    "close": {
        "hande_left_finger_joint": 0.0,
        "hande_right_finger_joint": 0.0,
    },
}

gripper_joint_cfg = {  
    "ur10e_2f85": ROBOTIQ_2F85_JOINT_CFG,
    "ur10e_hand_e": ROBOTIQ_HAND_E_JOINT_CFG,
}

ARM_JOINT = {
    "shoulder_pan_joint": 0.,
    "shoulder_lift_joint": -np.pi + bow_angle,
    "elbow_joint": np.pi / 2 * 1.5,
    "wrist_1_joint": -np.pi / 2 - bow_angle,
    "wrist_2_joint": -np.pi / 2,
    "wrist_3_joint": np.pi,
}
JOINT_SETUP = ARM_JOINT.copy()
JOINT_SETUP.update(gripper_joint_cfg[robot_name]["default"])

focus_distance = 44. # 400.0
focal_length_cm = 1.69 # 18.14756
horizontal_aperture_mm = 2.59 # 20.955
sensor_width_m = horizontal_aperture_mm / 1000
focal_length_m = focal_length_cm / 1000  # error in orbit, it ignore the unit of focal length


################# CAMERA CONFIG #################

disable_camera = n_multiple_cam == 0

if use_sb3:
    MODALITIES = {
        "rgb": 4,
        "distance_to_image_plane": 1
        }
else:
    MODALITIES = {
        "rgb": 4,
        "distance_to_image_plane": 1,
        "normals": 4,
        "instance_segmentation_fast": 1,
        }

CAMERA_CFG = CameraCfg(
    update_period=0.001,
    height=cam_height,
    width=cam_width,
    data_types=[*MODALITIES],
    colorize_instance_id_segmentation=False,
    colorize_semantic_segmentation=False,
    colorize_instance_segmentation=False,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=focal_length_cm,
        focus_distance=focus_distance,
        horizontal_aperture=horizontal_aperture_mm,
        clipping_range=(0.01, 1.0e5),
    ),
    offset=CameraCfg.OffsetCfg(pos=view_pos_cam if on_hand else view_pos_w, 
                               rot=view_quat_cam, 
                               convention="ros"),
    )

image_resolution = (cam_width, cam_height)
focal_length_pixels = (focal_length_m / sensor_width_m) * image_resolution[0]
cx, cy = image_resolution[0] / 2, image_resolution[1] / 2


n_random_cam = n_multiple_cam - 1
random_cam_names = [f"camera_{i+1}" for i in range(n_random_cam)]
fix_cam_angle = np.linspace(0, np.pi * 2, n_random_cam, endpoint=False)
    

CAMERA_RANDOM_CFG = [
    CameraCfg(
        update_period=0.001,
        height=cam_height,
        width=cam_width,
        data_types=[*MODALITIES],
        colorize_instance_id_segmentation=False,
        colorize_semantic_segmentation=False,
        colorize_instance_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length_cm,
            focus_distance=focus_distance,
            horizontal_aperture=horizontal_aperture_mm,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=MISSING,
    )
    for _ in range(n_random_cam)
    ]

# Viewpoint sampling paremeters
center_pt = torch.tensor(((ee_obj_default[0][0] + ee_obj_default[0][1]) / 2, 
                          (ee_obj_default[1][0] + ee_obj_default[1][1]) / 2, 
                          0.), 
                          dtype=torch.float32)


################# ROBOT CONFIG #################

# Reachable end effector goals
ee_goals_default = [
    (0.3, 0.9),
    (-0.4, 0.4),
    (0.4, 0.6),
    (-1, 1),
    (-1, 1),
    (-1, 1),
    (-1, 1),
]

UR10e_2F85_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/ur10e_with_2F85.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=50.0,
            linear_damping = 2,
            angular_damping = 2,
            max_contact_impulse = float("inf"),
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=JOINT_SETUP, pos=robot_pos),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=list(ARM_JOINT.keys()),
            velocity_limit=100.0,
            effort_limit=1e4,
            stiffness=5e3,
            damping=400.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[*ROBOTIQ_2F85_JOINT_CFG["default"]],
            velocity_limit=100.0,
            effort_limit=1e10,
            stiffness=1e10,
            damping=1,
        ),
    },
)

UR10e_HAND_E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/ur10e_with_hand_e_and_camera_mount.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=50.0,
            linear_damping = 2,
            angular_damping = 2,
            max_contact_impulse = float("inf"),
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=JOINT_SETUP, pos=robot_pos),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=list(ARM_JOINT.keys()),
            velocity_limit=100.0,
            effort_limit=1e4,
            stiffness=5e3,
            damping=400.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[*ROBOTIQ_HAND_E_JOINT_CFG["default"]],
            velocity_limit=0.001,
            effort_limit=1e3,
            stiffness=1e10,
            damping=1e3,
        ),
    },
)

end_effector_frame_name = {
    "ur10": "ee_link",
    "ur10e": "tool0",
    "ur10e_2f85": "tool0",
    "ur10e_hand_e": "hande_end",
}
ee_name = end_effector_frame_name[robot_name]

################# OBJECT CONFIG #################

rigid_props = sim_utils.RigidBodyPropertiesCfg(
    rigid_body_enabled=True,
    disable_gravity=False,
    max_depenetration_velocity=50.0,
    linear_damping = 1,
    angular_damping = 2,
    max_contact_impulse = float("inf"),
    max_linear_velocity=1,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=16,
    stabilization_threshold=0.1,
    )

TABLE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/Workspace/Table_sim.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=50.0,
            linear_damping = 1,
            angular_damping = 2,
            max_contact_impulse = float("inf"),
            max_angular_velocity=0.,
            max_linear_velocity=0.,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0, 0.0, 0.0),
        rot=(1, 0.0, 0.0, 0),
    ),
)

MARK1_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/obj.usd",
        rigid_props=rigid_props,
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(ee_goals_default[0][0], ee_goals_default[1][0], 0),
        rot=(1, 0.0, 0.0, 0),
    ),
)

MARK2_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/obj.usd",
        rigid_props=rigid_props,
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(ee_goals_default[0][1], ee_goals_default[1][0], 0),
        rot=(1, 0.0, 0.0, 0),
    ),
)

MARK3_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/obj.usd",
        rigid_props=rigid_props,
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(ee_goals_default[0][0], ee_goals_default[1][1], 0),
        rot=(1, 0.0, 0.0, 0),
    ),
)

MARK4_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/obj.usd",
        rigid_props=rigid_props,
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(ee_goals_default[0][1], ee_goals_default[1][1], 0),
        rot=(0, 0.0, 0.0, 0),
    ),
)

MGN_CFGs = [
    RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=mgn,
            rigid_props=rigid_props,
            mass_props = sim_utils.MassPropertiesCfg(density=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            semantic_tags=[("class", f"{mgn.split('/')[-2]}"), ("color", "red")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=robot_pos,
            rot=(1, 0.0, 0.0, 0),
        ),
        collision_group = 0
    )
    for mgn in MGN_PATHS
]

MGN_CFGs_URDF = [
    RigidObjectCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=mgn,
            rigid_props=rigid_props,
            mass_props = sim_utils.MassPropertiesCfg(density=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            fix_base=False,
            force_usd_conversion=True,
            make_instanceable=False,
            semantic_tags=[("class", f"{mgn.split('/')[-2]}"), ("color", "red")],
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=robot_pos,
            rot=(1, 0.0, 0.0, 0),
        ),
        collision_group = 0,
    )
    for mgn in MGN_PATHS_URDF
]

TARGO_CFGs_URDF = [
    RigidObjectCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=mgn,
            rigid_props=rigid_props,
            mass_props = sim_utils.MassPropertiesCfg(density=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            fix_base=False,
            force_usd_conversion=True,
            make_instanceable=False,
            semantic_tags=[("class", f"{mgn.split('/')[-2]}"), ("color", "red")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=robot_pos,
            rot=(1, 0.0, 0.0, 0),
        ),
    )
    for mgn in MGN_PATHS_URDF
]

################# OBJECT CONFIG #################
OBJ_CFGs = MGN_CFGs_URDF if use_urdf_converter else MGN_CFGs

################# CONTROLLER CONFIG #################
UR_IK_CFG = DifferentialIKControllerCfg(
    command_type="pose", use_relative_mode=False, ik_method="dls"
)

################# ROBOT CONFIG #################
ROBOT_CFGs = {
    "ur10e_2f85": UR10e_2F85_CFG,
    "ur10e_hand_e": UR10e_HAND_E_CFG,
}

################# MARKER CONFIG #################
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
