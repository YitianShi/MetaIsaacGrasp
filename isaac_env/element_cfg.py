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
import matplotlib.pyplot as plt

import isaaclab.sim as sim_utils
import torch
import numpy as np
from .read_targo import read_targo
from scipy.spatial.transform import Rotation as R

# from isaaclab.sensors.camera import Camera, PinholeCameraCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import *
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from dataclasses import MISSING
import trimesh as tm


# Hyperparameters
# Parameters frequently changed
use_sb3 = False # use stable baselines3 for training
on_hand = True # whether the camera is on the hand 
num_objs = 12 # number of objects
use_urdf_converter = False # use urdf converter to convert usd to urdf

# Task parameters 
read_from_hdf5 = False # read data from hdf5 file to get grasp pose

# Learning Environment parameters
successive_grasp_failure_limit = 12 # number of successive grasp failure before reset the environment
step_total = 30 # total number of steps for each episode

# Robot arm parameters
robot_name = "ur10e_hand_e" # Choose from "ur10", "ur10e", "ur10e_2f85", "ur10e_hand_e"
continuous_control = False # continuous control for robot
robot_pos = (0.1, 0.6, 0.925) # robot position
bow_angle = 0.1 # bow angle of the robot
down_ration = .75 # down ration of the robot
wrist_lift = .5 # wrist lift of the robot
approach_distance = 0.05 # distance to approach the object before grasp
lift_height = 0.4 # height to lift the object
grasp_distance = 0.1 # apprach distance before grasp the object
distance_limit = 2e-2 # distance limit for robot to reach the goal before state transition
ee_vel_limit = 5e-2 # velocity limit for robot to reach the goal before state transition
ee_grasp_quat_default = (.5, -0.5, 0.5, -0.5) # default quaternion after grasping
CONTROLLER = "NOT_IMPLEMENTED" # Not used, in future version, choose from "RMPFLOW"
remote_agent = False # use remote agent for training
targo = False

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
up_direction = torch.tensor([0., 0., 1.]) # up direction of the camera

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

#obj_exclude = [ '003', '033', '030', '055', '074', '078', '018',
#               #'017', '059', '003', '069', '068', '055', '065', '070',
#                #'022', '074', '077', '044', '056', '027', '060', '008',
#                #'026', '019', '043', '061', '076', '025', '018', '079',
#                #'028', '067', '010', '012', '004', '035', '020', '023'
#                ]
# obj_exclude = [ '003', '033', '030', '055', '074', '078', '018',
#                 "037", "032", "041", "071", "076", "070", "068",
#                 "038", "072", "035"]
obj_exclude = ["033"]

test_obj = False
if test_obj:
    num_objs = 1

################# PATH CONFIG #################
HOME_PATH =  "omniverse://nucleus.ifl.kit.edu/Users/yitian/" # 
#HOME_PATH = Path(os.getcwd())

nums = [f"{i:03}" for i in range(0, 83)]
for i in obj_exclude:
    nums = [j for j in nums if i not in j]

# for testing define single object

# 009 - scissors
# 078 - bookshelf

# no
#nums = ["037", "032", "041", "071", "076", "070", "068"]

    
MODEL_PATH = os.path.join(HOME_PATH, "models", "models_ifl") # path to the models
MGN_PATHS = [os.path.join(MODEL_PATH, a, "orbit_obj.usd") for a in nums] # path to the objects
 
CORNER_PATHS = [os.path.join(HOME_PATH, "models/obj.usd")] # path to the corner marker squares
MGN_PATHS_URDF = [os.path.join(MODEL_PATH, a, "textured.urdf") for a in nums] # path to the objects in urdf format

# Target object path and pose
if targo:
    view_pos_targo = (0.3, 0.18, 0.) # viewpoint wrt. targo object
    shift_targo = tuple(a + b for a, b in zip(view_pos_targo, robot_pos))
    TARGO_OBJ_PATHS, targo_obj_scales, targo_obj_positions, targo_obj_rotations, occ_targ_max, targo_extrinsic, targo_obj_chosen = read_targo(shift_targo)

USD_PATH = os.path.join(HOME_PATH, "models") # path to the usd files
IMG_PATH = os.path.join("./", "data") # path to the data storage

# Create directories if not exist
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)
    print(f"Directory {IMG_PATH} created since it does not exist")

# use urdf converter to convert usd to urdf or not
if targo:
    OBJ_PATH = TARGO_OBJ_PATHS
    num_objs = len(TARGO_OBJ_PATHS)
elif use_urdf_converter:    
    OBJ_PATH = MGN_PATHS_URDF  
else:
    OBJ_PATH = MGN_PATHS

if not targo:
    random.shuffle(OBJ_PATH)
    OBJ_LABLE = [i.split("/")[-2] for i in OBJ_PATH][:num_objs]
else:
    OBJ_LABLE = [i.split("/")[-1].split(".")[0] for i in OBJ_PATH][:num_objs]


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
    "shoulder_lift_joint": -np.pi * down_ration + bow_angle,
    "elbow_joint": np.pi * down_ration,
    "wrist_1_joint": -np.pi / 2 - bow_angle - wrist_lift,
    "wrist_2_joint": -np.pi / 2,
    "wrist_3_joint": np.pi,
}

if targo:
    ARM_JOINT.update({
        "shoulder_pan_joint": 0.133,
        "shoulder_lift_joint": -1.077,
        "elbow_joint": 2.852,
        "wrist_1_joint": -4.356,
        "wrist_2_joint": -1.683,
        "wrist_3_joint": 3.212,
    })

JOINT_SETUP = ARM_JOINT.copy()
JOINT_SETUP.update(gripper_joint_cfg[robot_name]["default"])

# Object parameters
object_name = "obj"
obj_vel_limit = 5e-1 # velocity limit for object to be regarded as stable
obj_height_limit = 0.1 # height limit for object to be regarded as stable
desk_center = (0.75, 0.2, 0) # center of the desk

ee_obj_default = [
    (desk_center[0] - 0.15, desk_center[0] + 0.15),
    (desk_center[1] - 0.15, desk_center[1] + 0.15),
    (0.05, 0.55),
    (-1, 1),
    (-1, 1),
    (-1, 1),
    (-1, 1),
]
obj_drop_pose = (0.3, 0.7, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0) # object drop pose

# Viewpoint sampling paremeters
if targo:
    desk_center = torch.tensor(targo_obj_positions.mean(axis=0) - robot_pos, dtype=torch.float32)

center_pt = torch.tensor((desk_center[0], desk_center[1], 0.), dtype=torch.float32)

################# CAMERA CONFIG #################

focus_distance = 44. # 400.0
focal_length_cm = 1.69 # 18.14756
horizontal_aperture_mm = 2.59 # 20.955
sensor_width_m = horizontal_aperture_mm / 1000
focal_length_m = focal_length_cm / 1000  # error in orbit, it ignore the unit of focal length

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

################# ROBOT CONFIG #################

# Reachable end effector goals
ee_goals_default = [
    (desk_center[0] - 0.5, desk_center[0] + 0.3),
    (desk_center[1] - 0.6, desk_center[1] + 0.4),
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
            velocity_limit_sim=100.0,
            effort_limit_sim=1e4,
            stiffness=5e3,
            damping=400.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[*ROBOTIQ_2F85_JOINT_CFG["default"]],
            velocity_limit_sim=100.0,
            effort_limit_sim=1e10,
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
            velocity_limit_sim=100.0,
            effort_limit_sim=1e4,
            stiffness=5e3,
            damping=400.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[*ROBOTIQ_HAND_E_JOINT_CFG["default"]],
            stiffness=7000.,#1e10,
            damping=10.,#1e3,
            friction=0.8
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
    max_depenetration_velocity=4.0,
    linear_damping = .1,
    angular_damping = .1,
    max_contact_impulse = float("inf"),
    max_angular_velocity=2,
    max_linear_velocity=2,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=10,
    stabilization_threshold=0.1,
    )

TABLE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/Workspace/Table_sim.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=10,
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
            joint_drive=None,
            mass_props = sim_utils.MassPropertiesCfg(density=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            fix_base=False,
            #force_usd_conversion=True,
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

if targo:
    TARGO_CFGs_URDF = [
        RigidObjectCfg(
            spawn=sim_utils.UrdfFileCfg(
                asset_path=tgo,
                rigid_props= rigid_props,
                mass_props = sim_utils.MassPropertiesCfg(density=5.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                ),
                fix_base=False,
                force_usd_conversion=True,
                make_instanceable=False,
                semantic_tags=[("class", f"{tgo.split('/')[-2]}"), ("color", "red")],
                scale=[scale] * 3,
                visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0., 0., 0.) if targo_obj_chosen == tgo else (random.random(), random.random(), random.random()),
                metallic=0.5
        ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=pos,
                rot=rot,
            ),
        )
        for tgo, pos, rot, scale in zip(TARGO_OBJ_PATHS, targo_obj_positions, targo_obj_rotations, targo_obj_scales)
    ]


TEST_OBJ = [RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            scale=(1.0, 1.0, 1.0),
        ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=robot_pos,
                rot=(1., 0., 0., 0),
            ),)
]

################# OBJECT CONFIG #################
if targo:
    OBJ_CFGs = TARGO_CFGs_URDF
elif use_urdf_converter:
    OBJ_CFGs = MGN_CFGs_URDF
elif test_obj:
    OBJ_CFGs = TEST_OBJ
else:
    OBJ_CFGs = MGN_CFGs

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
marker_scale = 0.025
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

################# REMOTE AGENT CONFIG #################

# server initialization
if remote_agent:
    import socket, sys
    chunk_size = 4096
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allows the socket to reuse the address
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(('172.22.222.222', 8081))
        server.listen(1)
        print("Simulation server is waiting for the agent...")
        conn, addr = server.accept()
        print(f"Connected to agent at {addr}. Adding remote agent inference to teleoperation interface.")
    except socket.error as e:
        print(f"Failed to start the server: {e}")
        server.close()
        sys.exit(1)  # Exit if we cannot start the server
