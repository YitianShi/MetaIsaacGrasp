# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="ur10", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip
import os
from pathlib import Path
#HOME_PATH =  "omniverse://nucleus.ifl.kit.edu/Users/yitian/"
HOME_PATH = Path(os.getcwd())
USD_PATH = os.path.join(HOME_PATH, "models")

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
ARM_JOINT = {
    "shoulder_pan_joint": 0.,
    "shoulder_lift_joint": -np.pi * down_ration + bow_angle,
    "elbow_joint": np.pi * down_ration,
    "wrist_1_joint": -np.pi / 2 - bow_angle - wrist_lift,
    "wrist_2_joint": -np.pi / 2,
    "wrist_3_joint": np.pi,
}
JOINT_SETUP = ARM_JOINT.copy()
JOINT_SETUP.update(ROBOTIQ_HAND_E_JOINT_CFG["default"])

MODEL_PATH = os.path.join(HOME_PATH, "models", "models_ifl") # path to the models
MGN_PATH = os.path.join(MODEL_PATH, "009", "orbit_obj.usd") # path to the objects

EE_GOALS = [[0., 0.51, 0.2, 0.707, -0.707, 0, 0],
            [0., 0.51, 0.05, 0.707, -0.707, 0, 0],
            [0.2, 0.51, 0.3, 0.707, -0.707, 0, 0],]

OBJ_ROT = (0.707, 0.707, 0., 0.)

#OBJ_ROT = (1., 1., 0., 0.)

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):

    
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    #obj = RigidObjectCfg(
    #    prim_path="/World/envs/env_.*/object",
    #    spawn=sim_utils.UsdFileCfg(
    #        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
    #            rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #            solver_position_iteration_count=16,
    #            solver_velocity_iteration_count=1,
    #            max_angular_velocity=1000.0,
    #            max_linear_velocity=1000.0,
    #            max_depenetration_velocity=5.0,
    #            disable_gravity=False,
    #        ),
    #        scale=(1.0, 1.0, 1.0),
    #    ),
    #        init_state=RigidObjectCfg.InitialStateCfg(
    #            pos=(0., .5, 0.2),
    #            rot=(1., 1., 0., 0),
    #        ),)

    
    obj = RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=MGN_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
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
                    ),
            mass_props = sim_utils.MassPropertiesCfg(density=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            semantic_tags=[("class", f"{MGN_PATH.split('/')[-2]}"), ("color", "red")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0., .5, 0.2),
            rot=OBJ_ROT,
        ),
        collision_group = 0,
        prim_path = "{ENV_REGEX_NS}/obj"
    )


    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":


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
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=JOINT_SETUP),#, pos=robot_pos),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=list(ARM_JOINT.keys()),
                    velocity_limit_sim=50.0,
                    effort_limit_sim=1e4,
                    stiffness=5e3,
                    damping=400.0,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=[*ROBOTIQ_HAND_E_JOINT_CFG["default"]],
                    stiffness=7000,
                    damping=100,
                    #friction=0.8
                ),
            },
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        robot = UR10e_HAND_E_CFG#.replace(prim_path="{ENV_REGEX_NS}/Robot")
        print("prim_path = ",UR10e_HAND_E_CFG.prim_path)
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    #view_pos_rob = self.ee_frame.data.target_pos_source.clone()[:, 0, :]
    #view_quat_rob = self.ee_frame.data.target_quat_source.clone()[:, 0, :]
    #ee_pos = torch.cat((view_pos_rob, view_quat_rob), -1)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    

    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        ee_goals = [[0, 0.5, 0.02, 0, 1., 0., 0.]]
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["hande_end"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    
    
    ee_goals = torch.tensor(EE_GOALS, device=sim.device)

    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
        # reset
    while simulation_app.is_running():
        if count < 200:
            # reset time
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        if count == 300:
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
        elif count == 500:
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)

        #print("count ", count)
        if count == 400:
            #print(joint_pos_des)
            joint_pos_des[0, -2:] = 0.0
        
        
        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)
        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])



def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()