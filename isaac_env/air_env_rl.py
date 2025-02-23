"""
This script demonstrates how to run the RL environment for the cartpole balancing task.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import os
import random
from typing import Dict, Tuple, Union
import time
import cv2
import numpy as np

# from omni.isaac.lab.envs.mdp.rewards import action_rate_l2, action_l2
import pandas as pd
import torch
from omni.isaac.lab.controllers import DifferentialIKController

# from omni.isaac.lab.controllers.rmp_flow import *
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_mul, combine_frame_transforms

from metagraspnet.Scripts.visualize_labels import (
    create_contact_pose,
    from_contact_to_6D,
    read_in_mesh_config,
)

from isaac_env import *
from .air_env_base import AIREnvBase

##
# Pre-defined configs
##


class AIREnvRL(AIREnvBase):
    def __init__(self, cfg: CellEnvCfg, 
                 render_mode: str | None = None,
                 save_camera_data: bool = False, 
                 **kwargs):
        # Setup RL environment
        super().__init__(cfg, render_mode, save_camera_data=save_camera_data)

        # For continuous grasp decision making
        self.gripper_state_con = torch.zeros(
            self.num_envs, device=self.device
        )  # gripper state for continuous control (open or close)
        # For continuous grasp decision making
        self.advance_frame_con = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )  # whether to advance the frame for continuous control
        self.frame_wait_time_con = torch.zeros(
            self.num_envs, device=self.device
        )  # wait time for the frame to advance for continuous control

        self.gripper_state_con_wp = wp.from_torch(self.gripper_state_con, wp.float32)        
        self.advance_frame_con_wp = wp.from_torch(self.advance_frame_con, wp.bool)
        self.frame_wait_time_con_wp = wp.from_torch(self.frame_wait_time_con, wp.float32)

        self.inference_state = STATE_MACHINE["execute"]

    def _advance_state_machine(self):
        """Compute the desired state of the robot's end-effector and the gripper."""

        # get the end-effector velocity
        ee_vel = self._get_ee_vel()

        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = self._get_ee_pose()[:, [0, 1, 2, 4, 5, 6, 3]]
        self.grasp_pose = self.grasp_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        ee_vel_wp = wp.from_torch(ee_vel, wp.float32)
        env_reachable_wp = wp.from_torch(self.env_reachable.contiguous(), wp.bool)
        env_reachable_and_stable_wp = wp.from_torch(
            self.env_reachable_and_stable.contiguous(), wp.bool
        )
        grasp_pose_wp = wp.from_torch(self.grasp_pose, wp.transform)

        wp.launch(
                kernel=infer_state_machine_con,
                dim=self.num_envs,
                inputs=[
                    # environment state machine recorders
                    self.sm_dt_wp,
                    self.sm_state_wp,
                    self.sm_wait_time_wp,
                    # environment time states
                    self.successive_grasp_failure_wp,
                    self.epi_count_wp,
                    self.step_count_wp,
                    # environment physical states
                    env_reachable_wp,
                    env_reachable_and_stable_wp,
                    # current robot end effector state
                    ee_pose_wp,
                    ee_vel_wp,
                    # desired robot end effector state
                    self.des_ee_pose_wp,
                    self.des_gripper_state_wp,
                    self.ee_quat_default_wp,
                    # proposed grasp pose
                    grasp_pose_wp,
                    self.gripper_state_con_wp,
                    # continuous control time recorder
                    self.advance_frame_con_wp,
                    self.frame_wait_time_con_wp,
                ],
                device=self.device,
            )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        ee_pose = ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        self.grasp_pose = self.grasp_pose[:, [0, 1, 2, 6, 3, 4, 5]]

        # convert to torch
        return torch.cat((des_ee_pose, self.des_gripper_state.unsqueeze(-1)), -1)
    
    def _policy_inf_criteria(self):
        return self.advance_frame_con.clone()