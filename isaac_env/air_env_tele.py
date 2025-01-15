"""
This script demonstrates how to run the RL environment for the cartpole balancing task.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import os
import random
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import open3d as o3d

import socket
import pickle
import struct
import sys

# from omni.isaac.lab.envs.mdp.rewards import action_rate_l2, action_l2
import pandas as pd
import torch
from omni.isaac.lab.controllers import DifferentialIKController
from omni.isaac.lab.devices import Se3Keyboard

# from omni.isaac.lab.controllers.rmp_flow import *
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_mul, combine_frame_transforms, apply_delta_pose

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


class AIREnvTele(AIREnvBase):
    def __init__(self, cfg: CellEnvCfg, render_mode: str | None = None, **kwargs):
        
        # Setup RL environment
        super().__init__(cfg, render_mode)

        self.inference_state = STATE_MACHINE["execute"]
         
        # Initialize the teleoperation interface
        self.teleop_interface = Se3Keyboard(
            pos_sensitivity=0.01, rot_sensitivity=0.01
        )
        self.teleop_interface.add_callback("L", self._manual_reset_env)
        self.teleop_interface.add_callback("R", self._manual_reset)
            
        self.teleop_interface.reset()

        print(self.teleop_interface)
        
        if remote_agent:
            self.teleop_interface.add_callback("N", self.get_action_remote)
            self.teleop_interface.add_callback("M", self.get_action_llm)

        if targo:
            frame_marker_targo = FRAME_MARKER_CFG.copy()
            frame_marker_targo.markers["frame"].scale = (marker_scale, marker_scale, marker_scale)
            self.targo_marker = VisualizationMarkers(
                frame_marker_targo.replace(prim_path="/Visuals/targo_view"))
            self.targo_pose_view = torch.tensor(targo_extrinsic, device=self.device)


    def _advance_state_machine(self):
        """Compute the desired state of the robot's end-effector and the gripper."""

        # get the end-effector velocity
        ee_vel = self._get_ee_vel()

        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = self._get_ee_pose()[:, [0, 1, 2, 4, 5, 6, 3]]
        self.grasp_pose = self.grasp_pose[:, [0, 1, 2, 4, 5, 6, 3]].to(self.device)

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        ee_vel_wp = wp.from_torch(ee_vel, wp.float32)
        env_reachable_wp = wp.from_torch(self.env_reachable.contiguous(), wp.bool)
        env_reachable_and_stable_wp = wp.from_torch(self.env_reachable_and_stable.contiguous(), wp.bool)
        grasp_pose_wp = wp.from_torch(self.grasp_pose, wp.transform)

        
        wp.launch(
            kernel=infer_state_machine_tele,
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
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        ee_pose = ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        self.grasp_pose = self.grasp_pose[:, [0, 1, 2, 6, 3, 4, 5]]

        # convert to torch
        return torch.cat((des_ee_pose, self.des_gripper_state.unsqueeze(-1)), -1)
    
    def get_action(self, ids, obs_buf):
        delta_pose, gripper_command = self.teleop_interface.advance()
        delta_pose = delta_pose.astype("float32")
        # print(f"Delta pose: {delta_pose}")
        # convert to torch
        delta_pose = torch.tensor(delta_pose, device=self.device).repeat(ids.shape[0], 1)
        if self.focus:
            # focus on certain object
            self.grasp_pose[..., 3:] = quat_from_matrix(
                rotation_matrix_from_view(eyes=self.grasp_pose[..., :3],
                                                targets=desk_center.to(self.device),
                                                device=self.device))
        # resolve gripper command
        self.des_gripper_state[ids] = -1. if gripper_command else 1
        # compute actions
        curr_pose = self.grasp_pose[ids]
        actions = apply_delta_pose(curr_pose[:, :3], curr_pose[:, 3:], delta_pose)
        actions = torch.cat(actions, dim=-1)
        # update the grasp pose
        self.grasp_pose[ids] = actions.to(self.device)
        return self.grasp_pose
    
    def get_action_llm(self):

        rgb = self.obs_buf["policy"]["rgb"]
        pcd = self.obs_buf["policy"]["pcd"]

        # save the rgb image
        rgb = cv2.cvtColor(rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{IMG_PATH}/rgb.png", rgb)
        depth = cv2.normalize(pcd[..., -1].cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(f"{IMG_PATH}/depth.png", depth)

        ## TODO: Implement the LLM 
        # robot_action = llm(rgb, depth, prompt)
        ## TODO: Teleoperate the robot with the robot_action

    def _manual_reset_env(self):
        """Manual reset of the environment."""
        self.teleop_interface.reset()
        self.sm_state[self.sm_state == STATE_MACHINE["execute"]] = STATE_MACHINE["init_env"]
    
    def _manual_reset(self):
        """Manual reset of the environment."""
        self.teleop_interface.reset()
        self.sm_state[self.sm_state == STATE_MACHINE["execute"]] = STATE_MACHINE["init"]

    def _vis(self, ee_goals):
        super()._vis(ee_goals)
        if targo:    
            self.targo_marker.visualize(
                self.targo_pose_view[:3].unsqueeze(0) + self.scene.env_origins,
                self.targo_pose_view[3:].unsqueeze(0),
            )

