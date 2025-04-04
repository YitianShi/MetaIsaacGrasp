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

# from isaaclab.envs.mdp.rewards import action_rate_l2, action_l2
import pandas as pd
import torch

from .wp_cfg import *
from .env_cfg import *
from isaac_env import AIREnvBase

##
# Pre-defined configs
##


class AIREnvGrasp(AIREnvBase):
    def __init__(self, cfg: CellEnvCfg, 
                 render_mode: str | None = None, 
                 save_camera_data: bool = False,
                 **kwargs):
        super().__init__(cfg, 
                         render_mode=render_mode, 
                         save_camera_data=save_camera_data)

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
            kernel=infer_state_machine_disc,
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
        if REMOTE_AGENT:
            # Get action from remote agent
            return self.get_action_remote(ids, obs_buf)
        else:
            return self.get_action_demo(ids, obs_buf)
        
    def get_action_remote(self, ids, obs_buf):
        data = obs_buf["policy"]["pcd"][ids]
        env_num, cam_id, h, w, _ = data.shape
        data = data.view(env_num * cam_id, h * w, 3)
        
        data = (data*1000).to(torch.int16) if data.dtype == torch.float32 else data
        data = pickle.dumps(data.cpu().numpy())
                    
        # Send the data to the agent
        print("Sending data to agent...")
        data_length = struct.pack('>I', len(data)) 
        conn.sendall(data_length + data)
        
        # Receive the actions from the agent
        print("Receiving actions from agent...")
        data = b""
        data_length_bin = None
        while not data_length_bin:
            data_length_bin = conn.recv(4)
        data_length = struct.unpack('>I', data_length_bin)[0]
        while len(data) < data_length:
            packet = conn.recv(chunk_size)
            if not packet:
                break
            data += packet
        actions = pickle.loads(data)
        no_action = actions[:, 3, -1] == 0
        rotactions = quat_from_matrix(actions[:, :3, :3])
        translation = actions[:, :3, 3]
        actions = torch.cat((translation, rotactions), dim=-1).to(self.device)
        if no_action.any():
            no_action_ids = ids[no_action]
            print("Agent failed to provide actions.")
            actions_remains = self.get_action_demo(no_action_ids, self.obs_buf)[no_action_ids]
            actions[no_action] = actions_remains
        
        self.grasp_pose[ids] = actions.to(self.device)
        return self.grasp_pose


    