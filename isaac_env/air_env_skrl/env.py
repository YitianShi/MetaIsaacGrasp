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

# from isaaclab.envs.mdp.rewards import action_rate_l2, action_l2
import pandas as pd
import torch
# from isaaclab.controllers.rmp_flow import *

from .wp_cfg import *
from isaac_env import AIREnvBase
from .env_cfg import *
##
# Pre-defined configs
##


class AIREnvSKRL(AIREnvBase):
    def __init__(self, cfg: CellEnvCfg, 
                 render_mode: str | None = None,
                 save_camera_data: bool = False, 
                 **kwargs):
        # Setup RL environment
        super().__init__(cfg, 
                         render_mode=render_mode, 
                         save_camera_data=save_camera_data)

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
        self.reward_state = STATE_MACHINE["execute"]

        self.RL_TRAIN_FLAG = True
        
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
        gripper_state_con_wp = wp.from_torch(self.gripper_state_con, wp.float32)

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
                    gripper_state_con_wp,
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
        # advance_frame_con 记录是否应该推进帧（用于连续控制），直接根据 advance_frame_con 变量的值来决定是否推进，和Base类不同
        return self.advance_frame_con.clone()
    
    
    def step(self, grasp_pose, policy_inference_criteria=torch.tensor([])):
        # Get the grasp pose from the policy
        grasp_pose = self.process_action(grasp_pose)

        obs_buf, reward_buf, reset_terminated, reset_time_outs, policy_inference_criteria = super().step(grasp_pose)
           
        #ids = self.env_idx.clone()[self.inference_criteria]
        return obs_buf, reward_buf, reset_terminated, reset_time_outs, dict()

    
    def process_action(self, grasp_pose):
        xyz = self.scene["ee_frame"].data.target_pos_source.clone()[:, 0, :]
        quat = self.scene["ee_frame"].data.target_quat_source.clone()[:, 0, :]
        # xyz + dxdydz
        grasp_pose = torch.tensor(grasp_pose, dtype=torch.float32, device='cuda:0')
        grasp_pose[:, :3] /= torch.norm(grasp_pose[:, :3], dim=-1, keepdim=True)
        pos = xyz + 0.05 * grasp_pose[:, :3]
        action_normalized = torch.cat((pos, quat), dim=1)
        self.gripper_state_con = grasp_pose[:, -1]  
        return action_normalized


    def _summerize_and_reset(self, reward_buf):
        """
        Calculate the reward and
        reset the indexed enviroments
        """
        # Judge in the terminate state
        judge_reward = (self.sm_state == 8.) | (self.sm_state == 7.) # 如果当前状态是lift状态
        # Calculate the reward
        #if reward_buf.any(): # 如果 reward_buf 里至少有一个非零奖励，就记录奖励并重置有奖励的机器人
        #    self._reset_robot(reward_buf.bool())

        # Get the reset id from the state machine 获取需要重置的索引
        init_id = self.env_idx.clone()[self.sm_state == STATE_MACHINE["init"]]
        init_env_id = self.env_idx.clone()[self.sm_state == STATE_MACHINE["init_env"]]

        # Reset the robot, environment and the teleoperation interface
        self._reset_robot(init_id)
        self._reset_idx(init_env_id)

        # To stabilize the robot, set the grasp pose to the current pose during the start state
        # 选出 start 状态的环境，然后将 grasp_pose 设为当前 EE 位姿 让机器人稳定。
        self.grasp_pose[self.sm_state == STATE_MACHINE["start"]] = self._get_ee_pose()[self.sm_state == STATE_MACHINE["start"]]
    
    def _record_reward(self, judge_reward):
        """Summarize the reward and print the success message."""
        for i, env_success in enumerate(self.reward_buf):
            if env_success and judge_reward[i]:
                self.sm_state[i] = STATE_MACHINE["init"]
