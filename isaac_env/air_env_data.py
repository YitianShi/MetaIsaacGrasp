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


class AIREnvData(AIREnvBase):
    def __init__(self, cfg: CellEnvCfg, 
                 render_mode: str | None = None, 
                 **kwargs):
        super().__init__(cfg, 
                         render_mode, 
                         random_drop_obj=True, 
                         save_camera_data=True)
    
    def _advance_state_machine(self):
        """Compute the desired state of the robot's end-effector and the gripper."""

        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = self._get_ee_pose()[:, [0, 1, 2, 4, 5, 6, 3]]
        self.grasp_pose = self.grasp_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        env_reachable_and_stable_wp = wp.from_torch(
            self.env_reachable_and_stable.contiguous(), wp.bool
        )

        wp.launch(
                kernel=infer_state_machine_data,
                dim=self.num_envs,
                inputs=[
                    # environment state machine recorders
                    self.sm_dt_wp,
                    self.sm_state_wp,
                    self.sm_wait_time_wp,
                    # environment time states
                    self.epi_count_wp,
                    # environment physical states
                    env_reachable_and_stable_wp,
                    # desired robot end effector state
                    self.des_ee_pose_wp,
                    self.des_gripper_state_wp,
                    # current robot end effector state
                    ee_pose_wp,
                ],
                device=self.device,
            )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        ee_pose = ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        self.grasp_pose = self.grasp_pose[:, [0, 1, 2, 6, 3, 4, 5]]

        # convert to torch
        return torch.cat((des_ee_pose, self.des_gripper_state.unsqueeze(-1)), -1)
