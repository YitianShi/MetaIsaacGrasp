# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from .. import agents
from isaac_env.utils import *
from .env_cfg import *
from .sim import AIRPickSmRL
from .env import AIREnvRL
from .element_cfg import *

__all__ = ["AIRPickSmRL", "AIREnvRL"]

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="AIR-v0-RL",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="isaac_env.air_env_rl:AIREnvRL",
    kwargs={
        "env_cfg_entry_point": CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

