# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from .. import agents
from isaac_env.utils import *
from .env_cfg import *
from .sim import AIRPickSmSB3
from .env import AIREnvSB3
from .element_cfg import *

__all__ = ["AIRPickSmSB3", "AIREnvSB3"]

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="AIR-v0-SB3",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="isaac_env.air_env_rl:AIREnvSB3",
    kwargs={
        "env_cfg_entry_point": CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

