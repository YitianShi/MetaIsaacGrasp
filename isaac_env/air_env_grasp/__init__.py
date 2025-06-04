# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from .. import agents
from isaac_env.utils import *
from .env_cfg import *
from .sim import AIRPickSmGrasp
from .env import AIREnvGrasp
from .element_cfg import *

__all__ = ["AIRPickSmGrasp", "AIREnvGrasp"]

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="AIR-v0-Grasp",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="isaac_env.air_env_grasp:AIREnvGrasp",
    kwargs={
        "env_cfg_entry_point": CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

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

gym.register(
    id="AIR-v0-Data",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="isaac_env.air_env_data:AIREnvData",
    kwargs={
        "env_cfg_entry_point": CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="AIR-v0-Tele",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="isaac_env.air_env_tele:AIREnvTele",
    kwargs={
        "env_cfg_entry_point": CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)


