# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, air_env_cfg
from .element_cfg import *
from .air_env_cfg import *
from .utils import *
from .wp_cfg import *
##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="AIR-v0-Grasp",
    # entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    entry_point="isaac_env.air_env_grasp:AIR_RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": air_env_cfg.CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="AIR-v0-Cont",
    # entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    entry_point="isaac_env.air_env_continuous:AIR_RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": air_env_cfg.CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="AIR-v0-Data",
    # entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    entry_point="isaac_env.air_env_data:AIR_RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": air_env_cfg.CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="AIR-v0-Tele",
    # entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    entry_point="isaac_env.air_env_tele:AIR_RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": air_env_cfg.CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)


