# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from .. import agents
from isaac_env.utils import *
from .env_cfg import *
from .sim import AIRPickSmSKRL
from .env import AIREnvSKRL
from .element_cfg import *

__all__ = ["AIRPickSmSKRL", "AIREnvSKRL"]

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="AIR-v0-SKRL",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="isaac_env.air_env_skrl:AIREnvSKRL",
    kwargs={
        "env_cfg_entry_point": CellEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

