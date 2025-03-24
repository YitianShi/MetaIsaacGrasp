import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry
from isaaclab.utils.math import subtract_frame_transforms, quat_from_matrix
from isaac_env.air_env_base.element_cfg import *
from isaac_env.utils import *
# initialize warp
from isaac_env.air_env_base.wp_cfg import *
from datetime import datetime
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.utils.dict import print_dict
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaac_env.agents.custom_extractor import CustomExtractor

from isaac_env.air_env_base.sim import AIRPickSm
from .element_cfg import OBJ_LABLE

class AIRPickSmData(AIRPickSm):
    def __init__(self, args_cli):
        super().__init__(args_cli)
        self.obj_label = OBJ_LABLE
    pass