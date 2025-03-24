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

class AIRPickSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. 

    """

    def __init__(self, args_cli):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # parse configuration
        env_cfg = parse_env_cfg(
            args_cli.task,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        self.env = gym.make(args_cli.task, cfg=env_cfg, save_camera_data=args_cli.save_camera_data)        
        self.device = self.env_unwrapped.device
        self.num_envs = self.env_unwrapped.num_envs
        # Environment index
        self.env_idx = torch.arange(args_cli.num_envs, dtype=torch.int64, device=self.device)
        self.inference_criteria = ~torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
    
    def init_run(self):
        """Initialize the simulation loop."""
        # Environment step
        obs_buf = self.env.reset()
        self.obs_buf = obs_buf[0]
        self.env_unwrapped.update_env_state()
        print("-" * 80)
        print("[INFO]: Reset finish...")
    
    
    def run(self):
        """Runs the simulation loop."""
        # Get the grasp pose
        actions = self.propose_action()

        # Advance the environment and get the observations
        self.obs_buf, reward_buf, reset_terminated, dones, self.inference_criteria = self.env.step(actions)

        
    def propose_action(self, get_pcd = False):
        
        # Get the envs that are in the choose object state
        ids = self.env_idx.clone()[self.inference_criteria]
        actions = self.env_unwrapped.get_action(ids, self.obs_buf)
        
        return actions

    @property
    def env_unwrapped(self):
        return self.env.unwrapped


