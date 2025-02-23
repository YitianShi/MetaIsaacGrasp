import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_from_matrix
from isaac_env.element_cfg import *
from isaac_env.utils import *
# initialize warp
from isaac_env.wp_cfg import *
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from isaac_env.air_env_grasp import AIR_RLTaskEnv
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import socket
import pickle
import struct
import open3d as o3d

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
        self.env = gym.make(args_cli.task, cfg=env_cfg)        
        self.device = self.env_unwrapped.device
        self.num_envs = self.env_unwrapped.num_envs
        # Environment index
        self.env_idx = torch.arange(args_cli.num_envs, dtype=torch.int64, device=self.device)
        self.inference_criteria = ~torch.empty(self.num_envs, dtype=torch.bool, device=self.device)

        if use_sb3:
            self._rlg_train(args_cli)

        self.teleop = "Tele" in args_cli.task
            

    def _rlg_train(self, args_cli):
        # directory for logging into
        
            # parse configuration
        env_cfg = parse_env_cfg(
            args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

        # override from command line
        if args_cli.seed is not None:
            agent_cfg["seed"] = args_cli.seed
        
        if args_cli.max_iterations:
            agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

        # specify directory for logging experiments
        self.log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        # dump the configuration into log-directory
        dump_yaml(os.path.join(self.log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(self.log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(self.log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(self.log_dir, "params", "agent.pkl"), agent_cfg)

        # post-process agent configuration
        agent_cfg = process_sb3_cfg(agent_cfg)
        # read configurations about the agent-training
        policy_arch = agent_cfg.pop("policy")
        self.n_timesteps = agent_cfg.pop("n_timesteps")

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(self.log_dir, "videos"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True}
            print_dict(video_kwargs, nesting=4)
            self.env = gym.wrappers.RecordVideo(self.env, **video_kwargs)
        # wrap around environment for rl-games
        self.env = Sb3VecEnvWrapper(self.env)

        # set the seed
        self.env.seed(seed=agent_cfg["seed"])

        if "normalize_input" in agent_cfg:
            self.env = VecNormalize(
                self.env,
                training=True,
                norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
                norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
                clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
                gamma=agent_cfg["gamma"],
                clip_reward=np.inf,
            )

        self.agent = SAC(policy_arch, self.env, verbose=1, **agent_cfg)

        # configure the logger
        new_logger = configure(self.log_dir, ["stdout", "tensorboard"])
        self.agent.set_logger(new_logger)

        # callbacks for agent
        self.checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=self.log_dir, name_prefix="model", verbose=2)
        
    
    def init_run(self):
        """Initialize the simulation loop."""
        # Environment step
        obs_buf = self.env.reset()
        self.obs_buf = obs_buf if use_sb3 else obs_buf[0]
        self.env_unwrapped.update_env_state()
        print("-" * 80)
        print("[INFO]: Reset finish...")
    
    def run_sb3(self):
        # train the agent
        self.agent.learn(total_timesteps=self.n_timesteps, callback=self.checkpoint_callback)
        # save the final model
        self.agent.save(os.path.join(self.log_dir, "model"))
    

    def run(self):
        """Runs the simulation loop."""
        # Get the grasp pose
        actions = self.propose_action()

        # Advance the environment and get the observations
        self.obs_buf, reward_buf, reset_terminated, dones, self.inference_criteria = self.env.step(actions)

        
    def propose_action(self, demo = not use_sb3, get_pcd = False):
        
        # Get the envs that are in the choose object state
        ids = self.env_idx.clone()[self.inference_criteria]
            
        if self.teleop:
            actions = self.env_unwrapped.get_teleop_action(ids, self.obs_buf)

        elif remote_agent:
            actions = self.env_unwrapped.remote_action(ids, self.obs_buf)
            
        elif demo or collect_data:
            actions = self.env_unwrapped.get_grasp_pose_demo(ids, self.obs_buf)
        else:
            # Use policy if not demo:
            # Get the camera data
            rgbs = self.obs_buf[:3]
            depths = self.obs_buf[3:]
            # Get the point cloud data
            pcds = self.env_unwrapped.get_pointcloud_map(ids) if get_pcd else None

            # Get the grasp pose from the policy
            actions = self.policy(rgbs, depths, pcds, ids)

            if save_data:
                for id, grasp_pose, rgb, depth in zip(ids, actions[:, :7], rgbs, depths):
                    self.env_unwrapped.save_data(id, grasp_pose, None, rgb, depth)
        
        return actions

    
    def policy(self, rgbs, depths, pcds, view_poses_rob, ids):
        """
        Get the grasp pose from the policy
        """
        # Get the grasp pose from the policy
        grasp_pose, gripper_state_con = None, None # TODO: Implement the policy

        return torch.cat((grasp_pose, gripper_state_con), dim=-1) if continuous_control else grasp_pose
    

    @property
    def env_unwrapped(self):
        return self.env.unwrapped


