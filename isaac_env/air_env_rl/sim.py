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

class AIRPickSmRL(AIRPickSm):
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. 

    """

    def __init__(self, args_cli):
        super().__init__(args_cli)
        self.obj_label = OBJ_LABLE
        self._rlg_train(args_cli)
            

    def _rlg_train(self, args_cli):
        # directory for logging into
        
            # parse configuration
        env_cfg = parse_env_cfg(
            args_cli.task, self.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
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

        self.agent = PPO( # AC共享特征提取
            policy="MultiInputPolicy", # 多输入策略
            env=self.env,
            verbose=1,
            policy_kwargs=dict(
                features_extractor_class=CustomExtractor,
                features_extractor_kwargs={},
                net_arch={'pi': [256, 256, 128, 64, 7], 'vf': [256, 256, 128, 64, 1]}, # 最后这个1可以不写
            ),
            **agent_cfg
        )


        # configure the logger
        new_logger = configure(self.log_dir, ["stdout", "tensorboard"])
        self.agent.set_logger(new_logger)

        # callbacks for agent
        self.checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=self.log_dir, name_prefix="model", verbose=2)
    
    
    def init_run(self):
        """Initialize the simulation loop."""
        # Environment step
        obs_buf = self.env.reset()
        self.obs_buf = obs_buf
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

        
    def propose_action(self, get_pcd = False):
        
        # Get the envs that are in the choose object state
        ids = self.env_idx.clone()[self.inference_criteria]
            
        # Use policy if not demo:
        # Get the camera data
        actions = self.policy(ids, get_pcd)

        if True:
            for id, grasp_pose, rgb, depth in zip(ids, actions[:, :7], rgbs = self.obs_buf['rgb'], depths = self.obs_buf['distance_to_image_plane']):
                self.env_unwrapped.save_data(id, grasp_pose, None, rgb, depth)
        
        return actions

    
    def policy(self, ids, get_pcd, view_poses_rob=None):
        """
        Get the grasp pose from the policy
        """
        rgbs = self.obs_buf['rgb'][ids]
        depths = self.obs_buf['distance_to_image_plane'][ids]
        pcds = self.obs_buf['pcd'][ids] if get_pcd else None
        rgbs = rgbs.float() / 255.0
        depths = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)

        obs_dict = {
            "rgb": rgbs.to(self.device),
            "distance_to_image_plane": depths.to(self.device),
        }

        if pcds is not None:
            obs_dict["pcd"] = pcds.to(self.device)
        
        # Get the grasp pose from the policy
        grasp_pose, gripper_state_con = None, None # TODO: Implement the policy
        actions, _ = self.agent.predict(obs_dict, deterministic=True) 
        grasp_pose, gripper_state_con = actions[:, :7], actions[:, 7] 

        return torch.cat((grasp_pose, gripper_state_con), dim=-1)


    @property
    def env_unwrapped(self):
        return self.env.unwrapped


