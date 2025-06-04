import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401

from datetime import datetime
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry
from isaaclab.utils.math import subtract_frame_transforms, quat_from_matrix
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.utils.dict import print_dict


from isaac_env.air_env_base.element_cfg import *
from isaac_env.utils import *
from isaac_env.air_env_base.wp_cfg import *
from isaac_env.air_env_base.sim import AIRPickSm
from .element_cfg import OBJ_LABLE
from isaac_env.agents.skrl_gaussian_model import PolicyMLP, ValueMLP


from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.models.torch import GaussianMixin
from isaaclab_rl.skrl import SkrlVecEnvWrapper


class AIRPickSmSKRL(AIRPickSm):
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. 
    """

    def __init__(self, args_cli):
        super().__init__(args_cli)
        self.obj_label = OBJ_LABLE

        if self.env_unwrapped.RL_TRAIN_FLAG:
            self._skrl_train(args_cli)
        else:
            self._load_model(args_cli)
            
    def _skrl_train(self, args_cli):
        # directory for logging into
        
        # parse configuration
        env_cfg = parse_env_cfg(
            args_cli.task, self.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")


        # override from command line
        if args_cli.seed is not None:
            agent_cfg["seed"] = args_cli.seed
        
        if args_cli.max_iterations:
            agent_cfg["timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

        self.n_timesteps = agent_cfg["timesteps"]
        agent_cfg["learning_rate"] = float(agent_cfg["learning_rate"])

        # specify directory for logging experiments
        self.log_dir = os.path.join("logs", "skrl", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        # dump the configuration into log-directory
        dump_yaml(os.path.join(self.log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(self.log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(self.log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(self.log_dir, "params", "agent.pkl"), agent_cfg)

        # wrap for video recording
        # if args_cli.video:
        #     video_kwargs = {
        #         "video_folder": os.path.join(self.log_dir, "videos"),
        #         "step_trigger": lambda step: step % args_cli.video_interval == 0,
        #         "video_length": args_cli.video_length,
        #         "disable_logger": True}
        #     print_dict(video_kwargs, nesting=4)
        #     self.env = gym.wrappers.RecordVideo(self.env, **video_kwargs)
        
        self.env = SkrlVecEnvWrapper(self.env)

        # set the seed
        self.env_unwrapped.seed(seed=agent_cfg["seed"])

        pmodel = PolicyMLP(
            observation_space=self.env._unwrapped.single_observation_space,
            action_space=self.env._unwrapped.single_action_space,
            device=self.device,
        )
        vmodel = ValueMLP(
            observation_space=self.env._unwrapped.single_observation_space,
            action_space=self.env._unwrapped.single_action_space,
            device=self.device,
        )

        # configure the logger
        agent_cfg["experiment"]["directory"] = self.log_dir
        agent_cfg["experiment"]["name"] = args_cli.task
        agent_cfg["experiment"]["checkpoint_interval"] = 50000

        # agent
        self.agent = PPO(
            models={
                "policy": pmodel,
                "value": vmodel
            },
            memory=RandomMemory(memory_size=512, num_envs=self.env.num_envs, device=self.device),
            cfg=agent_cfg,
            observation_space=self.env._unwrapped.single_observation_space,
            action_space=self.env._unwrapped.single_action_space,
            device=self.device
        )

        self.agent_cfg = agent_cfg

    
    def _load_model(self, args_cli):
        checkpoint = "2025-04-29_16-31-24" # modify as needed
        path = os.path.join("logs", "sb3", args_cli.task, checkpoint, "model_1000000_steps")
        self.model = PPO.load(path)
    
    def init_run(self):
        """Initialize the simulation loop."""
        # Environment step
        obs_buf = self.env.reset()
        self.obs_buf = obs_buf
        self.env_unwrapped.update_env_state()
        print("-" * 80)
        print("[INFO]: Reset finish...")
        
    def run_rl(self):
        # train the agent
        from skrl.trainers.torch import SequentialTrainer
        trainer = SequentialTrainer(env=self.env, agents=self.agent, cfg=self.agent_cfg)
        trainer.train()
        # save the final model
        self.agent.save(os.path.join(self.log_dir, "final_model"))
    

    def run(self):
        """Runs the simulation loop."""
        # Get the grasp pose
        actions = self.propose_action()

        # Advance the environment and get the observations
        self.obs_buf, reward_buf, reset_terminated, dones, self.inference_criteria = self.env.step(actions)

        
    def propose_action(self):
        required_keys = self.model.observation_space.spaces.keys()
        if isinstance(self.obs_buf, dict):
            obs_dict = self.obs_buf["policy"]
        elif isinstance(self.obs_buf, tuple):
            obs_dict = self.obs_buf[0]["policy"]
        obs_np = {
            k: obs_dict[k].detach().cpu().numpy()
            for k in required_keys if k in obs_dict
        }
        actions, _ = self.model.predict(obs_np, deterministic=True)

        return torch.tensor(actions, device=self.device)
