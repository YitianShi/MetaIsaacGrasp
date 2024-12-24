# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p source/standalone/environments/state_machine/lift_cube_sm.py --num_envs 32

"""

"""Launch Omniverse Toolkit first."""

import argparse
import torch

from omni.isaac.lab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")

parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--task", type=str, default="AIR-v0-Grasp", choices=["AIR-v0-Grasp", "AIR-v0-cont", "AIR-v0-Data", "AIR-v0-Tele"], help="Task name.")

parser.add_argument(
    "--distributed", action="store_true", help="Run training with multiple GPUs or nodes."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.settings
settings = carb.settings.get_settings()

# set different types into different keys
# guideline: each extension puts settings in /ext/[ext name]/ and lists them extension.toml for discoverability
settings.set("/renderer/multiGPU/enabled", True)
settings.set("/renderer/multiGPU/autoEnable", True)
settings.set("/rtx/realtime/mgpu/autoTiling/enabled", True)

from air_sim import AIRPickSm
"""Rest everything else."""

import time
from isaac_env import use_sb3

if __name__ == "__main__":
    # run the main function
    simulator = AIRPickSm(args_cli)
    simulator.init_run()
    if use_sb3:
        simulator.run_sb3()
    else:
        while simulation_app.is_running():
            # record run time
            with torch.inference_mode():
                simulator.run()
    # close the environment
    simulator.close()
    # close sim app
    simulation_app.close()