# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot
"""

from isaac_env.air_env_base.element_cfg import *

NUM_OBJS = 1
OBJ_LABLE = [i.split("/")[-2] for i in OBJ_PATH][:NUM_OBJS]