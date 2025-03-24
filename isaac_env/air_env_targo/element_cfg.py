# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt

from isaac_env.air_env_base.element_cfg import *
from .read_targo import read_targo

# Target object path and pose
view_pos_targo = (0.3, 0.18, 0.) # viewpoint wrt. targo object
shift_targo = tuple(a + b for a, b in zip(view_pos_targo, ROBOT_POS))
TARGO_OBJ_PATHS, targo_obj_scales, targo_obj_positions, targo_obj_rotations, occ_targ_max, targo_extrinsic, targo_obj_chosen = read_targo(shift_targo)

# use urdf converter to convert usd to urdf or not
OBJ_PATH = TARGO_OBJ_PATHS
NUM_OBJS = len(TARGO_OBJ_PATHS)
OBJ_LABLE = [i.split("/")[-1].split(".")[0] for i in OBJ_PATH][:NUM_OBJS]

desk_center = torch.tensor(targo_obj_positions.mean(axis=0) - ROBOT_POS, dtype=torch.float32)


TARGO_CFGs_URDF = [
    RigidObjectCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=tgo,
            rigid_props= rigid_props,
            mass_props = sim_utils.MassPropertiesCfg(density=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            fix_base=False,
            force_usd_conversion=True,
            make_instanceable=False,
            semantic_tags=[("class", f"{tgo.split('/')[-2]}"), ("color", "red")],
            scale=[scale] * 3,
            visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0., 0., 0.) if targo_obj_chosen == tgo else (random.random(), random.random(), random.random()),
            metallic=0.5
    ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
        ),
    )
    for tgo, pos, rot, scale in zip(TARGO_OBJ_PATHS, targo_obj_positions, targo_obj_rotations, targo_obj_scales)
]

OBJ_CFGs = TARGO_CFGs_URDF

################# REMOTE AGENT CONFIG #################
REMOTE_AGENT = False
# server initialization
if REMOTE_AGENT:
    import socket, sys
    chunk_size = 4096
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allows the socket to reuse the address
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(('172.22.222.222', 8081))
        server.listen(1)
        print("Simulation server is waiting for the agent...")
        conn, addr = server.accept()
        print(f"Connected to agent at {addr}. Adding remote agent inference to teleoperation interface.")
    except socket.error as e:
        print(f"Failed to start the server: {e}")
        server.close()
        sys.exit(1)  # Exit if we cannot start the server
