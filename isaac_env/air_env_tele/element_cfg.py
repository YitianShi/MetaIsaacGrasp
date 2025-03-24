# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot
"""

from isaac_env.air_env_base.element_cfg import *

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