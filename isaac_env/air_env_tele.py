"""
This script demonstrates how to run the RL environment for the cartpole balancing task.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import os
import random
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import open3d as o3d

import socket
import pickle
import struct

# from omni.isaac.lab.envs.mdp.rewards import action_rate_l2, action_l2
import pandas as pd
import torch
from omni.isaac.lab.controllers import DifferentialIKController
from omni.isaac.lab.devices import Se3Keyboard

# from omni.isaac.lab.controllers.rmp_flow import *
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_mul, combine_frame_transforms, apply_delta_pose

from metagraspnet.Scripts.visualize_labels import (
    create_contact_pose,
    from_contact_to_6D,
    read_in_mesh_config,
)

from isaac_env import *
from omni.isaac.lab.envs import ManagerBasedRLEnv

##
# Pre-defined configs
##


class AIR_RLTaskEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: CellEnvCfg, render_mode: str | None = None, **kwargs):
        env_cfg = CellEnvCfg()
        # Setup RL environment
        super().__init__(cfg, render_mode)
        env_cfg.scene.num_envs = self.num_envs
        self.dt = float(self.physics_dt * self.cfg.decimation)

        # Define entity
        self.robot = self.scene[robot_name]
        self.ee_frame = self.scene["ee_frame"]
        if not disable_camera:
            self.camera = [self.scene[f"camera_{i}"] for i in range(n_multiple_cam)]

        if self.sim.has_gui():
            cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
            cfg.markers["hit"].radius = 0.002
            self.pc_markers = [VisualizationMarkers(cfg) for _ in range(self.num_envs)]

        self.robot_origin = torch.tensor(
            self.robot.cfg.init_state.pos, device=self.device
        )

        # Specify UR self.robot-specific parameters
        self.robot_entity_cfg = SceneEntityCfg(
            robot_name, joint_names=ARM_JOINT, body_names=ee_name
        )
        self.objs = [self.scene[object_name + f"_{i}"] for i in range(num_objs)]

        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 2, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)
        self.obj_grasp_pose = torch.zeros((self.num_envs, 7), device=self.device)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)

        self.ee_quat_default = torch.tensor(
            ee_grasp_quat_default, device=self.device
        ).repeat(self.num_envs, 1)
        self.ee_quat_default_wp = wp.from_torch(
            self.ee_quat_default[:, [1, 2, 3, 0]], wp.quat
        )

        # Successive grasp failure recorder, if more than n times of failure, reset the environment
        self.successive_grasp_failure = torch.zeros(self.num_envs, device=self.device)
        self.successive_grasp_failure_wp = wp.from_torch(
            self.successive_grasp_failure, wp.float32
        )

        # Record episodes and steps
        self.epi_step_count = torch.zeros(
            (self.num_envs, 2), dtype=torch.int32, device=self.device
        )
        self.epi_count_wp = wp.from_torch(self.epi_step_count[:, 0], wp.int32)
        self.step_count_wp = wp.from_torch(self.epi_step_count[:, 1], wp.int32)

        # Initialize the teleoperation interface
        self.teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05, rot_sensitivity=0.05
        )
        self.teleop_interface.add_callback("L", self._manual_reset_env)
        self.teleop_interface.add_callback("R", self._manual_reset)

        self.teleop_interface.reset()

        print(self.teleop_interface)

        # Create controller
        if CONTROLLER == "RMPFLOW":
            pass
            # self.controller = RmpFlowController(UR_RMPFLOW_CFG, device=self.device)
            # self.controller.initialize(f"{self.scene.env_regex_ns}/"+robot_name)
        else:
            self.controller = DifferentialIKController(
                UR_IK_CFG, num_envs=self.num_envs, device=self.device
            )

        # Markers

        # Visualize goal of the end-effector
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.goal_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    
        # Visualize the current grasp pose
        frame_marker_grasp = FRAME_MARKER_CFG.copy()
        frame_marker_grasp.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.grasp_marker = VisualizationMarkers(
            frame_marker_grasp.replace(prim_path="/Visuals/grasp"))
        
        # Camera markers
        frame_marker_cfg_cam = FRAME_MARKER_CFG.copy()
        frame_marker_cfg_cam.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.camera_markers =[VisualizationMarkers(
            frame_marker_cfg_cam.replace(prim_path=f"/Visuals/camera_{cam_id}")
        ) for cam_id in range(n_multiple_cam)]

        # Resolving the self.scene entities
        self.robot_entity_cfg.resolve(self.scene)

        # Obtain the frame index of the end-effector
        # For a fixed base self.robot, the frame index is one less than the body index. This is because
        # The root body is not included in the returned Jacobians.
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # Get initial joint positions
        self.joint_pos_init = self.robot.data.default_joint_pos.clone()
        self.joint_vel_init = self.robot.data.default_joint_vel.clone()

        # Environment index
        self.env_idx = torch.arange(
            self.num_envs, dtype=torch.int64, device=self.device
        )

        # Define the reset triggers
        self.env_reset_id = torch.arange(
            self.num_envs, dtype=torch.int64, device=self.device
        )

        # Reward recorder
        self.reward_recorder = torch.zeros(
            (self.num_envs, 100, step_total), device=self.device
        )

        self.obj_drop_pose = torch.tensor(obj_drop_pose, device=self.device)[None, ...]

        # grasp and approach pose
        self.grasp_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.obj_chosen = torch.full(
            (self.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        
        # server initialization
        if remote_agent:
            self.chunk_size = 4096
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self.server.bind(('localhost', 8081))
            except:
                print("Server already running")
            self.server.listen(1)
            print("Simulation server is waiting for the agent...")
            self.conn, self.addr = self.server.accept()
            print(f"Connected to agent at {self.addr}")

    def update_env_state(self):
        """Update the environment state before taking action.
        Args:
            sm_wait_time: The time the robot needs to wait before taking action.
        Returns:
            obj_graspable: The objects that are graspable.
            env_reachable: The environments that are reachable.
            env_reachable_and_stable: The environments that are reachable and stable.
        """
        # Grasp target recorder (position only)
        object_pos = torch.zeros((self.num_envs, num_objs, 4), device=self.device)

        for id_obj in range(num_objs):
            # Record the object position
            object_pos[:, id_obj, :3] = self._get_obj_pos(id_obj)
            # Record the object velocity
            obj_vel_b = self._get_obj_vel(id_obj)
            object_pos[:, id_obj, -1] = torch.norm(obj_vel_b, dim=-1)

        # Object reachable
        # Initial condition: Check if the object is below a certain height limit
        self.obj_reachable = (
            (object_pos[:, :, 2] < obj_height_limit)
            & (object_pos[:, :, 2] > -5e-2)
            & (object_pos[:, :, 0] > ee_goals_default[0][0])
            & (object_pos[:, :, 0] < ee_goals_default[0][1])
            & (object_pos[:, :, 1] > ee_goals_default[1][0])
            & (object_pos[:, :, 1] < ee_goals_default[1][1])
        )

        # Object stable
        self.obj_stable = object_pos[:, :, -1] < obj_vel_limit

        # Stable objects are either slow speed or not reachable
        self.obj_stable = self.obj_stable | ~self.obj_reachable

        # At least one object is reachable
        self.env_reachable = self.obj_reachable.any(dim=1)

        # Object graspable is the object that is both reachable and stable
        self.obj_graspable = self.obj_reachable & self.obj_stable

        # In stable environment,
        # all objects are stable (to take picture) and at least one object is reachable
        # so robot can take photo and choose the object
        self.env_reachable_and_stable = self.obj_stable.all(dim=1) & self.env_reachable

        # Minimum time for the robot to be in the state of "start" before it can reach
        self.env_reachable_and_stable = self.env_reachable_and_stable & (self.sm_wait_time > 1.0)


    def _advance_state_machine(self):
        """Compute the desired state of the robot's end-effector and the gripper."""

        # get the end-effector velocity
        ee_vel = self._get_ee_vel()

        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = self._get_ee_pose()[:, [0, 1, 2, 4, 5, 6, 3]]
        self.grasp_pose = self.grasp_pose[:, [0, 1, 2, 4, 5, 6, 3]].to(self.device)

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        ee_vel_wp = wp.from_torch(ee_vel, wp.float32)
        env_reachable_wp = wp.from_torch(self.env_reachable.contiguous(), wp.bool)
        env_reachable_and_stable_wp = wp.from_torch(self.env_reachable_and_stable.contiguous(), wp.bool)
        grasp_pose_wp = wp.from_torch(self.grasp_pose, wp.transform)

        
        wp.launch(
            kernel=infer_state_machine_tele,
            dim=self.num_envs,
            inputs=[
                # environment state machine recorders
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                # environment time states
                self.successive_grasp_failure_wp,
                self.epi_count_wp,
                self.step_count_wp,
                # environment physical states
                env_reachable_wp,
                env_reachable_and_stable_wp,
                # current robot end effector state
                ee_pose_wp,
                ee_vel_wp,
                # desired robot end effector state
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.ee_quat_default_wp,
                # proposed grasp pose
                grasp_pose_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        ee_pose = ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        self.grasp_pose = self.grasp_pose[:, [0, 1, 2, 6, 3, 4, 5]]

        # convert to torch
        return torch.cat((des_ee_pose, self.des_gripper_state.unsqueeze(-1)), -1)

    def _action_plan(self):
        # Compute the joint commands
        if CONTROLLER == "RMPFLOW":
            joint_pos_des, joint_vel_des = self.controller.compute()
        else:
            jacobian = self.robot.root_physx_view.get_jacobians()[
                :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids
            ]

            joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

            ee_pose_w = self.robot.data.body_state_w[
                :, self.robot_entity_cfg.body_ids[0], 0:7
            ].clone()
            root_pose_w = self.robot.data.root_state_w[:, 0:7].clone()

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3],
                ee_pose_w[:, 3:7],
            )

            joint_pos_des = self.controller.compute(
                ee_pos_b, ee_quat_b, jacobian, joint_pos
            )
            self.joint_vel_des = None

        joint_pos_des_rel = joint_pos_des - self.joint_pos_init[:, :6]

        joint_pos_des_rel[self.sm_state == STATE_MACHINE["init"]] *= 0.0
        joint_pos_des_rel[self.sm_state == STATE_MACHINE["init_env"]] *= 0.0
        joint_pos_des_rel[self.sm_state == STATE_MACHINE["start"]] *= 0.0

        return joint_pos_des_rel

    def step(self, grasp_pose, policy_inference_criteria=torch.tensor([])):
        # Get the grasp pose from the policy
        self.grasp_pose = grasp_pose

        # Loop until the simulation frames until policy inference criteria is met
        while not policy_inference_criteria.any():

            # Update the environment state to know whether the environment is graspable or stable
            self.update_env_state()

            # Advance the state machine
            action_env = self._advance_state_machine()

            # Set the command
            self.controller.set_command(action_env[:, :7])
            # Compute the kinematics

            joint_pos_des_rel = self._action_plan()
            # Add the gripper command
            joint_pos_des_rel = torch.concatenate(
                (joint_pos_des_rel, action_env[:, -1:]), dim=1
            )

            # Step the simulation
            obs_buf, reward_buf, reset_terminated, reset_time_outs, extras = (
                super().step(joint_pos_des_rel)
            )
            # Visualize the markers
            self._vis(action_env[:, :7])

            # Reset
            self._summerize_and_reset(reward_buf)

            # Update policy inference criteria
            policy_inference_criteria = (
                self.sm_state == STATE_MACHINE["execute"]
            )

        # Get the camera info
        self.camera_info = [self.camera[can_id].data.info for can_id in range(n_multiple_cam)]
        #ids = self.env_idx.clone()[self.inference_criteria]
        if use_sb3:
            return obs_buf, reward_buf, reset_terminated, reset_time_outs, dict()

        return (
            obs_buf,
            reward_buf,
            reset_terminated,
            reset_time_outs,
            policy_inference_criteria,
        )

    def to_np(self, obs_buf):
        """Convert the observation buffer to numpy arrays."""
        for key, data in obs_buf.items():
            obs_buf[key] = data.to(self.device)
        return obs_buf

    def get_grasp_pose_demo(self, ids, obs_buf):
        """Get the grasp pose from the camera data."""

        # Get the camera data
        data_cam = obs_buf["policy"]

        # Get the instance segmentation
        instances_all_env = (
            data_cam["instance_segmentation_fast"]
            if "instance_segmentation_fast" in data_cam.keys()
            else None
        )

        # Get the depth image
        assert "distance_to_image_plane" in data_cam.keys(), "No depth image found"
        pcds_all_env = data_cam["pcd"]
        depths_all_env = data_cam["distance_to_image_plane"]

        # Get the normals
        normals_all_env = data_cam["normals"]

        for env_id in ids:
            pcds = pcds_all_env[env_id] if pcds_all_env is not None else None
            instances = instances_all_env[env_id] if instances_all_env is not None else None
            depths = depths_all_env[env_id] if depths_all_env is not None else None
            normals = normals_all_env[env_id] if normals_all_env is not None else None
            id_to_labels = []
            for cam_id in range(n_multiple_cam):
                pcd = pcds[cam_id]
                # Get the instance segmentation and grasp pose
                try:
                    # Get instance segmentation
                    instance = instances[cam_id].squeeze(-1)
                    # Get the object id from the instance segmentation
                    id_to_label = self.camera_info[cam_id][env_id]["instance_segmentation_fast"]["idToLabels"]
                    id_to_labels.append(id_to_label)
                    if cam_id == 0:
                        # Get the pixels with objects
                        grasp_criteria = instance > 1
                        graspable_pixels = grasp_criteria.nonzero()
                        # Randomly choose a grasp point
                        grasp_pos_img = graspable_pixels[
                            random.randint(0, graspable_pixels.size(0) - 1)
                        ]
                        # Get the object id of the grasp pixel
                        self.obj_chosen[env_id] = int(
                            id_to_label[
                                int(instance[grasp_pos_img[0], grasp_pos_img[1]])
                            ].split("_")[-1]
                        )
                    else:
                        grasp_pos_img = None
                except:
                    # if no instance segmentation available or no graspable pixels on the image
                    print("No object instance in the image found")
                    instance, id_to_label = None, None
                    # Randomly choose an object
                    if cam_id == 0:
                        self.obj_chosen[env_id] = (
                            random.choice(self.obj_graspable[env_id].nonzero())
                            if self.obj_graspable[env_id].any()
                            else random.randint(0, num_objs - 1)
                        )
                        # Get the grasp point as the center of the object in the robot frame
                        grasp_pos = self._get_obj_pos(self.obj_chosen[env_id])[env_id]
                        # Get the grasp point in the image
                        grasp_pos_img = robot_point_to_image(
                            grasp_pos, self.get_camera_pose(cam_id, env_id)
                        )
                    else:
                        grasp_pos_img = None

                if read_from_hdf5:

                    # Get id of chosen object
                    obj_id = OBJ_LABLE[self.obj_chosen[env_id]]
                    # Get grasp pose relative to object
                    self.grasp_pose[env_id] = self.get_grasp_poses_from_hdf5(obj_id, 
                                                                            env_id, 
                                                                            data_cam["rgb"][env_id].cpu().numpy(), 
                                                                            self.get_camera_pose(cam_id, env_id))

                    grasp_pos_img = robot_point_to_image(
                        self.grasp_pose[env_id, :3], self.get_camera_pose(cam_id, env_id)
                    )

                    """grasp_viz = cv2.circle(
                        data_cam["rgb"][env_id].cpu().numpy(),
                        (int(grasp_pos_img[0]), int(grasp_pos_img[1])),
                        2,
                        (0, 0, 255),
                        -1,
                        )
                    cv2.imwrite(f"env_{env_id}_grasp_viz.png", grasp_viz)
                    """

                else:
                    normal = normals[cam_id] if "normals" in data_cam.keys() else None   
                    if cam_id == 0:                     
                        # Get the grasp pose at the grasp point, which is opposite to the normal
                        if normal is not None:
                            grasp_normal = normal[grasp_pos_img[0], grasp_pos_img[1]]
                            grasp_quat = perpendicular_grasp_orientation(grasp_normal)
                        else:
                            # if the grasp point is out of the image, use default grasp orientation
                            grasp_quat = self.ee_quat_default[env_id]

                        grasp_pos = pcd[grasp_pos_img[0], grasp_pos_img[1]]
                        self.grasp_pose[env_id] = torch.cat((grasp_pos, grasp_quat), -1)

            if save_data:
                
                # Get the rgb image
                rgbs = data_cam["rgb"][env_id] if "rgb" in data_cam.keys() else None
                self.save_data(
                    env_id, 
                    self.grasp_pose[env_id], 
                    grasp_pos_img, 
                    rgbs, 
                    pcds,
                    depths, 
                    normals, 
                    instances, 
                    id_to_labels,
                    )

        return self.grasp_pose

    def get_teleop_action(self, ids, obs_buf):
        delta_pose, gripper_command = self.teleop_interface.advance()
        delta_pose = delta_pose.astype("float32")
        # print(f"Delta pose: {delta_pose}")
        # convert to torch
        delta_pose = torch.tensor(delta_pose, device=self.device).repeat(ids.shape[0], 1)
        # resolve gripper command
        gripper_vel = torch.zeros(ids.shape[0], 1, device=delta_pose.device)
        gripper_vel[:] = -1.0 if gripper_command else 1.0
        # compute actions
        curr_pose = self.grasp_pose[ids]
        actions = apply_delta_pose(curr_pose[:, :3], curr_pose[:, 3:], delta_pose)
        actions = torch.cat(actions, dim=-1)
        # update the grasp pose
        self.grasp_pose[ids] = actions.to(self.device)
        return self.grasp_pose

    def remote_action(self, ids, obs_buf):
        data = obs_buf["policy"]["pcd"][ids]
        env_num, cam_id, h, w, _ = data.shape
        data = data.view(env_num * cam_id, h * w, 3)
        data_o3d = [o3d.geometry.PointCloud() for _ in range(env_num * cam_id)]
        
        data = (data*1000).to(torch.int16) if data.dtype == torch.float32 else data
        data = pickle.dumps(data.cpu().numpy())
                    
        # Send the data to the agent
        print("Sending data to agent...")
        data_length = struct.pack('>I', len(data)) 
        self.conn.sendall(data_length + data)
        
        # Receive the actions from the agent
        print("Receiving actions from agent...")
        data = b""
        data_length_bin = None
        while not data_length_bin:
            data_length_bin = self.conn.recv(4)
        data_length = struct.unpack('>I', data_length_bin)[0]
        while len(data) < data_length:
            packet = self.conn.recv(self.chunk_size)
            if not packet:
                break
            data += packet
        actions = pickle.loads(data)
        no_action = actions[:, 3, -1] == 0
        rotactions = quat_from_matrix(actions[:, :3, :3])
        translation = actions[:, :3, 3]
        actions = torch.cat((translation, rotactions), dim=-1)
        if no_action.any():
            no_action_ids = ids[no_action]
            print("Agent failed to provide actions.")
            actions_remains = self.get_grasp_pose_demo(no_action_ids, self.obs_buf)[no_action_ids]
            actions[no_action] = actions_remains
        
        self.grasp_pose[ids] = actions.to(self.device)
        return self.grasp_pose
    
    def _manual_reset_env(self):
        """Manual reset of the environment."""
        self.teleop_interface.reset()
        self.sm_state[self.sm_state == STATE_MACHINE["execute"]] = STATE_MACHINE["init_env"]
    
    def _manual_reset(self):
        """Manual reset of the environment."""
        self.teleop_interface.reset()
        self.sm_state[self.sm_state == STATE_MACHINE["execute"]] = STATE_MACHINE["init"]

    def _summerize_and_reset(self, reward_buf):
        """
        Calculate the reward and
        reset the indexed enviroments
        """
        # Judge in the terminate state
        judge_reward = self.sm_state == STATE_MACHINE["execute"]
        # Calculate the reward
        if reward_buf.any():
            self._record_reward(judge_reward)
            self._reset_robot(reward_buf.bool())

        # Get the reset id from the state machine
        init_id = self.env_idx.clone()[self.sm_state == STATE_MACHINE["init"]]
        init_env_id = self.env_idx.clone()[self.sm_state == STATE_MACHINE["init_env"]]

        # Reset the robot, environment and the teleoperation interface
        self._reset_robot(init_id)
        self._reset_idx(init_env_id)
        #self.teleop_interface.reset()

        # To stabilize the robot, set the grasp pose to the current pose during the start state
        self.grasp_pose[self.sm_state == STATE_MACHINE["start"]] = self._get_ee_pose()[self.sm_state == STATE_MACHINE["start"]]

    def _record_reward(self, judge_reward):
        """Summarize the reward and print the success message."""
        for i, env_success in enumerate(self.reward_buf):
            if env_success and judge_reward[i]:
                self.reward_recorder[
                    i, self.epi_step_count[i, 0], self.epi_step_count[i, 1]
                ] += self.reward_buf[i]
                # Move object to somewhere away from the bin
                drop_pose_curr = self.obj_drop_pose.clone()
                id_tensor = torch.tensor([i], device=self.device)
                drop_pose_curr[:, :3] += self.scene.env_origins[i]
                self.scene.rigid_objects[
                    f"obj_{int(self.obj_chosen[i])}"
                ].write_root_state_to_sim(drop_pose_curr, id_tensor)
                print(
                    f"[INFO] Env {i} succeeded in "
                    + f"Episode {self.epi_step_count[i, 0]} "
                    + f"Step {self.epi_step_count[i, 1]} ! "
                    + f"Current reward: {torch.sum(self.reward_recorder[i, self.epi_step_count[i, 0]])} "
                )
                self.sm_state[i] = STATE_MACHINE["init"]
                self.obj_chosen[i] = -1

    def _reset_robot(self, robot_reset_id):
        """
        Reset robot states
        """
        self.robot.write_joint_state_to_sim(
            self.joint_pos_init[robot_reset_id],
            self.joint_vel_init[robot_reset_id],
            env_ids=robot_reset_id,
        )
        self.robot.reset(robot_reset_id)

        # Reset the controller
        if CONTROLLER == "RMPFLOW":
            self.controller.reset_idx(robot_reset_id)
        else:
            self.controller.reset(robot_reset_id)

    def _vis(self, ee_goals):
        """
        Visualize markers
        """
        # Obtain quantities from simulation
        self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        #
        
        self.goal_marker.visualize(
            ee_goals[:, 0:3] + self.scene.env_origins + self.robot_origin,
            ee_goals[:, 3:7],
        )
        self.grasp_marker.visualize(
        self._get_ee_pose()[:, :3] + self.scene.env_origins + self.robot_origin,
        self._get_ee_pose()[:, 3:],
        )

        # Update camera marker
        for cam_id in range(n_multiple_cam):
            self.camera_markers[cam_id].visualize(
                self.camera[cam_id].data.pos_w.clone(),
                self.camera[cam_id].data.quat_w_ros.clone(),
            )
            
        
    def save_data(
        self,
        env_id,
        grasp_pose,
        grasp_pos_img,
        rgbs,
        pcds,
        depths=None,
        normals=None,
        instances=None,
        id_to_labels=None,
    ):

        episode, step = self.epi_step_count[env_id, 0], self.epi_step_count[env_id, 1]

        if id_to_labels is not None:
            for id_to_label in id_to_labels:
                for k, v in id_to_label.items():
                    id_to_label[k] = (
                        OBJ_LABLE[int(v.split("_")[-1])] if "obj" in v else "-1"
                    )

        if pcds is not None:
            for cam_id in range(n_multiple_cam):
                dp = cv2.normalize(pcds[cam_id][..., -1].cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                dp = cv2.cvtColor(dp, cv2.COLOR_GRAY2RGB)
                if grasp_pos_img is not None:
                    cv2.circle(dp, (int(grasp_pos_img[1]), int(grasp_pos_img[0])), 5, (0, 0, 255), -1)
                # cv2.imwrite(f'{IMG_PATH}/env_{env_id}_epi_{episode}_step_{step}_camera_{cam_id}_depth.png', dp) #change to the dynamic name back
                rgb = cv2.cvtColor(rgbs[cam_id].cpu().numpy(), cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f'{IMG_PATH}/env_{env_id}_epi_{episode}_step_{step}_camera_{cam_id}_rgb.png', rgb)


        # Record object poses of the scene
        poses = []
        scene_obj_id = []
        for obj in range(num_objs):
            pose = self._get_obj_pose(obj, env_id)
            if pose[2] > -5e-2:
                # meter to centimeter
                pose[:3] *= 100
                # Get the object pose in 4x4 matrix
                pose = pose_vector_to_transformation_matrix(pose)                    
                poses.append(pose)
                scene_obj_id.append(OBJ_LABLE[obj])
        # Get the camera pose
        if len(poses) > 0: 
            obj_poses_robot = torch.stack(poses)
        else:
            obj_poses_robot = None
            print(f"{IMG_PATH}/env_{env_id}_epi_{episode}_step_{step}: No object on the table")
            
        data_to_save = {
                f"camera_{cam_id}": {
                                    "camera_intrinsics": self.camera[cam_id].data.intrinsic_matrices[env_id],
                                    "camera_pose": self.get_camera_pose(cam_id, env_id),
                                    "rgb": rgbs[cam_id] if rgbs is not None else None,
                                    "depth": depths[cam_id] if pcds is not None else None,
                                    "normal": normals[cam_id] if normals is not None else None,
                                    "instance": instances[cam_id] if instances is not None else None,
                                    "id_to_labels": id_to_labels[cam_id] if id_to_labels is not None else None,
                                    "pcd": pcds[cam_id] if pcds is not None else None,
                                     } 
                for cam_id in range(n_multiple_cam)
            }
        
        data_to_save["obj_poses_robot"] = obj_poses_robot
        data_to_save["obj_id"] = scene_obj_id
            
        if grasp_pose is not None:
            data_to_save["grasp_pose"] = grasp_pose
        
        torch.save(data_to_save,f"{IMG_PATH}/env_{env_id}_epi_{episode}_step_{step}_data.pt")
        print(f"{IMG_PATH}/env_{env_id}_epi_{episode}_step_{step}: Saved data")

        

    def _get_obj_pos(self, id_obj):
        root_pose_w = self.robot.data.root_state_w[:, 0:3].clone()
        return self.objs[id_obj].data.root_state_w[:, 0:3].clone() - root_pose_w

    def _get_obj_pose(self, id_obj, id_env):
        root_pose_w = self.robot.data.root_state_w[id_env, 0:3].clone()
        obj_pos = self.objs[id_obj].data.root_state_w[id_env, 0:3].clone() - root_pose_w
        obj_quat = self.objs[id_obj].data.root_state_w[id_env, 3:7].clone()
        return torch.cat((obj_pos, obj_quat), -1)

    def _get_obj_vel(self, id_obj):
        return self.objs[id_obj].data.root_state_w[:, 7:].clone()

    def _get_ee_pose(self):
        view_pos_rob = self.ee_frame.data.target_pos_source.clone()[:, 0, :]
        view_quat_rob = self.ee_frame.data.target_quat_source.clone()[:, 0, :]
        return torch.cat((view_pos_rob, view_quat_rob), -1)

    def _get_ee_vel(self):
        ee_vel = self.robot.data.body_state_w[
            :, self.robot_entity_cfg.body_ids[0], 7:
        ].clone()
        ee_vel_abs = torch.norm(ee_vel, dim=-1)
        return ee_vel_abs

    def get_camera_pose(self, cam_id, env_id = None):
        view_pos_w = self.scene[f"camera_{cam_id}"].data.pos_w.clone()
        view_quat_w = self.scene[f"camera_{cam_id}"].data.quat_w_ros.clone()
        view_pos_rob = view_pos_w - self.scene[robot_name].data.root_state_w[:, 0:3].clone()
        view_pose_rob = torch.cat((view_pos_rob, view_quat_w), -1)
        return view_pose_rob[env_id] if env_id is not None else view_pose_rob

    def get_pointcloud_map(self, ids, cam_id = 0, vis=True):
        pcds = []
        for env_id in ids:
            depth = self.camera[cam_id].data.output["distance_to_image_plane"][env_id]
            pointcloud = create_pointcloud_from_depth(
                intrinsic_matrix=self.camera.data.intrinsic_matrices[env_id],
                depth=depth,
                position=self.camera[cam_id].data.pos_w[env_id],
                orientation=self.camera[cam_id].data.quat_w_ros[env_id],
                device=self.device,
            )
            if pointcloud.size()[0] > 0 and vis and self.sim.has_gui():
                indices = torch.randperm(pointcloud.size()[0])[:5000]
                sampled_point_cloud = pointcloud[indices]
                self.pc_markers[env_id].visualize(translations=sampled_point_cloud)
            pcds.append(pointcloud.view(cam_width, cam_height, 3).permute(1, 0, 2))
        return pcds

    def rep_write(self, obs_buf, ids):
        # Get the view pose
        camera_info = self.camera.data.info

        for id in ids:
            episode, step = self.epi_step_count[id]

            # Write the replicator output
            rep_output = {"annotators": {}}
            data_cam = {}
            single_cam_data = convert_dict_to_backend(
                obs_buf["policy"], backend="numpy"
            )
            for key, data in zip(single_cam_data.keys(), single_cam_data.values()):
                info = camera_info[id][key]
                if info is not None:
                    rep_output["annotators"][key] = {
                        "render_product": {"data": data[id], **info}
                    }
                else:
                    rep_output["annotators"][key] = {
                        "render_product": {"data": data[id]}
                    }
                data_cam[key] = data
            rep_output["trigger_outputs"] = {
                "on_time": f"epi_{episode}_step_{step}_env"
            }

            self.rep_writer.write(rep_output)

    def recorder(self, file_name=f"{HOME_PATH}results.csv"):
        # Define the dictionary with the new data
        environment_ids = self.env_reset_id.cpu().numpy()
        episode_numbers = self.epi_step_count[self.env_reset_id, 0].cpu().numpy()
        grasp_tensor = self.reward_recorder[self.env_reset_id].cpu().numpy()

        # Check if environment_ids is empty and skip if true
        if len(environment_ids) == 0 or self.count == 0:
            return  # Exit the function early

        # Ensure the environment IDs and episode numbers match the first dimension of the grasp tensor
        if not (len(environment_ids) == len(episode_numbers) == grasp_tensor.shape[0]):
            raise ValueError(
                "Length of environment_ids and episode_numbers must match the first dimension of grasp_tensor."
            )

        # Prepare data for DataFrame construction
        data = {
            "Environment ID": np.repeat(
                environment_ids,
                [len(grasp_tensor[i]) for i in range(len(environment_ids))],
            ),
            "Episode Number": np.repeat(
                episode_numbers,
                [len(grasp_tensor[i]) for i in range(len(episode_numbers))],
            ),
            "Step Number": [
                step for episode in grasp_tensor for step in range(len(episode))
            ],
            "Grasp Success": [
                success for episode in grasp_tensor for success in episode
            ],
        }

        # Convert the dictionary to a DataFrame
        df_new_data = pd.DataFrame(data)

        # Check if the Excel file exists
        if os.path.exists(file_name):
            # Read the existing data
            df_existing = pd.read_csv(file_name)
            # Append new data
            df_combined = pd.concat([df_existing, df_new_data], ignore_index=True)
            # Write combined data back to the Excel file
            df_combined.to_csv(file_name, index=False)
        else:
            # If the file does not exist, create and write the data
            df_new_data.to_csv(file_name, index=False)


    def get_grasp_poses_from_hdf5(self, obj_id, env_id, img, camera_pose):
        # Load the grasp poses from the hdf5 file
        hdf5_path = os.path.join(
            MODEL_PATH.replace("*", obj_id), "textured.obj.hdf5"
        )

        grasp_dict = read_in_mesh_config(
            hdf5_path,
            parallel=True,
            keypts_byhand=True,
            keypts_com=True,
            analytical=True,
        )

        grasp_poses = []
        for grasp in grasp_dict["paralleljaw_pregrasp_transform"]:
            # Get the contact point and the second point on the gripper finger surface
            approach_vec = torch.tensor(grasp[0:3], device=self.device) 
            baseline = torch.tensor(grasp[3:6], device=self.device)
            contact_pt = torch.tensor(grasp[6:9], device=self.device) / 100 
            pt2 = contact_pt + baseline * grasp[9] / 100
            grasp_pos = pt2 - approach_vec * 0.1

            # Convert the grasp pose to 6D pose using the transformation matrix of the object from the simulation
            grasp_pose_obj = from_contact_to_6D(grasp)
            # Convert the transformation matrix to a pose vector
            grasp_pose_obj = transformation_matrix_to_pose_vector(torch.Tensor(grasp_pose_obj)).to(self.device)
            # Convert to meters
            grasp_pose_obj[:3] = grasp_pos
            
            # Transform the grasp pose to the robot frame
            obj_pose_w = self._get_obj_pose(self.obj_chosen[env_id], env_id)
            grasp_pos, grasp_quat = combine_frame_transforms(obj_pose_w[:3], obj_pose_w[3:], grasp_pose_obj[:3], grasp_pose_obj[3:])
            contact_pt, _ = combine_frame_transforms(obj_pose_w[:3], obj_pose_w[3:], contact_pt)
            pt2, _ = combine_frame_transforms(obj_pose_w[:3], obj_pose_w[3:], pt2)
            mid_pt = (contact_pt + pt2) / 2
            
            # Visualize the grasp pose
            grasp_pos_img = robot_point_to_image(grasp_pos, camera_pose)
            contact_pt_img = robot_point_to_image(contact_pt, camera_pose)
            pt2_img = robot_point_to_image(pt2, camera_pose)
            mid_pt_img = robot_point_to_image(mid_pt, camera_pose)
            img = cv2.line(
                img,
                (int(contact_pt_img[0]), int(contact_pt_img[1])),
                (int(pt2_img[0]), int(pt2_img[1])),
                (0, 255, 0),
                1,
            )
            img = cv2.line(
                img,
                (int(mid_pt_img[0]), int(mid_pt_img[1])),
                (int(grasp_pos_img[0]), int(grasp_pos_img[1])),
                (0, 0, 255),
                1,
            )
            img = cv2.circle(
                img, (int(mid_pt_img[0]), int(mid_pt_img[1])), 2, (0, 0, 0), -1
            )
            
            grasp_poses.append(torch.cat((grasp_pos, grasp_quat), -1))
        grasp_poses = torch.stack(grasp_poses)
        cv2.imwrite(f"env_{env_id}_grasp_viz.png", img)

        # Get grasp with maximum grasp score
        grasp_pose = grasp_poses[
            grasp_dict["paralleljaw_pregrasp_score"].index(
                max(grasp_dict["paralleljaw_pregrasp_score"])
            )]
        self.pc_markers[env_id].visualize(translations=grasp_poses[:, :3])
        return grasp_pose
