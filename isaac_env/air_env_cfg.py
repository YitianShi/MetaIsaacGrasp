# from omni.isaac.lab.sensors.camera import Camera, PinholeCameraCfg
from dataclasses import MISSING
from math import pi

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
import torch
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import (
    ManagerBasedEnv,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    mdp,
)
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.sensors.frame_transformer import OffsetCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
)
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import *

from .element_cfg import *


@configclass
class CellSceneCfg(InteractiveSceneCfg):
    """Configuration for a env scene."""

    def __init__(self, num_objs, disable_camera, **kwargs):
        # ground plane
        # lights
        self.dome_light = AssetBaseCfg(
            spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=300.0),
        ).replace(prim_path="{ENV_REGEX_NS}/Domelight")

        self.distant_light = AssetBaseCfg(
            spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=800.0),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 12), rot=(0, 0, 0, 0.0)),
        ).replace(prim_path="/World/Distantlight")

        self.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        )

        self.table: AssetBaseCfg = TABLE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Table"
        )

        setattr(self, robot_name, ROBOT_CFGs[robot_name].replace(
            prim_path="{ENV_REGEX_NS}/" + robot_name
        ))

        # self.mark = MARK1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/Mark1")
        # self.mark2 = MARK2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/Mark2")
        # self.mark3 = MARK3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/Mark3")
        # self.mark4 = MARK4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/Mark4")

        super().__init__(**kwargs)
        
        for i in range(num_objs):
            exec(
                f"self.obj_{i}: RigidObjectCfg = OBJ_CFGs[{i}]"
                + '.replace(prim_path="{ENV_REGEX_NS}/obj_"+'
                + f"str({i}))"
            )

        """Random camera initialization."""
        if not disable_camera:
            self.camera_0 = CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}"+f"/{robot_name}/{camera_name}camera_0")
            if n_random_cam:
                for i in range(n_random_cam):
                    setattr(self, random_cam_names[i], CAMERA_RANDOM_CFG[i].replace(
                        prim_path="{ENV_REGEX_NS}"+f"/{robot_name}/{random_cam_names[i]}", 
                        offset=CameraCfg.OffsetCfg(
                            pos=(0.1, 0.1, 0.1), 
                            rot=(0.1, 0.1, 0.1, 0.1), 
                            convention="ros")
                        )
                    )    
                
        """Pose initialization."""
        self.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/"+robot_name+"/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/"+robot_name+"/"+ee_name,
                name=ee_name,
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.),
                ),
            )
        ],)

        if on_hand and not collect_data:
            self.view_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/"+robot_name+"/base_link",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path = "/Visuals/ViewFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}"+f"/{robot_name}/VMS3D_Femto_Mega_S_0",
                    name=ee_name,
                    offset=OffsetCfg(
                        pos=view_pos_cam,
                        rot=view_quat_cam
                    ),
                )
            ],)
##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name=robot_name, joint_names=list(ARM_JOINT.keys()), scale=1.0
    )
    gripper_pos = mdp.BinaryJointPositionActionCfg(
        asset_name=robot_name,
        joint_names=[*gripper_joint_cfg[robot_name]["default"]],
        open_command_expr=gripper_joint_cfg[robot_name]["open"],
        close_command_expr=gripper_joint_cfg[robot_name]["close"],
    )


def rgb_capture(env):
    """Height scan from the given sensor w.r.t. the sensor's frame."""
    # extract the used quantities (to enable type-hinting)
    rgb = [env.scene[f"camera_{i}"].data.output["rgb"][..., :3].float() for i in range(n_multiple_cam)]
    rgb = torch.stack(rgb, 0).transpose(0, 1)
    if not use_sb3:
        return rgb
    rgb_m = rgb.float().mean(dim=(2,3), keepdim=True)
    rgb_std = rgb.std(dim=(2,3), keepdim=True)
    rgb = (rgb - rgb_m) / rgb_std
    return rgb


def normal_capture(env):
    """Height scan from the given sensor w.r.t. the sensor's frame."""
    # extract the used quantities (to enable type-hinting)
    data = [env.scene[f"camera_{i}"].data.output["normals"][..., :3] for i in range(n_multiple_cam)]
    return torch.stack(data, 0).transpose(0, 1)


def inst_capture(env):
    """Height scan from the given sensor w.r.t. the sensor's frame."""
    # extract the used quantities (to enable type-hinting)
    data = [env.scene[f"camera_{i}"].data.output["instance_segmentation_fast"].unsqueeze(-1) for i in range(n_multiple_cam)]
    return torch.stack(data, 0).transpose(0, 1)


def depth_capture(env):
    """Height scan from the given sensor w.r.t. the sensor's frame."""
    # extract the used quantities (to enable type-hinting)
    data = [env.scene[f"camera_{i}"].data.output["distance_to_image_plane"].unsqueeze(-1) for i in range(n_multiple_cam)]
    # Maximum representable value for the tensor's dtype
    # Clip positive infinity to maximum value
    return torch.clip(torch.stack(data, 0).transpose(0, 1), 0., depth_max)


def pcd_capture(env):
    pcds = []
    for cam_id in range(n_multiple_cam):
        depths = env.scene[f"camera_{cam_id}"].data.output["distance_to_image_plane"]
        depths = torch.clip(depths, 0, depth_max)
        pcd_map = [create_pointcloud_from_depth(
            intrinsic_matrix=env.scene[f"camera_{cam_id}"].data.intrinsic_matrices[env_id],
            depth=depths[env_id],
            position=get_camera_pose(env, cam_id, env_id)[:3],
            orientation=get_camera_pose(env, cam_id, env_id)[3:],
            device=env.device).view(cam_width, cam_height, 3).permute(1,0,2) for env_id in range(env.scene.num_envs)]
        pcds.append(torch.stack(pcd_map, 0))
    return torch.stack(pcds, 0).transpose(0, 1)


def get_camera_pose(env, cam_id=None, env_id=None):
    if cam_id is None:
        return torch.stack([env.get_camera_pose(cam_id) for cam_id in range(n_multiple_cam)])
    return env.get_camera_pose(cam_id) if env_id is None else env.get_camera_pose(cam_id, env_id)


def get_obj_height(env):
    """Get whether the object is lifted above the minimal height"""
    objs_hight = torch.stack(
        [env.scene[f"obj_{i}"].data.root_pos_w[:, 2] for i in range(num_objs)]).T
    objs_hight -= env.scene[robot_name].data.root_pos_w[:, 2][..., None]
    # The obj is lifted if its height is above 90% of the lift height
    return objs_hight


@configclass
class ObservationsCfg:

    @configclass
    class ImageCfg(ObsGroup):

        def __init__(self):
            if "rgb" in MODALITIES:
                self.rgb = ObsTerm(func=rgb_capture)
            if "normals" in MODALITIES:
                self.normals = ObsTerm(func=normal_capture)
            if "instance_segmentation_fast" in MODALITIES:
                self.instance_segmentation_fast = ObsTerm(func=inst_capture)
            if "distance_to_image_plane" in MODALITIES:
                self.distance_to_image_plane = ObsTerm(func=depth_capture)
                self.pcd = ObsTerm(func=pcd_capture)

            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class PolicyCfg(ObsGroup):

        # observation terms (order preserved)
        pos = ObsTerm(func=get_camera_pose)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    cam_pos: PolicyCfg = PolicyCfg()
    policy: ImageCfg = ImageCfg()
    # objs_hight: ObjHeight = ObjHeight()


def reset_robot_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset the scene to the default state specified in the scene configuration."""
    # articulations
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_state_to_sim(default_root_state, env_ids=env_ids)
        # obtain default joint pts
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulationenv_cfg
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]

    # random object dropped on the ground
    if collect_data and random_drop_obj:
        z_mask = torch.bernoulli(torch.full((len(env_ids),), 0.2, device=asset.device)).bool()
        positions[z_mask, -1] = 0.1

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_sphere(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    sphere_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    angle_min = 0, angle_max = pi / 4
):
    """Reset the asset root state to a random position on a sphere surface.
    
    This function randomizes the root position of the asset.
    * It samples the root position from a sphere surface with the given radius and adds it to the default root position,
      before setting them into the physics simulation.
    * It samples a random orientation and sets it into the physics simulation.
    
    The function takes a sphere radius for position sampling and a dictionary of velocity ranges for each axis and rotation.
    The keys of the velocity dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples
    of the form ``(min, max)``. If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """

    angle_min = 0
    angle_max = pi / 3

    if len(env_ids):
        robot_pos = env.scene[robot_name].data.root_pos_w[env_ids]

        asset = env.scene[asset_cfg.name]

        cam_id = int(asset_cfg.name.split("_")[-1])

        num_envs = len(env_ids)
        if not fix_rand_camera:
            theta = 2 * pi * torch.rand(num_envs, device=env.device)  
        else: 
            theta = torch.full((num_envs,), fix_cam_angle[cam_id-1], device=env.device)

        if not fix_rand_camera:
            phi = (angle_max - angle_min) * torch.rand(num_envs, device=env.device) + angle_min 
        else: 
            phi = torch.full((num_envs,), angle_max, device=env.device)

        x = sphere_radius * torch.sin(phi) * torch.cos(theta)
        y = sphere_radius * torch.sin(phi) * torch.sin(theta)
        z = sphere_radius * torch.cos(phi)

        random_pts_ct = torch.stack([x, y, z], dim=1) 
        random_pts_r = random_pts_ct + center_pt.to(env.device) 
        random_pts_w = random_pts_r + robot_pos

        center_pt_w = center_pt.to(env.device) + robot_pos

        asset.set_world_poses_from_view(random_pts_w, center_pt_w, env_ids=env_ids)

@configclass
class EventCfg:
    def __init__(self, num_objs):
        self.reset_robot_pos = EventTerm(
            func=reset_robot_to_default,
            mode="reset",
        )
        for i in range(num_objs):
            setattr(
                self,
                f"reset_obj_pos_{i}",
                EventTerm(
                    func=reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": {
                            "x": ee_obj_default[0],
                            "y": ee_obj_default[1],
                            "z": ee_obj_default[2],
                            "roll": (-1.0, 1.0),
                            "pitch": (-1.0, 1.0),
                            "yaw": (-1.0, 1.0),
                        },
                        "velocity_range": {},
                        "asset_cfg": SceneEntityCfg(f"obj_{i}"),
                    },
                ),
            )
        for ix in range(n_random_cam):
            setattr(
                self,
                f"reset_camera_pos_{random_cam_names[ix]}",
                EventTerm(
                    func=reset_root_state_sphere,
                    mode="reset",
                    params={
                        "sphere_radius": sphere_radius,
                         "asset_cfg": SceneEntityCfg(random_cam_names[ix]),
                    },
                ),
            )


def grasp_success_compute(env) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    objs_hight = torch.stack([env._get_obj_pos(i)[:, -1] for i in range(num_objs)]).T
    # The obj is lifted if its height is above 90% of the lift height
    # The environment should reach the terminate state
    objs_hight_over_limit = objs_hight > lift_height * 0.9
    success_env = torch.any(objs_hight_over_limit, dim=1) & (STATE_MACHINE["lift"] == env.sm_state)
    # Successive grasp failure is reset if the env is successful
    env.successive_grasp_failure[success_env] = -1
    # The reward is 1 if any of the objs is lifted
    reward = success_env.float()
    # Update the chosen obj
    success_obj = objs_hight_over_limit.nonzero()
    if len(success_obj) > 0 and success_env.any():
        env.obj_chosen[success_obj[:, 0]] = success_obj[:, 1]
    return reward


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    grasp_success = RewTerm(func=grasp_success_compute, weight=1)


def time_out(env) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    cri1 = env.successive_grasp_failure == successive_grasp_failure_limit
    cri2 = ~ env.env_reachable
    cri3 = env.epi_step_count[:, 1] >= step_total
    cri4 = env.sm_state == STATE_MACHINE["init_env"]

    return (cri1 | cri2 | cri3) & cri4


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=time_out, time_out=True)
    pass


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    pass

@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass

##
# Environment configuration
##

@configclass
class CellEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    def __init__(self, disable_camera = False, **kwargs):
        """Initialization."""
        super().__init__(**kwargs)
        # Basic settings
        self.observations: ObservationsCfg = ObservationsCfg()
        self.actions: ActionsCfg = ActionsCfg()

        # MDP settings
        self.curriculum: CurriculumCfg = CurriculumCfg()
        self.rewards: RewardsCfg = RewardsCfg()
        self.terminations: TerminationsCfg = TerminationsCfg()
        # No command generator
        self.commands: CommandsCfg = CommandsCfg()

        # Scene settings
        self.scene: CellSceneCfg = CellSceneCfg(
            num_envs=4096,
            env_spacing=2.5,
            replicate_physics=True,
            num_objs=num_objs,
            disable_camera=disable_camera,
        )
        self.events: EventCfg = EventCfg(num_objs=num_objs)
        # environment settings
        self.num_envs = 1
        self.num_objs = num_objs
        self.episode_length_s = 5

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        # self.sim.substeps = 8

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.enable_ccd = True
        #self.sim.device = "cuda"
