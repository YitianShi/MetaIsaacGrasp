# from isaaclab.sensors.camera import Camera, PinholeCameraCfg
from dataclasses import MISSING
from math import pi

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import (
    ManagerBasedEnv,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    mdp,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
)

from isaaclab.utils import configclass
from isaaclab.utils.math import *

from isaac_env.air_env_base.env_cfg import *
from .element_cfg import *


@configclass
class CellSceneCfg(InteractiveSceneCfg):
    """Configuration for a env scene."""

    def __init__(self, disable_camera, **kwargs):
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
        setattr(self, ROBOT_NAME, ROBOT_CFGs[ROBOT_NAME].replace(
            prim_path="{ENV_REGEX_NS}/" + ROBOT_NAME
        ))

        # self.mark = MARK1_CFG.replace(prim_path="{ENV_REGEX_NS}/" + ROBOT_NAME + "/Mark1")
        # self.mark2 = MARK2_CFG.replace(prim_path="{ENV_REGEX_NS}/" + ROBOT_NAME + "/Mark2")

        super().__init__(**kwargs)
        
        self.objs: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            f"obj_{i}": OBJ_CFGs[i].replace(
                prim_path="{ENV_REGEX_NS}/obj_"+str(i)
            )
            for i in range(NUM_OBJS)
        },
        )

        """Random camera initialization."""
        if not disable_camera:
            self.camera_0 = CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}"+f"/{ROBOT_NAME}/{CAMERA_NAME}camera_0")
            if n_random_cam:
                for i in range(n_random_cam):
                    setattr(self, random_cam_names[i], CAMERA_RANDOM_CFG[i].replace(
                        prim_path="{ENV_REGEX_NS}"+f"/{ROBOT_NAME}/{random_cam_names[i]}", 
                        offset=CameraCfg.OffsetCfg(
                            pos=(0.1, 0.1, 0.1), 
                            rot=(0.1, 0.1, 0.1, 0.1), 
                            convention="ros")
                        )
                    )    
                
        """Pose initialization."""
        self.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/"+ROBOT_NAME+"/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/"+ROBOT_NAME+"/"+ee_name,
                name=ee_name,
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.),
                ),
            )
        ],)

        if ON_HAND:
            self.view_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/"+ROBOT_NAME+"/base_link",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path = "/Visuals/ViewFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}"+f"/{ROBOT_NAME}/VMS3D_Femto_Mega_S_0",
                    name=ee_name,
                    offset=OffsetCfg(
                        pos=VIEW_POS_CAM,
                        rot=VIEW_QUAT_CAM
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
        asset_name=ROBOT_NAME, joint_names=list(ARM_JOINT.keys()), scale=1.0
    )
    gripper_pos = mdp.BinaryJointPositionActionCfg(
        asset_name=ROBOT_NAME,
        joint_names=[*gripper_joint_cfg[ROBOT_NAME]["default"]],
        open_command_expr=gripper_joint_cfg[ROBOT_NAME]["open"],
        close_command_expr=gripper_joint_cfg[ROBOT_NAME]["close"],
    )

@configclass
class ObservationsCfg:

    @configclass
    class ImageCfg(ObsGroup):

        def __init__(self):
            super().__init__()
            if "rgb" in MODALITIES:
                self.rgb = ObsTerm(func=rgb_capture)
            if "normals" in MODALITIES:
                self.normals = ObsTerm(func=normal_capture)
            if "instance_segmentation_fast" in MODALITIES:
                self.instance_segmentation_fast = ObsTerm(func=inst_capture)
            if "distance_to_image_plane" in MODALITIES:
                self.distance_to_image_plane = ObsTerm(func=depth_capture)
                self.pcd = ObsTerm(func=pcd_capture)

            self.gripper_pose = ObsTerm(func=gripper_pose_capture)
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


@configclass
class EventCfg:
    def __init__(self):
        self.reset_robot_pos = EventTerm(
            func=reset_robot_to_default,
            mode="reset",
        )
        self.reset_obj_pos = EventTerm(
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
                "asset_cfg": SceneEntityCfg(f"objs"),
            },
        )
                
        for ix in range(n_random_cam):
            setattr(
                self,
                f"reset_camera_pos_{random_cam_names[ix]}",
                EventTerm(
                    func=reset_root_state_sphere,
                    mode="reset",
                    params={
                        "SPHERE_RADIUS": SPHERE_RADIUS,
                         "asset_cfg": SceneEntityCfg(random_cam_names[ix]),
                    },
                ),
            )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    grasp_success = RewTerm(func=grasp_success_compute, weight=1)


def time_out(env) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    cri1 = env.successive_grasp_failure == SUCCESSIVE_GRASP_FAILURE_LIMIT
    cri2 = ~ env.env_reachable
    cri3 = env.epi_step_count[:, 1] >= STEP_TOTAL
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
            disable_camera=disable_camera,
        )
        self.events: EventCfg = EventCfg()
        # environment settings
        self.num_envs = 1
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
