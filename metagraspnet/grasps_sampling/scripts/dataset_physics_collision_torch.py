#!/usr/bin/env python
# Author : Maximilian Gilles, IFL, KIT

import concurrent.futures
import copy
import glob
import multiprocessing as mp
import os
import pathlib
import random
import sys
import threading as th

import h5py
import numpy as np
import torch
import trimesh

# Print maximum length in console.
np.set_printoptions(threshold=sys.maxsize)

# Set backend for headless rendering in pyrender
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Setup default generation variables
# Value are (min, max) ranges
RANDOM_TRANSLATION_X = (-6.0, 6.0)
RANDOM_TRANSLATION_Y = (-3.0, 3.0)
RANDOM_TRANSLATION_Z = (30.0, 500.0)
RANDOM_ROTATION_X = (0.0, 360.0)
RANDOM_ROTATION_Y = (0.0, 360.0)
RANDOM_ROTATION_Z = (0.0, 360.0)
# CAMERA_DISTANCE = 200.0

# Gripper Parameters ParallelJaw and Suction Cup
GRIPPER_HEIGHT = 11.21  # cm
AREA_CENTER = 0.8  # cm
PREGRASP_PARALLEL_DISTANCE = 1  # cm note: this is added to the predicted gripper width.
GRIPPER_WIDTH_MAX = 8  # cm

test = False  # for debugging
scene_root_dir = "../data_all/data5"

def rad2deg(val):
    ret = (val * 180) / np.pi
    return ret


def get_parallel_gripper_collision_mesh(root, grasp_width, transform):
    """
    Based on grasp with return different collision manager for franka gripper.
    Maximum grasp width is 8 cm.
    """
    # hand
    # hand_dir = os.path.join(root, "../hand_collision.stl")
    hand_mesh = trimesh.load(root + "/hand_collision.stl")
    hand_mesh = hand_mesh.apply_scale(100)  # convert m to cm
    # finger left
    # finger_left_dir = os.path.join(root, "../finger_collsion_left.stl")
    finger_left_mesh = trimesh.load(root + "/finger_collision_left.stl")
    finger_left_mesh = finger_left_mesh.apply_scale(100)
    # finger right
    # finger_right_dir = os.path.join(root, "../finger_collsion_right.stl")
    finger_right_mesh = trimesh.load(root + "/finger_collision_right.stl")
    finger_right_mesh = finger_right_mesh.apply_scale(100)

    # create collsision manager for franka hand
    franka_hand_collision_manager = trimesh.collision.CollisionManager()
    # add hand
    hand_trans = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    hand_trans_world = np.dot(transform, hand_trans)
    franka_hand_collision_manager.add_object(
        name="hand", mesh=hand_mesh, transform=np.array(hand_trans_world)
    )

    # add finger left
    finger_left_trans = [
        [1, 0, 0, -grasp_width / 2],
        [0, 1, 0, 0],
        [0, 0, 1, 5.81],
        [0, 0, 0, 1],
    ]
    finger_left_trans_world_ = np.dot(hand_trans_world, finger_left_trans)
    franka_hand_collision_manager.add_object(
        name="finger_left",
        mesh=finger_left_mesh,
        transform=np.array(finger_left_trans_world_),
    )

    # add finger right
    finger_right_trans = [
        [1, 0, 0, grasp_width / 2],
        [0, 1, 0, 0],
        [0, 0, 1, 5.81],
        [0, 0, 0, 1],
    ]
    finger_right_trans_world = np.dot(hand_trans_world, finger_right_trans)
    franka_hand_collision_manager.add_object(
        name="finger_right",
        mesh=finger_right_mesh,
        transform=np.array(finger_right_trans_world),
    )
    return franka_hand_collision_manager


def load_single_mesh(_path, output_on=True, scale=100):
    if output_on:
        print(f"-> load with scale 100 [m]->[cm]{_path}")
    # Load mesh and rescale with factor 100
    mesh = trimesh.load(_path)
    mesh = mesh.apply_scale(scale)  # convert m to cm
    return mesh


def load_single_grasp_config(hdf5_path, num_samples=2000):
    # print(f"-> load {hdf5_path}")
    f = h5py.File(str(hdf5_path), "r")
    # access grasp datasets
    dset_parallel_grasps = f["grasps"]["paralleljaw"]["pregrasp_transform"]
    dset_parallel_score = f["grasps"]["paralleljaw"]["quality_score"]
    dset_parallel_grasps = list(dset_parallel_grasps)
    dset_parallel_score = list(dset_parallel_score)
    # random sample
    sample_index = list(range(len(dset_parallel_grasps)))
    random.shuffle(sample_index)
    sample_index = sample_index[:num_samples]
    #
    dset_parallel_grasps = [dset_parallel_grasps[i] for i in sample_index]
    dset_parallel_score = [dset_parallel_score[i] for i in sample_index]
    ret = {
        "paralleljaw_pregrasp_transform": dset_parallel_grasps,
        "paralleljaw_pregrasp_score": dset_parallel_score,
    }
    return ret


def load_single_keypts_config(hdf5_path):
    # print(f"-> load {hdf5_path}")
    f = h5py.File(str(hdf5_path), "r")
    # access grasp datasets
    try:
        dset_keypts_com = f["keypts"]["com"]
        dset_keypts_byhand = f["keypts"]["byhand"]

        ret = {
            "keypts_com": list(dset_keypts_com),
            "keypts_byhand": list(dset_keypts_byhand),
        }
    except:
        ret = {
            "keypts_com": [],
            "keypts_byhand": [],
        }
    return ret


def create_easy_gripper(
    color=[0, 255, 0, 140], sections=6, show_axis=False, width=None
):
    if width:
        w = min((width + PREGRASP_PARALLEL_DISTANCE) / 2, 4.1)
    else:
        w = 4.1

    l_center_grasps = GRIPPER_HEIGHT - AREA_CENTER  # gripper length till grasp contact

    cfl = trimesh.creation.cylinder(
        radius=0.1,
        sections=sections,
        segment=[
            [w, -7.27595772e-10, 6.59999996],
            [w, -7.27595772e-10, l_center_grasps],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.1,
        sections=sections,
        segment=[
            [-w, -7.27595772e-10, 6.59999996],
            [-w, -7.27595772e-10, l_center_grasps],
        ],
    )
    # arm
    cb1 = trimesh.creation.cylinder(
        radius=0.1, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996]]
    )
    # queer
    cb2 = trimesh.creation.cylinder(
        radius=0.1, sections=sections, segment=[[-w, 0, 6.59999996], [w, 0, 6.59999996]]
    )
    # coordinate system
    if show_axis:
        cos_system = trimesh.creation.axis(
            origin_size=0.04,
            transform=None,
            origin_color=None,
            axis_radius=None,
            axis_length=None,
        )
        tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl, cos_system])
    else:
        tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    # tmp.visual.face_colors = color
    return tmp


def interpolate_between_red_and_green(score, alpha=255):
    """
    Returns RGBA color between RED and GREEN.
    """
    delta = int(score * 255)
    COLOR = [255 - delta, 0 + delta, 0, alpha]  # RGBA
    return COLOR


def create_contact_pose(grasp_config):
    parallel_vec_a = np.array([grasp_config[0], grasp_config[1], grasp_config[2]])

    parallel_vec_b = np.array([grasp_config[3], grasp_config[4], grasp_config[5]])

    parallel_contact_pt1 = np.array([grasp_config[6], grasp_config[7], grasp_config[8]])

    parallel_width = grasp_config[9]

    # create 4x4 Matrix
    c_ = np.cross(parallel_vec_a, parallel_vec_b)
    R_ = [
        [parallel_vec_b[0], c_[0], parallel_vec_a[0]],
        [parallel_vec_b[1], c_[1], parallel_vec_a[1]],
        [parallel_vec_b[2], c_[2], parallel_vec_a[2]],
    ]

    t_ = parallel_contact_pt1
    # create 4x4 transform matrix of grasp
    contact_pt_transform_ = [
        [R_[0][0], R_[0][1], R_[0][2], t_[0]],
        [R_[1][0], R_[1][1], R_[1][2], t_[1]],
        [R_[2][0], R_[2][1], R_[2][2], t_[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]

    cos_system = trimesh.creation.axis(
        origin_size=0.1, transform=np.array(contact_pt_transform_)
    )

    return cos_system


def from_contact_to_6D(grasp_config, obj_to_world_transform=None):
    """
    Convert from hdf5 contact representation to 4x4 Matrix in World COS.
    """
    parallel_vec_a = np.array([grasp_config[0], grasp_config[1], grasp_config[2]])

    parallel_vec_b = np.array([grasp_config[3], grasp_config[4], grasp_config[5]])

    parallel_contact_pt1 = np.array([grasp_config[6], grasp_config[7], grasp_config[8]])

    parallel_width = grasp_config[9]
    # print("parallel width", parallel_width)

    parallel_pregrasp_transform = convert_to_franka_6DOF(
        vec_a=parallel_vec_a,
        vec_b=parallel_vec_b,
        contact_pt=parallel_contact_pt1,
        width=parallel_width,
    )

    if obj_to_world_transform is None:
        obj_to_world_transform = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    parallel_pregrasp_transform_world = np.dot(
        obj_to_world_transform, parallel_pregrasp_transform
    )

    return parallel_pregrasp_transform_world


def convert_to_franka_6DOF(vec_a, vec_b, contact_pt, width):
    """
    Convert Contact-Point Pregrasp representation to 6DOF Gripper Pose (4x4).
    """
    # get 3rd unit vector
    c_ = np.cross(vec_a, vec_b)
    # rotation matrix
    R_ = [
        [vec_b[0], c_[0], vec_a[0]],
        [vec_b[1], c_[1], vec_a[1]],
        [vec_b[2], c_[2], vec_a[2]],
    ]
    # translation t
    t_ = contact_pt + width / 2 * vec_b + (GRIPPER_HEIGHT - AREA_CENTER) * vec_a * (-1)
    # create 4x4 transform matrix of grasp
    pregrasp_transform_ = [
        [R_[0][0], R_[0][1], R_[0][2], t_[0]],
        [R_[1][0], R_[1][1], R_[1][2], t_[1]],
        [R_[2][0], R_[2][1], R_[2][2], t_[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return np.array(pregrasp_transform_)


def convert_to_contact_6DOF(vec_a, vec_b, contact_pt):
    """
    Convert Contact-Point Pregrasp representation to 6DOF Contact Point Pose (4x4).
    """
    # get 3rd unit vector
    c_ = np.cross(vec_a, vec_b)
    # rotation matrix
    R_ = [
        [vec_b[0], c_[0], vec_a[0]],
        [vec_b[1], c_[1], vec_a[1]],
        [vec_b[2], c_[2], vec_a[2]],
    ]
    # translation t
    t_ = contact_pt
    # create 4x4 transform matrix of grasp
    contact_transform_ = [
        [R_[0][0], R_[0][1], R_[0][2], t_[0]],
        [R_[1][0], R_[1][1], R_[1][2], t_[1]],
        [R_[2][0], R_[2][1], R_[2][2], t_[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return np.array(contact_transform_)


class CollisionWithScene:
    """
    Description not yet created.
    """

    def __init__(self, sid, meshes, poses, grasps, root_dir, box_dir, assets=None):
        self.trimesh_scene = None
        self.collision_scene = None
        self.all_meshes = meshes
        self.all_poses = poses
        self.all_grasps = grasps
        self.all_assets = assets
        self.box_dir = box_dir
        self.root = root_dir
        self.sid = sid
        self.scene = torch.load(
                f"{scene_root_dir}/{self.sid}.pt", map_location=torch.device("cpu")
            )

    def load_potential_grasps_and_generate_trimesh_scene(self):
        """1) Generate a twin scene in trimesh (much simpler) and check for gripper collision.
        2) Create scene description .hp5y file + .usd stage
        """
        print(f"{self.sid} -> evaluate for collision with scene.")

        if self.trimesh_scene is None:
            print(f"{self.sid} -> new trimesh scene.")
            # generate new trimesh scene
            self.trimesh_scene = trimesh.scene.Scene()
            self.collision_manager = trimesh.collision.CollisionManager()

            # add world cos
            world_cos = trimesh.creation.axis(
                origin_size=1.0,
                transform=None,
                origin_color=None,
                axis_radius=None,
                axis_length=None,
            )

            self.trimesh_scene.add_geometry(geometry=world_cos, transform=None)

            epsilon = 0.7
            epsilon_scale = 0.9

            # add ground plane
            plane_width = 1.5
            plane_trans = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -plane_width / 2 - 3 * epsilon],
                [0, 0, 0, 1],
            ]
            ground_mesh = trimesh.creation.box(
                extents=(200, 200, plane_width), face_colors=[55, 55, 255, 140]
            )

            self.collision_manager.add_object(
                name="groundplane", mesh=ground_mesh, transform=np.array(plane_trans)
            )

            self.trimesh_scene.add_geometry(
                geometry=ground_mesh, transform=np.array(plane_trans)
            )

            # add box, rotated by 90 deg around x-axis, and increase 3 cm in height
            # usd object is already modified
            # box_dir = "/home/isaac/GraspNet_Models/KLT/BOX_VOL4.obj"

            """
            self.collision_manager.add_object(
                    name = "box",
                    mesh = box_mesh_collision,
                    transform = None)

            self.trimesh_scene.add_geometry(
                    geometry = box_mesh,
                    transform = None)
            
            self.collision_manager.add_object(
                    name = "box",
                    mesh = box_mesh_collision,
                    transform = None)

            self.trimesh_scene.add_geometry(
                    geometry = box_mesh,
                    transform = None)"""

            # add meshes in collision scene with poses from isaac sim

            self.pcd_gt = []
            for idx, obj_mesh in enumerate(self.all_meshes):
                obj_mesh_collision = copy.deepcopy(obj_mesh)
                pose = self.all_poses[idx]
                transform = pose
                obj_mesh.apply_transform(transform)
                transform[:2, :2] *= epsilon_scale
                obj_mesh_collision.apply_transform(transform)

                self.collision_manager.add_object(
                    name=str(idx), mesh=obj_mesh_collision, transform=None
                )

                self.trimesh_scene.add_geometry(
                    node_name=str(idx), geometry=obj_mesh, transform=None
                )

                obj_mesh_area = obj_mesh.area

                contact_points, index_tri = trimesh.sample.sample_surface_even(
                    mesh=obj_mesh, count=int(obj_mesh_area * 40), radius=None
                )  # removes points below this radius

                self.pcd_gt.append(contact_points)

            self.pcd_gt = np.concatenate(self.pcd_gt, axis=0)/10

            # pcd = o3d.geometry.PointCloud()
            # points = self.scene["camera_5"]["pcd"].view(-1, 3).numpy()/100
            # pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points) * [0.5, 0.5, 0.5])

            # # points2 = self.scene["non_colliding_parallel_contact_poses"][:, :3, -1].numpy()
            # # pcd2 = o3d.geometry.PointCloud()
            # # pcd2.points = o3d.utility.Vector3dVector(points2)
            # pcd3 = o3d.geometry.PointCloud()
            # pcd3.points = o3d.utility.Vector3dVector(self.pcd_gt)
            # #pcd4 = o3d.geometry.PointCloud()
            # #pcd4.points = o3d.utility.Vector3dVector(self.scene["pcd_gt"].numpy()/100)
            # axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([pcd, pcd3, axis_frame])

            # print(f"{self.sid} -> {self.pcd_gt.shape[0]} contact points sampled")

        else:
            print(f"{self.sid} -> update trimesh scene.")
            # empty old grasps configs, get them by geometry name
            self.trimesh_scene.delete_geometry(names="pregrasp_config")

            # trimesh scene already exists, only update pose of objects
            for idx, obj_mesh in enumerate(self.all_meshes):
                # new pose
                pose = self.all_poses[idx]
                transform = pose
                # update trimesh scene
                self.trimesh_scene.graph.update(frame_to=str(idx), matrix=transform)
                # update collision scene
                self.collision_manager.set_transform(name=str(idx), transform=transform)

        return True

    def check_for_collision(self, show_scene=False, parallel_gripper=True):
        """
        Iterate through grasps for each asset in scene and check for collision.
        """

        print(f"{self.sid} -> evaluate for collision with scene.")

        # add data lists to store collision information
        self.non_colliding_parallel_gripper_poses = []
        self.non_colliding_parallel_analytical_score = []
        self.non_colliding_parallel_object_id = []
        self.non_colliding_parallel_asset_names = []
        self.non_colliding_parallel_contact_poses = []
        self.non_colliding_parallel_contact_width = []

        self.colliding_parallel_gripper_poses = []
        self.colliding_parallel_analytical_score = []
        self.colliding_parallel_object_id = []
        self.colliding_parallel_asset_names = []
        self.colliding_parallel_contact_poses = []
        self.colliding_parallel_contact_width = []
        self.colliding_parallel_names = []

        # iterate through all objects in scene
        for idx, (mesh, pose, grasps) in enumerate(
            zip(self.all_meshes, self.all_poses, self.all_grasps)
        ):
            self.check_mesh_for_collision_with_scene(
                idx=idx,
                obj_pose=pose,
                obj_grasp_dict=grasps,
                prim_path=None,
                approaching=True,
                debug=True,
                parallel_gripper=parallel_gripper,
            )

        print(
            f"{self.sid} -> Collision check over, found {len(self.non_colliding_parallel_gripper_poses)} valid parallel grasps"
        )

        if len(self.non_colliding_parallel_gripper_poses) > 0:
            data = dict(
                pcd_gt=(torch.tensor(self.pcd_gt) * 1000).to(torch.int16),
                non_colliding_parallel_gripper_poses=(
                    torch.tensor(np.stack(self.non_colliding_parallel_gripper_poses))
                    if len(self.non_colliding_parallel_gripper_poses) > 0
                    else None
                ),
                non_colliding_parallel_analytical_score=(
                    torch.tensor(np.stack(self.non_colliding_parallel_analytical_score))
                    if len(self.non_colliding_parallel_analytical_score) > 0
                    else None
                ),
                non_colliding_parallel_object_id=(
                    torch.tensor(np.stack(self.non_colliding_parallel_object_id))
                    if len(self.non_colliding_parallel_object_id) > 0
                    else None
                ),
                non_colliding_parallel_contact_poses=(
                    torch.tensor(np.stack(self.non_colliding_parallel_contact_poses))
                    if len(self.non_colliding_parallel_contact_poses) > 0
                    else None
                ),
                non_colliding_parallel_contact_width=(
                    torch.tensor(np.stack(self.non_colliding_parallel_contact_width))
                    if len(self.non_colliding_parallel_contact_width) > 0
                    else None
                ),
            )
            self.scene.update(data)
            torch.save(self.scene, f"{scene_root_dir}/{self.sid}.pt")

        return len(self.non_colliding_parallel_gripper_poses)

    def check_mesh_for_collision_with_scene(
        self,
        idx,
        obj_pose,
        obj_grasp_dict,
        prim_path=None,
        debug=False,
        approaching=True,
        parallel_gripper=True,
    ):
        """
        Check if grasps collide with the scene.
        """
        # print(f"{self.sid} -> check for collision.")
        # load grasps for objects in scene.
        grasp_dict = obj_grasp_dict
        num_parallel_grasps = len(grasp_dict["paralleljaw_pregrasp_transform"])

        # check if object is in scene by looking a z-coordinate.
        z_value_obj = obj_pose[2, 3]
        if z_value_obj < -100.0:
            print(f"{self.sid} -> mesh not in scene. z values {z_value_obj}")
            return

        # for debugging purpose show axis of objects
        if debug:
            obj_cos = trimesh.creation.axis(
                origin_size=0.4,
                transform=obj_pose,
                origin_color=None,
                axis_radius=None,
                axis_length=None,
            )
            self.trimesh_scene.add_geometry(geometry=obj_cos)

        # idea normalize score for each object in between [0,1]
        parallel_score_max = np.array(grasp_dict["paralleljaw_pregrasp_score"]).max()

        ## paralleljaw
        if parallel_gripper:
            # Accelerate this part by using multiprocessing
            # mp.set_start_method('spawn', force=True)
            save_data_lock = th.Lock()
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=200)

            # iterate through all parallel grasps
            for j in range(num_parallel_grasps):
                parallel_pregrasp_config = grasp_dict["paralleljaw_pregrasp_transform"][
                    j
                ]
                parallel_score = grasp_dict["paralleljaw_pregrasp_score"][j] / (
                    parallel_score_max + 1e-12
                )
                pool.submit(
                    self.check_collison,
                    j,
                    parallel_pregrasp_config,
                    parallel_score,
                    obj_pose,
                    prim_path,
                    approaching,
                    save_data_lock,
                )

            pool.shutdown(wait=True)

        print(f"{self.sid} -> collision check done idx {idx}.")
        return True

    def check_collison(
        self,
        idx,
        parallel_pregrasp_config,
        parallel_score,
        obj_pose,
        prim_path,
        approaching,
        save_data_lock,
    ):

        # filter out zero entries, sometimes grasp sampling cannot find enough antipodal grasps

        # Generate grasp for franka gripper (6DOF) from contact point
        # pose and transform to world cos.
        parallel_vec_a = np.array(
            [
                parallel_pregrasp_config[0],
                parallel_pregrasp_config[1],
                parallel_pregrasp_config[2],
            ]
        )

        parallel_vec_b = np.array(
            [
                parallel_pregrasp_config[3],
                parallel_pregrasp_config[4],
                parallel_pregrasp_config[5],
            ]
        )

        parallel_contact_pt1 = np.array(
            [
                parallel_pregrasp_config[6],
                parallel_pregrasp_config[7],
                parallel_pregrasp_config[8],
            ]
        )

        parallel_width = parallel_pregrasp_config[9]

        parallel_pregrasp_transform = convert_to_franka_6DOF(
            vec_a=parallel_vec_a,
            vec_b=parallel_vec_b,
            contact_pt=parallel_contact_pt1,
            width=parallel_width,
        )

        parallel_pregrasp_transform_world = np.dot(
            obj_pose, parallel_pregrasp_transform
        )

        parallel_contact_transform = convert_to_contact_6DOF(
            vec_a=parallel_vec_a, vec_b=parallel_vec_b, contact_pt=parallel_contact_pt1
        )

        parallel_contact_transform_world = np.dot(obj_pose, parallel_contact_transform)

        # if not colliding and approach set to True, check also for approach dir
        if approaching:
            approach_distances = [0.0, 10.0, 20.0]  # cm -> # removed 20 cm to speed up.
        else:
            approach_distances = [0.0]

        parallel_colliding_names = set()
        parallel_collision_free = True

        ## check collision when approaching
        for dist in approach_distances:
            # compute different gripper poses at approach line
            artificial_contact_pt_ = parallel_contact_pt1 + parallel_vec_a * (-1) * dist
            parallel_gripper_transform_ = convert_to_franka_6DOF(
                vec_a=parallel_vec_a,
                vec_b=parallel_vec_b,
                contact_pt=artificial_contact_pt_,
                width=parallel_width,
            )

            parallel_gripper_transform_world_ = np.dot(
                obj_pose, parallel_gripper_transform_
            )

            pregrasp_width = min(
                parallel_width + PREGRASP_PARALLEL_DISTANCE, GRIPPER_WIDTH_MAX
            )
            parallel_colission_manager = get_parallel_gripper_collision_mesh(
                self.root, pregrasp_width, parallel_gripper_transform_world_
            )

            parallel_colliding, parallel_names = (
                self.collision_manager.in_collision_other(
                    other_manager=parallel_colission_manager,
                    return_names=True,
                    return_data=False,
                )
            )

            parallel_colliding_names.update(
                [collision_pair[0] for collision_pair in parallel_names]
            )

            if parallel_colliding:
                parallel_collision_free = (
                    False  # once a collision is detected, mark as not collision free
                )
                break

        ## check collision when closing the finger
        if not parallel_collision_free:
            # new version : check collsion when closing the gripper
            parallel_gripper_transform_ = convert_to_franka_6DOF(
                vec_a=parallel_vec_a,
                vec_b=parallel_vec_b,
                contact_pt=parallel_contact_pt1,
                width=parallel_width,
            )

            parallel_gripper_transform_world_ = np.dot(
                obj_pose, parallel_gripper_transform_
            )

            pregrasp_width = min(
                parallel_width + PREGRASP_PARALLEL_DISTANCE, GRIPPER_WIDTH_MAX
            )

            # check for collision when closing the gripper till 1cm before
            for curr_gripper_width in np.linspace(
                pregrasp_width, parallel_width + 1, 2
            ):  # changed from 3 -> 2
                parallel_colission_manager = get_parallel_gripper_collision_mesh(
                    self.root, curr_gripper_width, parallel_gripper_transform_world_
                )

                parallel_colliding, parallel_names = (
                    self.collision_manager.in_collision_other(
                        other_manager=parallel_colission_manager,
                        return_names=True,
                        return_data=False,
                    )
                )

                parallel_colliding_names.update(
                    [collision_pair[0] for collision_pair in parallel_names]
                )

                if parallel_colliding:
                    parallel_collision_free = False  # once a collision is detected, mark as not collision free
                    break

        ## save
        # if parallel_collision_free:
        # print(f"{self.sid} -> Grasp idx {idx}: {parallel_collision_free}")
        with save_data_lock:
            if parallel_collision_free is True and parallel_width > 0.0:
                # save for later use in dataset
                # in world cos
                self.non_colliding_parallel_gripper_poses.append(
                    parallel_pregrasp_transform_world
                )

                self.non_colliding_parallel_analytical_score.append(parallel_score)

                # same order as self.all_poses
                self.non_colliding_parallel_object_id.append(idx)

                # add asset nameenv_1_epi_173_step_0_scene
                self.non_colliding_parallel_asset_names.append(str(prim_path))

                # in world cos
                self.non_colliding_parallel_contact_poses.append(
                    parallel_contact_transform_world
                )

                self.non_colliding_parallel_contact_width.append(parallel_width)

            elif parallel_collision_free is False:
                # save for later use in dataset
                # in world cos
                self.colliding_parallel_gripper_poses.append(
                    parallel_pregrasp_transform_world
                )

                self.colliding_parallel_analytical_score.append(parallel_score)


                self.colliding_parallel_names.append(parallel_colliding_names)

                # same order as self.all_poses
                self.colliding_parallel_object_id.append(idx)

                # in world cos
                self.colliding_parallel_contact_poses.append(
                    parallel_contact_transform_world
                )

                self.colliding_parallel_contact_width.append(parallel_width)
            else:
                raise NotImplementedError

    def visualize_grasps(self):
        #self.trimesh_scene.show()

        ## visualize parallel gripper
        for i, parallel_grasp_config in enumerate(
            self.non_colliding_parallel_gripper_poses
        ):
            score = self.non_colliding_parallel_analytical_score[i]
            # Transform of gripper COS (G) relative to world (W)
            # T_W_G = from_contact_to_6D(parallel_grasp_config)
            grasp_width = self.non_colliding_parallel_contact_width[i]
            # add to scene
            self.trimesh_scene.add_geometry(
                geometry=create_easy_gripper(
                    color=interpolate_between_red_and_green(score, 150),
                    sections=3,
                    show_axis=True,
                    width=grasp_width,
                ),
                transform=parallel_grasp_config,
            )
            cos_system = trimesh.creation.axis(
                origin_size=0.1,
                transform=np.array(self.non_colliding_parallel_contact_poses[i]),
            )
            self.trimesh_scene.add_geometry(geometry=cos_system)

        print(f"{self.sid} -> show scene ...")
        self.trimesh_scene.show()

    def load_grasps_from_file(self):
        filter = self.scene["non_colliding_parallel_contact_width"].numpy() > 0.0
        self.non_colliding_parallel_contact_width = self.scene[
            "non_colliding_parallel_contact_width"
        ].numpy()[filter]
        self.non_colliding_parallel_gripper_poses = self.scene[
            "non_colliding_parallel_gripper_poses"
        ].numpy()[filter]
        self.non_colliding_parallel_analytical_score = self.scene[
            "non_colliding_parallel_analytical_score"
        ].numpy()[filter]
        self.non_colliding_parallel_object_id = self.scene[
            "non_colliding_parallel_object_id"
        ].numpy()[filter]
        # non_colliding_parallel_asset_names = grasps['non_colliding_parallel_asset_names'].numpy()
        self.non_colliding_parallel_contact_poses = self.scene[
            "non_colliding_parallel_contact_poses"
        ].numpy()[filter]

        return len(self.non_colliding_parallel_contact_width)

def read_in_scene(
    scene_dir,
    model_root,
    viewpt=0,
    load_meshes=True,
    load_grasps=True,
):
    """
    Read in viewpt of scene and return obj_poses_rel_world as well as categories.
    """
    # hdf5_path = pathlib.Path(scene_dir) / f"{viewpt}_scene.hdf5"

    ## read in hdf5
    # f = h5py.File(str(hdf5_path), 'r')

    try:
        f = torch.load(scene_dir, map_location="cpu")  # pt file
        assert f["obj_poses_robot"] is not None
    except:
        print(f"Scene {scene_dir} -> File is incorrect, the scene will be removed.")
        return None, None, None, None

    ## get meshes with poses
    dset_obj_categories = f["obj_id"]  # int
    dset_cam_pose_rel_world = f["obj_poses_robot"].numpy()  # 4x4 np matrix

    obj_poses_rel_world = []
    obj_meshes_paths = []
    hdf5_paths = []
    categories = []
    for obj_pose_rel_world, cat in zip(dset_cam_pose_rel_world, dset_obj_categories):
        ## transform obj pose to world
        obj_poses_rel_world.append(
            obj_pose_rel_world
        )  # ! to fit structure from RandomObject Class

        obj_meshes_paths.append(model_root / "{:03}".format(int(cat)) / "textured.obj")
        hdf5_paths.append(model_root / "{:03}".format(int(cat)) / "textured.obj.hdf5")
        categories.append("{:03}".format(int(cat)))

    obj_poses_rel_world = np.array(obj_poses_rel_world)
    # load meshes and return
    poses = obj_poses_rel_world
    if load_meshes:
        meshes = [
            load_single_mesh(str(path), output_on=False) for path in obj_meshes_paths
        ]
    else:
        meshes = []
    if load_grasps:
        grasps = [load_single_grasp_config(str(path)) for path in hdf5_paths]
    else:
        grasps = []

    # close file
    # f.close()
    return poses, meshes, grasps, categories


def evaluate_scene(scene_dir):
    path_to_root = pathlib.Path("models")
    path_to_models = pathlib.Path("models/models_ifl")
    scene_id = scene_dir.split("/")[-1].split(".")[0]

    ## read in scene
    obj_poses_rel_world, meshes, grasps_obj, categories = read_in_scene(
        scene_dir, path_to_models, load_meshes=True, load_grasps=True
    )

    if obj_poses_rel_world is None:
        target = glob.glob(f"{scene_root_dir}/{scene_id}*")
        print(
            f"Scene {scene_id} -> No objects in the scene, the scene will be removed."
        )
        for t in target:
            os.remove(t)
        return

    ## check collision
    collision_instance = CollisionWithScene(
        sid=scene_id,
        meshes=meshes,
        poses=obj_poses_rel_world,
        grasps=grasps_obj,
        root_dir=str(path_to_root),
        box_dir=str(path_to_root) + "/KLT/BOX_VOL4.obj",
    )

    if "non_colliding_parallel_gripper_poses" in collision_instance.scene.keys():
        print(f"Scene {collision_instance.sid} -> already evaluated.")
        if not test:
            return
    collision_instance.load_potential_grasps_and_generate_trimesh_scene()

    if "non_colliding_parallel_gripper_poses" in collision_instance.scene.keys():
        valid_grasp_file = collision_instance.load_grasps_from_file()
    else:
        valid_grasp_file = collision_instance.check_for_collision(show_scene=test)

    if not valid_grasp_file:
        print(f"Scene {scene_id} -> No valid grasps, the data will be removed.")
        target = glob.glob(scene_dir)
        for t in target:
            os.remove(t)
        return
    print(f"Scene {scene_id} -> {valid_grasp_file} valid grasps.")

    if test:
        collision_instance.visualize_grasps()


if __name__ == "__main__":
    "Typical usage"
    file_name = "env_*_epi_*_step_*_data.pt"
    scene_dir = pathlib.Path(scene_root_dir)

    scenes = glob.glob(str(scene_dir / file_name))
    if test:
        for scene in scenes:
            evaluate_scene(scene)
    else:
        Pool = mp.Pool(processes=80)
        Pool.map(evaluate_scene, scenes)
        Pool.close()
        Pool.join()
