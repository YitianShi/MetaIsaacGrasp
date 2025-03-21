#!/usr/bin/env python
# Author : Maximilian Gilles, IFL, KIT

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import pathlib
import h5py
import random
import cv2

def filter_contact_points(pcd, contacts, threshold_radius=0.005):
    """
    Filter out contact points for given sensor cloud.
    Args:
        pcd [o3d.geometry.PointCloud] : colored point cloud
        contacts [list] : list of contact transformations
        radius [float] : search radius for radius_vector_search in meter
    Return: 
        valid_contacts [list] : list of True, False for each contact
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    ret = []
    for contact_transform in contacts:
        pt = contact_transform[0:3,3] / 100
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pt, 1)
        pcd_pt = np.asarray(pcd.points)[idx][0]
        dist_vec = pcd_pt - pt
        dist = np.linalg.norm(dist_vec)
        if dist < threshold_radius:
            ret.append(True)
        else:
            ret.append(False)
    return ret

def get_franka_gripper_to_contact_transform(grasp_width, 
        gripper_length=0.1121):
    T_contact_relative_to_gripper = np.array([
        [1.0, 0.0, 0.0, -grasp_width/2],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, gripper_length],
        [0.0, 0.0, 0.0, 1.0]])
    return T_contact_relative_to_gripper

def random_rgb_color(seed, score=1.0):
    random.seed(seed)
    red = random.random()
    green = random.random()
    blue = random.random()
    return np.array([red, green, blue])

def rgb_score_color(score):
    red = float(1.0-score)
    green = float(score)
    blue = 0.0
    return np.array([red, green, blue])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PCL Generation.")
    parser.add_argument(
        "--data_root",
        type=str, 
        default="data/",
        help="Path to data.")
    parser.add_argument(
        "--scene",
        type=int, default=0,
        help="Specify which scene to load.")
    parser.add_argument(
        "--viewpt",
        type=int, default=0,
        help="Specify which viwpt to load.")
    parser.add_argument(
        "--save_pcl",
        action="store_true",
        help="Set flag to save pcl in same location.")
    parser.add_argument(
        "--visualize_parallel_gripper",
        action="store_true",
        help="Set flag to visualize two finger grasps.")
    parser.add_argument(
        "--visualize_suction_cup",
        action="store_true",
        help="Set flag to visualize suctioncup.")
    parser.add_argument(
        "--visualize_colliding_grasps",
        action='store_true',
        help="Set flag to visualize colliding grasps.")
    parser.add_argument(
        "--visualize_keypts_com",
        action="store_true",
        help="Set flag to visualize center of mass points.")
    parser.add_argument(
        "--visualize_keypts_byhand",
        action="store_true",
        help="Set flag to visualize byhand keytps.")
    parser.add_argument(
        "--visualize_parallel_contacts_from_dataset",
        action="store_true",
        help="Set flag to visualize contact points from dataset.")
    parser.add_argument(
        "--visualize_parallel_contacts_from_gripper_pose",
        action="store_true",
        help="Set flag to generate contact points from gripper pose.")
    parser.add_argument(
        "--colorize_per_score",
        type=str,
        help="Colorize Grasps [analytical] or [simulation] or [l2norm].")
    parser.add_argument(
        "--colorize_per_object",
        action="store_true", default=True,
        help="Set flag to colorize based on object id.")
    parser.add_argument(
        "--score_threshold",
        type=float, default=-1,
        help="Only show grasps above threshold.")
    parser.add_argument(
        "--visualize_pose",
        action="store_true",
        help="Set flag to visualize object pose.")
    args = parser.parse_args()

    PATH_TO_DATA = pathlib.Path(args.data_root)
    SCENE = args.scene
    VIEWPT = args.viewpt

    PATH_TO_SCENE = PATH_TO_DATA / f"scene{SCENE}"
    PATH_TO_RGB = PATH_TO_SCENE / f"{VIEWPT}_rgb.png"
    PATH_TO_DEPTH = PATH_TO_SCENE / f"{VIEWPT}.npz"
    PATH_TO_CAMPARAMS = PATH_TO_SCENE / f"{VIEWPT}_camera_params.json"
    PATH_TO_HDF5 = PATH_TO_SCENE / f"{VIEWPT}_scene.hdf5"
    PATH_TO_PCL = PATH_TO_SCENE / f"{VIEWPT}_scene_pcl.ply"

    # load np arrays
    color_raw = o3d.io.read_image(str(PATH_TO_RGB))
    with np.load(str(PATH_TO_DEPTH)) as data:
        depth_np = data['depth']

    height, width = depth_np.shape

    print("height [px] x width [px]", depth_np.shape)
    print("depth min [cm]", np.amin(depth_np))
    print("depth max [cm]", np.amax(depth_np))

    # inpait NaN values as zeros
    for i in range(depth_np.shape[0]):
        for j in range(depth_np.shape[1]):
            if np.isnan(depth_np[i,j]):
                depth_np[i,j] = 0.0

    depth_raw = o3d.geometry.Image(depth_np)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_raw, 
        depth=depth_raw,
        depth_scale=100, # important! cm -> m
        depth_trunc=1,
        convert_rgb_to_intensity=False)

    with open(str(PATH_TO_CAMPARAMS)) as json_file:
        f = json.load(json_file)
    fx = float(f['fx'])
    fy = float(f['fy'])


    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    camera_intrinsics.set_intrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=width/2,
        cy=height/2)

    print("intrinsic matrix : \n", camera_intrinsics.intrinsic_matrix)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics)

    gripper_meshes = []
    gripper_cos = []
    if args.visualize_parallel_gripper:
        # load two finger gripper as mesh
        # load gripper config and transform
        f = h5py.File(str(PATH_TO_HDF5), 'r')

        grasps_selected = 'non_colliding_grasps' if not \
            args.visualize_colliding_grasps else 'colliding_grasps'

        dset_grasps = f[grasps_selected]['paralleljaw']['franka_poses_relative_to_camera']
        dset_obj_id = f[grasps_selected]['paralleljaw']['object_id']
        dset_score_analytical = f[grasps_selected]['paralleljaw']['score_analytical']
        dset_score_simulation = f[grasps_selected]['paralleljaw']['score_simulation']
        
        parallel_grasps = list(dset_grasps)
        object_ids = list(dset_obj_id)
        scores_analytical = list(dset_score_analytical)
        scores_simulation = list(dset_score_simulation)
        
        for grasp_transform, id, score_analytical, score_simulation in zip(parallel_grasps, object_ids, scores_analytical, scores_simulation):
            grasp_transform[0,3] /=100
            grasp_transform[1,3] /=100
            grasp_transform[2,3] /=100
            print(grasp_transform)
            gripper = o3d.io.read_triangle_mesh("./utils/Meshes/parallel_gripper.ply")
            cos = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            gripper.scale(scale=0.01, center=np.array([0,0,0]))
            if args.colorize_per_score == "analytical":
                _score = score_analytical
            elif args.colorize_per_score == "simulation":
                _score = score_simulation
            elif args.colorize_per_score == "l2norm":
                _score = np.sqrt(score_analytical*score_analytical + score_simulation*score_simulation)/np.sqrt(2)
            else:
                print("no score selected!")
            if args.colorize_per_score == "analytical" or args.colorize_per_score == "simulation" or args.colorize_per_score == "l2norm":
                color = rgb_score_color(score=_score)
                gripper.paint_uniform_color(color)
            elif args.colorize_per_object:
                color = random_rgb_color(seed=id)
                gripper.paint_uniform_color(color)
            if _score > args.score_threshold:
                gripper_meshes.append(gripper.transform(grasp_transform))

    parallel_contacts = []
    if args.visualize_parallel_contacts_from_dataset:
        f = h5py.File(str(PATH_TO_HDF5), 'r')
        grasps_selected = 'non_colliding_grasps' if not \
            args.visualize_colliding_grasps else 'colliding_grasps'
        dset_contacts = f[grasps_selected]['paralleljaw']['contact_poses_relative_to_camera']
        contact_poses = list(dset_contacts)
        for contact_transform in contact_poses:
            contact_transform[0,3] /=100
            contact_transform[1,3] /=100
            contact_transform[2,3] /=100
            contact_cos = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
            parallel_contacts.append(contact_cos.transform(contact_transform))
    if args.visualize_parallel_contacts_from_gripper_pose:
        f = h5py.File(str(PATH_TO_HDF5), 'r')
        grasps_selected = 'non_colliding_grasps' if not \
            args.visualize_colliding_grasps else 'colliding_grasps'
        dset_grasps = f[grasps_selected]['paralleljaw']['franka_poses_relative_to_camera']
        dset_grasp_width = f[grasps_selected]['paralleljaw']['contact_width']
        parallel_grasps = list(dset_grasps)
        grasp_widths = list(dset_grasp_width)
        for grasp_transform, width in zip(parallel_grasps, grasp_widths):
            # from gripper cos -> to contact cos
            relative_transform_contact_to_gripper = get_franka_gripper_to_contact_transform(
                grasp_width=width/100)
            grasp_transform[0,3] /=100
            grasp_transform[1,3] /=100
            grasp_transform[2,3] /=100
            contact_transform = np.dot(
                grasp_transform,
                relative_transform_contact_to_gripper)
            contact_cos = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
            parallel_contacts.append(contact_cos.transform(contact_transform))

    suctioncup_meshes = []
    suctioncup_cos = []
    if args.visualize_suction_cup:
        # load two finger gripper as mesh
        # load gripper config and transform
        f = h5py.File(str(PATH_TO_HDF5), 'r')
        grasps_selected = 'non_colliding_grasps' if not \
            args.visualize_colliding_grasps else 'colliding_grasps'
        dset_grasps = f[grasps_selected]['suctioncup']['suction_poses_relative_to_camera']
        dset_obj_id = f[grasps_selected]['suctioncup']['object_id']
        dset_score_analytical = f[grasps_selected]['suctioncup']['score_analytical']
        dset_score_simulation = f[grasps_selected]['suctioncup']['score_simulation']
        
        suctioncup_grasps = list(dset_grasps)
        object_ids = list(dset_obj_id)
        scores_analytical = list(dset_score_analytical)
        scores_simulation = list(dset_score_simulation)

        for grasp_transform, id, score_analytical, score_simulation in zip(suctioncup_grasps, object_ids, scores_analytical, scores_simulation):
            grasp_transform[0,3] /=100
            grasp_transform[1,3] /=100
            grasp_transform[2,3] /=100

            # gripper = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            gripper = o3d.io.read_triangle_mesh("./utils/Meshes/suction_gripper.ply")
            cos = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            gripper.scale(scale=0.01, center=np.array([0,0,0]))

            if args.colorize_per_score == "analytical":
                _score = score_analytical
            elif args.colorize_per_score == "simulation":
                _score = score_simulation
            elif args.colorize_per_score == "l2norm":
                _score = np.sqrt(score_analytical*score_analytical + score_simulation*score_simulation)/np.sqrt(2)
            else:
                print("no score selected!")
            if args.colorize_per_score == "analytical" or args.colorize_per_score == "simulation" or args.colorize_per_score == "l2norm":
                color = rgb_score_color(score=_score)
                gripper.paint_uniform_color(color)
            elif args.colorize_per_object:
                color = random_rgb_color(seed=id)
                gripper.paint_uniform_color(color)

            if _score > args.score_threshold:
                suctioncup_meshes.append(gripper.transform(grasp_transform))

    keypts_com_vis = []
    if args.visualize_keypts_com:
        f = h5py.File(str(PATH_TO_HDF5), 'r')
        keypts_com_dset = f['keypts']['com']['keypts_relative_to_camera']
        keypts_com_object_id_dset = f['keypts']['com']['object_id']
        keypts_com = list(keypts_com_dset)
        for keypt in keypts_com:
            score_ = keypt[0]
            keypt_x = keypt[1]/100
            keypt_y = keypt[2]/100
            keypt_z = keypt[3]/100
            keypt_com = o3d.geometry.TriangleMesh.create_sphere()
            keypt_com.scale(scale=0.002, center=np.array([keypt_x,keypt_y,keypt_z]))
            color = rgb_score_color(score=score_)
            keypt_com.paint_uniform_color(color)
            keypts_com_vis.append(keypt_com)
    
    keypts_byhand_vis = []
    if args.visualize_keypts_byhand:
        f = h5py.File(str(PATH_TO_HDF5), 'r')
        keypts_byhand_dset = f['keypts']['byhand']['keypts_relative_to_camera']
        keypts_byhand_object_id_dset = f['keypts']['byhand']['object_id']
        keypts_byhand = list(keypts_byhand_dset)
        print(keypts_byhand)
        print(keypts_byhand_object_id_dset)
        for keypt in keypts_byhand:
            id_ = keypt[0]
            keypt_x = keypt[1]/100
            keypt_y = keypt[2]/100
            keypt_z = keypt[3]/100
            keypt_byhand = o3d.geometry.TriangleMesh.create_sphere()
            keypt_byhand.scale(scale=0.01, center=np.array([keypt_x,keypt_y,keypt_z]))
            color = random_rgb_color(seed=id_)
            keypt_byhand.paint_uniform_color(color)
            keypts_byhand_vis.append(keypt_byhand)

    camera_cos = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    meshes_cos = []
    if args.visualize_pose:
        # load json
        PATH_TO_POSE = PATH_TO_SCENE / 'scene_gt.json'
        with open(str(PATH_TO_POSE), 'r') as f:
            poses = json.load(f)
        # iterate over json
        poses_in_vpt = poses[str(VIEWPT)]
        for idx, pose in enumerate(poses_in_vpt):
            R_m2c = pose['cam_R_m2c']
            t_m2c = pose['cam_t_m2c']
            mesh_class = pose['obj_id']

            def hex_to_rgb(value):
                value = value.lstrip('#')
                lv = len(value)
                return tuple(int(value[i:i + lv // 3], 16)/256. for i in range(0, lv, lv // 3))

            color_list_hex = ['#be254a', '#dc484c', '#ef6645', '#f88c51', '#fdb365', '#fed27f', '#feeb9d', '#fffebe',
                          '#f0f9a7', '#d8ef9b', '#b3e0a2', '#89d0a4', '#60bba8', '#3f97b7', '#4273b3']
            
            color_list = [hex_to_rgb(color) for color in color_list_hex]

            T_m2c = np.array([[R_m2c[0][0], R_m2c[0][1], R_m2c[0][2], t_m2c[0]/100.],
                            [R_m2c[1][0], R_m2c[1][1], R_m2c[1][2], t_m2c[1]/100.],
                            [R_m2c[2][0], R_m2c[2][1], R_m2c[2][2], t_m2c[2]/100.],
                            [0, 0, 0, 1]])

            mesh_cos = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

            meshes_cos.append(mesh_cos.transform(T_m2c))

    # world_cos
    world_cos = []

    # world_cos
    world_cos = []
    o3d.visualization.draw_geometries([
        pcd,
        *gripper_meshes,
        *suctioncup_meshes,
        *gripper_cos,
        *suctioncup_cos,
        *parallel_contacts,
        *keypts_byhand_vis,
        *meshes_cos,
        *keypts_com_vis])

    if args.save_pcl:
        pcl_path = PATH_TO_PCL
        o3d.io.write_point_cloud(str(pcl_path), pcd)
        print(f"-> save pcl {pcl_path}")