#!/usr/bin/env python
# Author : Maximilian Gilles, IFL, KIT
import trimesh
import multiprocessing as mp
import argparse
import pathlib
import glob
import numpy as np
import os
import random
import transforms3d
import h5py
import open3d as o3d

# HARDWARE SPECIFIC PARAMETERS
GRIPPER_HEIGHT = 11.21  # cm
AREA_CENTER = 0.8  # cm
GRIPPER_WIDTH_MAX = 8  # cm
PREGRASP_PARALLEL_DISTANCE = 3  # cm
                                # note: added to the predicted gripper width.


def get_pj_collission_manager(grasp_width, transform):
    """
        Based on grasp width, return different collision manager
        for franka gripper. Maximum grasp width is 8 cm.
    """
    # hand
    hand_mesh = trimesh.load("./models/hand_collision.stl")
    hand_mesh = hand_mesh.apply_scale(100)  # convert m to cm
    # finger left
    finger_left_mesh = trimesh.load("./models/finger_collision_left.stl")
    finger_left_mesh = finger_left_mesh.apply_scale(100)
    # finger right
    finger_right_mesh = trimesh.load("./models/finger_collision_right.stl")
    finger_right_mesh = finger_right_mesh.apply_scale(100)

    # create collsision manager for franka hand
    franka_hand_collision_manager = trimesh.collision.CollisionManager()
    # add hand
    hand_trans = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]

    hand_trans_world = np.dot(
        transform,
        hand_trans)
    franka_hand_collision_manager.add_object(
        name="hand",
        mesh=hand_mesh,
        transform=np.array(hand_trans_world))

    # add finger left
    finger_left_trans = [
        [1, 0, 0, -grasp_width/2],
        [0, 1, 0, 0],
        [0, 0, 1, 5.81],
        [0, 0, 0, 1]]

    finger_left_trans_world_ = np.dot(
        hand_trans_world,
        finger_left_trans)
    franka_hand_collision_manager.add_object(
        name="finger_left",
        mesh=finger_left_mesh,
        transform=np.array(finger_left_trans_world_))

    # add finger right
    finger_right_trans = [
        [1, 0, 0, grasp_width/2],
        [0, 1, 0, 0],
        [0, 0, 1, 5.81],
        [0, 0, 0, 1]]

    finger_right_trans_world = np.dot(
        hand_trans_world,
        finger_right_trans)
    franka_hand_collision_manager.add_object(
        name="finger_right",
        mesh=finger_right_mesh,
        transform=np.array(finger_right_trans_world))
    return franka_hand_collision_manager


def create_file(rootdir, filepath):
    # create object
    path = rootdir / str(str(filepath) + ".hdf5")

    try:
        f = h5py.File(str(path), 'r+')
        print(f"[Info] open {path}")
    except OSError:
        f = h5py.File(str(path),'w')
        print(f"[Info] create {path}")

    # create groupsparalleljaw_subgrp
    _ = f.require_group("grasps")
    _ = f.require_group("metadata")

    return f


def load_mesh(root, relative, scale_factor=None):
    # load mesh and check for watertight
    path = root / relative
    obj_mesh = trimesh.load(str(path))
    if scale_factor is not None:
        obj_mesh = obj_mesh.apply_scale(scale_factor)
        new_scale = obj_mesh.scale
    print("[Info] loaded mesh {} | scale {:03}".format(str(path), new_scale))
    return obj_mesh, True


def sample_antipodal_grasps_geometrically(mesh, contact_points,
                                          contact_indexes, hdf5_file,
                                          mesh_name="", **config):
    """
    Description:
        Sample antipodal grasps by assuming soft-contact-model.
        Intersecting line from contact points must lie within
        friction cone at both contact locations. See "A mathematical
        introduction to robotic manipulation" for prove.
    Returns:
        counter         :   (n,)        number of found positive grasps
        grasp_config    :   (n,4)       stores pre-grasp config gripper_width, 
                                        rotation matrix, translation and 4x4 
                                        transform matrix
    """
    # load config
    tries = config['samples_per_point']
    max_angle_sampling = config['max_angle_sampling']
    max_angle_friction = config['max_angle_friction']
    max_translation = config['max_translation']
    max_gripper_length = config['antipodal']['gripper_length']
    distance_gripper = config['antipodal']['distance_gripper']
    max_collision_tryouts = config['antipodal']['collision_samples']
    # max_rotation_per_grasp = config['antipodal']['max_rotation_per_grasp']
    max_grasps = config['max_grasps']
    grasps_config = []
    counter = 0
    # initialize dataset
    # save in hdf5 file
    # get grasp group first and create paralleljaw subgroup

    grp = hdf5_file['grasps']
    # add non colliding grasps but first delete old dataset
    try:
        del hdf5_file['grasps']["paralleljaw"]
        print("delete old paralleljaw group.")
    except:
        print("[grasps][paralleljaw] not found.")
    paralleljaw_subgrp = grp.require_group("paralleljaw")

    dset_grasp_transform = paralleljaw_subgrp.create_dataset(
        name="pregrasp_transform",
        shape=(max_grasps, 10),
        dtype='f')

    dset_grasp_score = paralleljaw_subgrp.create_dataset(
        name="quality_score",
        shape=(max_grasps,),
        dtype='f')

    # initialize collision manager
    collision_manager = trimesh.collision.CollisionManager()
    collision_manager.add_object(
        name="mesh_object",
        mesh=mesh,
        transform=np.identity(4))

    for i in range(len(contact_points)):
        if counter >= max_grasps:
            # do not sample more than max_grasps
            # max number of grasps reached. stop here.
            break

        succ_counter_rot = 0
        pos, neg = 0, 0
        successful_candidates_ = []
        for k in range(tries):
            approach_vec_, center_ = generate_random_approach(
                max_rotation=max_angle_sampling,
                max_translation=max_translation)
            # get surface normal
            surface_norm_ = mesh.face_normals[contact_indexes[i]]  
            # get transformation matrix between z-axis and surface normal
            H, _ = trimesh.geometry.align_vectors(
                surface_norm_, [0, 0, 1], True)
            # rotate approach vector to surface normal
            approach_vec_rot = np.dot(H.T[:3, :3], approach_vec_)
            # rotate center to surface normal
            center_rot = np.dot(H.T[:3, :3], center_)
            # transform to contact point pregrasp
            center_pregrasp = center_rot + contact_points[i] + (-1) * approach_vec_rot * (1.0)
            faces_idx, _, pts_projected_ = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh).intersects_id(
                    ray_origins=[center_pregrasp],
                    ray_directions=[approach_vec_rot],
                    multiple_hits=True,
                    return_locations=True)
            # handle differenct scenarios for intersecting rays
            if len(pts_projected_) < 2:
                # if less then two contacts -> no valid grasp
                pass
            elif len(pts_projected_) >= 2:
                # check angle between rays
                p1 = pts_projected_[0]
                p1_idx = faces_idx[0]
                p1_surface_norm = mesh.face_normals[p1_idx]
                # chose random point in between
                random_idx = random.randint(1, len(pts_projected_)-1)
                p2 = pts_projected_[random_idx]
                p2_idx = faces_idx[random_idx]
                p2_surface_norm = mesh.face_normals[p2_idx]
                v = p2 - p1
                v_l2 = np.linalg.norm(v)
                v = v / (v_l2 + 0.0000001)
                # check for max gripper opening
                if v_l2 < max_gripper_length:
                    # check for angle
                    p1_angle = trimesh.geometry.vector_angle([-v, p1_surface_norm])
                    p2_angle = trimesh.geometry.vector_angle([v, p2_surface_norm])
                    if p1_angle <= max_angle_friction and p2_angle <= max_angle_friction:
                        successful_candidates_.append([p1, p2, v])
                        pos += 1
                    else:
                        neg += 1
                else:
                    # if greater then max gripper width -> no valid grasp
                    neg += 1
        # continue if we have at least one successful candidate k for c_i
        if len(successful_candidates_) >= 1:
            # idea 1 : select random successful antipodal candidate
            # idea 2 : compute mean p1, p2 and v over all successful candidates

            # idea 3 : take all successfull candidates c_i_k
            random.shuffle(successful_candidates_)
            for succ_idx in range(len(successful_candidates_)-1):
                succ_counter_rot = 0
                p1_, p2_, v_ = successful_candidates_[succ_idx]
                ray = [p1_ - v_*2, p2_ + v_*2]

                # compute grasp config (end effector pose) for G_i_k_l
                # choose rotation (vector b) randomly but check for collision
                angles = np.linspace(0, 2 * np.pi, max_collision_tryouts)
                # width w_
                w_ = np.linalg.norm(p2_ - p1_)
                if w_ == 0:
                    continue
                for angle in angles:
                    R = rodrigues_rotation(v_, angle)

                    # Define an arbitrary vector to be rotated (e.g., orthogonal to v_)
                    if v_[0] == 0 and v_[1] == 0:
                        orthogonal_vector = np.array([1, 0, 0])
                    else:
                        orthogonal_vector = np.cross(v_, [0, 0, 1])
                    orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
                    
                    # Rotate the orthogonal vector to get the approaching vector
                    a_ = np.dot(R, orthogonal_vector)
                    a_ = a_ / np.linalg.norm(a_)

                    c_ = np.cross(a_, v_)
                    R_ = [[v_[0], c_[0], a_[0]],
                          [v_[1], c_[1], a_[1]],
                          [v_[2], c_[2], a_[2]]]
                    
                    # translation t_
                    t_ = p1_ + w_/2 * v_ + distance_gripper * a_ * (-1)
                    # create 4x4 transform matrix G of grasp
                    grasp_transform_ = [[R_[0][0], R_[0][1], R_[0][2], t_[0]],
                                        [R_[1][0], R_[1][1], R_[1][2], t_[1]],
                                        [R_[2][0], R_[2][1], R_[2][2], t_[2]],
                                        [0.0, 0.0, 0.0, 1.0]]
                    # check for collision with moving gripper fingers
                    # -> additional filtering, useful for complex items
                    #    we do not want to pregrasp with maximimum width!
                    grasp_width = min((w_ + PREGRASP_PARALLEL_DISTANCE), GRIPPER_WIDTH_MAX)
                    gripper_collision_mesh = get_pj_collission_manager(
                        grasp_width=grasp_width,
                        transform=np.array(grasp_transform_))
                    colliding = collision_manager.in_collision_other(
                        other_manager=gripper_collision_mesh,
                        return_names=False,
                        return_data=False)
                    if colliding is False:
                        #if succ_counter_rot >= max_rotation_per_grasp:
                            # do not sample more than
                            # max_rotation_per_grasp per c_i_k!
                        #    break
                        if counter >= max_grasps:
                            # do not sample more than
                            # max_grasps per object
                            break
                        # way to go if you want to store in hdf5 file
                        dset_grasp_transform[counter, 0] = a_[0]
                        dset_grasp_transform[counter, 1] = a_[1]
                        dset_grasp_transform[counter, 2] = a_[2]
                        dset_grasp_transform[counter, 3] = v_[0]
                        dset_grasp_transform[counter, 4] = v_[1]
                        dset_grasp_transform[counter, 5] = v_[2]
                        dset_grasp_transform[counter, 6] = p1_[0]
                        dset_grasp_transform[counter, 7] = p1_[1]
                        dset_grasp_transform[counter, 8] = p1_[2]
                        dset_grasp_transform[counter, 9] = w_
                        # save score
                        # score is defined as pos/tries
                        prec = pos/tries
                        dset_grasp_score[counter] = prec
                        # save grasp config with score and ray for vis
                        grasps_config.append([w_, R_, t_, grasp_transform_])
                        succ_counter_rot += 1
                        counter += 1
        else:
            pass
        print("PJ | mesh {} | pt {:04} | # grasps {:05} | \
               antipodal curr. score {:03} ".format(mesh_name, i, counter, pos/tries))
    return counter, grasps_config


def rodrigues_rotation(axis, angle):
    """
    Compute the rotation matrix using Rodrigues' rotation formula.
    :param axis: The rotation axis (should be a unit vector)
    :param angle: The rotation angle in radians
    :return: The rotation matrix
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R


def generate_random_approach(max_rotation, max_translation):
    # rotate and slightly move
    R = transforms3d.euler.euler2mat(
        ai=random.uniform(-max_rotation, max_rotation),
        aj=random.uniform(-max_rotation, max_rotation),
        ak=random.uniform(-max_rotation, max_rotation))
    T = [random.uniform(-max_translation, max_translation),
         random.uniform(-max_translation, max_translation),
         random.uniform(-max_translation, max_translation)]
    approach_vector = np.dot(R, np.array([0, 0, -1]).T)
    center = np.dot(R, np.array([0, 0, 0]).T) + T
    return approach_vector, center


def get_sc_collission_mesh(radius, height):
    ret = trimesh.creation.cone(
        radius=radius,
        height=height,
    )
    return ret


def sample_surface_points_com(mesh, num_points):
    # sample rays from COM
    mesh_com = mesh.center_mass
    ray_origins = []
    ray_directions = []
    for i in range(num_points):
        # sample in SE(3) for x,y,z
        ray_origins.append(mesh_com)
        ray_directions.append([random.uniform(-1, 1) for i in range(3)])
    # let them intersect with mesh
    contact_points, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions)
    return contact_points, index_tri


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for Parallel-Jaw and Vacuum Suction Grasp Sampling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--mesh_root", default="./models/models_ifl",
        help="Root directory for meshes.")
    parser.add_argument(
        "--categories", type=str, nargs="+",
        help="Specify specific object category to load. If not specified load all.")
    parser.add_argument(
        "--num_points", default=1000, type=int,
        help="Number of contact points to sample.")
    parser.add_argument(
        "--max_grasps", default=24000, type=int,
        help="Max. number of successful grasps to sample.")
    parser.add_argument(
        "--random", action='store_true', default=True,
        help="Samle points even over surface.")
    parser.add_argument(
        "--com", action='store_true',
        help="Samle points based on Center of Mass.")
    parser.add_argument(
        "--paralleljaw", action='store_true',
        help="Sample antipodal points for two-finger gripper.")
    parser.add_argument(
        "--suction", action='store_true',
        help="Sample suction grasps for vacuum gripper.")
    parser.add_argument(
        "--pool_size", type=int, default=None,
        help="How many cores to use? [default use all cores].")
    args = parser.parse_args()

    # create config dict
    config = {
        'max_angle_sampling': np.pi/16,
        'max_angle_friction': np.pi/8,
        'samples_per_point': 10,
        'max_translation': 0.5,
        'max_grasps': args.max_grasps,
        'antipodal': {
            'gripper_length': 8,
            'distance_gripper': (GRIPPER_HEIGHT - AREA_CENTER),
            'collision_samples': 360 // 24,
            #'max_rotation_per_grasp': 200,  # how many max grasps per successfull
                                           # antipodal sample differ in rotation
        },
        'suction': {
            # sampling
            'approach_height': 3.0,

            # dimension
            'length_gripper': 15,
            'radius_gripper': 2,

            'num_vertices': 8,
            'tolerance': 0.15,

            'cup_radius': 0.7,
            'num_mass_points': 128,
            'young': 0.02,
            'mu': 1.0,
            'pressure_diff': 0.7,
            'compensable_height_difference': 0.7
        }
    }

    category = args.categories
    # get path to all meshes in mesh_root
    obj_paths = []
    if args.categories is not None:
        for cat in category:
            obj_paths.append(glob.glob(
                os.path.join(
                    args.mesh_root,
                    cat,
                    "**/textured.obj"),
                recursive=True)[0])
    else:
        obj_paths = glob.glob(os.path.join(args.mesh_root, "**/textured.obj"),
                              recursive=True)

    print("Found following Object_paths : ", obj_paths)
    print("is this ok? press any key to continue.")
    input()

    mesh_root_path = pathlib.PosixPath(args.mesh_root)
    gripper_root_path = pathlib.PosixPath("./grasps_sampling")

    def do_in_parallel(obj_path):

        relative_obj_path = pathlib.PosixPath(obj_path).relative_to(mesh_root_path)
        # load mesh
        obj_mesh, _ = load_mesh(mesh_root_path, relative_obj_path, scale_factor=100)
        obj_mesh.fill_holes()

        # create hdf5 file
        hdf5_file = create_file(mesh_root_path, relative_obj_path)

        # sample contact points
        if args.com:
            # sample contact point from rays from COM
            contact_points, index_tri = sample_surface_points_com(
                mesh=obj_mesh,
                num_points=args.num_points)
        elif args.random:
            # sample contact points randomly
            contact_points, index_tri = trimesh.sample.sample_surface_even(
                mesh=obj_mesh,
                count=args.num_points,
                radius=None)  # removes points below this radius
        
        # visualize sampled points using open3d
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(contact_points)
        #o3d.visualization.draw_geometries([pcd])         

        # parallel jaw gripper
        counter, grasps_config = sample_antipodal_grasps_geometrically(
            obj_mesh,
            contact_points,
            contact_indexes=index_tri,
            hdf5_file=hdf5_file,
            mesh_name=relative_obj_path,
            **config)

        hdf5_file.close()
        return True

    if args.pool_size is None:
        pool = mp.Pool(mp.cpu_count())
    else:
        pool = mp.Pool(args.pool_size)
    results = pool.map(do_in_parallel, [path for path in obj_paths])
    pool.close()
