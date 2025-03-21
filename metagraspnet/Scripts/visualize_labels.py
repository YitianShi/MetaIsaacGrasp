#!/usr/bin/env python
# Author : Maximilian Gilles, IFL, KIT

import trimesh
import numpy as np
import argparse
import pathlib
import h5py
import random
import os

# Gripper Parameters ParallelJaw and Suction Cup
GRIPPER_HEIGHT = 11.21 # cm
AREA_CENTER = 0.8 # cm
PREGRASP_PARALLEL_DISTANCE = 3 # cm note: this is added to the predicted gripper width.
GRIPPER_WIDTH_MAX = 8 #cm
SUCTIONCUP_HEIGHT = 11.21 #cm
SUCTIONCUP_APPROACH_DISTANCE = 2 # cm


def interpolate_between_red_and_green(score, alpha = 255):
    """
    Returns RGBA color between RED and GREEN.
    """
    delta = int(score*255)
    COLOR = [255-delta, 0+delta, 0, alpha] # RGBA
    return COLOR


def random_rgb_color(seed, score=1.0):
    random.seed(seed)
    red = random.random()
    green = random.random()
    blue = random.random()
    return np.array([int(red*255), int(green*255), int(blue*255), 255])


def create_easy_gripper(
        color=[0, 255, 0, 140], sections=6, show_axis=False, width=None):
    if width:
        w = min((width + PREGRASP_PARALLEL_DISTANCE)/2, 4.1)
    else:
        w = 4.1

    l_center_grasps = GRIPPER_HEIGHT - AREA_CENTER  # gripper length till grasp contact

    cfl = trimesh.creation.cylinder(
        radius=0.1,
        sections=sections,
        segment=[[w, -7.27595772e-10, 6.59999996],
                 [w, -7.27595772e-10, l_center_grasps]])
    cfr = trimesh.creation.cylinder(
        radius=0.1,
        sections=sections,
        segment=[[-w, -7.27595772e-10, 6.59999996],
                 [-w, -7.27595772e-10, l_center_grasps]])
    # arm
    cb1 = trimesh.creation.cylinder(
        radius=0.1, sections=sections,
        segment=[[0, 0, 0],
                 [0, 0, 6.59999996]])
    # queer
    cb2 = trimesh.creation.cylinder(
        radius=0.1,
        sections=sections,
        segment=[[-w, 0, 6.59999996],
                 [w, 0, 6.59999996]])
    # coordinate system
    if show_axis:
        cos_system = trimesh.creation.axis(
            origin_size=0.04,
            transform=None,
            origin_color=None,
            axis_radius=None,
            axis_length=None)
        tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl, cos_system])
    else:
        tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color
    return tmp


def convert_to_franka_6DOF(vec_a, vec_b, contact_pt, width):
    """Convert Contact-Point Pregrasp representation to 6DOF Gripper Pose (4x4)."""
    # get 3rd unit vector
    c_ = np.cross(vec_a, vec_b)
    # rotation matrix
    R_ = [[vec_b[0], c_[0], vec_a[0]],
          [vec_b[1], c_[1], vec_a[1]],
          [vec_b[2], c_[2], vec_a[2]]]
    # translation t
    t_ = contact_pt + width/2 * vec_b + (GRIPPER_HEIGHT-AREA_CENTER) * vec_a * (-1)
    # create 4x4 transform matrix of grasp
    pregrasp_transform_ = [[R_[0][0], R_[0][1], R_[0][2], t_[0]],
                           [R_[1][0], R_[1][1], R_[1][2], t_[1]],
                           [R_[2][0], R_[2][1], R_[2][2], t_[2]],
                           [0.0, 0.0, 0.0, 1.0]]
    return np.array(pregrasp_transform_)


def read_in_mesh_config(file, parallel=False, suctioncup=False, simulation=False, keypts_byhand=False, keypts_com=False, analytical=False):
    f = h5py.File(str(file), 'r')
    dset_parallel_grasps = f['grasps']['paralleljaw']['pregrasp_transform']
    dset_parallel_score = f['grasps']['paralleljaw']['quality_score'] if parallel and analytical else []
    dset_parallel_score_simulation = f['grasps']['paralleljaw']['quality_score_simulation'] if parallel and simulation else []
    dset_suctioncup_grasps = f['grasps']['suctioncup']['pregrasp_transform'] if suctioncup and analytical else []
    dset_suctioncup_score = f['grasps']['suctioncup']['quality_score'] if suctioncup and analytical else []
    dset_suctioncup_score_simulation = dset_suctioncup_score if suctioncup and simulation else []
    dset_keypts_byhand = f['keypts']['byhand']if keypts_byhand else []
    dset_keypts_com = f['keypts']['com']if keypts_com else []

    ret = {"paralleljaw_pregrasp_transform": list(dset_parallel_grasps),
           "paralleljaw_pregrasp_score": list(dset_parallel_score),
           "paralleljaw_pregrasp_score_simulation": list(dset_parallel_score_simulation),
           "suctioncup_pregrasp_transform": list(dset_suctioncup_grasps),
           "suctioncup_pregrasp_score": list(dset_suctioncup_score),
           "suctioncup_pregrasp_score_simulation": list(dset_suctioncup_score_simulation),
           "keypts_com": list(dset_keypts_com),
           "keypts_byhand": list(dset_keypts_byhand)}
    return ret


def load_single_mesh(file):
    # Load mesh and rescale with factor 100
    mesh = trimesh.load(str(file))
    mesh = mesh.apply_scale(100) # convert m to cm
    return mesh


def from_contact_to_6D(grasp_config, obj_to_world_transform=None):
    """
    Convert from hdf5 contact representation to 4x4 Matrix in World COS.
    """
    parallel_vec_a = np.array([grasp_config[0],
                               grasp_config[1],
                               grasp_config[2]])

    parallel_vec_b = np.array([grasp_config[3],
                               grasp_config[4],
                               grasp_config[5]])

    parallel_contact_pt1 = np.array([grasp_config[6],
                                     grasp_config[7],
                                     grasp_config[8]])

    parallel_width = grasp_config[9]
    print("parallel width", parallel_width)

    parallel_pregrasp_transform = convert_to_franka_6DOF(
        vec_a=parallel_vec_a,
        vec_b=parallel_vec_b,
        contact_pt=parallel_contact_pt1,
        width=parallel_width)

    if obj_to_world_transform is None:
        obj_to_world_transform = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])

    parallel_pregrasp_transform_world = np.dot(
        obj_to_world_transform,
        parallel_pregrasp_transform)

    return parallel_pregrasp_transform_world


def generate_6DOF_from_SE2(vec_a, contact_pt):
    """
    Generate a rotation matrix from only one approach vector! Just for visualization.
    Keep in mind, that this might have to change later on!
    """
    # translation t
    t_ = contact_pt - (SUCTIONCUP_APPROACH_DISTANCE + SUCTIONCUP_HEIGHT) * vec_a
    # rotation R
    alphax, alphay = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
    b_ = [alphax,
          alphay,
          vec_a[0]/(vec_a[2]+1e-12)*(-alphax) + vec_a[1]/(vec_a[2]+1e-12)*(-alphay)]

    b_ = b_ / np.linalg.norm(b_)
    c_ = np.cross(vec_a, b_)
    R_ = [[b_[0], c_[0], vec_a[0]],
          [b_[1], c_[1], vec_a[1]],
          [b_[2], c_[2], vec_a[2]]]
    pregrasp_transform_ = [[R_[0][0], R_[0][1], R_[0][2], t_[0]],
                           [R_[1][0], R_[1][1], R_[1][2], t_[1]],
                           [R_[2][0], R_[2][1], R_[2][2], t_[2]],
                           [0.0, 0.0, 0.0, 1.0]]
    contact_transform_ = [[R_[0][0], R_[0][1], R_[0][2], contact_pt[0]],
                          [R_[1][0], R_[1][1], R_[1][2], contact_pt[1]],
                          [R_[2][0], R_[2][1], R_[2][2], contact_pt[2]],
                          [0.0, 0.0, 0.0, 1.0]]
    return np.array(pregrasp_transform_), np.array(contact_transform_)


def from_SE2_to_6D(grasp_config, obj_to_world_transform=None):
    suction_vec_a = np.array([grasp_config[0],
                              grasp_config[1],
                              grasp_config[2]])
    suction_contact_pt = np.array([grasp_config[3],
                                   grasp_config[4],
                                   grasp_config[5]])

    suctioncup_pregrasp_transform, suctioncup_contact_transform = generate_6DOF_from_SE2(
        vec_a=suction_vec_a,
        contact_pt=suction_contact_pt)

    if not obj_to_world_transform:
        obj_to_world_transform = np.array([[1., 0., 0., 0.],
                                           [0., 1., 0., 0.],
                                           [0., 0., 1., 0.],
                                           [0., 0., 0., 1.]])

    suctioncup_contact_transform_world = np.dot(
        obj_to_world_transform,
        suctioncup_contact_transform)

    return suctioncup_contact_transform_world


def create_contact_pose(grasp_config):
    parallel_vec_a = np.array([grasp_config[0],
                               grasp_config[1],
                               grasp_config[2]])

    parallel_vec_b = np.array([grasp_config[3],
                               grasp_config[4],
                               grasp_config[5]])

    parallel_contact_pt1 = np.array([grasp_config[6],
                                     grasp_config[7],
                                     grasp_config[8]])

    parallel_width = grasp_config[9]

    # create 4x4 Matrix
    c_ = np.cross(parallel_vec_a, parallel_vec_b)
    R_ = [[parallel_vec_b[0], c_[0], parallel_vec_a[0]],
          [parallel_vec_b[1], c_[1], parallel_vec_a[1]],
          [parallel_vec_b[2], c_[2], parallel_vec_a[2]]]

    t_ = parallel_contact_pt1
    # create 4x4 transform matrix of grasp
    contact_pt_transform_ = [[R_[0][0], R_[0][1], R_[0][2], t_[0]],
                             [R_[1][0], R_[1][1], R_[1][2], t_[1]],
                             [R_[2][0], R_[2][1], R_[2][2], t_[2]],
                             [0.0, 0.0, 0.0, 1.0]]

    cos_system = trimesh.creation.axis(
            origin_size=0.1,
            transform=np.array(contact_pt_transform_))

    return cos_system


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/yitian/Research/simulation_isaaclab_rl/models")
    parser.add_argument("--dataset_name", type=str, default="models_ifl")
    parser.add_argument("--object", type=str, default="009")
    parser.add_argument("--parallel_grasps", default=True)
    parser.add_argument("--keypts_byhand", action="store_true")
    parser.add_argument("--keypts_com", action="store_true")
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--analytical", default=True)
    parser.add_argument("--l2norm", action="store_true")
    parser.add_argument("--suction_grasps", action="store_true")
    parser.add_argument("--score_min", type=float, default=0.0, help="lower bound of interval")
    parser.add_argument("--score_max", type=float, default=1.0, help="upper bound of interval")
    parser.add_argument("--max_grasps", type=int, default=24000, help="max grasps to show.",)
    args = parser.parse_args()

    PATH_TO_DATA = pathlib.Path(os.path.join(args.root, args.dataset_name))
    PATH_TO_OBJ = PATH_TO_DATA / args.object / "textured.obj"
    PATH_TO_HDF5 = PATH_TO_DATA / args.object / "textured.obj.hdf5"

    if args.l2norm:
        args.simulation = True
        args.analytical = True

    ## read in dataset
    grasp_dict = read_in_mesh_config(
        file=PATH_TO_HDF5,
        parallel=args.parallel_grasps,
        suctioncup=args.suction_grasps,
        simulation=args.simulation,
        analytical=args.analytical,
        keypts_byhand=args.keypts_byhand,
        keypts_com=args.keypts_com)
    num_parallel_grasps = len(grasp_dict['paralleljaw_pregrasp_transform'])
    num_suctioncup_grasps = len(grasp_dict['suctioncup_pregrasp_transform'])
    num_keypts_byhand = len(grasp_dict['keypts_byhand'])
    num_keypts_com = len(grasp_dict['keypts_com'])

    ## load object
    obj_mesh = load_single_mesh(PATH_TO_OBJ)

    ## create trimesh scene
    trimesh_scene = trimesh.scene.Scene()
    trimesh_scene.add_geometry(
        geometry = obj_mesh)

    if args.parallel_grasps:
        k = min(args.max_grasps, num_parallel_grasps) if args.max_grasps else num_parallel_grasps
        grasps_ids = random.sample(range(num_parallel_grasps), k)

        score_counter = 0

        ## visualize parallel gripper
        for id in grasps_ids:
            parallel_grasp_config = grasp_dict['paralleljaw_pregrasp_transform'][id]
            if args.simulation:
                score = grasp_dict['paralleljaw_pregrasp_score_simulation'][id]
            elif args.analytical:
                score = grasp_dict['paralleljaw_pregrasp_score'][id]
            elif args.l2norm:
                score_analytical = grasp_dict['paralleljaw_pregrasp_score'][id]
                score_simulation = grasp_dict['paralleljaw_pregrasp_score_simulation'][id]
                score = np.sqrt(score_analytical*score_analytical + score_simulation*score_simulation)/np.sqrt(2)
            else:
                print("WARNING : no score type selected")
                exit()
            # Transform of gripper COS (G) relative to world (W)
            T_W_G = from_contact_to_6D(parallel_grasp_config)
            grasp_width = parallel_grasp_config[9]
            # add to scene
            if score >= args.score_min and score <= args.score_max:
                trimesh_scene.add_geometry(
                    geometry=create_easy_gripper(
                        color=interpolate_between_red_and_green(
                                score, 150),
                        sections=3,
                        show_axis=True,
                        width=grasp_width),
                    transform=T_W_G)
                score_counter += 1
                trimesh_scene.add_geometry(
                    geometry=create_contact_pose(parallel_grasp_config))

    if args.suction_grasps:
        k = min(args.max_grasps, num_suctioncup_grasps) if args.max_grasps else num_suctioncup_grasps
        #print("k", k)
        grasps_ids = random.sample(range(num_suctioncup_grasps), k)
        score_counter = 0
        ## visualize vacuum gripper
        for id in grasps_ids:
            vacuum_grasp_config = grasp_dict['suctioncup_pregrasp_transform'][id]
            if args.simulation:
                score = grasp_dict['suctioncup_pregrasp_score'][id]
            elif args.analytical:
                score = grasp_dict['suctioncup_pregrasp_score'][id]
            elif args.l2norm:
                score_analytical = grasp_dict['suctioncup_pregrasp_score'][id]
                score_simulation = grasp_dict['suctioncup_pregrasp_score_simulation'][id]
                score = np.sqrt(score_analytical*score_analytical + score_simulation*score_simulation)/np.sqrt(2)
            else:
                print("WARNING : no score type selected")
                exit()
            # Transform of gripper COS (G) relative to world (W)
            T_W_CONTACT = from_SE2_to_6D(vacuum_grasp_config)
            # add to scene
            if score >= args.score_min and score <= args.score_max:
                contact_pt = [T_W_CONTACT[0,3],T_W_CONTACT[1,3],T_W_CONTACT[2,3]]
                #print(score)
                vacuum_contact = trimesh.primitives.Sphere(radius=0.7, center=contact_pt)
                vacuum_contact.visual.vertex_colors = interpolate_between_red_and_green(score)
                trimesh_scene.add_geometry(
                    geometry=vacuum_contact)
                score_counter += 1
        ## show
        print(f"{score_counter} # grasps in [{args.score_min},{args.score_max}]")

    if args.keypts_byhand:
        score_counter = 0
        for id in range(num_keypts_byhand):
            keypt = grasp_dict['keypts_byhand'][id]
            pt_ = [keypt[1],keypt[2],keypt[3]]
            id_ = keypt[0]
            keypt_vis = trimesh.primitives.Sphere(radius=0.3, center=pt_)
            keypt_vis.visual.vertex_colors = random_rgb_color(id_)
            trimesh_scene.add_geometry(
                geometry=keypt_vis)
            score_counter += 1
        ## show
        print(f"{score_counter} # keypts by hand.")

    if args.keypts_com:
        score_counter = 0
        for id in range(num_keypts_com):
            keypt = grasp_dict['keypts_com'][id]
            pt_ = [keypt[1],keypt[2],keypt[3]]
            score_ = keypt[0]
            keypt_vis = trimesh.primitives.Sphere(radius=0.15, center=pt_)
            keypt_vis.visual.vertex_colors = interpolate_between_red_and_green(score_)
            trimesh_scene.add_geometry(
                geometry=keypt_vis)
            score_counter += 1
        ## show
        print(f"{score_counter} # keypts by score.")

    trimesh_scene.show()
