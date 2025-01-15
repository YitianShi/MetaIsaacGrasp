import trimesh as tm
import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch
from pathlib import Path

HOME_PATH_LOCAL = Path(os.getcwd())

def transformation_from_parameters(params):
    # unpack the parameters
    translation = params[4:]
    quaternion = params[:4]
    # convert quaternion to rotation matrix
    rotation = R.from_quat(quaternion).inv()
    translation = -rotation.apply(translation)
    # rotate the orientation by 90 degrees around the z-axis
    theta = - np.pi / 2  # 90 degrees
    rot_z= R.from_euler('z', theta, degrees=False)
    rotation = rot_z * rotation
    # xyzw to wxyz
    rotation = rotation.as_quat()
    rotation = (rotation[-1], *rotation[:-1])
    return np.hstack([translation, rotation])

def read_targo(shift):
    
    TARGO_PATH = os.path.join(HOME_PATH_LOCAL, "..", "dataset", "syn_train", "combined") # path to the target object
    TARGO_PATHS_POSE = glob.glob(f"{TARGO_PATH}/mesh_pose_dict/*_c*.npz") # path to the target object pose
    # Randomly choose one pose
    random.shuffle(TARGO_PATHS_POSE)
    occ_targ_max = 0.
    i = 0
    while occ_targ_max < 0.8:
        file = TARGO_PATHS_POSE[i]        
        # Load the scenes
        scenes_all = glob.glob(f"{TARGO_PATH}/scenes/{file.split('/')[-1].split('.')[0]}*.npz") # path to the target object scene
        target_obj_id = []
        for scene in scenes_all:
            obj = scene.split('/')[-1].split('.')[0].split('_')[-1]
            target_obj_id.append(obj)
            scene_dict = np.load(scene, allow_pickle=True)
            if scene_dict["occ_targ"] > occ_targ_max:
                occ_targ_max = scene_dict["occ_targ"]
                obj_chosen = obj
                pose = transformation_from_parameters(scene_dict["extrinsics"][0])
                extrinsic = pose
                seg_map = scene_dict["segmentation_map"][0]
                plt.imsave(f"seg_map.png", seg_map)
        i += 1
    print(f"Chosen object: {obj_chosen}, maximum occlusion: {occ_targ_max}")

    theta = - np.pi / 2  # 90 degrees
    rot_z= R.from_euler('z', theta, degrees=False)

    # Load the target object
    mesh_pose_dict = np.load(file, allow_pickle=True)["pc"]
    TARGO_OBJ_PATHS, obj_scales, obj_positions, obj_rotations = [], [], [], []
    for k in mesh_pose_dict.item():
        path, scale, pose = mesh_pose_dict.item()[k]
        if "plane" in path:
            plane_loc = pose[:3, -1]
            continue
        path = os.path.join(TARGO_PATH, path)
        path_urdf = path.replace("_visual.obj", ".urdf")
        if not os.path.exists(path_urdf):
            print(f"Path {path_urdf} does not exist")  
        TARGO_OBJ_PATHS.append(path_urdf)
        
        # Transform the pose
        pos = pose[:3, -1] - extrinsic[:3]
        rot = R.from_matrix(pose[:3, :3])

        # Rotate the object by 90 degrees around the z-axis
        pos = rot_z.apply(pos)
        rot = rot_z * rot

        # Convert to quaternion with wxyz format
        rot = rot.as_quat()
        rot = (rot[-1], *rot[:-1])
        obj_scales.append(scale)
        obj_positions.append(pos.squeeze())
        obj_rotations.append(rot)

        # object_mesh = tm.load(path)
        # object_mesh.apply_scale(scale)
        # object_mesh.apply_transform(pose)
        # tm_scene.add_geometry(object_mesh)
    # tm_scene.show()
    
    # Convert to numpy array
    obj_scales = np.array(obj_scales)
    obj_positions = np.array(obj_positions)
    obj_rotations = np.array(obj_rotations)

    obj_positions += shift
    extrinsic[:2] = shift[:2]

    obj_positions[:, 2] = obj_positions[:, 2] + extrinsic[2] - plane_loc[2] - 0.025
    extrinsic[2] = extrinsic[2] + shift[2] - plane_loc[2] - 0.025


    return TARGO_OBJ_PATHS, obj_scales, obj_positions, obj_rotations, occ_targ_max, extrinsic, TARGO_OBJ_PATHS[int(obj_chosen)-1]
