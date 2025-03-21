#!/usr/bin/env python
# Author : Maximilian Gilles, IFL, KIT

import numpy as np
import trimesh
import open3d
from trimesh.viewer import windowed
import pyglet
import argparse
import pathlib
import random
import h5py
import glob

def interpolate_between_red_and_green(score):
    """
    Returns RGBA color between RED and GREEN.
    """
    delta = int(score*255)
    COLOR = [255 - delta, 0 + delta, 0, 255] # RGBA
    return COLOR

def load_hdf5(path):
    f = h5py.File(str(path), 'r+')
    # create subgroup
    grp = f.require_group("keypts")
    return f

def get_idx_color(seed):
    """
    Returns color based on seed.
    """
    random.seed(seed)
    r = random.randrange(0,255)
    g = random.randrange(0,255)
    b = random.randrange(0,255)
    return [r,g,b,255]


class KeypointLabels:
    def __init__(self, path, obj_prefix="textured.obj", hdf5_prefix="textured.obj.hdf5"):
        
        # Variables defined by user
        self.obj_path = path / obj_prefix
        self.hdf5_path = path / hdf5_prefix

        # Variables defined during processing
        self.contact_point = ()
        self.faces_idx = ()
        self.reference_point = ()
        self.distance_vec = ()
        self.min_index = ()
        self.scene = ()
        self.permissible = True
        
        # labels
        self.keypts = []
        self.keypt_index = 0
        # Variables for visualization
        self.spheres = []

    # Calculates the contact point by right-clicking on the object mesh
    def calculate_contact_point(self,x,y):

        # calculate contact point with camera rays
        origins, vectors, pixels = self.scene.camera_rays()
        res = np.where(np.all(pixels == np.array([x, y]), axis=1))      
        self.faces_idx, ray_idx, self.contact_point = trimesh.ray.ray_pyembree.RayMeshIntersector(self.obj_mesh).intersects_id(
            ray_origins=origins[res],
            ray_directions=vectors[res],
            multiple_hits=False,
            return_locations=True
        )
        
        # check whether contact point lies on object
        if len(self.contact_point) < 1:
            self.permissible = False
            return False
            
        # Change variable form for postprocessing
        self.faces_idx = self.faces_idx[0]
        self.contact_point = self.contact_point[0]

        return True
    
    # Load the object mesh
    def load_obj_mesh(self):
        self.obj_mesh = trimesh.load(str(self.obj_path))
        self.obj_mesh.apply_scale(100) # convert m to cm
        self.scene = trimesh.scene.Scene(self.obj_mesh)

    # Callback function for the window: deletes the displayed geometries and reload them
    def callback(self, scene):
        for i in range(100):
            self.scene.delete_geometry(str(i))
        for id, keypt in enumerate(self.keypts):
            assert id == keypt[0]
            self.scene.add_geometry(self.spheres[id], geom_name=str(id))

    def save_to_file(self):
        keypt_subgrp = self.f['keypts']
        try:
            del keypt_subgrp['byhand']
        except:
            pass
        keypt_dset = keypt_subgrp.create_dataset(
            name="byhand",
            shape=(len(self.keypts), 4),
            dtype='f')
        for id, keypt in enumerate(self.keypts):
            keypt_entry = np.array([keypt[0], keypt[1][0], keypt[1][1], keypt[1][2]]) # id,x,y,z
            keypt_dset[id] = keypt_entry
        self.f.close()
        return True

    def get_keypt_id(self):
        """
        If [ENTER] increment, else use provided number.
        """
        input("Enter keypt id or skip with [ENTER].")

    # creates the window with the object mesh
    def create_window(self):
        self.load_obj_mesh()
        self.f = load_hdf5(self.hdf5_path)
        window = windowed.SceneViewer(self.scene, start_loop=False, visible=True, callback = self.callback)
            
        @window.event
        def on_mouse_press(x, y, buttons, modifiers):
            if buttons == 4:
                # add keypt
                self.permissible = True 
                # calculates the contact point                 
                if self.calculate_contact_point(x,y):
                    # add point to scene
                    sphere = trimesh.primitives.Sphere(radius=0.1, center=self.contact_point)
                    sphere.visual.vertex_colors = get_idx_color(seed=self.keypt_index)
                    self.spheres.append(sphere)
                    self.keypts.append([self.keypt_index, self.contact_point])
                    print("-> keypt {self.keypt_index} with {self.contact_point}")
                    self.keypt_index += 1

            if buttons == 2:
                # delete the last keypoint
                self.permissible = True
                # calculates the contact point                 
                if self.calculate_contact_point(x,y) and len(self.keypts) > 0:
                        keypt = self.keypts[-1]
                        v_ = keypt[1] - self.contact_point
                        if np.linalg.norm(v_) < 1:
                            print(f"remove the last {keypt}")
                            self.keypts.pop(-1)
                            self.keypt_index -= 1
                            self.spheres.pop(-1)

        @window.event
        def on_key_press(buttons, modifiers):
            if buttons == 115:
                self.save_to_file()
                print("-> saved to file.")
            else:
                pass


class CenterOfMassLabels:
    def __init__(self, path, obj_prefix="textured.obj", hdf5_prefix="textured.obj.hdf5"):
        # variables defined by user
        self.obj_path = path / obj_prefix
        self.hdf5_path = path / hdf5_prefix

        # keypts
        self.pts = []

    def compute_com_keypts(self, surface_pts=1000):
        # compute surface pts
        self.com_surface_points, index_tri, distance_to_com  = self.sample_surface_com_points(
            samples=surface_pts)

        # score distance relative
        min_dist = np.amin(distance_to_com, axis=0)
        max_dist = np.amax(distance_to_com, axis=0)
        self.com_dist_score = [(max_dist - d) / (max_dist- min_dist) for d in distance_to_com]

        # save in hdf5 file
        self.f = load_hdf5(self.hdf5_path)
        self.save_to_file()

        # visualize
        self.visualize()

        return True

    def visualize(self):
        pcl_com = trimesh.points.PointCloud(
            vertices=self.com_surface_points,
            colors=[interpolate_between_red_and_green(score) for score in self.com_dist_score]
            )
        scene = trimesh.scene.Scene([self.obj_mesh, pcl_com])
        scene.show()

    def save_to_file(self):
        keypt_subgrp = self.f['keypts']
        try:
            del keypt_subgrp['com']
        except:
            pass
        keypt_dset = keypt_subgrp.create_dataset(
            name="com",
            shape=(len(self.com_surface_points), 4),
            dtype='f')
        for id, keypt in enumerate(zip(self.com_dist_score, self.com_surface_points)):
            keypt_entry = np.array([keypt[0], keypt[1][0], keypt[1][1], keypt[1][2]]) # score,x,y,z
            keypt_dset[id] = keypt_entry
        self.f.close()
        return True

    def load_obj_mesh(self):
        # Load the object mesh
        self.obj_mesh = trimesh.load(str(self.obj_path))
        self.obj_mesh.apply_scale(100) # convert m to cm

    def sample_surface_com_points(self, samples=1000, random=True):
        # load mesh
        self.load_obj_mesh()

        # sample rays from COM
        mesh_com = self.obj_mesh.center_mass
        if not random:
            sphere = trimesh.creation.icosphere(subdivisions=3)
            ray_origins = []
            ray_directions = []
            for idx in range(len(sphere.vertices)):
                # sample in SE(3) for x,y,z
                ray_origins.append(mesh_com)
                ray_directions.append(sphere.vertices[idx])
            ## let them intersect with mesh
            contact_points, index_ray, index_tri = self.obj_mesh.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                multiple_hits=False)
        else:
            contact_points, index_tri = trimesh.sample.sample_surface_even(
                mesh=self.obj_mesh,
                count=samples,
                radius=None  # removes points below this radius
            )
        distances = np.array([np.linalg.norm(pt - mesh_com) for pt in contact_points])
        return contact_points, index_tri, distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./models")
    parser.add_argument("--dataset_name", default="models_ifl")
    parser.add_argument("--object_idx", default="008")
    parser.add_argument("--compute_com", action="store_true")
    args = parser.parse_args()

    print("save with (s)")

    PATH_TO_DATASET = pathlib.Path(args.data_root, args.dataset_name)
    PATH_TO_OBJ = PATH_TO_DATASET / args.object_idx

    if args.compute_com:
        # compute com
        com_viewer = CenterOfMassLabels(path=PATH_TO_OBJ)
        com_viewer.compute_com_keypts()

    obj_viewer = KeypointLabels(path=PATH_TO_OBJ)

    # Create window and stay in window
    window = obj_viewer.create_window()
    pyglet.app.run()
