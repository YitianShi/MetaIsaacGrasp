#!/usr/bin/env python3
# Author : Maximilian Gilles, IFL, KIT
from isaacgym import gymapi
import random
import math
import argparse
import numpy as np
import h5py
import scipy.spatial.transform
import glob
import os
import pathlib


class State:
    Initialized = -1
    Pending = 0
    Object_loaded = 1
    Grasped = 2
    Shaked_trans = 3
    Shaked_rot = 4
    Grasp_successfull = 5
    Grasp_failed = 6
    Reset = 7
    Finished = 8
    Wait_for_finger_closing = 9
    Wait_for_gripper_shaked_trans = 10
    Wait_for_gripper_shaked_rot = 11

    state_dict = {
        "Initialized": -1,
        "Pending": 0,
        "Object_loaded": 1,
        "Grasped": 2,
        "Shaked_trans": 3,
        "Shaked_rot": 4,
        "Grasp_successfull": 5,
        "Grasp_failed": 6,
        "Reset": 7,
        "Finished": 8,
        "Wait_for_finger_closing": 9,
        "Wait_for_gripper_shaked_trans": 10,
        "Wait_for_gripper_shaked_rot": 11,

    }
    to_sem = {v: k for k, v in state_dict.items()}


GRASP_WIDTH = 0.04  # m
GRIPPER_HEIGHT = 0.1121  # m
AREA_CENTER = 0.008  # m


class GraspsDatabase():
    def __init__(self, mesh_root, category, step_size=1, min_friction=0.5,
                 max_friction=2.0, min_density=0.25, max_density=1.0):
        # grasp config index
        self.grasp_idx = 0
        self.idx = 0
        # store values
        self.all_configs = []
        self.all_success = {}
        # paths
        self.mesh_root = mesh_root
        self.category = category
        # simulation parameters
        self.friction_intervall = np.linspace(min_friction, max_friction, num=step_size)
        self.rolling_friction_intervall = np.linspace(min_friction, max_friction, num=step_size)
        self.torsion_friction_intervall = np.linspace(min_friction, max_friction, num=step_size)
        self.density_intervall = np.linspace(min_density, max_density, num=step_size)

        print("INFO : friction intervall set to ", self.friction_intervall)
        print("INFO : density intervall set to ", self.density_intervall)
        # default values
        self.successful_grasps = 0
        self.failed_grasps = 0
        self.total_grasps = 0

        # init
        self.load_hdf5()
        self.generate_list()

    def load_hdf5(self):
        path_hdf5 = glob.glob(os.path.join(self.mesh_root, self.category, "*.hdf5"), recursive=True)
        print(f"open {path_hdf5[0]}.")
        assert len(path_hdf5) == 1
        self.f = h5py.File(str(path_hdf5[0]), 'r+')

        # access grasp datasets
        self.pregrasp_dataset = self.f['grasps']['paralleljaw']['pregrasp_transform']
        # create new subgroup
        grp = self.f['/grasps']['paralleljaw']

        self.max_grasps = len(list(self.pregrasp_dataset))

        self.dset_phsics_score = grp.require_dataset(
            name="quality_score_simulation",
            shape=(self.max_grasps,),
            dtype='f')
        return True

    def generate_list(self):
        for grasp_id, parallel_pregrasp_config in enumerate(self.pregrasp_dataset):
            # convert to 6DOF pregrasp transform
            parallel_vec_a = np.array([parallel_pregrasp_config[0], parallel_pregrasp_config[1], parallel_pregrasp_config[2]])
            parallel_vec_b = np.array([parallel_pregrasp_config[3], parallel_pregrasp_config[4], parallel_pregrasp_config[5]])
            parallel_contact_pt1 = np.array([parallel_pregrasp_config[6]/100.0, parallel_pregrasp_config[7]/100.0, parallel_pregrasp_config[8]/100.0]) # in cm -> m !
            parallel_width = parallel_pregrasp_config[9]/100.0
            parallel_pregrasp_transform = self._convert_to_6DOF(
                vec_a=parallel_vec_a,
                vec_b=parallel_vec_b,
                contact_pt=parallel_contact_pt1,
                width=parallel_width)
            for friction in self.friction_intervall:
                for density in self.density_intervall:
                    ret = {"grasp_id": grasp_id,
                           "pregrasp": parallel_pregrasp_transform,
                           "friction": friction,
                           "rolling_friction": friction,
                           "torsional_friction": friction,
                           "density_factor": density}
                    self.all_configs.append(ret)
        print(f"finished. {len(self.pregrasp_dataset)} grasps and in total {len(self.all_configs)} grasp configs.")
        return True

    def _convert_to_6DOF(self, vec_a, vec_b, contact_pt, width):
        """Convert Contact-Point Pregrasp representation to 6DOF Gripper Pose (4x4)."""
        # get 3rd unit vector
        c_ = np.cross(vec_a, vec_b)
        # rotation matrix
        R_ = [[vec_b[0], c_[0], vec_a[0]],
              [vec_b[1], c_[1], vec_a[1]],
              [vec_b[2], c_[2], vec_a[2]]]
        q_ = scipy.spatial.transform.Rotation.from_matrix(R_).as_quat()
        # translation t
        t_ = contact_pt + width/2 * vec_b + (GRIPPER_HEIGHT-AREA_CENTER) * vec_a * (-1)
        trans_ = gymapi.Vec3(t_[0], t_[1], t_[2])
        quat_ = gymapi.Quat(q_[0], q_[1], q_[2], q_[3])
        ret = {'trans': trans_, 'quat': quat_}
        return ret

    def get_new_grasp_config(self):
        """returns a different grasp config everytime it is called."""
        if self.idx < len(self.all_configs):
            grasp_config = self.all_configs[self.idx]
            self.idx += 1
            return grasp_config
        else:
            print("reached end.")
            return False

    def save_grasp_success_in_database(self, grasp_config, result):
        """Client calls this method to store grasp success in database."""
        # check for value of result
        assert (result == 1) or (result == 0) 

        # save in dict
        grasp_id = str(grasp_config["grasp_id"])
        if grasp_id not in self.all_success:
            # compute score for previous grasp_id
            prev_grasp_id = str(int(grasp_id) - 1)
            if prev_grasp_id in self.all_success:
                # score is mean value over all different physcis grasps for this grasp_config
                score = np.mean(np.array(self.all_success[prev_grasp_id]))
                # save in hdf5 file
                self.dset_phsics_score[int(prev_grasp_id)] = score
            # generate new entry for given grasp_id
            self.all_success[grasp_id] = [result]

        else:
            # add result to existing entry in dict
            self.all_success[grasp_id].append(result)
        return True

    def close_hdf5_file(self):
        self.f.close()


class StateObserver():
    def __init__(self, sim, env, actor_gripper_handle, 
                 actor_object_handle, grasp_database, debug=False,
                 dumb_everything=False):
        self.state = State.Initialized
        self.grasp_width = GRASP_WIDTH
        self.env = env
        self.sim = sim
        self.gripper_handle = actor_gripper_handle
        self.object_handle = actor_object_handle
        self.grasp_database = grasp_database
        self.debug = debug
        self.dumb_everything = dumb_everything

        self.default_mass = None
        self.curr_grasp_id = None
        self.curr_mass = None
        self.curr_friction = None

        self.default_rigid_body_state_gripper = np.copy(
            gym.get_actor_rigid_body_states(self.env, self.gripper_handle, gymapi.STATE_ALL))
        self.default_dof_state_gripper = np.copy(
            gym.get_actor_dof_states(self.env, self.gripper_handle, gymapi.STATE_ALL))

        self.shaking_linvel = None
        self.counter_wait_for_finger_closing = None
        self.counter_wait_for_shaking = None

    def updateSM(self):
        """Update State in every simulation step. """

        # ~~ debug ~~ #
        if self.debug:
            ## filter out Waiting_* states
            if not self.state == State.Wait_for_finger_closing \
                and not self.state == State.Wait_for_gripper_shaked_trans \
                and not self.state == State.Wait_for_gripper_shaked_rot:
                print(f"current state : {self.state} / {State.to_sem[self.state]}")
                if self.dumb_everything:
                    self._get_current_rigid_body_state(self.gripper_handle)
                    self._get_current_dof_state(self.gripper_handle)
                    self._get_current_dof_target_position(self.gripper_handle, "hand_joint")
                    self._get_current_dof_target_position(self.gripper_handle, "hand_rotating")
                    self._get_current_dof_target_position(self.gripper_handle, "finger_joint_left")
                    self._get_current_dof_target_position(self.gripper_handle, "finger_joint_right")
        # ~~~~~~~~~~~ #

        if self.state == State.Initialized:
            self.state = State.Pending
            return

        if self.state == State.Pending:
            # waiting
            gym.set_rigid_body_color(self.env, self.object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.2, 0.8))
            # if new grasp config available -> load it
            valid = self.load_grasp_from_database()
            if valid:
                self.state = State.Object_loaded
            else:
                self.state = State.Finished
                if self.debug:
                    print("-> reached end of grasp database.")
            return

        if self.state == State.Object_loaded:
            # object is loaded with different config -> start grasping
            self.grasp()
            self.state = State.Wait_for_finger_closing
            self.counter_wait_for_finger_closing = 400
            return

        if self.state == State.Wait_for_finger_closing:
            # wait for defined wait of time till fingers are closed
            self.counter_wait_for_finger_closing -= 1
            if self.counter_wait_for_finger_closing <= 0:
                self.state = State.Grasped
                self.counter_wait_for_finger_closing = None
            return

        if self.state == State.Grasped:
            # check if object is still in contact with gripper
            if self.check_for_contact():
                # start shaking
                self.shaking_target = 1.0  # m
                self.counter_wait_for_shaking = 400
                self.shake_trans()
                self.state = State.Wait_for_gripper_shaked_trans
            else:
                # grasp failed
                self.state = State.Grasp_failed
            return

        if self.state == State.Wait_for_gripper_shaked_trans:
            # wait till shake is completed
            self.shake_trans()
            self.counter_wait_for_shaking -= 1
            if self.counter_wait_for_shaking <= 0:
                self.state = State.Shaked_trans
                self.counter_wait_for_shaking = None
                self.shaking_target = None
                self.remove_joint_target(self.gripper_handle, 'hand_joint')
            return

        if self.state == State.Shaked_trans:
            # check if object is still in contact with gripper
            succ = self.check_for_contact()
            if succ:
                self.shaking_target = 0.3  # rad
                self.counter_wait_for_shaking = 300
                self.shake_rot()
                self.state = State.Wait_for_gripper_shaked_rot
            else:
                self.state = State.Grasp_failed
            return

        if self.state == State.Wait_for_gripper_shaked_rot:
            # wait till shake is completed
            self.shake_rot()
            self.counter_wait_for_shaking -= 1
            if self.counter_wait_for_shaking <= 0:
                self.state = State.Shaked_rot
                self.counter_wait_for_shaking = None
                self.shaking_target = None
                self.remove_joint_target(self.gripper_handle, 'hand_rotating')
            return

        if self.state == State.Shaked_rot:
            # check if object still in contact after shaking
            succ = self.check_for_contact()
            if succ:
                self.state = State.Grasp_successfull
            else:
                self.state = State.Grasp_failed
            return

        if self.state == State.Grasp_successfull:
            # set color to GREEN and save it in database
            gym.set_rigid_body_color(self.env, self.object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0, 0.8, 0.2))
            self.save_result_in_database(result=True)
            print(f"Grasp {self.curr_grasp_id} SUCCEEDED : {self.curr_mass:.3f} [kg] {self.curr_friction} friction.")
            # reset gripper
            self.reset_gripper()
            self.state = State.Reset
            return

        if self.state == State.Grasp_failed:
            # set color to RED and save it in database
            gym.set_rigid_body_color(self.env, self.object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.2, 0.0))
            self.save_result_in_database(result=False)
            print(f"Grasp {self.curr_grasp_id} FAILED : {self.curr_mass:.3f} [kg] {self.curr_friction} [friction].")
            # reset gripper
            self.reset_gripper()
            self.state = State.Reset
            return

        if self.state == State.Reset:
            # do nothing in here
            self.state = State.Pending
            return

        if self.state == State.Finished:
            # finished, do nothing
            return

    def grasp(self):
        # controlled by position (max effort and velocity specified in gripper urdf)
        _lf_dof_handle = gym.find_actor_dof_handle(self.env, self.gripper_handle, 'finger_joint_left')
        _rf_dof_handle = gym.find_actor_dof_handle(self.env, self.gripper_handle, 'finger_joint_right')
        gym.set_dof_target_position(self.env, _lf_dof_handle, 0.04)
        gym.set_dof_target_position(self.env, _rf_dof_handle, -0.04)
        return

    def shake_trans(self):
        # translation
        _trans_dof_handle = gym.find_actor_dof_handle(self.env, self.gripper_handle, 'hand_joint')
        if self.counter_wait_for_shaking % 100 == 0:
            # controlled by POS
            self.shaking_target = self.shaking_target * (-1)
            gym.set_dof_target_position(self.env, _trans_dof_handle, self.shaking_target)
        return

    def shake_rot(self):
        # rotation
        _rot_dof_handle = gym.find_actor_dof_handle(self.env, self.gripper_handle, 'hand_rotating')
        if self.counter_wait_for_shaking % 100 == 0:
            # controlled by POS
            self.shaking_target = self.shaking_target * (-1)
            gym.set_dof_target_position(self.env, _rot_dof_handle, self.shaking_target)
        return

    def load_grasp_from_database(self):
        """
            Connect to database. Get object, gripper_pose, densitiy,
            friction, rolling_friction, torsion_friction
        """
        # get config from database
        succ = True
        self.grasp_config = self.grasp_database.get_new_grasp_config()
        if self.grasp_config is False:
            succ = False
            return succ

        # transform grasp config so that gripper stays and object is moved and oriented
        # get current gripper transform T_W_G relative to World (W)
        T_W_G = self._get_current_gripper_transform()
        # get pregrasp config T_O_G relative to Object (O)
        T_O_G = gymapi.Transform(self.grasp_config["pregrasp"]["trans"], self.grasp_config["pregrasp"]["quat"])
        # get new object transform T_W_O according to pregrasp config relative to World (W)
        self.T_W_O = T_W_G * T_O_G.inverse()

        # define rigid body states
        _body_states = gym.get_actor_rigid_body_states(self.env, self.object_handle, gymapi.STATE_ALL)            
        _body_states["pose"]["p"].fill((self.T_W_O.p.x, self.T_W_O.p.y, self.T_W_O.p.z))
        _body_states["pose"]["r"].fill((self.T_W_O.r.x, self.T_W_O.r.y, self.T_W_O.r.z, self.T_W_O.r.w))
        _body_states["vel"]["linear"].fill((0.0, 0.0, 0.0))  # all linear velocities (Vec3: x, y, z)
        _body_states["vel"]["angular"].fill((0.0, 0.0, 0.0))  # all angular velocities (Vec3: x, y, z)

        gym.set_actor_rigid_body_states(self.env, self.object_handle, _body_states, gymapi.STATE_ALL)

        # define rigid body properties
        _rigid_props = gym.get_actor_rigid_body_properties(self.env, self.object_handle)
        if self.default_mass is None:
            # small hack to change density at sim time -> get default mass and alternate it
            self.default_mass = _rigid_props[0].mass  # kg
            # print(f"-> set default object mass to {self.default_mass}")
        _rigid_props[0].mass = self.default_mass * self.grasp_config['density_factor'] 
        gym.set_actor_rigid_body_properties(self.env, self.object_handle, _rigid_props)

        # define rigid body shape properties for object
        _shape_props = gym.get_actor_rigid_shape_properties(self.env, self.object_handle)

        # Properties include friction, rolling_friction, torsion_friction, restitution etc.
        for _props in _shape_props:
            _props.friction = self.grasp_config['friction']
            _props.torsion_friction = self.grasp_config['torsional_friction']
            _props.rolling_friction = self.grasp_config['rolling_friction']       

        gym.set_actor_rigid_shape_properties(self.env, self.object_handle, _shape_props)

        # define rigid body shape properties for gripper
        _shape_props = gym.get_actor_rigid_shape_properties(self.env, self.gripper_handle)

        # Properties include friction, rolling_friction, torsion_friction, restitution etc.
        for _props in _shape_props:
            _props.friction = self.grasp_config['friction']
            _props.torsion_friction = self.grasp_config['torsional_friction']
            _props.rolling_friction = self.grasp_config['rolling_friction']     

        gym.set_actor_rigid_shape_properties(self.env, self.gripper_handle, _shape_props)

        # set object color to neutral
        gym.set_rigid_body_color(self.env, self.object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.8, 0.8))

        # display grasp config
        num_bodies = gym.get_actor_rigid_body_count(self.env, self.object_handle)
        num_joints = gym.get_actor_joint_count(self.env, self.object_handle)
        num_dofs = gym.get_actor_dof_count(self.env, self.object_handle)

        object_base_idx = gym.find_actor_rigid_body_index(self.env, self.object_handle, "base", gymapi.DOMAIN_ACTOR)

        for i in range(num_bodies):
            self.curr_grasp_id = self.grasp_config['grasp_id']
            self.curr_friction = gym.get_actor_rigid_shape_properties(self.env, self.object_handle)[i].friction
            self.curr_mass = gym.get_actor_rigid_body_properties(self.env, self.object_handle)[i].mass
        return succ

    def save_result_in_database(self, result):
        self.grasp_database.save_grasp_success_in_database(self.grasp_config, int(result))
        return True

    def remove_joint_target(self, handle, joint):
        _dof_handle = gym.find_actor_dof_handle(self.env, handle, joint)
        gym.set_dof_target_position(self.env, _dof_handle, 0.0)
        return True

    def reset_gripper(self):
        # remove target positions for dof
        _lf_dof_handle = gym.find_actor_dof_handle(self.env, self.gripper_handle, 'finger_joint_left')
        _rf_dof_handle = gym.find_actor_dof_handle(self.env, self.gripper_handle, 'finger_joint_right')
        _trans_dof_handle = gym.find_actor_dof_handle(self.env, self.gripper_handle, 'hand_joint')
        _rot_dof_handle = gym.find_actor_dof_handle(self.env, self.gripper_handle, 'hand_rotating')
        gym.set_dof_target_position(self.env, _lf_dof_handle, 0.0)
        gym.set_dof_target_position(self.env, _rf_dof_handle, 0.0)
        gym.set_dof_target_position(self.env, _trans_dof_handle, 0.0)
        gym.set_dof_target_position(self.env, _rot_dof_handle, 0.0)

        # reset gripper state
        gym.set_actor_rigid_body_states(
            self.env,
            self.gripper_handle,
            self.default_rigid_body_state_gripper,
            gymapi.STATE_POS)

        # reset gripper fingers 
        gym.set_actor_dof_states(
            self.env,
            self.gripper_handle,
            self.default_dof_state_gripper,
            gymapi.STATE_POS)

        return True

    def check_for_contact(self):
        """ check for contact between gripper finger and object. """
        lf_contact = False
        rf_contact = False
        # get contact between gripper and object
        contacts = gym.get_env_rigid_contacts(self.env)
        for i in range(len(contacts)):
            rigid_contact = contacts[i]
            if rigid_contact['body0'] == 5 and rigid_contact['body1'] == 3:
                lf_contact = True
            elif rigid_contact['body0'] == 5 and rigid_contact['body1'] == 4:
                rf_contact = True
            else:
                pass
        if lf_contact and rf_contact:
            return True
        else:
            return False

    def _get_current_rigid_body_state(self, handle):
        print("current rigid body state")
        print("pose")
        print(gym.get_actor_rigid_body_states(self.env, handle, gymapi.STATE_ALL)['pose'])
        print("vel")
        print(gym.get_actor_rigid_body_states(self.env, handle, gymapi.STATE_ALL)['vel'])

    def _get_current_dof_state(self, handle):
        print("current dof state")
        print("pos")
        print(gym.get_actor_dof_states(self.env, handle, gymapi.STATE_ALL)['pos'])
        print("vel")
        print(gym.get_actor_dof_states(self.env, handle, gymapi.STATE_ALL)['vel'])

    def _get_current_dof_target_position(self, handle, joint):
        _trans_dof_handle = gym.find_actor_dof_handle(self.env, handle, joint)
        print(f"target position {joint} of {handle}")
        print(gym.get_dof_target_position(self.env, _trans_dof_handle))
        print(f"target velocity {joint} of {handle}")
        print(gym.get_dof_target_velocity(self.env, _trans_dof_handle))

    def _get_current_gripper_transform(self):
        # get transform of gripper relative to world
        gripper_state = gym.get_actor_rigid_body_states(self.env, self.gripper_handle, gymapi.STATE_POS)
        hand_idx = gym.find_actor_rigid_body_index(self.env, self.gripper_handle, "hand", gymapi.DOMAIN_ACTOR)
        _t_g = gripper_state["pose"]["p"][hand_idx]
        _q_g = gripper_state["pose"]["r"][hand_idx]
        T_W_G = gymapi.Transform(_t_g, _q_g)
        return T_W_G

    def check_for_collision(self):
        # check for collision object with hand
        # in case of collision mark grasp as failed!
        if not self.state == State.PENDING:
            contacts = gym.get_env_rigid_contacts(self.env)
            for i in range(len(contacts)):
                rigid_contact = contacts[i]
                if rigid_contact['body0'] == 4 and rigid_contact['body1'] == 1:
                    print("-> collision")
                    self.state = State.FAILED


def get_sim_params():
    # get default set of parameters
    sim_params = gymapi.SimParams()

    # set common parameters
    sim_params.dt = 1 / 200
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    # set PhysX-specific parameters
    sim_params.physx.solver_type = 1  # temporal gauss seidel solver (default)
    sim_params.physx.num_position_iterations = 200  # increasing this number helps converge the system
    sim_params.physx.num_velocity_iterations = 0  # default value for TGS solver
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.always_use_articulations = True  # avaibles force sensor for single-object actors
    sim_params.physx.contact_offset = 0.0001
    sim_params.physx.use_gpu = True
    return sim_params


def get_ground_plane_params():
    # configure the ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 100
    plane_params.dynamic_friction = 100
    plane_params.restitution = 100
    return plane_params  


gym = gymapi.acquire_gym()


def main(
        visualize, num_envs, root, category, num_steps, debug, dumb_everything,
        device_number=0):
    # create simulation
    DEVICE = device_number
    sim_params = get_sim_params()
    sim = gym.create_sim(DEVICE, DEVICE, gymapi.SIM_PHYSX, sim_params)

    # load assets
    asset_root = "./physics_simulation/assets"
    asset_file = "urdf/" + "franka_description/robots/franka_hand_acronym_improved_collision_rotating.urdf"
    # load gripper
    asset_options_gripper = gymapi.AssetOptions()
    asset_options_gripper.fix_base_link = True  # !important!
    asset_options_gripper.armature = 0.01
    asset_options_gripper.disable_gravity = True
    asset_options_gripper.linear_damping = 1000

    asset_options_gripper.override_com = True
    asset_options_gripper.override_inertia = True
    asset_options_gripper.vhacd_enabled = True
    asset_options_gripper.vhacd_params = gymapi.VhacdParams()
    asset_options_gripper.vhacd_params.resolution = 500000
    asset_gripper = gym.load_asset(sim, asset_root, asset_file, asset_options_gripper)

    # generate asset
    asset_options_obj = gymapi.AssetOptions()
    asset_options_obj.fix_base_link = False  # !important!
    asset_options_obj.armature = 0.01
    asset_options_obj.disable_gravity = True
    asset_options_obj.use_mesh_materials = True
    asset_options_obj.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options_obj.override_com = True
    asset_options_obj.override_inertia = True
    asset_options_obj.vhacd_enabled = True
    asset_options_obj.vhacd_params = gymapi.VhacdParams()
    asset_options_obj.vhacd_params.resolution = 500000

    mesh_path = category + "/textured.urdf"
    object_asset = gym.load_asset(sim, root, mesh_path, asset_options_obj)

    # set up the env grid
    num_envs = int(num_envs)
    num_per_row = int(math.sqrt(num_envs))
    env_spacing = 0.5
    env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

    # cache some common handles for later use
    envs = []
    actor_handles_grippers = []
    state_machines = []

    grasp_database = GraspsDatabase(
        mesh_root=root,
        category=category,
        step_size=num_steps,
        min_friction=0.5,
        max_friction=2.0,
        min_density=0.25,
        max_density=1.0)

    # create and populate the environments
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add gripper
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), math.pi)
        actor_handle_gripper = gym.create_actor(env, asset_gripper, pose, "gripper", i, 1)

        # enable joint force sensors
        print(gym.enable_actor_dof_force_sensors(env, actor_handle_gripper))

        # move gripper by POS
        props = gym.get_actor_dof_properties(env, actor_handle_gripper)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(800.0)  # default value from examples
        props["damping"].fill(40.0)  # default value from examples 40.0
        lf_dof_handle = gym.find_actor_dof_handle(env, actor_handle_gripper, 'finger_joint_left')
        rf_dof_handle = gym.find_actor_dof_handle(env, actor_handle_gripper, 'finger_joint_right')
        rot_dof_handle = gym.find_actor_dof_handle(env, actor_handle_gripper, 'hand_rotating')
        props["velocity"][lf_dof_handle] = (0.05) 
        props["velocity"][rf_dof_handle] = (0.05)
        props["velocity"][rot_dof_handle] = (0.05)
        gym.set_actor_dof_properties(env, actor_handle_gripper, props)
        actor_handles_grippers.append(actor_handle_gripper)

        # load obejct with parameters
        spawn_pose = gymapi.Transform()
        spawn_pose.p = gymapi.Vec3(0.0, 0.0, 0.01)
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0)
        # create actor with same collision group as gripper! 
        actor_handle_object = gym.create_actor(env, object_asset, spawn_pose, "object", i, 0)

        # state machines
        state_machine = StateObserver(
            sim,
            env,
            actor_handle_gripper,
            actor_handle_object,
            grasp_database,
            debug,
            dumb_everything)

        state_machines.append(state_machine)

    # adding viewer
    if visualize:
        cam_props = gymapi.CameraProperties()
        cam_props.use_collision_geometry = True
        viewer = gym.create_viewer(sim, cam_props)

        # point camera at middle env
        cam_pos = gymapi.Vec3(3, 3, 5)
        cam_target = gymapi.Vec3(0, 0, 0)
        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    # running the simulation
    while True:
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # refresh latest sensor readings
        gym.refresh_force_sensor_tensor(sim)

        finished_envs = 0
        for machine in state_machines:
            machine.updateSM()
            if machine.state == State.Finished:
                finished_envs += 1
            else:
                pass

        if finished_envs == num_envs:
            print("finished.")
            break

        if visualize:
            # syncs visual representation of simulation with physics state
            gym.step_graphics(sim)
            # renders latest snapshot of viewer, independent of step_graphics!
            gym.draw_viewer(viewer, sim, True)
            # synchronize visual update freq with real time (! makes sim slower !)
            gym.sync_frame_time(sim)

    # clean up
    if visualize:
        gym.destroy_viewer(viewer)

    gym.destroy_sim(sim)

    # close database
    grasp_database.close_hdf5_file()


if __name__ == "__main__":
    # optional arguments from cmd line
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", help="visualize viewer", action="store_true")
    parser.add_argument("--debug", help="debug viewer", action="store_true")
    parser.add_argument("--dumb_everything", help="dump a lot in cmd line.", action="store_true")
    parser.add_argument("--num_envs", help="number of environments", default=1)
    parser.add_argument("--root", type=str, default="", help="root directory for models")
    parser.add_argument("--category", type=str, default="", help="mesh folder name")
    parser.add_argument("--num_steps", type=int, default=10)
    args = parser.parse_args()

    main(
        visualize=args.visualize,
        num_envs=args.num_envs,
        root=args.root,
        category=args.category,
        num_steps=args.num_steps,
        debug=args.debug,
        dumb_everything=args.dumb_everything)
