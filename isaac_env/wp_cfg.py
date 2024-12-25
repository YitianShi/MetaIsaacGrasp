import warp as wp
from .element_cfg import *
wp.init()

class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)

class PickSmState:
    """States for the pick state machine."""
    init_env=wp.constant(STATE_MACHINE["init_env"])
    init = wp.constant(STATE_MACHINE["init"])
    start = wp.constant(STATE_MACHINE["start"])
    choose_object = wp.constant(STATE_MACHINE["choose_object"])
    reach = wp.constant(STATE_MACHINE["reach"])
    approach = wp.constant(STATE_MACHINE["approach"])
    grasp = wp.constant(STATE_MACHINE["grasp"])
    lift = wp.constant(STATE_MACHINE["lift"])
    execute = wp.constant(STATE_MACHINE["execute"])

class PickSmLimitTime:
    """Additional wait times (in s) for states for before switching."""

    start = wp.constant(1.)
    reach = wp.constant(1.)
    approach = wp.constant(1.5)
    grasp = wp.constant(.5)
    lift = wp.constant(2.5)
    execute = wp.constant(2.)
    frame = wp.constant(0.1)
    
@wp.func
def dist_transforms(a: wp.transform, b: wp.transform):
    """Compute the distance between two transformations."""
    return wp.length(wp.transform_get_translation(a) - wp.transform_get_translation(b))

@wp.func
def approach_pose_from_grasp_pose(grasp_pose: wp.transform):
    """Compute the approach pose from the grasp pose, 
    approach pose have the same rotation as grasp pose but have certain distance in the opposite direction to the grasp pose."""
    v2 = wp.quat_rotate(wp.transform_get_rotation(grasp_pose), wp.vec3(0.0, approach_distance, 0.0))
    return wp.transform(wp.transform_get_translation(grasp_pose) - v2,
                        wp.transform_get_rotation(grasp_pose))

@wp.kernel
def infer_state_machine_data(
    # environment state machine recorders
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    # environment time states
    epi_count_wp: wp.array(dtype=wp.int32),
    # environment physical states
    env_stable: wp.array(dtype=bool),
    # desired robot end effector state
    des_ee_pose: wp.array(dtype=wp.transform),
    des_gripper_state: wp.array(dtype=float),
     # current robot end effector state
    ee_pose: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.init_env:
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        # reset the environment including the robot and objects   
        epi_count_wp[tid] = wp.add(epi_count_wp[tid], 1)
        sm_state[tid] = PickSmState.start
        sm_wait_time[tid] = 0.
        
    elif state == PickSmState.start:
        # start means objects start to fall down on the table
        # or wait until the environment is stable after grasping
        # when the environment is stable, 
        # robot starts to take photo and choose the object
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        if env_stable[tid] == True or sm_wait_time[tid] >= PickSmLimitTime.start:
            sm_state[tid] = PickSmState.choose_object
    
    elif state == PickSmState.choose_object:
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        # not wait for a while
        sm_state[tid] = PickSmState.init_env
        sm_wait_time[tid] = 0.0

    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
    

@wp.kernel
def infer_state_machine_tele(
    # environment state machine recorders
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    # environment time states
    successive_grasp_failure: wp.array(dtype=wp.float32),
    epi_count_wp: wp.array(dtype=wp.int32),
    step_count_wp: wp.array(dtype=wp.int32),
    # environment physical states
    env_reachable: wp.array(dtype=bool),
    env_stable: wp.array(dtype=bool),
    # current robot end effector state
    ee_pose: wp.array(dtype=wp.transform),
    ee_vel: wp.array(dtype=float),
    # desired robot end effector state
    des_ee_pose: wp.array(dtype=wp.transform),
    des_gripper_state: wp.array(dtype=float),
    ee_quat_default: wp.array(dtype=wp.quat),
    # proposed grasp pose
    grasp_pose: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.init_env:
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        # reset the environment including the robot and objects   
        step_count_wp[tid] = 0
        epi_count_wp[tid] = wp.add(epi_count_wp[tid], 1)
        successive_grasp_failure[tid] = 0.
        sm_state[tid] = PickSmState.start
        sm_wait_time[tid] = 0.

    elif state == PickSmState.init:
        # increment the step count and the successive grasp failure count
        step_count_wp[tid] = wp.add(step_count_wp[tid], 1)
        successive_grasp_failure[tid] = wp.add(successive_grasp_failure[tid], 1.)

        # check the termination conditions
        if (successive_grasp_failure[tid] == successive_grasp_failure_limit) or (env_reachable[tid] == False) or (step_count_wp[tid] == step_total):
            sm_state[tid] = PickSmState.init_env
        else:
            sm_state[tid] = PickSmState.start

        # reset the robot only
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        sm_wait_time[tid] = 0.
        
    elif state == PickSmState.start:
        # start means objects start to fall down on the table
        # or wait until the environment is stable after grasping
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        # when the environment is stable, 
        # robot starts to take photo and choose the object
        if env_stable[tid] == True or sm_wait_time[tid] >= PickSmLimitTime.start:
            sm_state[tid] = PickSmState.execute
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.execute:
        # robot start to execute the grasp based on continuous control
        des_ee_pose[tid] = grasp_pose[tid]
        
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


@wp.kernel
def infer_state_machine_disc(
    # environment state machine recorders
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    # environment time states
    successive_grasp_failure: wp.array(dtype=wp.float32),
    epi_count_wp: wp.array(dtype=wp.int32),
    step_count_wp: wp.array(dtype=wp.int32),
    # environment physical states
    env_reachable: wp.array(dtype=bool),
    env_stable: wp.array(dtype=bool),
    # current robot end effector state
    ee_pose: wp.array(dtype=wp.transform),
    ee_vel: wp.array(dtype=float),
    # desired robot end effector state
    des_ee_pose: wp.array(dtype=wp.transform),
    des_gripper_state: wp.array(dtype=float),
    ee_quat_default: wp.array(dtype=wp.quat),
    # proposed grasp pose
    grasp_pose: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.init_env:
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        # reset the environment including the robot and objects   
        step_count_wp[tid] = 0
        epi_count_wp[tid] = wp.add(epi_count_wp[tid], 1)
        successive_grasp_failure[tid] = 0.
        sm_state[tid] = PickSmState.start
        sm_wait_time[tid] = 0.

    elif state == PickSmState.init:
        # increment the step count and the successive grasp failure count
        step_count_wp[tid] = wp.add(step_count_wp[tid], 1)
        successive_grasp_failure[tid] = wp.add(successive_grasp_failure[tid], 1.)

        # check the termination conditions
        if (successive_grasp_failure[tid] == successive_grasp_failure_limit) or (env_reachable[tid] == False) or (step_count_wp[tid] == step_total):
            sm_state[tid] = PickSmState.init_env
        else:
            sm_state[tid] = PickSmState.start

        # reset the robot only
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        sm_wait_time[tid] = 0.
        
    elif state == PickSmState.start:
        # start means objects start to fall down on the table
        # or wait until the environment is stable after grasping
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        # when the environment is stable, 
        # robot starts to take photo and choose the object
        if env_stable[tid] == True or sm_wait_time[tid] >= PickSmLimitTime.start:
            sm_state[tid] = PickSmState.choose_object
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.choose_object:
        des_ee_pose[tid] = ee_pose[tid]
        # take photo and robot start to choose the object
        des_gripper_state[tid] = GripperState.OPEN
        # not wait for a while
        sm_state[tid] = PickSmState.reach
        sm_wait_time[tid] = 0.0

    elif state == PickSmState.reach:
        # robot moves to the object above
        des_ee_pose[tid] = approach_pose_from_grasp_pose(grasp_pose[tid])
        des_gripper_state[tid] = GripperState.OPEN
        # error between current and desired ee pose below threshold
        if dist_transforms(ee_pose[tid], des_ee_pose[tid])<distance_limit and ee_vel[tid]<ee_vel_limit:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.approach
        # wait for a while
        elif sm_wait_time[tid] >= PickSmLimitTime.reach:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.init

    elif state == PickSmState.approach:
        # robot moves to the grasp position
        des_ee_pose[tid] = grasp_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        # Error between current and desired ee pose below threshold
        # Or the approaching time is longer than the limit
        if dist_transforms(ee_pose[tid], des_ee_pose[tid])<distance_limit and ee_vel[tid]<ee_vel_limit or sm_wait_time[tid] >= PickSmLimitTime.approach:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.grasp
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.grasp:
        # robot grasps the object
        des_ee_pose[tid] = grasp_pose[tid]
        des_gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmLimitTime.grasp:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.lift
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.lift:
        # robot lifts the object
        grasp_pos = wp.add(wp.transform_get_translation(grasp_pose[tid]), wp.vec3(0.0, 0.0, lift_height))
        des_ee_pose[tid] = wp.transform(grasp_pos, ee_quat_default[tid])
        des_gripper_state[tid] = GripperState.CLOSE
        if dist_transforms(ee_pose[tid], des_ee_pose[tid])<distance_limit and ee_vel[tid]<ee_vel_limit or sm_wait_time[tid] >= PickSmLimitTime.lift:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.init

    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
    

@wp.kernel
def infer_state_machine_con(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    # environment time states
    successive_grasp_failure: wp.array(dtype=wp.float32),
    epi_count_wp: wp.array(dtype=wp.int32),
    step_count_wp: wp.array(dtype=wp.int32),
    # environment physical states
    env_reachable: wp.array(dtype=bool),
    env_stable: wp.array(dtype=bool),
    # current robot end effector state
    ee_pose: wp.array(dtype=wp.transform),
    ee_vel: wp.array(dtype=float),
    # desired robot end effector state
    des_ee_pose: wp.array(dtype=wp.transform),
    des_gripper_state: wp.array(dtype=float),
    ee_quat_default: wp.array(dtype=wp.quat),
    # proposed grasp pose
    grasp_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    # continuous control time recorder
    advance_frame: wp.array(dtype=bool),
    frame_wait_time: wp.array(dtype=float),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.init_env:
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        # reset the environment including the robot and objects   
        step_count_wp[tid] = 0
        epi_count_wp[tid] = wp.add(epi_count_wp[tid], 1)
        successive_grasp_failure[tid] = 0.
        sm_state[tid] = PickSmState.start
        sm_wait_time[tid] = 0.

    elif state == PickSmState.init:
        # increment the step count and the successive grasp failure count
        step_count_wp[tid] = wp.add(step_count_wp[tid], 1)
        successive_grasp_failure[tid] = wp.add(successive_grasp_failure[tid], 1.)

        # check the termination conditions
        if (successive_grasp_failure[tid] == successive_grasp_failure_limit) or (env_reachable[tid] == False) or (step_count_wp[tid] == step_total):
            sm_state[tid] = PickSmState.init_env
        else:
            sm_state[tid] = PickSmState.start

        # reset the robot only
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        sm_wait_time[tid] = 0.
        
    elif state == PickSmState.start:
        # start means objects start to fall down on the table
        # or wait until the environment is stable after grasping
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        # when the environment is stable, 
        # robot starts to take photo and choose the object
        if env_stable[tid] == True:
            sm_state[tid] = PickSmState.execute
            sm_wait_time[tid] = 0.0
        elif sm_wait_time[tid] >= PickSmLimitTime.start:
            sm_state[tid] = PickSmState.init
    
    elif state == PickSmState.execute:
        # robot start to execute the grasp based on continuous control
        des_ee_pose[tid] = grasp_pose[tid]
        des_gripper_state[tid] = gripper_state[tid]
        advance_frame[tid] = False
        # error between current and desired ee pose below threshold
        if sm_wait_time[tid] >= PickSmLimitTime.frame:
            # step frame
            # if advance_frame is True, the controller need to make decision
            advance_frame[tid] = True
            frame_wait_time[tid] = 0.0
        else:
            frame_wait_time[tid] = frame_wait_time[tid] + dt[tid]
        if sm_wait_time[tid] >= PickSmLimitTime.execute:
            sm_state[tid] = PickSmState.lift
            advance_frame[tid] = False
            sm_wait_time[tid] = 0.0
            frame_wait_time[tid] = 0.0

    elif state == PickSmState.lift:
        # robot lifts the object
        grasp_pos = wp.add(wp.transform_get_translation(grasp_pose[tid]), wp.vec3(0.0, 0.0, lift_height))
        des_ee_pose[tid] = wp.transform(grasp_pos, ee_quat_default[tid])
        des_gripper_state[tid] = GripperState.CLOSE
        if dist_transforms(ee_pose[tid], des_ee_pose[tid])<distance_limit and ee_vel[tid]<ee_vel_limit or sm_wait_time[tid] >= PickSmLimitTime.lift:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.init
        
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]