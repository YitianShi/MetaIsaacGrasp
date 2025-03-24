import warp as wp
from isaac_env.air_env_base.wp_cfg import *
from .element_cfg import *

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
        if (successive_grasp_failure[tid] == SUCCESSIVE_GRASP_FAILURE_LIMIT) or (env_reachable[tid] == False) or (step_count_wp[tid] == STEP_TOTAL):
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
    #         sm_state[tid] = PickSmState.lift
    #         advance_frame[tid] = False
    #         sm_wait_time[tid] = 0.0
    #         frame_wait_time[tid] = 0.0

    # elif state == PickSmState.lift:
    #     # robot lifts the object
    #     grasp_pos = wp.add(wp.transform_get_translation(grasp_pose[tid]), wp.vec3(0.0, 0.0, LIFT_HEIGHT))
    #     des_ee_pose[tid] = wp.transform(grasp_pos, ee_quat_default[tid])
    #     des_gripper_state[tid] = GripperState.CLOSE
    #     if dist_transforms(ee_pose[tid], des_ee_pose[tid])<DISTANCE_LIMIT and ee_vel[tid]<EE_VEL_LIMIT or sm_wait_time[tid] >= PickSmLimitTime.lift:
    #         # move to next state and reset wait time
            sm_state[tid] = PickSmState.init
        
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]