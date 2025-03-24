import warp as wp
from .element_cfg import *
from isaac_env.air_env_base.wp_cfg import *

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
    