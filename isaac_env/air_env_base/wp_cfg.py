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
    execute = wp.constant(20.)
    frame = wp.constant(0.1)
    
@wp.func
def dist_transforms(a: wp.transform, b: wp.transform):
    """Compute the distance between two transformations."""
    return wp.length(wp.transform_get_translation(a) - wp.transform_get_translation(b))

@wp.func
def approach_pose_from_grasp_pose(grasp_pose: wp.transform):
    """Compute the approach pose from the grasp pose, 
    approach pose have the same rotation as grasp pose but have certain distance in the opposite direction to the grasp pose."""
    v2 = wp.quat_rotate(wp.transform_get_rotation(grasp_pose), wp.vec3(0.0, APPROACH_DISTANCE, 0.0))
    return wp.transform(wp.transform_get_translation(grasp_pose) - v2,
                        wp.transform_get_rotation(grasp_pose))
