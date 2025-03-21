import numpy as np
import torch
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from .element_cfg import *



def perpendicular_grasp_orientation(normal, tensor=True):
    """
    Compute a quaternion (w, x, y, z) to align the gripper's Z-axis with the given surface normal.
    This approach simplifies alignment by ignoring rotation around the Z-axis (approach direction).

    Args:
    - normal (torch.Tensor): The surface normal vector, expected shape [3].

    Returns:
    - quaternion (torch.Tensor): The quaternion (w, x, y, z) representing the required orientation.
    """
    # Ensure the normal is a unit vector

    if not isinstance(normal, torch.Tensor) and tensor:
        normal = torch.tensor(
            normal, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    normal = -normal
    # Gripper's default approach direction is assumed to be along Y-axis [0, 1, 0]
    default_direction = torch.tensor([0.0, 1.0, 0.0], device=normal.device)

    # Calculate the rotation axis by taking the cross product
    rotation_axis = torch.linalg.cross(default_direction, normal)
    rotation_axis_norm = torch.norm(rotation_axis)

    # Check for parallel or anti-parallel vectors
    if rotation_axis_norm.item() < 1e-4:
        # If the dot product is negative, they are anti-parallel (180-degree rotation needed)
        quaternion = torch.tensor(EE_GRASP_QUAT_DEFAULT, device=normal.device)
    else:
        # Calculate the angle for rotation
        cos_theta = torch.dot(default_direction, normal)
        angle = torch.acos(cos_theta)

        # Use the half-angle formula for quaternion from axis-angle
        rotation_axis = rotation_axis / rotation_axis_norm  # Normalize rotation axis
        sin_half_angle = torch.sin(angle / 2)
        cos_half_angle = torch.cos(angle / 2)

        # Construct the quaternion in (w, x, y, z) format
        quaternion = torch.cat(
            [cos_half_angle.view(-1), sin_half_angle * rotation_axis]  # w  # x, y, z
        )
    return quaternion


def robot_point_to_image(world_point, cam_pose):

    # Assuming the extrinsic parameters are known
    # R_inv and T_inv would be the inverse rotation and translation matrices
    # For simplicity, assuming no rotation (identity matrix) and no translation (zero vector)

    # Convert from world coordinates to camera coordinates
    if not isinstance(world_point, torch.Tensor):
        world_point = torch.tensor(world_point)

    rotation_matrix = matrix_from_quat(cam_pose[3:])
    # Translate the point to the camera's coordinate system
    point_translated = world_point - cam_pose[:3].to(world_point.device)

    # Rotate the point to align with the camera's orientation
    P_camera = torch.matmul(
        rotation_matrix.T.to(point_translated.device), point_translated)


    # Project camera coordinates onto the image plane
    x_image = int((P_camera[0] / P_camera[2]) * focal_length_pixels + cx)
    y_image = int((P_camera[1] / P_camera[2]) * focal_length_pixels + cy)

    image_point = torch.tensor(
        [x_image, y_image], dtype=torch.int, device=world_point.device
    )
    return image_point.to(world_point.device)


def pose_vector_to_transformation_matrix(pose_vec):
    rot_matrix = matrix_from_quat(pose_vec[3:7])
    transformation_matrix = torch.Tensor([
                [rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], pose_vec[0]],
                [rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], pose_vec[1]],
                [rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], pose_vec[2]],
                [0, 0, 0, 1],
            ])
    return transformation_matrix

def transformation_matrix_to_pose_vector(transformation_matrix):
    rot_matrix = transformation_matrix[:3, :3]
    pose_vec = torch.cat(
        (
            transformation_matrix[:3, 3],
            quat_from_matrix(rot_matrix).squeeze(),
        )
    )

    return pose_vec

def rotation_matrix_from_view(
    eyes: torch.Tensor,
    targets: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute the rotation matrix from world to view coordinates.

    This function takes a vector ''eyes'' which specifies the location
    of the camera in world coordinates and the vector ''targets'' which
    indicate the position of the object.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

        The inputs eyes and targets can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)

    Args:
        eyes: Position of the camera in world coordinates.
        targets: Position of the object in world coordinates.
        up_axis: The up axis of the camera. Defaults to "Z".
        device: The device to create torch tensors on. Defaults to "cpu".

    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (N, 3, 3) batched rotation matrices

    Reference:
    Based on PyTorch3D (https://github.com/facebookresearch/pytorch3d/blob/eaf0709d6af0025fe94d1ee7cec454bc3054826a/pytorch3d/renderer/cameras.py#L1635-L1685)
    """
    up_axis_vec = torch.tensor((0, 0, 1), device=device, dtype=torch.float32).repeat(eyes.shape[0], 1)

    # get rotation matrix in opengl format (-Z forward, +Y up)
    y_axis = torch.nn.functional.normalize(targets - eyes, eps=1e-5)
    x_axis = -torch.nn.functional.normalize(torch.cross(up_axis_vec, y_axis, dim=1), eps=1e-5)
    z_axis = torch.nn.functional.normalize(torch.cross(x_axis, y_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = torch.nn.functional.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)