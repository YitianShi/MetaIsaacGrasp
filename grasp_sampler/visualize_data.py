import torch
import numpy as np
import open3d as o3d
import cv2
import argparse
from pathlib import Path
import glob

# Function to create a cylinder between two points
def create_cylinder_between_points(p1, p2, radius=0.05, color=[0.1, 0.1, 0.7]):
    """
    Create a cylinder mesh between two points p1 and p2 with a specified radius and color.
    Args:
        p1 (numpy.ndarray): Starting point of the cylinder (3D coordinates).
        p2 (numpy.ndarray): Ending point of the cylinder (3D coordinates).
        radius (float): Radius of the cylinder.
        color (list): Color of the cylinder in RGB format.
    Returns:
        cylinder (open3d.geometry.TriangleMesh): Cylinder mesh object.
    """
    # Calculate the direction and length of the cylinder
    direction = p2 - p1
    length = np.linalg.norm(direction)
    direction /= length

    # Create a cylinder mesh
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cylinder.compute_vertex_normals()

    # Rotate the cylinder to align with the direction vector
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    rotation_angle = np.arccos(np.dot(z_axis, direction))
    if np.linalg.norm(rotation_axis) > 0:  # Check if rotation is needed
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis * rotation_angle
        )
        cylinder.rotate(rotation_matrix, center=(0, 0, 0))

    # Translate the cylinder to start at point p1
    cylinder.translate((p1 + p2) / 2)

    # Paint the cylinder with the specified color
    cylinder.paint_uniform_color(color)

    return cylinder


# Function to draw grasp lines and spheres
def draw_grasps(cp, cp2, approach, score=None,
                color=[0.7, 0.1, 0.1], graspline_width=5e-4, finger_length=0.025, arm_length=0.02):
    """
    Draw grasp lines and spheres between two contact points.
    Args:
        cp (numpy.ndarray): Contact points (N x 3).
        cp2 (numpy.ndarray): Contact points (N x 3).
        approach (numpy.ndarray): Approach vectors (N x 3).
        score (numpy.ndarray): Grasp scores (N, optional).
        color (list): Color of the grasp lines and spheres.
        graspline_width (float): Width of the grasp lines.
        finger_length (float): Length of the fingers.
        arm_length (float): Length of the arm.
    Returns:
        vis_list (list): List of Open3D geometries for visualization.
    """
    # Initialize the visualization list
    
    vis_list = []
    color_max = np.array([0, 0, 1])
    color_min = np.array([1, 0, 0])
    cp_half = (cp + cp2) / 2

    for i, (q, a, app, half_q, half_a) in enumerate(zip(cp, cp2, approach, 
                                                        cp_half - approach * finger_length, 
                                                        cp_half - approach * (finger_length + arm_length))):
        # Determine color based on score
        color = color_max * score[i] + color_min * (1 - score[i]) if score is not None else color
        
        # Draw fingers and arm cylinders
        vis_list.extend([
            create_cylinder_between_points(a - app * finger_length, a, radius=graspline_width, color=color),
            create_cylinder_between_points(q - app * finger_length, q, radius=graspline_width, color=color),
            create_cylinder_between_points(q - app * finger_length, a - app * finger_length, radius=graspline_width, color=color),
            create_cylinder_between_points(half_q, half_a, radius=graspline_width, color=color)
        ])
    
    return vis_list


# Function to visualize grasps
def vis_grasps(
        pcd,
        cp,
        cp2,
        approach,
        score
    ):
    """
    Visualize grasps using Open3D.
    Args:
        pcd (numpy.ndarray): Point cloud data (N x 3).
        pt (numpy.ndarray): Contact points (N x 3).
        pt_2 (numpy.ndarray): Contact points (N x 3).
        approach (numpy.ndarray): Approach vectors (N x 3).
        scores (numpy.ndarray): Grasp scores (N, optional).
    Returns:
        vis_list (list): List of Open3D geometries for visualization.
    """
    vis_list = []
    approach = (approach.detach().cpu().numpy() if isinstance(approach, torch.Tensor) else approach)
    cp = cp.detach().cpu().numpy() if isinstance(cp, torch.Tensor) else cp
    cp2 = cp2.detach().cpu().numpy() if isinstance(cp2, torch.Tensor) else cp2
    score = (score.detach().cpu().numpy() if isinstance(score, torch.Tensor) else score).reshape(-1)

    # Visualize the sampled points
    pcd = pcd.cpu().numpy() if isinstance(pcd, torch.Tensor) else pcd
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(pcd)
    vis_list.append(pcd_vis)
    vis_list += draw_grasps(cp, cp2, approach, score, color=[0.1, 0.7, 0.1])

    return vis_list


# Helper: visualize point cloud and camera pose
def show_point_cloud(points, normals=None, colors=None, camera_pose=None):
    """
    Visualize a point cloud with optional normals, colors, and camera pose.
    Args:
        points (numpy.ndarray): Point cloud data (N x 3).
        normals (numpy.ndarray): Normals for the point cloud (N x 3, optional).
        colors (numpy.ndarray): Colors for the point cloud (N x 3, optional).
        camera_pose (numpy.ndarray): Camera pose (7 elements: [x, y, z, qx, qy, qz, qw], optional).
    """
    # Create a list to hold the geometries for visualization
    show_list = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    show_list.append(pcd)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if camera_pose is not None:
        # Create a coordinate frame for the camera pose
        # quaternion to rotation matrix
        camera_pose_m = np.eye(3)
        camera_pose_m = o3d.geometry.get_rotation_matrix_from_quaternion(camera_pose[3:])

        # Create a coordinate frame for the camera
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=camera_pose[:3])
        camera_frame.rotate(camera_pose_m, center=camera_pose[:3])
        show_list.append(camera_frame)

    # the coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    show_list.append(axis)

    o3d.visualization.draw_geometries(show_list)


def main(pth_paths):
    pth_paths = glob.glob(f"{pth_paths}/*.pt")
    sorted(pth_paths)
    for pth_path in pth_paths:
        print(f"Loading {pth_path}")
        # Load the .pth data
        data = torch.load(pth_path, map_location="cpu", weights_only=False)
        
        # Get the ground truth point cloud
        pcd = data["pcd_gt"].float().numpy() / 1e4
                
        # Extract the point cloud data
        grasp_width = data["non_colliding_parallel_contact_width"] / 1e2 # Convert to meters
        grasp_filter = grasp_width > 0.0

        # Get the grasp poses and widths
        grasp_width = grasp_width[grasp_filter].to(torch.float32) 
        
        grasp_poses = data["non_colliding_parallel_contact_poses"]
        grasp_poses = grasp_poses[grasp_filter]
        grasp_poses[:, :3, -1] /= 100 # Convert to meters

        # Get the grasp quality scores
        score = data["non_colliding_parallel_analytical_score"]
        score = score[grasp_filter]

        # Translate the grasp poses to the contact grasp representation
        grasp_contact_pts = grasp_poses[:, :3, -1]  # Contact points
        baseline = grasp_poses[:, :3, 0] # Baseline vectors
        approach = grasp_poses[:, :3, 2] # Approach vectors

        grasp_vis_list = vis_grasps(
            pcd,
            grasp_contact_pts,
            grasp_contact_pts + baseline * grasp_width[:, None], # Contact points 2
            approach,
            score
        )

        # Visualize the grasp points
        o3d.visualization.draw_geometries(grasp_vis_list)
        
        for cam_id in [cam_id for cam_id in data.keys() if "camera" in cam_id]:
            cam_data = data[cam_id]

            if "pcd" in cam_data:
                camera_pose = cam_data["camera_pose"].numpy()
                pcd = cam_data["pcd"].float().numpy() / 1000.0  # Convert to meters
                show_point_cloud(pcd.reshape(-1, 3), camera_pose=camera_pose)
                
            if "rgb" in cam_data:
                rgb = cam_data["rgb"].permute(2, 0, 1) / 255.0
                img = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                cv2.imshow(f"{cam_id} - RGB", img[..., ::-1])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if "grasp_pose" in cam_data:
                print(f"{cam_id} Grasp Pose:\n", cam_data["grasp_pose"])

            if "depth" in cam_data:
                depth = cam_data["depth"].squeeze().numpy()
                cv2.imshow(f"{cam_id} - Depth", depth / depth.max())
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            if "normal" in cam_data:
                normal = cam_data["normal"].squeeze().numpy()
                cv2.imshow(f"{cam_id} - Normal", normal / normal.max())
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if "instance" in cam_data:
                instance = cam_data["instance"].squeeze().numpy().astype(np.int32)
                
                # Set background (1) to 0
                instance[instance == 1] = 0

                unique_ids = np.unique(instance)

                # Assign random colors to instance IDs (excluding background)
                colors = {
                    idx: np.random.randint(0, 255, size=3, dtype=np.uint8)
                    for idx in unique_ids if idx != 0
                }

                # Create RGB image and paint each instance
                color_img = np.zeros((*instance.shape, 3), dtype=np.uint8)
                for idx, color in colors.items():
                    color_img[instance == idx] = color

                # Show the image in BGR format for OpenCV
                cv2.imshow(f"{cam_id} - Instance", color_img[..., ::-1])
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_file", type=str, default="../vmf_data/data_debug", help="Path to the .pt file")
    args = parser.parse_args()

    main(args.pth_file)

