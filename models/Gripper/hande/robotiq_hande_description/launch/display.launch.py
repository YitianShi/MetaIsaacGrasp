from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ld = LaunchDescription()

    pkg_path = FindPackageShare('robotiq_hande_description')
    model_path = PathJoinSubstitution(['urdf', 'robotiq_hande_gripper.urdf.xacro'])
    rviz_config_path = PathJoinSubstitution([pkg_path, 'rviz', 'urdf.rviz'])
    run_joint_state_publisher_gui = 'true'

    ld.add_action(IncludeLaunchDescription(
        PathJoinSubstitution([FindPackageShare('urdf_launch'), 'launch', 'display.launch.py']),
        launch_arguments={
            'urdf_package': 'robotiq_hande_description',
            'urdf_package_path': model_path,
            'rviz_config': rviz_config_path,
            'jsp_gui': run_joint_state_publisher_gui}.items()
    ))

    return ld