"""
Launch the live seed-traced sonar pipe model node (`pinger_sonar_trace`).

The node subscribes to the sonar point cloud, seeds the pipe detector from a
solver estimate (default) or from a point clicked in the Foxglove 3D panel, and
publishes the segmented pipe model's vertices as a PoseArray in the sonar frame
(default n_segments=3 -> 4 poses).

Defaults come from config/sonar_trace.yaml (edit that file to change the seed
source/solver, segment count, mount offsets, ...); common overrides are exposed
as launch arguments here.

Usage:
    ros2 launch pinger_localization sonar_trace.launch.py

    # real-bag odom topic + stale-estimate-tolerant replay test:
    ros2 launch pinger_localization sonar_trace.launch.py \
        odom_topic:=/auv5/nav/odom_ned max_estimate_age_s:=0.0

    # pick a different solver's estimate / a different number of segments:
    ros2 launch pinger_localization sonar_trace.launch.py \
        estimate_topic:=/pinger_localization/particle_filter/estimate \
        n_segments:=5

    # seed manually by clicking the pipe in the Foxglove 3D panel (no solver,
    # no odometry -- the click is already in the sonar frame):
    ros2 launch pinger_localization sonar_trace.launch.py seed_source:=sonar_point

Outputs: /pinger_localization/sonar_trace/seed_used (geometry_msgs/PointStamped,
the seed actually used) and /pinger_localization/sonar_trace/vertices
(geometry_msgs/PoseArray), both in frame auv5/sonar.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

CONFIG_FILE = os.path.join(
    get_package_share_directory("pinger_localization"),
    "config", "sonar_trace.yaml",
)


def generate_launch_description():
    # Launch arguments (override the yaml defaults without editing the file).
    seed_source_arg = DeclareLaunchArgument(
        "seed_source", default_value="estimate",
        description="'estimate' (solver estimate + odom) or 'sonar_point' "
                    "(PointStamped clicked in the Foxglove 3D panel, already "
                    "in the sonar frame).",
    )
    sonar_seed_arg = DeclareLaunchArgument(
        "sonar_seed_topic", default_value="/pinger_localization/sonar_trace/seed",
        description="PointStamped seed topic, used when seed_source=sonar_point.",
    )
    estimate_arg = DeclareLaunchArgument(
        "estimate_topic",
        default_value="/pinger_localization/iterative_ekf/estimate",
        description="PointStamped topic seeding the trace (pick a solver).",
    )
    odom_arg = DeclareLaunchArgument(
        "odom_topic", default_value="/auv5/odom_ned",
        description="nav_msgs/Odometry in the NED world frame.",
    )
    pointcloud_arg = DeclareLaunchArgument(
        "pointcloud_topic", default_value="/oculus/pointcloud",
        description="sensor_msgs/PointCloud2 sonar cloud.",
    )
    n_segments_arg = DeclareLaunchArgument(
        "n_segments", default_value="3",
        description="Straight segments to split the trace into "
                    "(poses = n_segments + 1; 3 => 4 poses).",
    )
    seg_robust_arg = DeclareLaunchArgument(
        "seg_robust_trim", default_value="0.15",
        description="Robust-fit trim fraction (0 = raw least-squares fit).",
    )
    max_est_age_arg = DeclareLaunchArgument(
        "max_estimate_age_s", default_value="5.0",
        description="Skip stale estimates (0 disables the check).",
    )
    mount_fwd_arg = DeclareLaunchArgument(
        "sonar_mount_forward_m", default_value="0.29442",
        description="Sonar mount offset along +x (m).",
    )

    node = Node(
        package="pinger_localization",
        executable="sonar_trace_node.py",
        name="pinger_sonar_trace",
        output="screen",
        parameters=[
            CONFIG_FILE,   # base defaults from config/sonar_trace.yaml
            {
                "seed_source": ParameterValue(
                    LaunchConfiguration("seed_source"), value_type=str),
                "sonar_seed_topic": ParameterValue(
                    LaunchConfiguration("sonar_seed_topic"), value_type=str),
                "estimate_topic": ParameterValue(
                    LaunchConfiguration("estimate_topic"), value_type=str),
                "odom_topic": ParameterValue(
                    LaunchConfiguration("odom_topic"), value_type=str),
                "pointcloud_topic": ParameterValue(
                    LaunchConfiguration("pointcloud_topic"), value_type=str),
                "n_segments": ParameterValue(
                    LaunchConfiguration("n_segments"), value_type=int),
                "seg_robust_trim": ParameterValue(
                    LaunchConfiguration("seg_robust_trim"), value_type=float),
                "max_estimate_age_s": ParameterValue(
                    LaunchConfiguration("max_estimate_age_s"), value_type=float),
                "sonar_mount_forward_m": ParameterValue(
                    LaunchConfiguration("sonar_mount_forward_m"),
                    value_type=float),
            },
        ],
    )

    return LaunchDescription([
        seed_source_arg, sonar_seed_arg,
        estimate_arg, odom_arg, pointcloud_arg,
        n_segments_arg, seg_robust_arg, max_est_age_arg, mount_fwd_arg,
        node,
    ])
