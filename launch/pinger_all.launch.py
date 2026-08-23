"""
Launch everything: simulator + all 6 solvers + eval node.

This is the main launch file for a full comparison run.

Usage:
    ros2 launch pinger_localization pinger_all.launch.py
    ros2 launch pinger_localization pinger_all.launch.py \
        trajectory_type:=figure8 \
        noise_radius:=0.3 \
        pinger_north:=15.0 \
        pinger_east:=8.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # ---- Simulator arguments ----
    trajectory_type_arg = DeclareLaunchArgument(
        "trajectory_type", default_value="circle"
    )
    trajectory_radius_arg = DeclareLaunchArgument(
        "trajectory_radius", default_value="10.0"
    )
    trajectory_speed_arg = DeclareLaunchArgument(
        "trajectory_speed", default_value="1.0"
    )
    pinger_north_arg = DeclareLaunchArgument("pinger_north", default_value="10.0")
    pinger_east_arg = DeclareLaunchArgument("pinger_east", default_value="5.0")
    noise_radius_arg = DeclareLaunchArgument("noise_radius", default_value="0.5")
    noise_type_arg = DeclareLaunchArgument("noise_type", default_value="jitter")
    doa_noise_std_arg = DeclareLaunchArgument("doa_noise_std", default_value="3.0")
    ping_interval_arg = DeclareLaunchArgument("ping_interval", default_value="2.0")
    odom_rate_arg = DeclareLaunchArgument("odom_rate", default_value="20.0")
    world_frame_arg = DeclareLaunchArgument("world_frame", default_value="odom_ned")

    simulator_params = {
        "trajectory_type": ParameterValue(
            LaunchConfiguration("trajectory_type"), value_type=str
        ),
        "trajectory_radius": ParameterValue(
            LaunchConfiguration("trajectory_radius"), value_type=float
        ),
        "trajectory_speed": ParameterValue(
            LaunchConfiguration("trajectory_speed"), value_type=float
        ),
        "pinger_north": ParameterValue(
            LaunchConfiguration("pinger_north"), value_type=float
        ),
        "pinger_east": ParameterValue(
            LaunchConfiguration("pinger_east"), value_type=float
        ),
        "noise_radius": ParameterValue(
            LaunchConfiguration("noise_radius"), value_type=float
        ),
        "noise_type": ParameterValue(
            LaunchConfiguration("noise_type"), value_type=str
        ),
        "doa_noise_std": ParameterValue(
            LaunchConfiguration("doa_noise_std"), value_type=float
        ),
        "ping_interval": ParameterValue(
            LaunchConfiguration("ping_interval"), value_type=float
        ),
        "odom_rate": ParameterValue(
            LaunchConfiguration("odom_rate"), value_type=float
        ),
        "world_frame": ParameterValue(
            LaunchConfiguration("world_frame"), value_type=str
        ),
    }

    # ---- Simulator ----
    simulator_node = Node(
        package="pinger_localization",
        executable="pinger_simulator.py",
        name="pinger_simulator",
        output="screen",
        parameters=[simulator_params],
    )

    # ---- Solvers ----
    solver_names = [
        "batch_sliding",
        "batch_full",
        "particle_filter",
        "iterative_ekf",
        "iterative_rls",
        "iterative_gd",
    ]

    solver_nodes = []
    for name in solver_names:
        solver_nodes.append(Node(
            package="pinger_localization",
            executable=f"{name}.py",
            name=f"pinger_{name}",
            output="screen",
            parameters=[{
                "world_frame": ParameterValue(
                    LaunchConfiguration("world_frame"), value_type=str
                ),
            }],
        ))

    # ---- Eval node ----
    eval_node = Node(
        package="pinger_localization",
        executable="eval_node.py",
        name="pinger_eval",
        output="screen",
        parameters=[{
            "solver_names": solver_names,
        }],
    )

    # ---- Launch description ----
    return LaunchDescription([
        trajectory_type_arg,
        trajectory_radius_arg,
        trajectory_speed_arg,
        pinger_north_arg,
        pinger_east_arg,
        noise_radius_arg,
        noise_type_arg,
        doa_noise_std_arg,
        ping_interval_arg,
        odom_rate_arg,
        world_frame_arg,
        simulator_node,
        *solver_nodes,
        eval_node,
    ])
