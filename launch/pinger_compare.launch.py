"""
Launch a configurable subset of the simulator + solvers for focused testing.

Usage:
    ros2 launch pinger_localization pinger_compare.launch.py \
        solvers:="batch_sliding,particle_filter" \
        trajectory_type:=lawnmower
"""

import json

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ALL_SOLVER_INFO = {
    "batch_sliding":   ("batch_sliding.py",   "pinger_batch_sliding"),
    "batch_full":      ("batch_full.py",      "pinger_batch_full"),
    "particle_filter": ("particle_filter.py", "pinger_particle_filter"),
    "iterative_ekf":   ("iterative_ekf.py",   "pinger_iterative_ekf"),
    "iterative_rls":   ("iterative_rls.py",   "pinger_iterative_rls"),
    "iterative_gd":    ("iterative_gd.py",    "pinger_iterative_gd"),
}


def _parse_solvers(context):
    """Parse the 'solvers' launch argument: comma-separated or JSON list."""
    raw = LaunchConfiguration("solvers").perform(context)
    raw = raw.strip()
    if raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return [s.strip() for s in raw.split(",") if s.strip()]


def _create_nodes(context, *args, **kwargs):
    """OpaqueFunction: produce solver + eval nodes from the parsed solver list."""
    solver_names = _parse_solvers(context)
    world_frame = LaunchConfiguration("world_frame").perform(context)

    nodes = []
    for name in solver_names:
        if name not in ALL_SOLVER_INFO:
            raise ValueError(
                f"Unknown solver '{name}'. Choose from: {list(ALL_SOLVER_INFO.keys())}"
            )
        executable, node_name = ALL_SOLVER_INFO[name]
        nodes.append(Node(
            package="pinger_localization",
            executable=executable,
            name=node_name,
            output="screen",
            parameters=[{"world_frame": world_frame}],
        ))

    # Eval node — pass the parsed list directly.
    nodes.append(Node(
        package="pinger_localization",
        executable="eval_node.py",
        name="pinger_eval",
        output="screen",
        parameters=[{"solver_names": solver_names}],
    ))
    return nodes


def generate_launch_description():
    # ---- Simulator arguments (same as pinger_all) ----
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

    solvers_arg = DeclareLaunchArgument(
        "solvers",
        default_value="iterative_ekf,iterative_rls,iterative_gd",
        description="Comma-separated solver names (or JSON list).",
    )

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

    # ---- Solvers + eval (dynamic via OpaqueFunction) ----
    dynamic_nodes = OpaqueFunction(function=_create_nodes)

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
        solvers_arg,
        simulator_node,
        dynamic_nodes,
    ])
