"""
Launch the pinger localization solvers against REAL sensors (no simulator).

The solvers need exactly two live inputs, which in the sim come from
`pinger_simulator.py`. In real life those must be published by your own nodes:

  * Odometry on `odom_topic`   (nav_msgs/Odometry, pose in NED world frame)
  * Pings on `ping_topic`      (bb_sensor_msgs/Ping, body-relative DOA in degrees)

Topic remapping connects the solvers' hardcoded sim topic names
(`/auv5/odom_ned`, `/sensors/ping`) to whatever your real system publishes.

Usage:
    ros2 launch pinger_localization pinger_irl.launch.py \
        solvers:="iterative_ekf,iterative_rls,iterative_gd"

    # Your real topics / world frame:
    ros2 launch pinger_localization pinger_irl.launch.py \
        odom_topic:=/auv5/odom_ned \
        ping_topic:=/sensors/ping \
        world_frame:=world

Estimates appear on /pinger_localization/<name>/estimate (PointStamped) and
/pinger_localization/<name>/estimate_pose (PoseStamped).
"""

import json

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ALL_SOLVER_INFO = {
    "batch_sliding":   ("batch_sliding.py",   "pinger_batch_sliding"),
    "batch_full":      ("batch_full.py",      "pinger_batch_full"),
    "particle_filter": ("particle_filter.py", "pinger_particle_filter"),
    "iterative_ekf":   ("iterative_ekf.py",   "pinger_iterative_ekf"),
    "iterative_rls":   ("iterative_rls.py",   "pinger_iterative_rls"),
    "iterative_gd":    ("iterative_gd.py",    "pinger_iterative_gd"),
}

# Topics the solver code subscribes to (from the simulator). These get
# remapped to the real topics at launch time.
SIM_ODOM_TOPIC = "/auv5/odom_ned"
SIM_PING_TOPIC = "/sensors/ping"


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
    """OpaqueFunction: produce solver nodes, remapped to real topics."""
    solver_names = _parse_solvers(context)
    odom_topic = LaunchConfiguration("odom_topic").perform(context)
    ping_topic = LaunchConfiguration("ping_topic").perform(context)
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
            remappings=[
                (SIM_ODOM_TOPIC, odom_topic),
                (SIM_PING_TOPIC, ping_topic),
            ],
        ))
    return nodes


def generate_launch_description():
    solvers_arg = DeclareLaunchArgument(
        "solvers",
        default_value="iterative_ekf,iterative_rls,iterative_gd",
        description="Comma-separated solver names (or JSON list).",
    )
    odom_arg = DeclareLaunchArgument(
        "odom_topic",
        default_value=SIM_ODOM_TOPIC,
        description="Topic carrying nav_msgs/Odometry in the NED world frame.",
    )
    ping_arg = DeclareLaunchArgument(
        "ping_topic",
        default_value=SIM_PING_TOPIC,
        description="Topic carrying bb_sensor_msgs/Ping (body-relative DOA).",
    )
    world_frame_arg = DeclareLaunchArgument(
        "world_frame",
        default_value="odom_ned",
        description="Frame name the pinger estimate is expressed in.",
    )

    dynamic_nodes = OpaqueFunction(function=_create_nodes)

    return LaunchDescription([
        solvers_arg,
        odom_arg,
        ping_arg,
        world_frame_arg,
        dynamic_nodes,
    ])
