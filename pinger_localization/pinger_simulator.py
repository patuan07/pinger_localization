#!/usr/bin/python3
"""
Pinger Localization Simulator Node.

Simulates an AUV moving along a trajectory, publishing odometry in NED and
body-relative DOA pings with configurable noise models. Also publishes ground
truth pinger position for evaluation.
"""

import math
import random
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, Pose, Twist, Vector3, Quaternion
from bb_sensor_msgs.msg import Ping
def _quaternion_from_euler(roll, pitch, yaw):
    """Return [x, y, z, w] quaternion from Euler angles (radians)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qx, qy, qz, qw]

from pinger_localization.sim.trajectory import create_trajectory
from pinger_localization.solvers.base import wrap_angle_0_360

# === Parameters ===
PINGER_NORTH = 10.0           # True pinger North (m, NED)
PINGER_EAST = 5.0             # True pinger East (m, NED)
NOISE_RADIUS = 0.5            # Jitter radius around true pinger position (m)
NOISE_TYPE = "jitter"         # "jitter", "gaussian", or "both"
DOA_NOISE_STD = 3.0           # Std dev for Gaussian DOA noise (degrees)
TRAJECTORY_TYPE = "circle"    # circle, lawnmower, figure8, spiral
TRAJECTORY_RADIUS = 10.0      # Approximate distance from pinger (m)
TRAJECTORY_SPEED = 1.0        # Vehicle speed (m/s)
DEPTH = -2.0                  # Constant vehicle depth in NED (m)
PING_INTERVAL = 2.0           # Seconds between pings
ODOM_RATE = 20.0              # Odom publish rate (Hz)
# =================


class PingerSimulator(Node):
    """ROS2 Node that simulates vehicle motion and acoustic pinger measurements."""

    def __init__(self):
        super().__init__("pinger_simulator")

        # Read parameters (ROS2 params override module-level defaults).
        self.pinger_north = (
            self.declare_parameter("pinger_north", PINGER_NORTH)
            .get_parameter_value()
            .double_value
        )
        self.pinger_east = (
            self.declare_parameter("pinger_east", PINGER_EAST)
            .get_parameter_value()
            .double_value
        )
        self.noise_radius = (
            self.declare_parameter("noise_radius", NOISE_RADIUS)
            .get_parameter_value()
            .double_value
        )
        self.noise_type = (
            self.declare_parameter("noise_type", NOISE_TYPE)
            .get_parameter_value()
            .string_value
        )
        self.doa_noise_std = (
            self.declare_parameter("doa_noise_std", DOA_NOISE_STD)
            .get_parameter_value()
            .double_value
        )
        trajectory_type = (
            self.declare_parameter("trajectory_type", TRAJECTORY_TYPE)
            .get_parameter_value()
            .string_value
        )
        trajectory_radius = (
            self.declare_parameter("trajectory_radius", TRAJECTORY_RADIUS)
            .get_parameter_value()
            .double_value
        )
        trajectory_speed = (
            self.declare_parameter("trajectory_speed", TRAJECTORY_SPEED)
            .get_parameter_value()
            .double_value
        )
        self.depth = (
            self.declare_parameter("depth", DEPTH)
            .get_parameter_value()
            .double_value
        )
        self.ping_interval = (
            self.declare_parameter("ping_interval", PING_INTERVAL)
            .get_parameter_value()
            .double_value
        )
        odom_rate = (
            self.declare_parameter("odom_rate", ODOM_RATE)
            .get_parameter_value()
            .double_value
        )
        # Frame in which odom, ground truth, and pinger positions are expressed.
        # Override to your real-world frame name (e.g. "world") at launch time.
        self._world_frame = (
            self.declare_parameter("world_frame", "odom_ned")
            .get_parameter_value()
            .string_value
        )

        # Create trajectory generator.
        self._trajectory = create_trajectory(
            trajectory_type, self.pinger_north, self.pinger_east,
            trajectory_radius, trajectory_speed,
        )

        # Publishers.
        self._odom_pub = self.create_publisher(Odometry, "/auv5/odom_ned", 10)
        self._ping_pub = self.create_publisher(Ping, "/sensors/ping", 10)
        self._gt_pub = self.create_publisher(
            PointStamped, "/pinger_localization/ground_truth", 10
        )

        # Timers.
        self._start_time = time.time()
        self._last_ping_time = -self.ping_interval  # Fire first ping immediately.
        self._odom_timer = self.create_timer(1.0 / odom_rate, self._odom_tick)
        self._ping_timer = self.create_timer(0.1, self._ping_tick)

        # Tracking.
        self._ping_count = 0

        self.get_logger().info(
            f"Simulator started: pinger=({self.pinger_north:.1f}, {self.pinger_east:.1f}), "
            f"trajectory={trajectory_type}, noise={self.noise_type}"
        )

    # ------------------------------------------------------------------
    # Noise helpers
    # ------------------------------------------------------------------

    def _jittered_pinger_position(self):
        """Return (north, east) of the pinger with position jitter.

        Uniform random sample within a 2D circle of radius `noise_radius`
        around the true pinger position.
        """
        # Rejection sampling in a circle.
        while True:
            dn = random.uniform(-self.noise_radius, self.noise_radius)
            de = random.uniform(-self.noise_radius, self.noise_radius)
            if dn * dn + de * de <= self.noise_radius * self.noise_radius:
                return self.pinger_north + dn, self.pinger_east + de

    # ------------------------------------------------------------------
    # Timer callbacks
    # ------------------------------------------------------------------

    def _odom_tick(self):
        """Publish odometry at the configured rate."""
        elapsed = time.time() - self._start_time
        north, east, yaw = self._trajectory.generate(elapsed)

        # Build Odometry message in NED frame.
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._world_frame
        msg.child_frame_id = "base_link"

        # Position: x=North, y=East, z=Down.
        msg.pose.pose.position = Pose().position
        msg.pose.pose.position.x = north
        msg.pose.pose.position.y = east
        msg.pose.pose.position.z = self.depth

        # Orientation: yaw about Down axis, NED convention.
        # In NED, a positive yaw rotates from North toward East.
        q = _quaternion_from_euler(0.0, 0.0, yaw)
        msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # Twist (linear velocity in world NED frame).
        msg.twist.twist.linear = Vector3()
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        speed = TRAJECTORY_SPEED
        msg.twist.twist.linear.x = speed * cos_yaw
        msg.twist.twist.linear.y = speed * sin_yaw
        msg.twist.twist.linear.z = 0.0

        self._odom_pub.publish(msg)

    def _ping_tick(self):
        """Check if it's time to publish a ping, and do so."""
        elapsed = time.time() - self._start_time
        if elapsed - self._last_ping_time < self.ping_interval:
            return
        self._last_ping_time = elapsed

        # Current vehicle state from trajectory.
        veh_north, veh_east, veh_yaw = self._trajectory.generate(elapsed)

        # Determine perceived pinger position with noise.
        if self.noise_type in ("jitter", "both"):
            pinger_north, pinger_east = self._jittered_pinger_position()
        else:
            pinger_north, pinger_east = self.pinger_north, self.pinger_east

        # Compute world-frame bearing from vehicle to (noisy) pinger.
        world_bearing_rad = math.atan2(
            pinger_east - veh_east, pinger_north - veh_north
        )

        # Convert to body-relative DOA in the sensor's 0..360 format:
        # 0 = forward, 90 = starboard (right), 180 = behind, 270 = port (left),
        # increasing clockwise from the vehicle (body = world - yaw).
        doa_body_deg = math.degrees(world_bearing_rad - veh_yaw)
        doa_body_deg = wrap_angle_0_360(doa_body_deg)

        # Apply Gaussian noise to DOA if configured, then keep within [0, 360).
        if self.noise_type in ("gaussian", "both"):
            doa_body_deg = wrap_angle_0_360(
                doa_body_deg + random.gauss(0.0, self.doa_noise_std)
            )

        # Round to integer as per the Ping message type (360 wraps back to 0).
        doa_deg = int(round(doa_body_deg)) % 360

        # Publish ping.
        ping = Ping()
        ping.doa_deg = doa_deg
        ping.elevation = 0
        ping.frequency = 25000
        ping.confidence = 1.0
        self._ping_pub.publish(ping)

        # Publish ground truth.
        gt = PointStamped()
        gt.header.stamp = self.get_clock().now().to_msg()
        gt.header.frame_id = self._world_frame
        gt.point.x = self.pinger_north
        gt.point.y = self.pinger_east
        gt.point.z = 0.0
        self._gt_pub.publish(gt)

        self._ping_count += 1
        self.get_logger().info(
            f"Ping #{self._ping_count}: doa={doa_deg}deg, "
            f"vehicle=({veh_north:.1f},{veh_east:.1f}), yaw={math.degrees(veh_yaw):.0f}deg"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PingerSimulator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node is not None:
            node.get_logger().fatal(f"Simulator crashed: {e}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
