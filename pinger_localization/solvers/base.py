"""
Base solver class and shared utilities for pinger localization.

Provides the common interface that all solver nodes inherit:
- Odom history buffer
- World-bearing conversion (body-relative DOA → world frame)
- Expected bearing computation
- Angular error (shortest signed difference) helper
"""

import math
from collections import deque
from typing import Optional, Tuple

import numpy as np
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
# Local implementation of euler_from_quaternion to avoid depending on
# tf_transformations (which may not be available in all environments).
def _euler_from_quaternion(qx, qy, qz, qw):
    """Return (roll, pitch, yaw) from a quaternion [x, y, z, w].

    Uses the sxyz (static x-y-z) convention, matching tf_transformations.
    """
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap an angle in degrees to [-180, 180)."""
    return ((angle_deg + 180.0) % 360.0) - 180.0


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle in radians to [-pi, pi)."""
    return ((angle_rad + math.pi) % (2.0 * math.pi)) - math.pi


def wrap_angle_0_360(angle_deg: float) -> float:
    """Wrap an angle in degrees to [0, 360)."""
    return angle_deg % 360.0


def angular_error_deg(measured: float, expected: float) -> float:
    """Shortest signed angular difference in degrees: measured - expected, wrapped to [-180, 180)."""
    return wrap_angle_deg(measured - expected)


def yaw_from_odom(msg: Odometry) -> float:
    """Extract yaw (radians) from a nav_msgs/Odometry message.

    Assumes the orientation quaternion uses the ENU→NED conversion convention
    from the existing codebase (enu_ned_odom_repub.py): the NED yaw is derived
    from the published quaternion.
    """
    q = msg.pose.pose.orientation
    _, _, yaw = _euler_from_quaternion(q.x, q.y, q.z, q.w)
    return yaw


def compute_world_bearing(vehicle_north: float, vehicle_east: float,
                          pinger_north: float, pinger_east: float) -> float:
    """Compute the world-frame bearing (radians) from vehicle to pinger.

    Returns bearing in radians, 0 = North, positive = East (clockwise in NED).
    """
    return math.atan2(pinger_east - vehicle_east, pinger_north - vehicle_north)


def compute_body_doa(vehicle_north: float, vehicle_east: float, vehicle_yaw: float,
                     pinger_north: float, pinger_east: float) -> float:
    """Compute body-relative DOA (radians) from vehicle to pinger.

    0 = forward (ahead of vehicle), positive = starboard.
    """
    world_bearing = compute_world_bearing(vehicle_north, vehicle_east,
                                          pinger_north, pinger_east)
    return wrap_angle_rad(world_bearing - vehicle_yaw)


def triangulate_from_bearings(measurements) -> Optional[Tuple[float, float]]:
    """Roughly triangulate a 2D position from bearing measurements.

    Each measurement is a tuple (vehicle_north, vehicle_east, world_bearing_rad).
    Returns (north, east) or None if triangulation fails.

    Uses linear least-squares: each bearing gives a constraint
        (east - ve) * cos(b) - (north - vn) * sin(b) = 0
    →  -sin(b)*north + cos(b)*east = -sin(b)*vn + cos(b)*ve

    Stack as A * [north, east]^T = b and solve.
    """
    if len(measurements) < 2:
        return None

    A = []
    B = []
    for vn, ve, bearing in measurements:
        s = math.sin(bearing)
        c = math.cos(bearing)
        A.append([-s, c])
        B.append(-s * vn + c * ve)

    A = np.array(A)
    B = np.array(B)

    try:
        result, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        return float(result[0]), float(result[1])
    except np.linalg.LinAlgError:
        return None


class BaseSolver(Node):
    """Abstract base class for pinger localization solvers.

    Provides:
    - Odom history buffer keyed by timestamp
    - World-bearing conversion
    - Expected DOA computation
    - Estimate publishing
    """

    def __init__(self, solver_name: str):
        """
        Args:
            solver_name: Short name for this solver (used in topic names and logging).
        """
        super().__init__(f"pinger_{solver_name}")
        self._solver_name = solver_name

        # === Parameters ===
        self.odom_history_size = (
            self.declare_parameter("odom_history_size", 1000)
            .get_parameter_value()
            .integer_value
        )
        # Depth (m, NED) at which the visualization pose is rendered underwater.
        self.estimate_depth = (
            self.declare_parameter("estimate_depth", 1.0)
            .get_parameter_value()
            .double_value
        )
        # Frame in which the pinger estimate is expressed. Defaults to the
        # simulator's odom/world frame; override to your real-world frame name
        # (e.g. "world", "map") at launch time.
        self._world_frame = (
            self.declare_parameter("world_frame", "odom_ned")
            .get_parameter_value()
            .string_value
        )
        # =================

        # Odom history: deque of (stamp_ns, north, east, yaw_rad).
        self._odom_history: deque = deque(maxlen=self.odom_history_size)

        # Latest vehicle state (for quick access).
        self._latest_north: float = 0.0
        self._latest_east: float = 0.0
        self._latest_yaw: float = 0.0
        self._latest_stamp_ns: int = 0

        # Subscribers.
        self._odom_sub = self.create_subscription(
            Odometry, "/auv5/odom_ned", self._odom_callback, 10
        )

        # Publisher for estimate.
        self._est_pub = self.create_publisher(
            PointStamped,
            f"/pinger_localization/{solver_name}/estimate",
            10,
        )

        # Publisher for estimate pose (visualization).
        self._est_pose_pub = self.create_publisher(
            PoseStamped,
            f"/pinger_localization/{solver_name}/estimate_pose",
            10,
        )

        self.get_logger().info(f"Solver '{solver_name}' initialized")

    # ------------------------------------------------------------------
    # Public helpers for subclasses
    # ------------------------------------------------------------------

    @property
    def solver_name(self) -> str:
        return self._solver_name

    def get_vehicle_state_at(self, stamp_ns: int) -> Optional[Tuple[float, float, float]]:
        """Find the vehicle (north, east, yaw_rad) closest to the given timestamp.

        Returns None if no odometry has been received yet.
        """
        if not self._odom_history:
            return None
        # Linear search for closest timestamp (odom is usually small and sorted).
        best = None
        best_diff = float("inf")
        for sns, n, e, y in self._odom_history:
            diff = abs(sns - stamp_ns)
            if diff < best_diff:
                best_diff = diff
                best = (n, e, y)
        return best

    def get_latest_vehicle_state(self) -> Tuple[float, float, float, int]:
        """Return (north, east, yaw_rad, stamp_ns) of the most recent odometry."""
        return (self._latest_north, self._latest_east,
                self._latest_yaw, self._latest_stamp_ns)

    @staticmethod
    def world_bearing_from_doa(doa_body_deg: float, vehicle_yaw_rad: float) -> float:
        """Convert body-relative DOA (degrees) to world-frame bearing (radians).

        Incoming convention: 0..360 degrees, 0 = forward, increasing clockwise
        (90 = starboard/right, 180 = behind, 270 = port/left).

        world_bearing = wrap(vehicle_yaw + doa_body).
        """
        # Normalize the reading into [-180, 180) so any circular input
        # (including 0..360 or noise spill) is handled uniformly.
        doa_deg = wrap_angle_deg(doa_body_deg)
        doa_rad = math.radians(doa_deg)
        return wrap_angle_rad(vehicle_yaw_rad + doa_rad)

    @staticmethod
    def compute_expected_bearing(vehicle_north: float, vehicle_east: float,
                                 vehicle_yaw: float, pinger_north: float,
                                 pinger_east: float) -> float:
        """Compute the predicted body-relative DOA (radians) given a pinger hypothesis."""
        world_b = compute_world_bearing(vehicle_north, vehicle_east,
                                        pinger_north, pinger_east)
        return wrap_angle_rad(world_b - vehicle_yaw)

    def publish_estimate(self, north: float, east: float, stamp=None):
        """Publish a PointStamped estimate and a PoseStamped for visualization."""
        msg = PointStamped()
        if stamp is not None:
            msg.header.stamp = stamp
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._world_frame
        msg.point.x = float(north)
        msg.point.y = float(east)
        msg.point.z = 0.0
        self._est_pub.publish(msg)

        # Pose output (visualization): same world-frame position, but at a fixed
        # depth below the surface so it renders underwater in RViz.
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = float(north)
        pose.pose.position.y = float(east)
        pose.pose.position.z = self.estimate_depth
        pose.pose.orientation.w = 1.0
        self._est_pose_pub.publish(pose)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _odom_callback(self, msg: Odometry):
        """Buffer odometry for later lookup."""
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        north = msg.pose.pose.position.x
        east = msg.pose.pose.position.y
        yaw = yaw_from_odom(msg)

        self._odom_history.append((stamp_ns, north, east, yaw))
        self._latest_north = north
        self._latest_east = east
        self._latest_yaw = yaw
        self._latest_stamp_ns = stamp_ns
