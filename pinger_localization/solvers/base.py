"""
Base solver class and shared utilities for pinger localization.

Provides the common interface that all solver nodes inherit:
- Odom history buffer
- World-bearing conversion (body-relative DOA → world frame)
- Expected bearing computation
- Angular error (shortest signed difference) helper
"""

import math
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node


# Watchdog period (s) at which a solver re-reports an input that has produced
# nothing yet -- see BaseSolver._health_cb.
HEALTH_PERIOD_S = 5.0


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

        # Diagnostics: input counters, one-shot latches, and one throttle
        # budget per distinct message (see _throttled).  Everything a solver
        # silently gives up on is reported through these.  Named _pings_seen
        # (not _ping_count) because batch_full already uses _ping_count for its
        # own solve cadence.
        self._odom_count: int = 0
        self._pings_seen: int = 0
        self._odom_frame: str = ""
        self._logged_first_odom: bool = False
        self._logged_first_publish: bool = False
        self._throttle_last: dict = {}

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

        # Startup banner names the RESOLVED topics (topic_name is the name after
        # remapping), because a topic that was remapped elsewhere -- or simply
        # misnamed -- is the usual reason a solver never publishes anything.
        self.get_logger().info(
            f"Solver '{solver_name}' initialized: "
            f"odom<-'{self._odom_sub.topic_name}' "
            f"estimate->'{self._est_pub.topic_name}' "
            f"pose->'{self._est_pose_pub.topic_name}' "
            f"world_frame='{self._world_frame}' "
            f"odom_history_size={self.odom_history_size} "
            f"estimate_depth={self.estimate_depth}"
        )

        # Input watchdog: a callback cannot report an input that never arrives,
        # so this timer is the only thing that can tell the user why a running
        # solver is quiet.
        self._health_timer = self.create_timer(HEALTH_PERIOD_S, self._health_cb)

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

        if not self._logged_first_publish:
            # Confirms input -> estimate -> output topic end to end at least
            # once; after this each solver logs its own estimate values.
            self._logged_first_publish = True
            self.get_logger().info(
                f"first estimate published: ({north:.2f}, {east:.2f}) m in frame "
                f"'{self._world_frame}' on '{self._est_pub.topic_name}', pose on "
                f"'{self._est_pose_pub.topic_name}' at z={self.estimate_depth}"
            )

    # ------------------------------------------------------------------
    # Diagnostics (why is nothing being published?)
    # ------------------------------------------------------------------

    def _throttled(self, level: str, key: str, msg: str, every_s: float = 2.0):
        """Log `msg` at most once per `every_s`, with one budget per `key`.

        Each distinct message gets its own timer: a single shared budget would
        let a frequently-firing message suppress a rare one."""
        now = time.monotonic()
        if now - self._throttle_last.get(key, 0.0) >= every_s:
            self._throttle_last[key] = now
            getattr(self.get_logger(), level)(msg)

    def _ping_topic_name(self) -> str:
        """Resolved name of the ping subscription the subclass created.

        Every solver adds exactly one non-odom subscription (its Ping topic),
        so the first such subscription is it."""
        for sub in self.subscriptions:
            if sub is not self._odom_sub:
                return sub.topic_name
        return "<none>"

    def vehicle_state_for_ping(self, msg):
        """Vehicle (north, east, yaw_rad, stamp_ns) at this ping, or None to drop it.

        Every solver opens its ping callback with this call, because the two
        ways a ping gets silently discarded are both invisible from outside:

        * `msg.doa_deg` is not finite -- NaN/inf would quietly poison the state
          of any solver that folded it in; and
        * no usable odometry has been buffered, so there is no vehicle pose to
          convert the DOA against.

        Whichever fires is logged here (throttled) and re-reported by
        `_health_cb` until it stops happening, so a silent solver always has a
        stated reason.  `msg` is a bb_sensor_msgs/Ping."""
        self._pings_seen += 1

        if not math.isfinite(msg.doa_deg):
            self._throttled("warn", "ping_bad_doa",
                            f"dropping ping: doa_deg={msg.doa_deg!r} is not finite")
            return None

        vn, ve, yaw, stamp_ns = self.get_latest_vehicle_state()
        if stamp_ns == 0:
            self.warn_ping_without_odom()
            return None
        return vn, ve, yaw, stamp_ns

    def warn_ping_without_odom(self):
        """Explain dropping a ping for want of odometry (throttled)."""
        if self._odom_count == 0:
            self._throttled(
                "warn", "ping_no_odom",
                f"ping received but no odometry buffered yet on "
                f"'{self._odom_sub.topic_name}' -- dropping.  Check that the "
                f"topic is actually published and remapped (pinger_irl.launch.py "
                f"remaps /auv5/odom_ned onto the odom_topic argument).")
        else:
            # Distinct from "no odom": odometry IS arriving, but stamped 0 --
            # which this class uses as its "no odometry yet" sentinel, so every
            # ping is dropped forever and nothing is ever published.
            self._throttled(
                "error", "ping_zero_stamp_odom",
                f"odometry is arriving ({self._odom_count} msgs, frame "
                f"'{self._odom_frame}') but its header.stamp is 0, which this "
                f"solver uses as the 'no odometry yet' sentinel -- every ping is "
                f"dropped and no estimate can ever be published.  Publish real "
                f"stamps on '{self._odom_sub.topic_name}'.")

    def _health_cb(self):
        """Report inputs that have still produced nothing (silent once they flow).

        Logging from the callbacks cannot cover this: the failure being
        diagnosed is precisely that no message ever arrives."""
        if self._odom_count > 0 and self._pings_seen > 0:
            return
        self.get_logger().warn(
            f"waiting for input: odometry {self._odom_count} msgs on "
            f"'{self._odom_sub.topic_name}', pings {self._pings_seen} msgs on "
            f"'{self._ping_topic_name()}'")

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

        self._odom_count += 1
        if not self._odom_frame:
            self._odom_frame = msg.header.frame_id
        if not self._logged_first_odom:
            # One line showing the pose convention actually in use: a wrong yaw
            # or a frame that disagrees with `world_frame` makes every estimate
            # wrong in a way the estimate log alone cannot reveal.
            self._logged_first_odom = True
            q = msg.pose.pose.orientation
            self.get_logger().info(
                f"first odometry on '{self._odom_sub.topic_name}': "
                f"stamp={stamp_ns} frame='{msg.header.frame_id}' "
                f"child_frame='{msg.child_frame_id}' "
                f"north={north:.2f} east={east:.2f} "
                f"yaw={math.degrees(yaw):.1f}deg "
                f"(quat x={q.x:.4f} y={q.y:.4f} z={q.z:.4f} w={q.w:.4f}); "
                f"world_frame param='{self._world_frame}'"
            )
