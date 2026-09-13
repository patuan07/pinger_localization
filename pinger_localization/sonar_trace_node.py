#!/usr/bin/python3
"""
Live seed-traced sonar pipe model node.

Wires the offline seed-traced pipe detector to the live sensors: subscribes to
the Oculus sonar point cloud, seeds the detector, traces the pipe, segments the
dense trace into an n-segment polyline, and publishes the model's VERTICES as a
PoseArray in the sonar frame.

The seed comes from one of two selectable sources (parameter `seed_source`):

* "estimate" (default) -- a solver's world-NED pinger estimate
  (geometry_msgs/PointStamped) plus odometry, mapped world -> sonar at the
  cloud's own timestamp.
* "sonar_point" -- a geometry_msgs/PointStamped clicked in the Foxglove 3D panel
  and already expressed in the sonar frame (its z is 0 by construction and is
  ignored).  No odometry and no solver are needed; the latest click persists
  until the next one.

Inputs
------
* /oculus/pointcloud (sensor_msgs/PointCloud2, frame auv5/sonar) -- the sonar
  return. x = range*cos(bearing), y = -range*sin(bearing), z = 0, intensity.
* "estimate" mode only: an estimate topic (geometry_msgs/PointStamped) -- one
  solver's world-NED pinger estimate, e.g.
  /pinger_localization/iterative_ekf/estimate -- plus an odometry topic
  (nav_msgs/Odometry) -- vehicle pose in the NED world frame, so the seed can be
  mapped world -> sonar at each cloud's own timestamp.
* "sonar_point" mode only: a seed topic (geometry_msgs/PointStamped) carrying a
  point in the sonar frame, e.g. clicked in the Foxglove 3D panel.

Outputs
-------
* /pinger_localization/sonar_trace/seed_used (geometry_msgs/PointStamped, frame
  auv5/sonar) -- the seed this node actually used, published per cloud *before*
  the trace so a mis-placed or mis-framed seed is visible.
* /pinger_localization/sonar_trace/vertices (geometry_msgs/PoseArray, frame
  auv5/sonar) -- one pose per vertex of the segmented pipe model, outward from
  the seed, z = 0. Default n_segments=3 => 4 poses (= n_segments+1 vertices).
"""

import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PointStamped, Pose, PoseArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField

from pinger_localization.solvers.base import yaw_from_odom
from pinger_localization.sonar.detector import detect
from pinger_localization.sonar.segments import seg_cost_table, fit_from_cost
from pinger_localization.sonar.geometry import seed_in_sonar

# ============================================================================
# module-level constants (parameters override these at construction)
# ============================================================================
POINTCLOUD_TOPIC = "/oculus/pointcloud"
ESTIMATE_TOPIC = "/pinger_localization/iterative_ekf/estimate"
ODOM_TOPIC = "/auv5/odom_ned"          # real bag: /auv5/nav/odom_ned
SONAR_SEED_TOPIC = "/pinger_localization/sonar_trace/seed"
SEED_USED_TOPIC = "/pinger_localization/sonar_trace/seed_used"
SEED_SOURCE = "estimate"               # "estimate" | "sonar_point"
SONAR_FRAME = "auv5/sonar"
MOUNT_FORWARD_M = 0.29442               # base_link -> auv5/sonar x offset (bag)
MOUNT_LATERAL_M = 0.0
MIN_CLOUD_POINTS = 100
MAX_ODOM_AGE_S = 0.5                    # 0 = off
MAX_ESTIMATE_AGE_S = 5.0                # 0 = off
CHECK_ESTIMATE_FRAME = True             # warn-only
PUBLISH_FAILURES = False
CLOUD_STRIDE = 1
ODOM_HISTORY_SIZE = 1000
N_SEGMENTS = 3                          # n_segments => n_segments+1 poses
SEG_ROBUST_TRIM = 0.15                  # 0 = raw least-squares fit
# ============================================================================


# ----------------------------------------------------------------------------
# PointCloud2 -> float64 (N,4) [x, y, z, intensity]  (manual parser)
# ----------------------------------------------------------------------------
_PF_DT = {
    PointField.INT8: np.int8, PointField.UINT8: np.uint8,
    PointField.INT16: np.int16, PointField.UINT16: np.uint16,
    PointField.INT32: np.int32, PointField.UINT32: np.uint32,
    PointField.FLOAT32: np.float32, PointField.FLOAT64: np.float64,
}


def _pointcloud2_xyzi(msg: PointCloud2):
    """Parse a PointCloud2 into a float64 (N,4) [x, y, z, intensity] array.

    No sensor_msgs_py dependency: raw byte parsing with numpy.  Drops rows with
    any non-finite value (mirrors the experiment's skip_nans), so the returned
    array is a clean input for the range-normalization step."""
    npts = msg.width * msg.height if msg.height > 1 else msg.width
    if npts == 0 or msg.point_step < 16:
        return np.zeros((0, 4))

    names = [f.name for f in msg.fields]
    if (names == ["x", "y", "z", "intensity"]
            and msg.point_step == 16 and not msg.is_bigendian
            and all(f.datatype == PointField.FLOAT32 and f.count == 1
                    and f.offset == i * 4 for i, f in enumerate(msg.fields))):
        # Fast path: tight little-endian float32 xyzi (the sonar driver layout).
        data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
    else:
        cols = {}
        for f in msg.fields:
            dt = _PF_DT.get(f.datatype)
            if dt is None or f.name not in ("x", "y", "z", "intensity"):
                continue
            cols[f.name] = (f.offset, np.dtype((dt, f.count) if f.count > 1 else dt))
        if not {"x", "y"} <= set(cols):
            return np.zeros((0, 4))
        base = np.frombuffer(msg.data, dtype=np.uint8)  # entire byte buffer
        order = "<" if not msg.is_bigendian else ">"
        data = np.zeros((npts, 4))
        for nm in ("x", "y", "z", "intensity"):
            if nm not in cols:
                continue
            off, dt = cols[nm]
            arr = np.frombuffer(base, dtype=np.dtype(order + str(dt)),
                                count=npts, offset=off)
            data[:, ["x", "y", "z", "intensity"].index(nm)] = arr

    data = data.astype(np.float64)
    return data[np.isfinite(data).all(axis=1)]


def _norm_frame(frame_id: str) -> str:
    """Frame names compare equal regardless of a leading '/'.

    Foxglove and `ros2 topic pub` commonly send '/auv5/sonar' where the
    tf tree (and this package's `sonar_frame` parameter) uses 'auv5/sonar'."""
    return frame_id[1:] if frame_id.startswith("/") else frame_id


class PingerSonarTrace(Node):
    """Seed the sonar pipe detector; publish the seed used and the model vertices."""

    def __init__(self):
        super().__init__("pinger_sonar_trace")

        def _d(name, default, cvt):
            return cvt(self.declare_parameter(name, default).get_parameter_value())

        self.pointcloud_topic = _d("pointcloud_topic", POINTCLOUD_TOPIC, lambda v: v.string_value)
        self.estimate_topic = _d("estimate_topic", ESTIMATE_TOPIC, lambda v: v.string_value)
        self.odom_topic = _d("odom_topic", ODOM_TOPIC, lambda v: v.string_value)
        self.sonar_seed_topic = _d("sonar_seed_topic", SONAR_SEED_TOPIC, lambda v: v.string_value)
        self.seed_source = _d("seed_source", SEED_SOURCE, lambda v: v.string_value)
        if self.seed_source not in ("estimate", "sonar_point"):
            self.get_logger().warn(
                f"seed_source '{self.seed_source}' not recognised (expected "
                f"'estimate' or 'sonar_point') -- falling back to 'estimate'")
            self.seed_source = "estimate"
        self.sonar_frame = _d("sonar_frame", SONAR_FRAME, lambda v: v.string_value)
        self._body_frame = _d("body_frame", "auv5/base_link_ned", lambda v: v.string_value)
        self.mount_forward_m = _d("sonar_mount_forward_m", MOUNT_FORWARD_M, lambda v: v.double_value)
        self.mount_lateral_m = _d("sonar_mount_lateral_m", MOUNT_LATERAL_M, lambda v: v.double_value)
        self.min_cloud_points = _d("min_cloud_points", MIN_CLOUD_POINTS, lambda v: v.integer_value)
        self.max_odom_age_s = _d("max_odom_age_s", MAX_ODOM_AGE_S, lambda v: v.double_value)
        self.max_estimate_age_s = _d("max_estimate_age_s", MAX_ESTIMATE_AGE_S, lambda v: v.double_value)
        self.check_estimate_frame = _d("check_estimate_frame", CHECK_ESTIMATE_FRAME, lambda v: v.bool_value)
        self.publish_failures = _d("publish_failures", PUBLISH_FAILURES, lambda v: v.bool_value)
        self.cloud_stride = _d("cloud_stride", CLOUD_STRIDE, lambda v: v.integer_value)
        self.odom_history_size = _d("odom_history_size", ODOM_HISTORY_SIZE, lambda v: v.integer_value)
        self.n_segments = _d("n_segments", N_SEGMENTS, lambda v: v.integer_value)
        self.seg_robust_trim = _d("seg_robust_trim", SEG_ROBUST_TRIM, lambda v: v.double_value)
        self.n_segments = max(self.n_segments, 1)

        # Odom history: deque of (stamp_ns, north, east, yaw_rad).
        self._odom_history: deque = deque(maxlen=self.odom_history_size)
        self._odom_frame = None
        self._estimate = None            # (stamp_ns, x, y, frame_id) or None
        self._sonar_seed = None          # (stamp_ns, x, y, frame_id) or None

        self._est_pub = self.create_publisher(
            PoseArray, "/pinger_localization/sonar_trace/vertices", 10
        )
        self._seed_pub = self.create_publisher(
            PointStamped, SEED_USED_TOPIC, 10
        )

        # Cloud is sensor data from a real driver: subscribe best-effort so we
        # also work with a best-effort publisher.  Odom queue is deep so the
        # slow cloud callback cannot starve it.
        self._cloud_sub = self.create_subscription(
            PointCloud2, self.pointcloud_topic, self._cloud_cb,
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST, depth=5),
        )
        # Only the selected seed source is subscribed: in sonar_point mode there
        # is no solver and no odometry to wait for.
        self._est_sub = None
        self._odom_sub = None
        self._seed_sub = None
        if self.seed_source == "sonar_point":
            self._seed_sub = self.create_subscription(
                PointStamped, self.sonar_seed_topic, self._sonar_seed_cb, 10
            )
        else:
            self._est_sub = self.create_subscription(
                PointStamped, self.estimate_topic, self._estimate_cb, 10
            )
            self._odom_sub = self.create_subscription(
                Odometry, self.odom_topic, self._odom_cb, 50
            )

        self._warned_no_estimate = False
        self._warned_no_seed = False
        self._last_throttled = 0.0
        self.get_logger().info(
            f"sonar trace node up: cloud={self.pointcloud_topic} "
            f"seed_source={self.seed_source} "
            + (f"seed={self.sonar_seed_topic} (manual, sonar frame)"
               if self.seed_source == "sonar_point"
               else f"estimate={self.estimate_topic} odom={self.odom_topic}")
            + f" seed_used={SEED_USED_TOPIC} "
            f"n_segments={self.n_segments} (-> {self.n_segments + 1} poses) "
            f"seg_robust_trim={self.seg_robust_trim}"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _throttled(self, level, msg, every_s=2.0):
        now = time.monotonic()
        if now - self._last_throttled >= every_s:
            self._last_throttled = now
            getattr(self.get_logger(), level)(msg)

    def _vehicle_at(self, stamp_ns):
        """(north, east, yaw_rad, matched_stamp_ns) closest to `stamp_ns`, or None."""
        if not self._odom_history:
            return None
        best = min(self._odom_history, key=lambda t: abs(t[0] - stamp_ns))
        return best

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        north = msg.pose.pose.position.x
        east = msg.pose.pose.position.y
        yaw = yaw_from_odom(msg)
        self._odom_history.append((stamp_ns, north, east, yaw))
        if self._odom_frame is None:
            self._odom_frame = msg.header.frame_id

    def _estimate_cb(self, msg: PointStamped):
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self._estimate = (stamp_ns, msg.point.x, msg.point.y, msg.header.frame_id)

    def _sonar_seed_cb(self, msg: PointStamped):
        """Manual seed clicked in the Foxglove 3D panel, in the sonar frame.

        Only x/y are used -- the sonar frame is its z=0 plane, so a click's z
        carries no information.  The latest click wins and persists until the
        next one: a manual seed deliberately has no age limit, because the
        operator re-clicks whenever the seed needs correcting."""
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self._sonar_seed = (stamp_ns, msg.point.x, msg.point.y, msg.header.frame_id)
        if (msg.header.frame_id
                and _norm_frame(msg.header.frame_id) != _norm_frame(self.sonar_frame)):
            self._throttled(
                "warn",
                f"seed frame '{msg.header.frame_id}' != sonar frame "
                f"'{self.sonar_frame}' -- using the point as sonar coordinates "
                f"anyway (set the 3D panel's frame to '{self.sonar_frame}')")

    def _seed_from_estimate(self, cloud_stamp_ns):
        """World-NED estimate -> sonar seed at the cloud's timestamp, or None.

        Logs (throttled) the reason whenever it returns None, so a silent
        no-output stream always has an explanation."""
        if self._estimate is None:
            if not self._warned_no_estimate:
                self._warned_no_estimate = True
                self.get_logger().warn(
                    f"no estimate yet on '{self.estimate_topic}' -- waiting")
            return None
        est_stamp_ns, est_x, est_y, est_frame = self._estimate
        if self.max_estimate_age_s > 0:
            age_s = abs(cloud_stamp_ns - est_stamp_ns) / 1e9
            if age_s > self.max_estimate_age_s:
                self._throttled("warn",
                                f"estimate stale by {age_s:.1f}s "
                                f"(max {self.max_estimate_age_s}s) -- skipping")
                return None
        if (self.check_estimate_frame and self._odom_frame is not None
                and est_frame and self._odom_frame != est_frame):
            self._throttled(
                "warn",
                f"estimate frame '{est_frame}' != odom frame '{self._odom_frame}' -- "
                f"seed assumes both are the same NED world frame")

        # --- odometry at cloud time -------------------------------------------
        veh = self._vehicle_at(cloud_stamp_ns)
        if veh is None:
            self._throttled("warn", "no odometry received yet -- skipping")
            return None
        odom_stamp_ns, vn, ve, yaw = veh
        if self.max_odom_age_s > 0:
            odom_age_s = abs(cloud_stamp_ns - odom_stamp_ns) / 1e9
            if odom_age_s > self.max_odom_age_s:
                self._throttled(
                    "warn",
                    f"odom {odom_age_s * 1e3:.0f} ms away from cloud stamp "
                    f"(max {self.max_odom_age_s}s) -- skipping")
                return None

        # --- map the world estimate into sonar data coords at this instant ----
        return seed_in_sonar(est_x, est_y, vn, ve, yaw,
                             self.mount_forward_m, self.mount_lateral_m)

    def _cloud_cb(self, msg: PointCloud2):
        cloud_stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        cloud_frame = msg.header.frame_id or self.sonar_frame

        arr = _pointcloud2_xyzi(msg)
        if self.cloud_stride > 1:
            arr = arr[:: self.cloud_stride]
        if len(arr) < self.min_cloud_points:
            self._throttled("warn",
                            f"cloud too small ({len(arr)} pts < {self.min_cloud_points})")
            return

        # --- seed -------------------------------------------------------------
        if self.seed_source == "sonar_point":
            # Manual seed from the 3D panel: already sonar-frame coordinates, so
            # no odometry, no estimate and no age check are involved.
            if self._sonar_seed is None:
                if not self._warned_no_seed:
                    self._warned_no_seed = True
                    self.get_logger().warn(
                        f"no seed yet on '{self.sonar_seed_topic}' -- click a "
                        f"point in the 3D panel (frame '{self.sonar_frame}')")
                return
            seed = (self._sonar_seed[1], self._sonar_seed[2])
        else:
            seed = self._seed_from_estimate(cloud_stamp_ns)
            if seed is None:
                return

        # Publish the seed actually used *before* tracing: when the trace fails
        # or the seed lands outside the ROI, this is what shows where we seeded.
        seed_msg = PointStamped()
        seed_msg.header.stamp = msg.header.stamp
        seed_msg.header.frame_id = cloud_frame
        seed_msg.point.x = float(seed[0])
        seed_msg.point.y = float(seed[1])
        seed_msg.point.z = 0.0
        self._seed_pub.publish(seed_msg)

        # --- trace + segment --------------------------------------------------
        chain, poly, info, _scorer = detect(arr, np.asarray(seed))
        if info['status'] != 'ok' or poly is None or len(poly) < 2:
            if self.publish_failures:
                empty = PoseArray()
                empty.header.stamp = msg.header.stamp
                empty.header.frame_id = cloud_frame
                self._est_pub.publish(empty)
            self._throttled("info",
                            f"trace status={info['status']} seed=({seed[0]:.2f},"
                            f"{seed[1]:.2f}) -- no model published")
            return

        nseg = min(self.n_segments, len(poly) - 1)
        if nseg < self.n_segments:
            self._throttled(
                "warn",
                f"clamped n_segments {self.n_segments} -> {nseg} "
                f"(trace only {len(poly)} samples)")
        C = seg_cost_table(poly, robust_trim=self.seg_robust_trim)
        model, _idx = fit_from_cost(poly, C, nseg)

        pa = PoseArray()
        pa.header.stamp = msg.header.stamp
        pa.header.frame_id = cloud_frame
        for i in range(len(model)):
            pose = Pose()
            pose.position.x = float(model[i, 0])
            pose.position.y = float(model[i, 1])
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        self._est_pub.publish(pa)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PingerSonarTrace()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node is not None:
            node.get_logger().fatal(f"sonar trace node crashed: {e}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
