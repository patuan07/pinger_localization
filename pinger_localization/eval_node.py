#!/usr/bin/python3
"""
Live Evaluation Node.

Subscribes to:
- /pinger_localization/ground_truth — true pinger position
- /pinger_localization/<solver_name>/estimate — from each solver

On each ground truth update, logs position error for each solver that has
published an estimate. Writes all data to a CSV log file.

Also publishes /pinger_localization/eval/summary as a string log.
"""

import csv
import os
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String

# === Parameters ===
LOG_DIR = os.path.expanduser("~/pinger_logs")
SOLVER_NAMES = [
    "batch_sliding",
    "batch_full",
    "particle_filter",
    "iterative_ekf",
    "iterative_rls",
    "iterative_gd",
]
# =================


class EvalNode(Node):
    """Live evaluation node for pinger localization solvers."""

    def __init__(self):
        super().__init__("pinger_eval")

        self.log_dir = (
            self.declare_parameter("log_dir", LOG_DIR)
            .get_parameter_value()
            .string_value
        )
        solver_names_param = (
            self.declare_parameter("solver_names", SOLVER_NAMES)
            .get_parameter_value()
            .string_array_value
        )
        self.solver_names = list(solver_names_param)

        os.makedirs(self.log_dir, exist_ok=True)

        # Per-solver tracking: latest estimate (north, east).
        self._estimates: dict[
            str, tuple[float, float, float]
        ] = {}  # name -> (n, e, stamp_ns)

        # Ground truth: (north, east, stamp_ns).
        self._gt_north = 0.0
        self._gt_east = 0.0
        self._gt_stamp_ns = 0

        # CSV log file.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._csv_path = os.path.join(self.log_dir, f"summary_{timestamp}.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        header = ["timestamp_ns", "gt_north", "gt_east"]
        for name in self.solver_names:
            header.extend([f"{name}_north", f"{name}_east", f"{name}_error_m"])
        self._csv_writer.writerow(header)

        # Subscribers.
        self._gt_sub = self.create_subscription(
            PointStamped,
            "/pinger_localization/ground_truth",
            self._gt_callback,
            10,
        )

        for name in self.solver_names:
            self.create_subscription(
                PointStamped,
                f"/pinger_localization/{name}/estimate",
                lambda msg, n=name: self._estimate_callback(msg, n),
                10,
            )

        # Publisher for text summary.
        self._summary_pub = self.create_publisher(
            String, "/pinger_localization/eval/summary", 10
        )

        self.get_logger().info(
            f"Eval node ready: watching {len(self.solver_names)} solvers, "
            f"logging to {self._csv_path}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _gt_callback(self, msg: PointStamped):
        """New ground truth — log errors for all solvers that have estimates."""
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self._gt_north = msg.point.x
        self._gt_east = msg.point.y
        self._gt_stamp_ns = stamp_ns

        # Build log row.
        row = [stamp_ns, self._gt_north, self._gt_east]
        status_lines = [
            f"[{stamp_ns / 1e9:.1f}s] Ground truth: "
            f"({self._gt_north:.2f}, {self._gt_east:.2f})"
        ]

        for name in self.solver_names:
            if name in self._estimates:
                en, ee, _ = self._estimates[name]
                err = ((en - self._gt_north) ** 2 + (ee - self._gt_east) ** 2) ** 0.5
                row.extend([en, ee, f"{err:.3f}"])
                status_lines.append(f"  {name}: err={err:.3f}m")
            else:
                row.extend(["", "", ""])

        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # Log to console.
        self.get_logger().info("\n".join(status_lines))

        # Publish summary string.
        summary = String()
        summary.data = "\n".join(status_lines)
        self._summary_pub.publish(summary)

    def _estimate_callback(self, msg: PointStamped, solver_name: str):
        """Cache latest estimate from a solver."""
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self._estimates[solver_name] = (msg.point.x, msg.point.y, stamp_ns)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self):
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = EvalNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node is not None:
            node.get_logger().fatal(f"Eval node crashed: {e}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
