#!/usr/bin/python3
"""
Batch Sliding-Window Solver.

Accumulates the last WINDOW_SIZE pings (or pings within WINDOW_SEC seconds)
and solves a nonlinear least-squares problem for the 2D pinger position that
minimizes squared angular error over the window.

Uses scipy.optimize.minimize (L-BFGS-B) for the optimization.
"""

import math
import time
from collections import deque
from typing import List, Tuple

import numpy as np
import rclpy
from bb_sensor_msgs.msg import Ping
from scipy.optimize import minimize

from pinger_localization.solvers.base import BaseSolver, wrap_angle_deg, angular_error_deg

# === Parameters ===
WINDOW_SIZE = 20              # Number of pings in the sliding window
WINDOW_SEC = 60.0             # Alternative: time-based window (seconds); 0 to disable
INIT_RANGE = 20.0             # Initial guess spread around vehicle (m)
# =================


class BatchSlidingSolver(BaseSolver):
    """Sliding-window nonlinear least-squares solver."""

    def __init__(self):
        super().__init__("batch_sliding")

        self.window_size = (
            self.declare_parameter("window_size", WINDOW_SIZE)
            .get_parameter_value()
            .integer_value
        )
        self.window_sec = (
            self.declare_parameter("window_sec", WINDOW_SEC)
            .get_parameter_value()
            .double_value
        )
        self.init_range = (
            self.declare_parameter("init_range", INIT_RANGE)
            .get_parameter_value()
            .double_value
        )

        # Store measurements as (vehicle_north, vehicle_east, vehicle_yaw, world_bearing_rad, stamp_ns).
        self._measurements: deque = deque(maxlen=self.window_size)
        self._initialized = False
        self._estimate_north = 0.0
        self._estimate_east = 0.0

        self._ping_sub = self.create_subscription(
            Ping, "/sensors/ping", self._ping_callback, 10
        )

        self.get_logger().info(
            f"BatchSliding ready: window_size={self.window_size}, "
            f"window_sec={self.window_sec} (0 = window_size only), "
            f"init_range={self.init_range}m, "
            f"ping<-'{self._ping_sub.topic_name}'"
        )

    def _evict_old(self, now_ns: int):
        """Remove measurements older than window_sec from the front."""
        if self.window_sec <= 0:
            return
        cutoff = now_ns - int(self.window_sec * 1e9)
        while self._measurements and self._measurements[0][4] < cutoff:
            self._measurements.popleft()

    def _loss(self, pos: np.ndarray, measurements: List[Tuple]) -> float:
        """Sum of squared angular errors for a candidate position."""
        total = 0.0
        pn, pe = pos[0], pos[1]
        for vn, ve, vyaw, wb, _ in measurements:
            # Expected world bearing from vehicle to candidate pinger.
            exp_wb = math.atan2(pe - ve, pn - vn)
            err = angular_error_deg(math.degrees(wb), math.degrees(exp_wb))
            total += err * err
        return total

    def _ping_callback(self, msg: Ping):
        """Process a new ping: add to window, evict old, re-optimize."""
        state = self.vehicle_state_for_ping(msg)
        if state is None:
            return
        vn, ve, vyaw, stamp_ns = state

        # Convert body-relative DOA to world bearing.
        world_bearing_rad = self.world_bearing_from_doa(msg.doa_deg, vyaw)

        self._measurements.append((vn, ve, vyaw, world_bearing_rad, stamp_ns))
        self._evict_old(stamp_ns)

        if len(self._measurements) < 2:
            # Need at least 2 measurements -- also reached when window_sec
            # evicts pings as fast as they arrive, which is otherwise invisible.
            self._throttled(
                "info", "sliding_needs_pings",
                f"not solving: {len(self._measurements)}/2 measurements in the "
                f"window (window_size={self.window_size}, window_sec="
                f"{self.window_sec} drops anything older) -- waiting for pings")
            return

        # Initial guess: previous estimate, or in front of vehicle.
        if not self._initialized:
            x0 = np.array([
                vn + self.init_range * math.cos(vyaw),
                ve + self.init_range * math.sin(vyaw),
            ])
            self._initialized = True
        else:
            x0 = np.array([self._estimate_north, self._estimate_east])

        # Optimize.
        mlist = list(self._measurements)
        result = minimize(
            self._loss, x0, args=(mlist,),
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-8},
        )

        if result.success:
            self._estimate_north = float(result.x[0])
            self._estimate_east = float(result.x[1])
            # Solved without logging before; a silently working solver looked
            # exactly like a broken one.
            self._throttled(
                "info", "sliding_solved",
                f"solved from {len(mlist)} measurements: est=("
                f"{self._estimate_north:.2f}, {self._estimate_east:.2f}), "
                f"cost={result.fun:.0f}deg^2", every_s=2.0)
        else:
            # The previous estimate is published regardless (it is (0, 0) if
            # this was the first solve), so say what went out and why.
            self._throttled(
                "warn", "sliding_opt_failed",
                f"optimization failed ({result.message}); publishing the previous "
                f"estimate ({self._estimate_north:.2f}, {self._estimate_east:.2f})")

        self.publish_estimate(self._estimate_north, self._estimate_east)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BatchSlidingSolver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node is not None:
            node.get_logger().fatal(f"Solver crashed: {e}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
