#!/usr/bin/python3
"""
Iterative Recursive Least Squares (RLS) Solver.

Each bearing measurement provides a linear constraint in the unknown pinger
position (north, east). For a vehicle at (vn, ve) with world bearing b:

    (east - ve) * cos(b) - (north - vn) * sin(b) = 0
 →  -sin(b) * north + cos(b) * east = -sin(b) * vn + cos(b) * ve

This is the form a_i^T * x = b_i, where:
    a_i = [-sin(b), cos(b)]^T
    b_i = -sin(b) * vn + cos(b) * ve
    x   = [north, east]^T

RLS updates the estimate incrementally without storing all past measurements.
"""

import math

import numpy as np
import rclpy
from bb_sensor_msgs.msg import Ping

from pinger_localization.solvers.base import BaseSolver, triangulate_from_bearings

# === Parameters ===
FORGETTING_FACTOR = 0.995     # Exponential forgetting factor (1.0 = no forgetting)
REGULARIZATION = 1e-3         # Initial P scaling (smaller = wider prior)
INIT_RANGE = 20.0             # Initial state uncertainty (m)
MIN_SEPARATION = 5.0          # Min distance between first 2 pings for triangulation (m)
# =================


class IterativeRLSSolver(BaseSolver):
    """RLS solver for stationary pinger localization."""

    def __init__(self):
        super().__init__("iterative_rls")

        self.forgetting_factor = (
            self.declare_parameter("forgetting_factor", FORGETTING_FACTOR)
            .get_parameter_value()
            .double_value
        )
        self.regularization = (
            self.declare_parameter("regularization", REGULARIZATION)
            .get_parameter_value()
            .double_value
        )
        self.init_range = (
            self.declare_parameter("init_range", INIT_RANGE)
            .get_parameter_value()
            .double_value
        )
        self.min_separation = (
            self.declare_parameter("min_separation", MIN_SEPARATION)
            .get_parameter_value()
            .double_value
        )

        # RLS state.
        self._x: np.ndarray = None     # (2,) estimate [north, east]
        self._P: np.ndarray = None     # (2, 2) inverse information matrix
        self._lam = self.forgetting_factor

        # Delayed initialization.
        self._init_measurements = []
        self._initialized = False

        self._ping_sub = self.create_subscription(
            Ping, "/sensors/ping", self._ping_callback, 10
        )

        self.get_logger().info(
            f"RLS ready: forgetting_factor={self.forgetting_factor}, "
            f"regularization={self.regularization}"
        )

    # ------------------------------------------------------------------
    # RLS
    # ------------------------------------------------------------------

    def _try_initialize(self):
        """Triangulate from buffered measurements to initialize RLS."""
        if len(self._init_measurements) < 2:
            return False

        vn0, ve0, _ = self._init_measurements[0]
        vn1, ve1, _ = self._init_measurements[-1]
        if math.hypot(vn1 - vn0, ve1 - ve0) < self.min_separation:
            return False

        result = triangulate_from_bearings(self._init_measurements)
        if result is None:
            return False

        self._x = np.array(result, dtype=float)
        self._P = np.eye(2) / self.regularization
        self._initialized = True
        self._init_measurements.clear()

        self.get_logger().info(
            f"RLS initialized: x0=({self._x[0]:.2f}, {self._x[1]:.2f})"
        )
        return True

    def _rls_update(self, veh_north: float, veh_east: float, world_bearing_rad: float):
        """Single RLS update step.

        Measurement model: a^T * x = b
            a = [-sin(b), cos(b)]^T
            b = -sin(b) * vn + cos(b) * ve
        """
        s = math.sin(world_bearing_rad)
        c = math.cos(world_bearing_rad)

        a = np.array([-s, c])  # (2,)
        b = -s * veh_north + c * veh_east

        # RLS update.
        # S = lambda + a^T * P * a
        Pa = self._P @ a
        S = self._lam + a @ Pa

        # K = P * a / S
        K = Pa / S

        # Innovation: b - a^T * x
        innovation = b - a @ self._x

        # Update.
        self._x = self._x + K * innovation
        self._P = (self._P - np.outer(K, Pa)) / self._lam

    # ------------------------------------------------------------------
    # Ping callback
    # ------------------------------------------------------------------

    def _ping_callback(self, msg: Ping):
        """Process a new ping."""
        vn, ve, vyaw, stamp_ns = self.get_latest_vehicle_state()
        if stamp_ns == 0:
            return

        world_bearing_rad = self.world_bearing_from_doa(msg.doa_deg, vyaw)

        if not self._initialized:
            self._init_measurements.append((vn, ve, world_bearing_rad))
            if self._try_initialize():
                self.publish_estimate(self._x[0], self._x[1])
            return

        self._rls_update(vn, ve, world_bearing_rad)
        self.publish_estimate(self._x[0], self._x[1])

        sigma_n = math.sqrt(self._P[0, 0])
        sigma_e = math.sqrt(self._P[1, 1])
        self.get_logger().info(
            f"RLS est=({self._x[0]:.2f}, {self._x[1]:.2f}), "
            f"sigma_n={sigma_n:.2f}m, sigma_e={sigma_e:.2f}m"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = IterativeRLSSolver()
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
