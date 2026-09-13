#!/usr/bin/python3
"""
Iterative Online Gradient Descent Solver.

Maintains a single estimate (north, east) of the pinger position. On each new
ping, runs K gradient descent steps minimizing the sum of squared angular errors
over all measurements seen so far (or a sliding window). Uses a decaying learning
rate for stability.

Cost for one ping i:
    L_i(north, east) = 0.5 * [angular_error(measured_wb, expected_wb(north, east))]^2

Gradient is computed analytically for each measurement.
"""

import math
from typing import List, Tuple

import numpy as np
import rclpy
from bb_sensor_msgs.msg import Ping

from pinger_localization.solvers.base import BaseSolver, angular_error_deg

# === Parameters ===
LEARNING_RATE = 0.05          # Initial learning rate (small — gradient is in deg²/m)
LR_DECAY = 0.995              # Multiplicative decay per ping
GD_STEPS = 20                 # Gradient descent steps per new ping
MAX_STEP_M = 2.0              # Max position change per gradient step (m)
WINDOW_SIZE = 0               # Max measurements stored (0 = unlimited)
MIN_SEPARATION = 5.0          # Min distance between initial pings (m)
INIT_RANGE = 20.0             # Initial guess range (m)
# =================


class IterativeGDSolver(BaseSolver):
    """Online gradient descent solver for stationary pinger localization."""

    def __init__(self):
        super().__init__("iterative_gd")

        self.learning_rate = (
            self.declare_parameter("learning_rate", LEARNING_RATE)
            .get_parameter_value()
            .double_value
        )
        self.lr_decay = (
            self.declare_parameter("lr_decay", LR_DECAY)
            .get_parameter_value()
            .double_value
        )
        self.gd_steps = (
            self.declare_parameter("gd_steps", GD_STEPS)
            .get_parameter_value()
            .integer_value
        )
        self.window_size = (
            self.declare_parameter("window_size", WINDOW_SIZE)
            .get_parameter_value()
            .integer_value
        )
        self.min_separation = (
            self.declare_parameter("min_separation", MIN_SEPARATION)
            .get_parameter_value()
            .double_value
        )
        self.init_range = (
            self.declare_parameter("init_range", INIT_RANGE)
            .get_parameter_value()
            .double_value
        )
        self.max_step_m = (
            self.declare_parameter("max_step_m", MAX_STEP_M)
            .get_parameter_value()
            .double_value
        )

        # Measurement store: (vn, ve, world_bearing_deg).
        self._measurements: List[Tuple] = []
        self._initialized = False
        self._estimate_north = 0.0
        self._estimate_east = 0.0
        self._lr = self.learning_rate

        # For delayed initialization.
        self._init_meas = []  # (vn, ve, world_bearing_rad).

        self._ping_sub = self.create_subscription(
            Ping, "/sensors/ping", self._ping_callback, 10
        )

        self.get_logger().info(
            f"GD ready: lr={self.learning_rate}, decay={self.lr_decay}, "
            f"steps={self.gd_steps}, max_step={self.max_step_m}m, "
            f"window_size={self.window_size} (0 = keep every measurement), "
            f"min_separation={self.min_separation}m, "
            f"ping<-'{self._ping_sub.topic_name}'"
        )

    # ------------------------------------------------------------------
    # Gradient descent
    # ------------------------------------------------------------------

    @staticmethod
    def _bearing_gradient(veh_north: float, veh_east: float,
                          world_bearing_deg: float,
                          pn: float, pe: float):
        """Compute the gradient of the squared angular error for one measurement.

        Cost: 0.5 * err^2 where err = wrap(measured_bearing - expected_bearing(pn, pe)).
        Gradient: -err * d(expected_bearing)/d[pn, pe].

        expected_bearing = atan2(pe - veh_east, pn - veh_north)  (rad)
        d(expected)/d_pn = -(pe - ve) / r^2
        d(expected)/d_pe =  (pn - vn) / r^2

        Returns (grad_north, grad_east).
        """
        dn = pn - veh_north
        de = pe - veh_east
        r2 = dn * dn + de * de

        if r2 < 1e-6:
            return 0.0, 0.0

        expected_rad = math.atan2(de, dn)
        expected_deg = math.degrees(expected_rad)

        err = angular_error_deg(world_bearing_deg, expected_deg)

        # Chain rule: d(0.5*err^2)/dp = err * d(err)/dp = -err * d(expected)/dp
        # d(expected_deg)/d_pn = -de/r^2 * 180/pi
        # d(expected_deg)/d_pe =  dn/r^2 * 180/pi
        deg_per_rad = 180.0 / math.pi
        d_exp_n = -de / r2 * deg_per_rad
        d_exp_e = dn / r2 * deg_per_rad

        grad_n = -err * d_exp_n
        grad_e = -err * d_exp_e
        return grad_n, grad_e

    def _gradient_step(self):
        """One gradient descent step over all measurements."""
        if not self._measurements:
            return

        total_gn = 0.0
        total_ge = 0.0

        pn = self._estimate_north
        pe = self._estimate_east

        for vn, ve, wb_deg in self._measurements:
            gn, ge = self._bearing_gradient(vn, ve, wb_deg, pn, pe)
            total_gn += gn
            total_ge += ge

        n = len(self._measurements)
        # Average gradient with per-dimension clipping.
        step_n = self._lr * total_gn / n
        step_e = self._lr * total_ge / n
        step_n = max(-self.max_step_m, min(self.max_step_m, step_n))
        step_e = max(-self.max_step_m, min(self.max_step_m, step_e))
        self._estimate_north -= step_n
        self._estimate_east -= step_e

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _try_initialize(self, vn: float, ve: float, vyaw: float, doa_deg: float):
        """Attempt to initialize from first few pings."""
        world_b = self.world_bearing_from_doa(doa_deg, vyaw)
        self._init_meas.append((vn, ve, world_b))

        if len(self._init_meas) < 2:
            self._throttled(
                "info", "gd_init_needs_pings",
                f"GD not initialized: {len(self._init_meas)}/2 initial pings "
                f"buffered -- waiting for another ping")
            return False

        vn0, ve0, _ = self._init_meas[0]
        vn1, ve1, _ = self._init_meas[-1]
        moved = math.hypot(vn1 - vn0, ve1 - ve0)
        if moved < self.min_separation:
            self._throttled(
                "info", "gd_init_needs_motion",
                f"GD not initialized: vehicle has moved only {moved:.2f} m since "
                f"the first buffered ping ({len(self._init_meas)} pings buffered, "
                f"needs {self.min_separation:.2f} m to separate the bearings) -- "
                f"drive on")
            return False

        # Simple triangulation.
        from pinger_localization.solvers.base import triangulate_from_bearings
        result = triangulate_from_bearings(self._init_meas)
        if result is None:
            # Fallback: guess in front of vehicle.
            self._throttled(
                "warn", "gd_init_triangulation_failed",
                f"triangulation from {len(self._init_meas)} bearings returned "
                f"nothing (degenerate geometry) -- starting from a guess "
                f"{self.init_range:.1f} m ahead of the vehicle instead")
            self._estimate_north = vn + self.init_range * math.cos(vyaw)
            self._estimate_east = ve + self.init_range * math.sin(vyaw)
        else:
            self._estimate_north = result[0]
            self._estimate_east = result[1]

        self._initialized = True
        self._init_meas.clear()
        self.get_logger().info(
            f"GD initialized: ({self._estimate_north:.2f}, {self._estimate_east:.2f})"
        )
        return True

    # ------------------------------------------------------------------
    # Ping callback
    # ------------------------------------------------------------------

    def _ping_callback(self, msg: Ping):
        """Process a new ping."""
        state = self.vehicle_state_for_ping(msg)
        if state is None:
            return
        vn, ve, vyaw, stamp_ns = state

        # Delayed initialization.
        if not self._initialized:
            if self._try_initialize(vn, ve, vyaw, msg.doa_deg):
                # Store first measurement and do initial publish.
                world_b_deg = math.degrees(
                    self.world_bearing_from_doa(msg.doa_deg, vyaw)
                )
                self._measurements.append((vn, ve, world_b_deg))
                self.publish_estimate(self._estimate_north, self._estimate_east)
            return

        # Add new measurement.
        world_b_deg = math.degrees(self.world_bearing_from_doa(msg.doa_deg, vyaw))
        self._measurements.append((vn, ve, world_b_deg))

        # Enforce window size.
        if self.window_size > 0 and len(self._measurements) > self.window_size:
            self._measurements = self._measurements[-self.window_size:]

        # Run K gradient descent steps.
        for _ in range(self.gd_steps):
            self._gradient_step()

        # Decay learning rate.
        self._lr *= self.lr_decay
        if self._lr < 0.01 * self.learning_rate:
            # Steps have become so small that the estimate no longer tracks
            # anything -- it keeps publishing, so this is invisible otherwise.
            self._throttled(
                "warn", "gd_lr_exhausted",
                f"learning rate has decayed to {self._lr:.2e} "
                f"({self._lr / self.learning_rate:.1e} of the initial "
                f"{self.learning_rate}) after {self._pings_seen} pings -- each "
                f"step now moves the estimate by a negligible amount, so it has "
                f"stopped tracking.  Raise lr_decay toward 1.0 to keep it "
                f"learning.", every_s=30.0)

        self.publish_estimate(self._estimate_north, self._estimate_east)

        # Compute current loss for logging.
        total_loss = 0.0
        for vn_m, ve_m, wb_deg in self._measurements:
            exp = math.degrees(math.atan2(
                self._estimate_east - ve_m, self._estimate_north - vn_m
            ))
            err = angular_error_deg(wb_deg, exp)
            total_loss += err * err
        rms = math.sqrt(total_loss / len(self._measurements))

        self.get_logger().info(
            f"GD est=({self._estimate_north:.2f}, {self._estimate_east:.2f}), "
            f"lr={self._lr:.4f}, RMS_err={rms:.2f}deg, "
            f"measurements={len(self._measurements)}, pings={self._pings_seen}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = IterativeGDSolver()
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
