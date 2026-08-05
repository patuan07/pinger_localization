#!/usr/bin/python3
"""
Iterative Extended Kalman Filter (EKF) Solver.

Estimates the stationary pinger position [north, east]^T using bearing
measurements. The measurement model is nonlinear, so we linearize via the
analytical Jacobian.

State:      x = [pinger_north, pinger_east]^T
Process:    x_k = x_{k-1}  (stationary, small Q for filter flexibility)
Measurement: h(x) = atan2(east - veh_east, north - veh_north) - veh_yaw
                    = body-relative DOA (radians)

Jacobian:   H = [ dh/dnorth, dh/deast ]
               = [ -de / r^2,  dn / r^2 ]   where dn = north - veh_north,
                                                de = east - veh_east,
                                                r^2 = dn^2 + de^2
"""

import math

import numpy as np
import rclpy
from bb_sensor_msgs.msg import Ping

from pinger_localization.solvers.base import BaseSolver, wrap_angle_rad, triangulate_from_bearings

# === Parameters ===
PROCESS_NOISE = 0.01           # Process noise variance (Q diagonal, m^2)
MEASUREMENT_NOISE_RAD = 0.087  # Measurement noise std dev (radians, ~5 deg)
INIT_RANGE = 20.0              # Initial state uncertainty half-side (m)
MIN_SEPARATION = 5.0           # Min distance (m) between first 2 pings for triangulation
# =================


class IterativeEKFSolver(BaseSolver):
    """EKF solver for stationary pinger localization."""

    def __init__(self):
        super().__init__("iterative_ekf")

        self.process_noise = (
            self.declare_parameter("process_noise", PROCESS_NOISE)
            .get_parameter_value()
            .double_value
        )
        self.measurement_noise_rad = (
            self.declare_parameter("measurement_noise_rad", MEASUREMENT_NOISE_RAD)
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

        # State: [north, east]^T
        self._x: np.ndarray = None           # (2,)
        self._P: np.ndarray = None           # (2, 2)

        # Identity process model (stationary).
        self._F = np.eye(2)
        self._Q = np.eye(2) * self.process_noise
        self._R = np.array([[self.measurement_noise_rad ** 2]])

        # For delayed initialization via triangulation.
        self._init_measurements = []  # (vn, ve, world_bearing_rad) tuples.
        self._initialized = False

        self._ping_sub = self.create_subscription(
            Ping, "/sensors/ping", self._ping_callback, 10
        )

        self.get_logger().info(
            f"EKF ready: process_noise={self.process_noise}, "
            f"meas_noise={math.degrees(self.measurement_noise_rad):.1f}deg"
        )

    # ------------------------------------------------------------------
    # EKF
    # ------------------------------------------------------------------

    def _try_initialize(self):
        """Attempt triangulation from buffered initial measurements."""
        if len(self._init_measurements) < 2:
            return False

        # Check minimum separation between first and last measurements.
        vn0, ve0, _ = self._init_measurements[0]
        vn1, ve1, _ = self._init_measurements[-1]
        if math.hypot(vn1 - vn0, ve1 - ve0) < self.min_separation:
            return False

        result = triangulate_from_bearings(self._init_measurements)
        if result is None:
            return False

        self._x = np.array(result, dtype=float)
        self._P = np.eye(2) * (self.init_range ** 2)
        self._initialized = True
        self._init_measurements.clear()

        self.get_logger().info(
            f"EKF initialized: x0=({self._x[0]:.2f}, {self._x[1]:.2f})"
        )
        return True

    def _predict(self):
        """Stationary prediction: x = x, P = P + Q."""
        self._P = self._F @ self._P @ self._F.T + self._Q

    def _update(self, veh_north: float, veh_east: float,
                veh_yaw: float, doa_body_deg: float):
        """EKF update with bearing measurement.

        Measurement model h(x):
            h = atan2(east - ve, north - vn) - veh_yaw
        Jacobian H wrt [north, east]:
            Let  dn = north - veh_north, de = east - veh_east, r2 = dn^2 + de^2.
            H = [ -de/r2,  dn/r2 ]
        """
        dn = self._x[0] - veh_north
        de = self._x[1] - veh_east
        r2 = dn * dn + de * de

        if r2 < 1e-6:
            return  # Too close to vehicle; skip update.

        # Predicted body-relative DOA.
        world_bearing = math.atan2(de, dn)
        h_pred = wrap_angle_rad(world_bearing - veh_yaw)

        # Measurement.
        z = math.radians(doa_body_deg)

        # Innovation (shortest angular difference).
        y = z - h_pred
        y = math.atan2(math.sin(y), math.cos(y))

        # Jacobian.
        H = np.array([[-de / r2, dn / r2]])

        # Kalman gain.
        S = H @ self._P @ H.T + self._R
        K = self._P @ H.T @ np.linalg.inv(S)

        # Update state and covariance.
        self._x = self._x + K.flatten() * y
        self._P = (np.eye(2) - K @ H) @ self._P

    # ------------------------------------------------------------------
    # Ping callback
    # ------------------------------------------------------------------

    def _ping_callback(self, msg: Ping):
        """Process a new ping."""
        vn, ve, vyaw, stamp_ns = self.get_latest_vehicle_state()
        if stamp_ns == 0:
            return

        world_bearing_rad = self.world_bearing_from_doa(msg.doa_deg, vyaw)

        # Delayed initialization.
        if not self._initialized:
            self._init_measurements.append((vn, ve, world_bearing_rad))
            if self._try_initialize():
                self.publish_estimate(self._x[0], self._x[1])
            return

        # EKF cycle.
        self._predict()
        self._update(vn, ve, vyaw, msg.doa_deg)

        self.publish_estimate(self._x[0], self._x[1])

        # Compute error ellipse info for logging.
        eigvals = np.linalg.eigvalsh(self._P)
        sigma = math.sqrt(max(eigvals[0], eigvals[1]))
        self.get_logger().info(
            f"EKF est=({self._x[0]:.2f}, {self._x[1]:.2f}), sigma_max={sigma:.2f}m"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = IterativeEKFSolver()
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
