#!/usr/bin/python3
"""
Particle Filter Solver for 2D stationary pinger localization.

Maintains a cloud of particles representing the pinger's possible (North, East)
position. Each ping measurement re-weights particles based on how well their
expected bearing matches the measured DOA.

- State: 2D stationary (no prediction step)
- Update: Gaussian likelihood on angular error
- Resample: systematic resampling when effective sample size drops below threshold
"""

import math
import random
from typing import List, Tuple

import numpy as np
import rclpy
from bb_sensor_msgs.msg import Ping

from pinger_localization.solvers.base import BaseSolver, wrap_angle_rad

# === Parameters ===
NUM_PARTICLES = 5000          # Number of particles
SIGMA_DOA_DEG = 5.0           # DOA measurement noise std dev (degrees)
INIT_RANGE = 20.0             # Initial particle spread half-side (m)
RESAMPLE_THRESHOLD = 0.5      # Resample when N_eff/N < this
ROUGHENING_STD = 0.05         # Small roughening jitter std dev (m), 0 to disable
# =================


class ParticleFilterSolver(BaseSolver):
    """Stationary 2D particle filter for pinger localization."""

    def __init__(self):
        super().__init__("particle_filter")

        self.num_particles = (
            self.declare_parameter("num_particles", NUM_PARTICLES)
            .get_parameter_value()
            .integer_value
        )
        self.sigma_doa_rad = math.radians(
            self.declare_parameter("sigma_doa_deg", SIGMA_DOA_DEG)
            .get_parameter_value()
            .double_value
        )
        self.init_range = (
            self.declare_parameter("init_range", INIT_RANGE)
            .get_parameter_value()
            .double_value
        )
        self.resample_threshold = (
            self.declare_parameter("resample_threshold", RESAMPLE_THRESHOLD)
            .get_parameter_value()
            .double_value
        )
        self.roughening_std = (
            self.declare_parameter("roughening_std", ROUGHENING_STD)
            .get_parameter_value()
            .double_value
        )

        # Particle state.
        self._initialized = False
        self._particles: np.ndarray = None       # (N, 2): [north, east]
        self._weights: np.ndarray = None          # (N,)

        # For output.
        self._estimate_north = 0.0
        self._estimate_east = 0.0

        self._ping_sub = self.create_subscription(
            Ping, "/sensors/ping", self._ping_callback, 10
        )

        self.get_logger().info(
            f"ParticleFilter ready: N={self.num_particles}, "
            f"sigma={math.degrees(self.sigma_doa_rad):.1f}deg, "
            f"init_range={self.init_range}m, resample_threshold="
            f"{self.resample_threshold}, roughening_std={self.roughening_std}m, "
            f"ping<-'{self._ping_sub.topic_name}'"
        )

    # ------------------------------------------------------------------
    # Particle filter core
    # ------------------------------------------------------------------

    def _init_particles(self, veh_north: float, veh_east: float):
        """Initialize particles uniformly around the vehicle's starting position."""
        r = self.init_range
        self._particles = np.column_stack([
            np.random.uniform(veh_north - r, veh_north + r, self.num_particles),
            np.random.uniform(veh_east - r, veh_east + r, self.num_particles),
        ])
        self._weights = np.ones(self.num_particles) / self.num_particles
        self._initialized = True

    def _update_weights(self, veh_north: float, veh_east: float,
                        veh_yaw: float, doa_body_deg: float):
        """Re-weight particles based on measured bearing."""
        doa_body_rad = math.radians(doa_body_deg)
        world_bearing = wrap_angle_rad(veh_yaw + doa_body_rad)

        # Vectorized: compute expected world bearing for each particle.
        de = self._particles[:, 1] - veh_east
        dn = self._particles[:, 0] - veh_north
        expected_bearing = np.arctan2(de, dn)

        # Angular error (shortest).
        err = expected_bearing - world_bearing
        err = np.arctan2(np.sin(err), np.cos(err))

        # Gaussian likelihood.
        self._weights *= np.exp(-0.5 * (err / self.sigma_doa_rad) ** 2)
        wsum = np.sum(self._weights)
        if wsum > 0:
            self._weights /= wsum
        else:
            # Degenerate: reset to uniform.  Every particle disagrees so
            # strongly with this bearing that the likelihoods underflowed --
            # the estimate is then just the (uninformative) prior again.
            self._throttled(
                "warn", "pf_weights_underflow",
                f"all {self.num_particles} particle weights underflowed to zero "
                f"(doa={doa_body_deg:.1f}deg contradicts every particle) -- "
                f"resetting to uniform")
            self._weights = np.ones(self.num_particles) / self.num_particles

    def _systematic_resample(self):
        """Systematic resampling. Returns new particle indices."""
        N = self.num_particles
        positions = (np.random.random() + np.arange(N)) / N
        cumsum = np.cumsum(self._weights)
        indices = np.zeros(N, dtype=int)
        i = 0
        for j in range(N):
            while i < N and positions[j] > cumsum[i]:
                i += 1
            indices[j] = min(i, N - 1)
        return indices

    def _roughen(self):
        """Add small Gaussian jitter to prevent sample impoverishment."""
        if self.roughening_std <= 0:
            return
        self._particles[:, 0] += np.random.normal(0, self.roughening_std, self.num_particles)
        self._particles[:, 1] += np.random.normal(0, self.roughening_std, self.num_particles)

    # ------------------------------------------------------------------
    # Ping callback
    # ------------------------------------------------------------------

    def _ping_callback(self, msg: Ping):
        """Process a new ping: update weights, resample if needed, publish estimate."""
        state = self.vehicle_state_for_ping(msg)
        if state is None:
            return
        vn, ve, vyaw, stamp_ns = state

        # Initialize on first ping.
        if not self._initialized:
            self._init_particles(vn, ve)
            self.get_logger().info(
                f"Initialized {self.num_particles} particles around ({vn:.1f}, {ve:.1f})"
            )

        # Always update weights with the current DOA measurement.
        self._update_weights(vn, ve, vyaw, msg.doa_deg)

        # Effective sample size.
        N_eff = 1.0 / np.sum(self._weights ** 2) if np.sum(self._weights) > 0 else 0

        # Resample if needed.
        if N_eff / self.num_particles < self.resample_threshold:
            self._throttled(
                "info", "pf_resampled",
                f"resampling: N_eff={N_eff:.0f}/{self.num_particles} "
                f"(< {self.resample_threshold:.2f} * N) -- the cloud has "
                f"collapsed onto few particles, which is expected while the "
                f"filter converges but not on every ping", every_s=5.0)
            indices = self._systematic_resample()
            self._particles = self._particles[indices]
            self._weights = np.ones(self.num_particles) / self.num_particles
            self._roughen()

        # Weighted mean estimate.
        self._estimate_north = float(np.average(self._particles[:, 0], weights=self._weights))
        self._estimate_east = float(np.average(self._particles[:, 1], weights=self._weights))

        self.publish_estimate(self._estimate_north, self._estimate_east)

        self.get_logger().info(
            f"N_eff={N_eff:.0f}/{self.num_particles}, "
            f"est=({self._estimate_north:.2f}, {self._estimate_east:.2f}), "
            f"doa={msg.doa_deg:.1f}deg, pings={self._pings_seen}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ParticleFilterSolver()
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
