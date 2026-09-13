#!/usr/bin/python3
"""
Batch Full-Trajectory Solver.

Accumulates all measurements over the entire trajectory and periodically solves
for the best-fit 2D pinger position. Uses two strategies:

1. **Least-squares**: nonlinear optimization over all measurements.
2. **RANSAC**: repeatedly sample 2 bearing pairs, compute their intersection,
   score against all measurements, keep the best consensus set.

The RANSAC variant is robust to outliers.
"""

import math
import random
from typing import List, Tuple

import numpy as np
import rclpy
from bb_sensor_msgs.msg import Ping
from scipy.optimize import minimize

from pinger_localization.solvers.base import BaseSolver, wrap_angle_deg, angular_error_deg

# === Parameters ===
USE_RANSAC = True              # Use RANSAC for robust estimation
RANSAC_ITERATIONS = 200        # Number of RANSAC rounds
RANSAC_INLIER_THRESH = 5.0     # Inlier threshold (degrees of angular error)
SOLVE_EVERY_N_PINGS = 5        # Re-solve every N pings (0 = only at end)
INIT_RANGE = 20.0              # Initial guess around vehicle (m)
# =================


class BatchFullSolver(BaseSolver):
    """Full-trajectory batch solver with optional RANSAC."""

    def __init__(self):
        super().__init__("batch_full")

        self.use_ransac = (
            self.declare_parameter("use_ransac", USE_RANSAC)
            .get_parameter_value()
            .bool_value
        )
        self.ransac_iterations = (
            self.declare_parameter("ransac_iterations", RANSAC_ITERATIONS)
            .get_parameter_value()
            .integer_value
        )
        self.ransac_inlier_thresh = (
            self.declare_parameter("ransac_inlier_thresh", RANSAC_INLIER_THRESH)
            .get_parameter_value()
            .double_value
        )
        self.solve_every_n = (
            self.declare_parameter("solve_every_n_pings", SOLVE_EVERY_N_PINGS)
            .get_parameter_value()
            .integer_value
        )
        self.init_range = (
            self.declare_parameter("init_range", INIT_RANGE)
            .get_parameter_value()
            .double_value
        )

        # Accumulate all measurements: (vn, ve, vyaw, world_bearing_rad, stamp_ns).
        self._measurements: List[Tuple] = []
        self._ping_count = 0
        self._estimate_north = 0.0
        self._estimate_east = 0.0
        self._initialized = False
        self._solved_once = False   # _initialized only seeds the starting guess

        self._ping_sub = self.create_subscription(
            Ping, "/sensors/ping", self._ping_callback, 10
        )

        self.get_logger().info(
            f"BatchFull ready: use_ransac={self.use_ransac}, "
            f"ransac_iters={self.ransac_iterations}, "
            f"ransac_inlier_thresh={self.ransac_inlier_thresh}deg, "
            f"solve_every_n_pings={self.solve_every_n} (0 = never re-solve), "
            f"init_range={self.init_range}m, "
            f"ping<-'{self._ping_sub.topic_name}'"
        )

    # ------------------------------------------------------------------
    # Intersection of two bearing lines
    # ------------------------------------------------------------------

    @staticmethod
    def _bearing_intersection(vn1: float, ve1: float, b1: float,
                              vn2: float, ve2: float, b2: float) -> Tuple[float, float]:
        """Compute the intersection of two bearing rays in 2D.

        Line 1: point (vn1, ve1), direction (sin(b1), cos(b1)) in NED.
                In the NED plane, bearing 0 = North, +90 = East.
                Direction = (cos(b), sin(b)) — cos for North, sin for East.
        Line 2: point (vn2, ve2), direction (cos(b2), sin(b2)).

        Solves: vn1 + t1*cos(b1) = vn2 + t2*cos(b2)
                ve1 + t1*sin(b1) = ve2 + t2*sin(b2)

        Returns (north, east) intersection, or falls back to midpoint if lines
        are parallel.
        """
        c1, s1 = math.cos(b1), math.sin(b1)
        c2, s2 = math.cos(b2), math.sin(b2)

        det = c1 * s2 - s1 * c2
        if abs(det) < 1e-10:
            # Parallel or nearly parallel — fall back to midpoint of vehicle positions.
            return (vn1 + vn2) / 2.0, (ve1 + ve2) / 2.0

        t1 = ((vn2 - vn1) * s2 - (ve2 - ve1) * c2) / det
        return vn1 + t1 * c1, ve1 + t1 * s1

    # ------------------------------------------------------------------
    # RANSAC
    # ------------------------------------------------------------------

    def _ransac_solve(self) -> Tuple[float, float, int]:
        """RANSAC: sample pairs, score, return best consensus estimate."""
        best_score = -1
        best_north = 0.0
        best_east = 0.0
        best_inliers = 0

        n = len(self._measurements)
        if n < 2:
            return 0.0, 0.0, 0

        for _ in range(self.ransac_iterations):
            # Sample 2 distinct measurements.
            i1, i2 = random.sample(range(n), 2)
            vn1, ve1, _, wb1, _ = self._measurements[i1]
            vn2, ve2, _, wb2, _ = self._measurements[i2]

            # Intersection of the two bearing lines.
            pn, pe = self._bearing_intersection(vn1, ve1, wb1, vn2, ve2, wb2)

            # Score against all measurements.
            score = 0
            inliers = 0
            for vn, ve, _, wb, _ in self._measurements:
                exp_wb = math.atan2(pe - ve, pn - vn)
                err = abs(angular_error_deg(math.degrees(wb), math.degrees(exp_wb)))
                if err < self.ransac_inlier_thresh:
                    inliers += 1
                    score += 1.0 - (err / self.ransac_inlier_thresh)

            if inliers > best_inliers or (inliers == best_inliers and score > best_score):
                best_inliers = inliers
                best_score = score
                best_north = pn
                best_east = pe

        # Refit using all inliers (least-squares on the best consensus set).
        inlier_measurements = []
        for vn, ve, _, wb, _ in self._measurements:
            exp_wb = math.atan2(best_east - ve, best_north - vn)
            err = abs(angular_error_deg(math.degrees(wb), math.degrees(exp_wb)))
            if err < self.ransac_inlier_thresh:
                inlier_measurements.append((vn, ve, wb))

        if len(inlier_measurements) >= 2:
            pn, pe = self._least_squares_solve(inlier_measurements, (best_north, best_east))
            return pn, pe, len(inlier_measurements)

        # No consensus: the returned point is one sample pair's intersection, or
        # the (0, 0) of a run that never agreed on anything.
        self._throttled(
            "warn", "full_ransac_no_consensus",
            f"RANSAC found only {len(inlier_measurements)} inlier(s) among "
            f"{len(self._measurements)} measurements (threshold "
            f"{self.ransac_inlier_thresh}deg) -- ({best_north:.2f}, "
            f"{best_east:.2f}) is unsupported by the data")
        return best_north, best_east, best_inliers

    # ------------------------------------------------------------------
    # Least-squares
    # ------------------------------------------------------------------

    def _loss(self, pos: np.ndarray, measurements) -> float:
        """Sum of squared angular errors."""
        total = 0.0
        pn, pe = pos[0], pos[1]
        for vn, ve, wb in measurements:
            exp_wb = math.atan2(pe - ve, pn - vn)
            err = angular_error_deg(math.degrees(wb), math.degrees(exp_wb))
            total += err * err
        return total

    def _least_squares_solve(self, measurements, x0) -> Tuple[float, float]:
        """Nonlinear least-squares for pinger position."""
        result = minimize(
            self._loss, np.array(x0), args=(measurements,),
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if result.success:
            return float(result.x[0]), float(result.x[1])
        # The old code returned x0 here without a word -- a non-converging
        # solve looked identical to a good one.
        self._throttled(
            "warn", "full_ls_not_converged",
            f"least-squares did not converge ({result.message}) on "
            f"{len(measurements)} measurements -- keeping the initial guess "
            f"({x0[0]:.2f}, {x0[1]:.2f})")
        return x0[0], x0[1]

    # ------------------------------------------------------------------
    # Ping callback
    # ------------------------------------------------------------------

    def _ping_callback(self, msg: Ping):
        """Accumulate measurement and solve."""
        state = self.vehicle_state_for_ping(msg)
        if state is None:
            return
        vn, ve, vyaw, stamp_ns = state

        world_bearing_rad = self.world_bearing_from_doa(msg.doa_deg, vyaw)
        self._measurements.append((vn, ve, vyaw, world_bearing_rad, stamp_ns))
        self._ping_count += 1

        # Initial guess: use measured bearing to place guess in the right direction.
        if not self._initialized and len(self._measurements) >= 2:
            world_bearing = vyaw + math.radians(msg.doa_deg)
            x0_n = vn + self.init_range * math.cos(world_bearing)
            x0_e = ve + self.init_range * math.sin(world_bearing)
            self._initialized = True

        # Determine if we should solve now.
        should_solve = False
        if self.solve_every_n > 0 and self._ping_count % self.solve_every_n == 0:
            should_solve = True
        if len(self._measurements) < 2:
            self._throttled(
                "info", "full_needs_pings",
                f"not solving: {len(self._measurements)}/2 measurements "
                f"accumulated -- waiting for pings")
            return

        if not should_solve:
            # Still publish best guess so far (useful for live monitoring).
            if self._initialized:
                if not self._solved_once:
                    # _initialized only means the starting guess was seeded, so
                    # what is being published until the first solve is a
                    # placeholder, not an estimate of anything.
                    self._throttled(
                        "info", "full_unsolved",
                        f"holding at ping {self._ping_count}: no solve has run "
                        f"yet (solve_every_n_pings={self.solve_every_n}; 0 would "
                        f"disable re-solving entirely), so the published estimate "
                        f"is still the placeholder "
                        f"({self._estimate_north:.2f}, {self._estimate_east:.2f}) "
                        f"-- drive on")
                self.publish_estimate(self._estimate_north, self._estimate_east)
            return

        x0 = (self._estimate_north, self._estimate_east) if self._initialized else (
            vn + self.init_range * math.cos(vyaw),
            ve + self.init_range * math.sin(vyaw),
        )

        if self.use_ransac and len(self._measurements) >= 5:
            pn, pe, n_inliers = self._ransac_solve()
            self.get_logger().info(
                f"RANSAC: n_inliers={n_inliers}/{len(self._measurements)}, "
                f"pos=({pn:.2f}, {pe:.2f})"
            )
        else:
            # Simple measurements list for LS: (vn, ve, wb) tuple.
            ls_m = [(vn, ve, wb) for vn, ve, _, wb, _ in self._measurements]
            pn, pe = self._least_squares_solve(ls_m, x0)
            # RANSAC is silently unavailable for the first few pings -- say so,
            # or the switch from RANSAC to LS looks arbitrary.
            why = ("use_ransac=False" if not self.use_ransac
                   else f"only {len(self._measurements)}/5 measurements")
            self.get_logger().info(
                f"LS: n={len(self._measurements)}, pos=({pn:.2f}, {pe:.2f}) "
                f"[{why}]"
            )

        self._estimate_north = pn
        self._estimate_east = pe
        self._solved_once = True
        self.publish_estimate(pn, pe)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BatchFullSolver()
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
