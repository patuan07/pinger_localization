"""
Trajectory generation for the pinger localization simulator.

Each trajectory class implements `generate(t, params) -> (north, east, yaw)`
where t is elapsed time in seconds. The vehicle moves in the NED horizontal
plane at a constant depth.
"""

import math
import numpy as np


class CircleTrajectory:
    """Vehicle moves in a circle centered near the pinger."""

    def __init__(self, pinger_north: float, pinger_east: float, radius: float, speed: float):
        """
        Args:
            pinger_north: True pinger North (m).
            pinger_east: True pinger East (m).
            radius: Circle radius (m). Center is offset from pinger so the pinger
                    sits inside the circle for diverse bearings.
            speed: Linear speed along the circle (m/s).
        """
        # Offset the circle center from the pinger so bearings vary more.
        self._cx = pinger_north + radius * 0.3
        self._cy = pinger_east - radius * 0.3
        self._radius = radius
        self._speed = speed
        # Angular velocity (rad/s).
        self._omega = speed / radius

    def generate(self, t: float):
        """Return (north, east, yaw_rad) at elapsed time t (seconds)."""
        angle = self._omega * t
        # Position on circle.
        north = self._cx + self._radius * math.cos(angle)
        east = self._cy + self._radius * math.sin(angle)
        # Heading is tangent to the circle (pointing forward along the path).
        yaw = angle + math.pi / 2.0
        return north, east, yaw


class LawnmowerTrajectory:
    """Back-and-forth survey lines covering an area around the pinger.

    The vehicle passes the pinger region from different angles on each leg.
    """

    def __init__(self, pinger_north: float, pinger_east: float, speed: float,
                 leg_length: float = 20.0, leg_spacing: float = 4.0, num_legs: int = 5):
        """
        Args:
            pinger_north, pinger_east: Pinger position (center of survey area).
            speed: Vehicle speed (m/s).
            leg_length: Length of each survey leg (m).
            leg_spacing: Distance between adjacent legs (m).
            num_legs: Number of back-and-forth legs.
        """
        self._speed = speed
        self._leg_length = leg_length
        self._leg_spacing = leg_spacing
        self._num_legs = num_legs

        # Build waypoints for the lawnmower pattern centered on the pinger.
        half_spacing = leg_spacing * (num_legs - 1) / 2.0
        half_length = leg_length / 2.0

        self._waypoints = []
        for i in range(num_legs):
            north_offset = -half_length if i % 2 == 0 else half_length
            east_offset = -half_spacing + i * leg_spacing
            self._waypoints.append((
                pinger_north + north_offset,
                pinger_east + east_offset,
            ))
        # Store total distance for time calculation.
        self._total_dist = self._compute_total_distance()

    def _compute_total_distance(self):
        dist = 0.0
        for i in range(len(self._waypoints) - 1):
            n0, e0 = self._waypoints[i]
            n1, e1 = self._waypoints[i + 1]
            dist += math.hypot(n1 - n0, e1 - e0)
        return dist

    def _get_pose_at_distance(self, dist: float):
        """Interpolate position and heading at a given distance along the path."""
        accum = 0.0
        for i in range(1, len(self._waypoints)):
            n0, e0 = self._waypoints[i - 1]
            n1, e1 = self._waypoints[i]
            seg_dist = math.hypot(n1 - n0, e1 - e0)
            if accum + seg_dist >= dist or i == len(self._waypoints) - 1:
                frac = (dist - accum) / seg_dist if seg_dist > 0 else 0.0
                frac = max(0.0, min(1.0, frac))
                north = n0 + frac * (n1 - n0)
                east = e0 + frac * (e1 - e0)
                yaw = math.atan2(e1 - e0, n1 - n0)
                return north, east, yaw
            accum += seg_dist
        # Past the end – stay at last waypoint.
        n0, e0 = self._waypoints[-1]
        n1, e1 = self._waypoints[-2]
        yaw = math.atan2(e0 - e1, n0 - n1)
        return n0, e0, yaw

    def generate(self, t: float):
        """Return (north, east, yaw_rad) at elapsed time t (seconds)."""
        dist = (t * self._speed) % self._total_dist
        return self._get_pose_at_distance(dist)


class Figure8Trajectory:
    """Figure-8 (lemniscate) pattern around the pinger.

    x = a * sin(t), y = a * sin(2*t) * 0.5 — gives two connected loops for diverse bearings.
    """

    def __init__(self, pinger_north: float, pinger_east: float, scale: float, speed: float):
        """
        Args:
            pinger_north, pinger_east: Pinger position (center of figure-8).
            scale: Size of the figure-8 (m) — half the major axis.
            speed: Average vehicle speed (m/s).
        """
        self._pinger_north = pinger_north
        self._pinger_east = pinger_east
        self._scale = scale
        self._speed = speed
        # Full period of the parametric curve (in the parameter theta).
        self._period = 2.0 * math.pi
        # Precompute arc length numerically for constant-speed mapping.
        self._arc_table = self._build_arc_table()

    def _build_arc_table(self, n_samples: int = 1000):
        """Build a lookup table mapping cumulative arc length -> theta parameter."""
        thetas = np.linspace(0, self._period, n_samples)
        # Lemniscate parametric: north = scale * sin(theta), east = scale * sin(2*theta) * 0.5.
        # Derivatives: dn = scale * cos(theta), de = scale * cos(2*theta).
        dtheta = self._period / (n_samples - 1)
        arc = [0.0]
        for i in range(1, n_samples):
            t_mid = thetas[i - 1] + dtheta / 2.0
            dn = self._scale * math.cos(t_mid)
            de = self._scale * math.cos(2.0 * t_mid)
            ds = math.hypot(dn, de) * dtheta
            arc.append(arc[-1] + ds)
        return thetas, np.array(arc)

    def _theta_from_arc(self, s: float):
        """Map an arc-length distance s to the parametric theta."""
        thetas, arcs = self._arc_table
        total = arcs[-1]
        s = s % total
        idx = np.searchsorted(arcs, s)
        if idx <= 0:
            return thetas[0]
        if idx >= len(thetas):
            return thetas[-1]
        # Linear interpolation.
        frac = (s - arcs[idx - 1]) / (arcs[idx] - arcs[idx - 1])
        return thetas[idx - 1] + frac * (thetas[idx] - thetas[idx - 1])

    def generate(self, t: float):
        """Return (north, east, yaw_rad) at elapsed time t (seconds)."""
        dist = t * self._speed
        theta = self._theta_from_arc(dist)
        # Lemniscate: north-south oriented main lobe.
        north = self._pinger_north + self._scale * math.sin(theta)
        east = self._pinger_east + self._scale * math.sin(2.0 * theta) * 0.5
        # Heading from derivative.
        dn = self._scale * math.cos(theta)
        de = self._scale * math.cos(2.0 * theta)
        yaw = math.atan2(de, dn)
        return north, east, yaw


class SpiralTrajectory:
    """Expanding/contracting spiral around the pinger.

    The vehicle spirals outward from the pinger center, giving a mix of bearing
    angles at varying distances.
    """

    def __init__(self, pinger_north: float, pinger_east: float,
                 speed: float, max_radius: float = 15.0, num_turns: float = 4.0):
        """
        Args:
            pinger_north, pinger_east: Pinger position (spiral center).
            speed: Vehicle speed (m/s).
            max_radius: Maximum radius of the spiral (m).
            num_turns: Number of full turns to reach max_radius.
        """
        self._pinger_north = pinger_north
        self._pinger_east = pinger_east
        self._speed = speed
        self._max_radius = max_radius
        self._num_turns = num_turns
        # Total angle swept.
        self._total_angle = 2.0 * math.pi * num_turns
        # Radius slope: r(theta) = (max_radius / total_angle) * theta.
        self._r_slope = max_radius / self._total_angle
        # Approximate arc length for time-to-angle mapping.
        self._arc_length = self._compute_arc_length()

    def _compute_arc_length(self):
        """Numerically integrate the spiral arc length."""
        n = 2000
        thetas = np.linspace(0, self._total_angle, n)
        total = 0.0
        for i in range(1, n):
            th = thetas[i - 1]
            dth = thetas[i] - thetas[i - 1]
            r = self._r_slope * th
            dr = self._r_slope
            ds = math.hypot(r, dr) * dth
            total += ds
        return total

    def _theta_from_arc(self, s: float):
        """Map arc-length distance s to theta via numerical inversion.

        Uses Newton iteration since we have a closed-form expression for
        r(theta) = a * theta where a = r_slope.
        Arc length: s(theta) = 0.5 * a * (theta * sqrt(1 + theta^2) + arcsinh(theta)).
        We binary-search instead for simplicity.
        """
        target = s % self._arc_length
        lo, hi = 0.0, self._total_angle
        for _ in range(50):
            mid = (lo + hi) / 2.0
            # s = 0.5 * a * (mid * sqrt(1 + mid^2) + arcsinh(mid)).
            sm = 0.5 * self._r_slope * (
                mid * math.sqrt(1.0 + mid * mid) + math.asinh(mid)
            )
            if sm < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def generate(self, t: float):
        """Return (north, east, yaw_rad) at elapsed time t (seconds)."""
        dist = t * self._speed
        theta = self._theta_from_arc(dist)
        r = self._r_slope * theta
        # Position on spiral.
        north = self._pinger_north + r * math.cos(theta)
        east = self._pinger_east + r * math.sin(theta)
        # Heading tangent to spiral: derivative of (r*cos(th), r*sin(th)).
        dr = self._r_slope
        dn = dr * math.cos(theta) - r * math.sin(theta)
        de = dr * math.sin(theta) + r * math.cos(theta)
        yaw = math.atan2(de, dn)
        return north, east, yaw


def create_trajectory(trajectory_type: str, pinger_north: float, pinger_east: float,
                      radius: float, speed: float):
    """Factory function to create a trajectory by name.

    Args:
        trajectory_type: One of 'circle', 'lawnmower', 'figure8', 'spiral'.
        pinger_north, pinger_east: True pinger position.
        radius: Approximate size / distance from pinger (m).
        speed: Vehicle speed (m/s).

    Returns:
        A trajectory object with a `generate(t) -> (north, east, yaw)` method.
    """
    ttype = trajectory_type.lower()
    if ttype == "circle":
        return CircleTrajectory(pinger_north, pinger_east, radius, speed)
    elif ttype == "lawnmower":
        return LawnmowerTrajectory(pinger_north, pinger_east, speed,
                                   leg_length=radius * 2.0, leg_spacing=radius * 0.4,
                                   num_legs=5)
    elif ttype == "figure8":
        return Figure8Trajectory(pinger_north, pinger_east, radius, speed)
    elif ttype == "spiral":
        return SpiralTrajectory(pinger_north, pinger_east, speed,
                                max_radius=radius * 1.5, num_turns=4.0)
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}. "
                         f"Choose from: circle, lawnmower, figure8, spiral")
