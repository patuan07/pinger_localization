"""
Seed mapping between the NED world frame and the sonar data frame (pure math).

The detector operates in DATA coordinates (X = range, Y = lateral) which match
the /oculus/pointcloud frame `auv5/sonar` exactly.  This module maps a
WORLD-NED pinger estimate into those data coords at the moment a sonar ping was
taken, and back again (for tests / synthetic-estimate helpers).

Transforms (extracted from the sep3_2dsonar_low bag's /tf_static, and matching
the Oculus driver's point-cloud construction in sonar_pointcloud_node.cpp):

    auv5/base_link -> auv5/sonar        : identity rotation, t = (0.29442, 0, 0.1354)
    auv5/base_link -> auv5/base_link_ned: q = (1, 0, 0, 0)   (180 deg about x)
    world -> world_ned                  : q = (0.707107, 0.707107, 0, 0)

The 180 deg-about-x base_link->base_link_ned flip makes the sonar's +y point to
the vehicle's LEFT, and the driver emits x = range*cos(bearing),
y = -range*sin(bearing) for a +bearing-starboard sensor, so:

    sonar_x =  forward   - mount_forward      (mount_forward = +0.29442 m)
    sonar_y = -starboard + mount_lateral       (mount_lateral  = 0.0)

where, for a vehicle pose (veh_north, veh_east, veh_yaw) in the NED world frame
and a world pinger at (pinger_north, pinger_east):

    forward   =  dN * cos(yaw) + dE * sin(yaw)
    starboard = -dN * sin(yaw) + dE * cos(yaw)
    dN = pinger_north - veh_north
    dE = pinger_east  - veh_east

Vertical (z) offsets never mix into x/y because the mount rotation has no
roll/pitch component -- the horizontal plane is decoupled, so z is ignored.
"""

import math

# Sonar mount offset along the vehicle +x axis (base_link -> auv5/sonar), m.
DEFAULT_MOUNT_FORWARD_M = 0.29442
# Sonar mount offset along the vehicle +y (starboard) axis, m (0 from the bag).
DEFAULT_MOUNT_LATERAL_M = 0.0


def seed_in_sonar(pinger_north, pinger_east,
                  veh_north, veh_east, veh_yaw_rad,
                  mount_forward_m=DEFAULT_MOUNT_FORWARD_M,
                  mount_lateral_m=DEFAULT_MOUNT_LATERAL_M):
    """Map a world-NED pinger estimate to sonar data coordinates (X, Y).

    Returns (seed_x, seed_y): the pipe-terminus location in the sonar frame,
    i.e. the seed the detector expects (pipe runs OUTWARD toward +X)."""
    dn = pinger_north - veh_north
    de = pinger_east - veh_east
    cos_yaw = math.cos(veh_yaw_rad)
    sin_yaw = math.sin(veh_yaw_rad)
    forward = dn * cos_yaw + de * sin_yaw
    starboard = -dn * sin_yaw + de * cos_yaw
    seed_x = forward - mount_forward_m
    seed_y = -starboard + mount_lateral_m
    return seed_x, seed_y


def sonar_to_world(seed_x, seed_y,
                   veh_north, veh_east, veh_yaw_rad,
                   mount_forward_m=DEFAULT_MOUNT_FORWARD_M,
                   mount_lateral_m=DEFAULT_MOUNT_LATERAL_M):
    """Inverse of seed_in_sonar: sonar data coords -> world-NED (north, east).

    Used by tests for the round-trip check and by the synthetic-estimate helper
    that must choose a world point whose seed lands inside the sonar ROI."""
    forward = seed_x + mount_forward_m
    starboard = -seed_y + mount_lateral_m
    cos_yaw = math.cos(veh_yaw_rad)
    sin_yaw = math.sin(veh_yaw_rad)
    # Forward/starboard are body axes of the NED yaw rotation (see module doc):
    #   forward   =  dN*cos(yaw) + dE*sin(yaw)
    #   starboard = -dN*sin(yaw) + dE*cos(yaw)
    # Invert the 2x2 (its determinant is cos^2 + sin^2 = 1).
    dn = forward * cos_yaw - starboard * sin_yaw
    de = forward * sin_yaw + starboard * cos_yaw
    return veh_north + dn, veh_east + de
