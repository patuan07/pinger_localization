# pinger_localization

Underwater acoustic pinger 2D localization — simulation, solver comparison, and evaluation.

A ROS2 Humble Python package that simulates an AUV moving around a stationary acoustic pinger, runs six different localization algorithms in parallel, and evaluates their accuracy over time.

## Overview

An AUV moves along a configurable trajectory around a fixed underwater pinger. The pinger emits acoustic pulses; the AUV measures the body-relative **direction of arrival (DOA)** of each ping (with configurable noise). Six solver nodes independently estimate the pinger's 2D position (North, East in NED frame) using only DOA bearings and vehicle odometry. An evaluation node logs all estimates and an offline analysis script produces comparison plots.

```
┌─────────────────┐     ┌──────────────────────────────┐
│  PingerSimulator │────▶│  /auv5/odom_ned (20 Hz)      │
│                  │     │  /sensors/ping (~0.5 Hz)     │
│  trajectory +    │     │  /pinger_localization/gt     │
│  noise models    │     └──────────────────────────────┘
└─────────────────┘                  │
       ┌─────────────────────────────┤
       ▼             ▼               ▼
┌────────────┐ ┌───────────┐ ┌──────────────┐
│batch_      │ │particle_  │ │iterative_    │   ...6 solvers total
│sliding     │ │filter     │ │ekf           │
└────────────┘ └───────────┘ └──────────────┘
       │             │               │
       ▼             ▼               ▼
┌──────────────────────────────────────────┐
│  EvalNode  ──▶  ~/pinger_logs/*.csv      │
│  analyze.py ──▶  eval/plots/*.png        │
└──────────────────────────────────────────┘
```

### Solvers

| Solver | Type | Description |
|---|---|---|
| `batch_sliding` | Batch | Sliding-window nonlinear least-squares |
| `batch_full` | Batch | Full-trajectory least-squares + RANSAC variant |
| `particle_filter` | Sequential Monte Carlo | 5K stationary 2D particles with systematic resampling |
| `iterative_ekf` | Kalman filter | Extended Kalman Filter with analytical bearing Jacobian |
| `iterative_rls` | Recursive | Recursive Least Squares via linearized bearing constraints |
| `iterative_gd` | Gradient | Online gradient descent with decaying learning rate |

## Prerequisites

- **ROS2 Humble** (Ubuntu 22.04)
- **Python 3.10+** with: `numpy`, `scipy`, `matplotlib`
- **colcon** build tool
- **`bb_msgs`** — custom ROS2 interface package providing `bb_sensor_msgs/Ping`

### The `bb_msgs` package

This package depends on `bb_sensor_msgs` (for the `Ping` message type). The `bb_msgs` metapackage lives alongside this package in the same workspace:

```
src/
├── bb_msgs/
│   ├── bb_sensor_msgs/   ← build this first
│   ├── bb_asv_msgs/
│   └── ...
└── pinger_localization/  ← this package
```

The `Ping` message has these fields:
```python
int32 doa_deg      # Body-relative DOA (deg), 0..360 clockwise: 0 fwd, 90 right
int32 elevation    # Elevation angle (degrees)
int32 frequency    # Acoustic frequency (Hz)
float32 confidence # Detection confidence [0, 1]
```

## Build

```bash
# From the workspace root (e.g., ~/workspaces/isaac_ros-dev)

# 1. Build the message package first
colcon build --packages-up-to bb_sensor_msgs

# 2. Source the overlay
source install/setup.bash

# 3. Build this package
colcon build --packages-up-to pinger_localization

# 4. Source again
source install/setup.bash
```

> **Note:** If you get CMake cache errors about mismatched paths, clean the stale build first:
> ```bash
> rm -rf build/bb_sensor_msgs install/bb_sensor_msgs
> colcon build --packages-up-to bb_sensor_msgs pinger_localization
> ```

## Quick Start

### 1. Simulator only (test trajectory types and noise)

```bash
# Default: circle trajectory, jitter noise
ros2 launch pinger_localization pinger_sim.launch.py

# Try different trajectories
ros2 launch pinger_localization pinger_sim.launch.py trajectory_type:=figure8
ros2 launch pinger_localization pinger_sim.launch.py trajectory_type:=spiral noise_radius:=0.3

# All available overrides
ros2 launch pinger_localization pinger_sim.launch.py \
    trajectory_type:=lawnmower \
    trajectory_radius:=15.0 \
    trajectory_speed:=1.5 \
    pinger_north:=12.0 \
    pinger_east:=8.0 \
    noise_type:=both \
    doa_noise_std:=5.0 \
    ping_interval:=1.5
```

### 2. Everything (simulator + all 6 solvers + eval)

```bash
ros2 launch pinger_localization pinger_all.launch.py

# With custom config
ros2 launch pinger_localization pinger_all.launch.py \
    trajectory_type:=figure8 \
    noise_radius:=0.3 \
    pinger_north:=15.0 \
    pinger_east:=8.0
```

This produces a CSV log at `~/pinger_logs/summary_<timestamp>.csv`.

### 3. Compare specific solvers

```bash
# Only the iterative solvers
ros2 launch pinger_localization pinger_compare.launch.py \
    solvers:="iterative_ekf,iterative_rls,iterative_gd"

# Batch vs particle filter
ros2 launch pinger_localization pinger_compare.launch.py \
    solvers:="batch_sliding,particle_filter"

# With a different trajectory
ros2 launch pinger_localization pinger_compare.launch.py \
    solvers:="batch_full,batch_sliding" \
    trajectory_type:=spiral

# All available solver keys
ros2 launch pinger_localization pinger_compare.launch.py \
    solvers:="batch_sliding,batch_full,particle_filter,iterative_ekf,iterative_rls,iterative_gd"
```

### 4. Real-world operation (no simulator)

Run the solvers against your **real** sensors instead of the simulator:

```bash
ros2 launch pinger_localization pinger_irl.launch.py \
    solvers:="iterative_ekf,iterative_rls,iterative_gd" \
    odom_topic:=/auv5/odom_ned \
    ping_topic:=/sensors/ping \
    world_frame:=world
```

This launches only the solver nodes (no simulator, no eval node — there is no ground truth in real life). Topic remapping points the solvers at your real data sources instead of the simulator's `/auv5/odom_ned` and `/sensors/ping`.

The solvers need exactly two live inputs:

| Input | Type | Convention required |
|---|---|---|
| `odom_topic` | `nav_msgs/Odometry` | Pose in NED: `x`=North, `y`=East, `z`=Down; quaternion yaw measured from North toward East (same convention `base.py`'s `yaw_from_odom` expects). If your vehicle publishes ENU odom, convert to NED first. |
| `ping_topic` | `bb_sensor_msgs/Ping` | `doa_deg` is **body-relative** on `[0, 360)`, clockwise from the bow: 0° = forward, 90° = starboard, 180° = behind, 270° = port. The solver computes `world_bearing = vehicle_yaw + doa`. |

Estimates are published on `/pinger_localization/<name>/estimate` (`PointStamped`) and `/pinger_localization/<name>/estimate_pose` (`PoseStamped`), both in the `world_frame` frame. Record them with `ros2 bag record /pinger_localization/.../estimate_pose`.

## Offline Analysis

After a run produces a CSV log, analyze it with the plotting script:

```bash
# From anywhere (the script is installed with the package)
python3 -m pinger_localization.eval.analyze ~/pinger_logs/summary_20250101_120000.csv

# Compare multiple runs (Monte Carlo)
python3 -m pinger_localization.eval.analyze ~/pinger_logs/summary_*.csv

# Custom output directory
python3 -m pinger_localization.eval.analyze ~/pinger_logs/summary_*.csv --out ./my_plots
```

**Output plots** (saved to `./plots/` by default):

| Plot | Description |
|---|---|
| `error_vs_time.png` | Position error (m) vs elapsed time — one curve per solver |
| `error_vs_ping.png` | Position error vs ping count |
| `scatter.png` | True pinger position vs estimated positions (per solver) |
| `monte_carlo.png` | Mean ± 1σ band across multiple runs (when ≥2 CSVs provided) |

Plus a **convergence table** printed to stdout:
- RMSE over the full trajectory
- Final position error (m)
- Time to converge below 1 m and 0.5 m

## Topics

| Topic | Type | Publisher | Rate | Notes |
|---|---|---|---|---|
| `/auv5/odom_ned` | `nav_msgs/Odometry` | Simulator | 20 Hz | Vehicle pose + twist in NED |
| `/sensors/ping` | `bb_sensor_msgs/Ping` | Simulator | ~0.5 Hz | Body-relative DOA measurement |
| `/pinger_localization/ground_truth` | `geometry_msgs/PointStamped` | Simulator | Per ping | True pinger position (solvers do NOT see this) |
| `/pinger_localization/<name>/estimate` | `geometry_msgs/PointStamped` | Each solver | Per ping | Estimated pinger position in NED |
| `/pinger_localization/<name>/estimate_pose` | `geometry_msgs/PoseStamped` | Each solver | Per ping | Estimated pinger pose in NED, rendered at `estimate_depth` (visualization) |
| `/pinger_localization/eval/summary` | `std_msgs/String` | Eval node | Per GT | Text summary of all solver errors |

## Parameters

### Simulator (`pinger_simulator`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pinger_north` | float | `10.0` | True pinger North (m, NED) |
| `pinger_east` | float | `5.0` | True pinger East (m, NED) |
| `noise_radius` | float | `0.5` | Position jitter radius around true pinger (m) |
| `noise_type` | string | `"jitter"` | `"jitter"`, `"gaussian"`, or `"both"` |
| `doa_noise_std` | float | `3.0` | Gaussian DOA noise std dev (degrees) |
| `trajectory_type` | string | `"circle"` | `circle`, `lawnmower`, `figure8`, `spiral` |
| `trajectory_radius` | float | `10.0` | Approximate distance from pinger (m) |
| `trajectory_speed` | float | `1.0` | Vehicle speed (m/s) |
| `depth` | float | `-2.0` | Constant vehicle depth (m, NED) |
| `ping_interval` | float | `2.0` | Seconds between pings |
| `odom_rate` | float | `20.0` | Odometry publish rate (Hz) |
| `world_frame` | string | `"odom_ned"` | Frame name for odom, ground truth, and pinger positions |

### Per-solver parameters

Each solver has module-level constants at the top of its source file (tunable via ROS2 parameter overrides). Key ones:

| Solver | Key Parameter | Default | Description |
|---|---|---|---|
| `batch_sliding` | `window_size` | `20` | Number of recent measurements in window |
| `batch_full` | `use_ransac` | `True` | Enable RANSAC for outlier robustness |
| `batch_full` | `ransac_inlier_thresh` | `5.0` | RANSAC inlier threshold (degrees) |
| `batch_full` | `solve_every_n_pings` | `5` | Re-solve frequency |
| `particle_filter` | `num_particles` | `5000` | Number of particles |
| `particle_filter` | `sigma_doa_deg` | `5.0` | DOA measurement noise std dev |
| `iterative_ekf` | `q_diag` | `0.01` | Process noise diagonal |
| `iterative_rls` | `forgetting_factor` | `0.98` | RLS forgetting factor |
| `iterative_gd` | `learning_rate` | `0.05` | Initial learning rate |
| `iterative_gd` | `gd_steps` | `20` | Gradient descent steps per ping |

All solvers additionally share a common `estimate_depth` parameter (base class):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `world_frame` | string | `"odom_ned"` | Frame name the pinger estimate is published in (e.g. `world`, `map`) |
| `estimate_depth` | float | `1.0` | Depth (m, NED) of the visualization pose (`estimate_pose`) — renders underwater |
| `odom_history_size` | int | `1000` | Number of odometry samples buffered per solver |

## Coordinate Convention

- **World frame:** NED (North-East-Down)
  - X = North, Y = East, Z = Down (negative altitude)
- **Frame name:** the odom, ground truth, and pinger estimates are all published in the frame named by the `world_frame` parameter (default `"odom_ned"`). Set it to your real-world frame (e.g. `world_frame:=world`) at launch — the estimates are computed in whatever frame the odometry pose is expressed in, so relabeling is only correct when that matches your world frame.
- **DOA**: Body-relative degrees, reported on `[0, 360)` and increasing clockwise from the vehicle (viewed from above)
  - 0° = directly ahead of the vehicle (bow)
  - 90° = starboard (right side)
  - 180° = directly behind
  - 270° = port (left side)
- **Transformation:** `world_bearing = vehicle_yaw + doa_body` (wrap handled internally)

## Noise Models

The simulator supports three noise modes (set via `noise_type`):

| Mode | Behavior |
|---|---|
| `jitter` | Each ping uses a random pinger position sampled uniformly from a 2D circle (radius = `noise_radius`) around the true position — simulates sound-speed uncertainty, refraction, multipath |
| `gaussian` | Direct Gaussian noise added to the DOA measurement (std dev = `doa_noise_std` degrees) — simulates hydrophone array measurement noise |
| `both` | Both position jitter and DOA noise applied |

## Trajectory Types

| Type | Description |
|---|---|
| `circle` | Vehicle moves in a circle offset from the pinger |
| `lawnmower` | Back-and-forth survey lines covering the pinger area |
| `figure8` | Two connected loops (lemniscate) for diverse bearing angles, constant-speed arc-length parameterized |
| `spiral` | Expanding spiral from the pinger center |

## Package Structure

```
pinger_localization/
├── README.md
├── package.xml
├── CMakeLists.txt
├── config/
│   └── pinger_sim.yaml
├── launch/
│   ├── pinger_sim.launch.py       # Simulator only
│   ├── pinger_all.launch.py       # Simulator + all solvers + eval
│   ├── pinger_compare.launch.py   # Configurable solver subset
│   └── pinger_irl.launch.py       # Real sensors only (no simulator)
└── pinger_localization/
    ├── __init__.py
    ├── pinger_simulator.py         # Simulator node
    ├── eval_node.py                # Live evaluation + CSV logging node
    ├── sim/
    │   ├── __init__.py
    │   └── trajectory.py           # Circle, Lawnmower, Figure8, Spiral
    ├── solvers/
    │   ├── __init__.py
    │   ├── base.py                 # BaseSolver: odom buffer, bearing math, helpers
    │   ├── batch_sliding.py        # Sliding-window least-squares
    │   ├── batch_full.py           # Full-trajectory + RANSAC
    │   ├── particle_filter.py      # 2D stationary particle filter
    │   ├── iterative_ekf.py        # Extended Kalman Filter
    │   ├── iterative_rls.py        # Recursive Least Squares
    │   └── iterative_gd.py         # Online Gradient Descent
    └── eval/
        ├── __init__.py
        ├── metrics.py              # RMSE, convergence, summary helpers
        └── analyze.py              # Offline plotting & comparison script
```

## Dependencies

| Dependency | Purpose |
|---|---|
| `rclpy` | ROS2 Python client library |
| `nav_msgs` | `Odometry` message |
| `geometry_msgs` | `PointStamped`, `PoseStamped` messages |
| `bb_sensor_msgs` | `Ping` message (from `bb_msgs` metapackage) |
| `std_msgs` | `String` message (eval summary) |
| `numpy` | Numerical arrays, vectorized particle ops |
| `scipy` | `scipy.optimize.minimize` for batch solvers |
| `matplotlib` | Offline analysis plots |
| `launch` / `launch_ros` | ROS2 launch file support |
