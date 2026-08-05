#!/usr/bin/python3
"""
Offline Analysis Script for Pinger Localization Evaluation.

Reads CSV log files produced by the eval node and generates comparison plots.

Usage:
    python3 analyze.py ~/pinger_logs/summary_20250101_120000.csv
    python3 analyze.py ~/pinger_logs/summary_*.csv  # Compare multiple runs

Output:
    Saves figures to ./plots/ (or the directory specified by --out).
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Headless backend.
import matplotlib.pyplot as plt
import numpy as np

from pinger_localization.eval.metrics import solver_summary, convergence_time, rmse


# === Parameters ===
OUT_DIR = "./plots"
FIG_SIZE = (14, 10)
DPI = 150
# =================


def parse_solver_columns(header: List[str]) -> List[str]:
    """Extract solver names from CSV header columns.

    Header pattern: timestamp_ns, gt_north, gt_east, [solver]_north, [solver]_east, [solver]_error_m ...
    """
    names = []
    for col in header:
        if col.endswith("_north") and not col.startswith("gt_"):
            name = col[:-6]  # Strip "_north".
            names.append(name)
    return names


def load_csv(csv_path: str) -> Optional[Dict]:
    """Load a single eval CSV file.

    Returns a dictionary:
        {
            "timestamps": [t1, t2, ...],  # seconds from first GT
            "gt_north": [...],
            "gt_east": [...],
            "solvers": {
                "name": {
                    "north": [...],
                    "east": [...],
                    "error": [...],
                }
            }
        }
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        solver_names = parse_solver_columns(header)

        # Build column index map.
        col_idx = {h: i for i, h in enumerate(header)}

        data = {
            "timestamps": [],
            "gt_north": [],
            "gt_east": [],
            "solvers": {name: {"north": [], "east": [], "error": []}
                        for name in solver_names},
        }

        t0_ns = None
        for row in reader:
            ts_ns = int(row[col_idx["timestamp_ns"]])
            if t0_ns is None:
                t0_ns = ts_ns

            data["timestamps"].append((ts_ns - t0_ns) / 1e9)
            data["gt_north"].append(float(row[col_idx["gt_north"]]))
            data["gt_east"].append(float(row[col_idx["gt_east"]]))

            for name in solver_names:
                n_col = col_idx.get(f"{name}_north")
                e_col = col_idx.get(f"{name}_east")
                err_col = col_idx.get(f"{name}_error_m")
                if (n_col is not None and e_col is not None
                        and err_col is not None
                        and row[n_col] and row[e_col]):
                    data["solvers"][name]["north"].append(float(row[n_col]))
                    data["solvers"][name]["east"].append(float(row[e_col]))
                    data["solvers"][name]["error"].append(
                        float(row[err_col]) if row[err_col] else float("nan")
                    )
                else:
                    data["solvers"][name]["north"].append(float("nan"))
                    data["solvers"][name]["east"].append(float("nan"))
                    data["solvers"][name]["error"].append(float("nan"))

    return data


# ------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------

def plot_error_vs_time(data: Dict, out_dir: str, filename: str = ""):
    """Error (m) vs time for all solvers."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    colors = plt.cm.tab10.colors
    for i, (name, sdata) in enumerate(data["solvers"].items()):
        color = colors[i % len(colors)]
        t = data["timestamps"]
        err = sdata["error"]
        # Only plot at pings where this solver had an estimate.
        valid = [(ti, ei) for ti, ei in zip(t, err) if not np.isnan(ei)]
        if valid:
            vt, ve = zip(*valid)
            ax.plot(vt, ve, color=color, label=name, linewidth=1.0, alpha=0.9)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Pinger Localization: Position Error vs. Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fname = filename or "error_vs_time.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_error_vs_ping(data: Dict, out_dir: str):
    """Error vs ping count."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    colors = plt.cm.tab10.colors
    for i, (name, sdata) in enumerate(data["solvers"].items()):
        color = colors[i % len(colors)]
        err = sdata["error"]
        valid = [(j, err[j]) for j in range(len(err)) if not np.isnan(err[j])]
        if valid:
            pi, pe = zip(*valid)
            ax.plot(pi, pe, color=color, label=name, linewidth=1.0, alpha=0.9)

    ax.set_xlabel("Ping Number")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Pinger Localization: Position Error vs. Ping Count")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.savefig(os.path.join(out_dir, "error_vs_ping.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved error_vs_ping.png")


def plot_scatter(data: Dict, out_dir: str):
    """Per-solver scatter: true pinger vs estimated positions."""
    n_solvers = len(data["solvers"])
    cols = min(3, n_solvers)
    rows = (n_solvers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (name, sdata) in enumerate(data["solvers"].items()):
        ax = axes[i]
        # True pinger position.
        gt_n = np.mean(data["gt_north"])
        gt_e = np.mean(data["gt_east"])
        ax.plot(gt_n, gt_e, "r*", markersize=15, label="True Pinger")

        # Estimated positions (downsampled for clarity).
        n = sdata["north"]
        e = sdata["east"]
        step = max(1, len(n) // 200)
        valid_n = [n[j] for j in range(0, len(n), step) if not np.isnan(n[j])]
        valid_e = [e[j] for j in range(0, len(e), step) if not np.isnan(e[j])]

        ax.scatter(valid_n, valid_e, c=range(len(valid_n)), cmap="viridis",
                   s=15, alpha=0.6, edgecolors="none")
        ax.set_xlabel("North (m)")
        ax.set_ylabel("East (m)")
        ax.set_title(name)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    # Hide unused subplots.
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Pinger Localization: Estimated vs. True Position", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved scatter.png")


def print_convergence_table(data: Dict):
    """Print convergence statistics table."""
    print("\n" + "=" * 80)
    print(f"{'Solver':<20} {'RMSE(m)':>8} {'Final(m)':>8} {'Conv<1m(s)':>11} {'Conv<0.5m(s)':>13}")
    print("-" * 80)

    for name, sdata in data["solvers"].items():
        t = data["timestamps"]
        err = sdata["error"]
        # Filter valid entries.
        valid_t = [ti for ti, ei in zip(t, err) if not np.isnan(ei)]
        valid_e = [ei for ei in err if not np.isnan(ei)]

        summary = solver_summary(valid_t, valid_e, name)
        conv1 = f"{summary['convergence_1m']:.0f}" if summary['convergence_1m'] != float("inf") else "never"
        conv05 = f"{summary['convergence_0_5m']:.0f}" if summary['convergence_0_5m'] != float("inf") else "never"
        print(f"{name:<20} {summary['rmse']:>8.2f} {summary['final_error']:>8.2f} {conv1:>11} {conv05:>13}")

    print("=" * 80)


def plot_monte_carlo(data_list: List[Dict], out_dir: str):
    """If multiple runs provided, plot mean ± 1σ band per solver."""
    if len(data_list) < 2:
        return

    # Align all runs to a common time axis (interpolate).
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    colors = plt.cm.tab10.colors
    solver_names = list(data_list[0]["solvers"].keys())

    for i, name in enumerate(solver_names):
        color = colors[i % len(colors)]
        # Collect all error curves for this solver.
        all_t = []
        all_e = []
        for data in data_list:
            t = data["timestamps"]
            err = data["solvers"][name]["error"]
            valid = [(tj, ej) for tj, ej in zip(t, err) if not np.isnan(ej)]
            if valid:
                vt, ve = zip(*valid)
                all_t.append(np.array(vt))
                all_e.append(np.array(ve))

        if not all_t:
            continue

        # Interpolate all curves to the longest time axis.
        max_len = max(len(arr) for arr in all_t)
        ref_t = np.linspace(
            min(arr.min() for arr in all_t),
            max(arr.max() for arr in all_t),
            max_len,
        )
        interp_curves = []
        for t_arr, e_arr in zip(all_t, all_e):
            interp_curves.append(np.interp(ref_t, t_arr, e_arr))

        curves = np.array(interp_curves)
        mean = np.mean(curves, axis=0)
        std = np.std(curves, axis=0)

        ax.plot(ref_t, mean, color=color, label=name, linewidth=1.5)
        ax.fill_between(ref_t, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title(f"Monte Carlo: Mean ± 1σ ({len(data_list)} runs)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.savefig(os.path.join(out_dir, "monte_carlo.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved monte_carlo.png")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze pinger localization evaluation CSV logs."
    )
    parser.add_argument(
        "csv_files", nargs="+",
        help="One or more summary CSV files from the eval node.",
    )
    parser.add_argument(
        "--out", default=OUT_DIR,
        help=f"Output directory for plots (default: {OUT_DIR})",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load all CSV files.
    all_data = []
    for path in args.csv_files:
        print(f"Loading {path} ...")
        data = load_csv(path)
        if data is not None:
            all_data.append(data)

    if not all_data:
        print("No valid data loaded. Aborting.")
        sys.exit(1)

    # Use the first (or only) file for single-run plots.
    primary = all_data[0]

    print(f"\nGenerating plots in {args.out}/ ...")
    plot_error_vs_time(primary, args.out)
    plot_error_vs_ping(primary, args.out)
    plot_scatter(primary, args.out)

    print_convergence_table(primary)

    if len(all_data) > 1:
        plot_monte_carlo(all_data, args.out)

    print(f"\nDone. Plots saved to {args.out}/")


if __name__ == "__main__":
    main()
