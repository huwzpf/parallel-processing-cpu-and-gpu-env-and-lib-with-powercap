"""
Plot the EDP optimisation trajectory from a cudampilib run log.

The library emits one averaged optimiser objective per step as a

    ... [WARN] [EDP] edp=<value>

line (the value the optimiser actually consumes, i.e. the mean EDP over the
N-window averaging interval). This script parses those lines in order and saves a
plot of EDP vs optimiser step next to the input log as `trajectory_<name>.png`,
annotating the start point, the minimum, and the end point.

Usage:
    python generate_trajectory.py <log_file>
    python generate_trajectory.py -d <directory>   # all *.txt in the directory
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ... [<timestamp>] ... [WARN] [EDP] edp=<value>
_EDP_RE = re.compile(r"\[EDP\]\s+edp=(?P<edp>[\d.eE+\-]+)")


def parse_edp_trajectory(log_path: str) -> list[float]:
    """Return the per-step EDP values in log order from the [EDP] lines."""
    edps = []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            if "[EDP]" not in line:
                continue
            m = _EDP_RE.search(line)
            if m:
                edps.append(float(m.group("edp")))
    return edps


def plot_trajectory(edps: list[float], out_path: str, title: str) -> None:
    steps = list(range(len(edps)))
    min_i = min(steps, key=lambda i: edps[i])
    end_i = len(edps) - 1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, edps, "-", color="steelblue", linewidth=1.2, alpha=0.8, zorder=1)

    # Annotated points: (index, label, colour, vertical offset for the text)
    start_edp = edps[0]
    points = [
        (0, "start", "tab:green", 14),
        (min_i, "minimum", "tab:red", -18),
        (end_i, "end", "tab:purple", 14),
    ]
    for i, label, colour, dy in points:
        text = f"{label}\nstep {i}, edp={edps[i]:.4f}"
        if i != 0:  # % change vs start for minimum and end
            pct = (edps[i] - start_edp) / start_edp * 100.0
            text += f"\n{pct:+.1f}% vs start"
        ax.scatter([i], [edps[i]], color=colour, s=70, zorder=4,
                   edgecolor="black", linewidth=0.5)
        ax.annotate(text,
                    xy=(i, edps[i]), xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom" if dy > 0 else "top", fontsize=8,
                    color=colour, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              edgecolor=colour, alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.6))

    ax.set_xlabel("optimiser step")
    ax.set_ylabel("EDP (averaged objective)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def process_log(log_path: str) -> bool:
    """Plot one log's trajectory. Returns True if a plot was written, False if the
    file has no [EDP] lines (skipped, no plot generated)."""
    edps = parse_edp_trajectory(log_path)
    if not edps:
        print(f"[trajectory] skip (no [EDP] lines): {log_path}")
        return False

    name = os.path.splitext(os.path.basename(log_path))[0]
    out_path = os.path.join(os.path.dirname(log_path), f"trajectory_{name}.png")
    plot_trajectory(edps, out_path, title=name)
    print(f"[trajectory] {os.path.basename(log_path)}: {len(edps)} steps, "
          f"first={edps[0]:.6f} last={edps[-1]:.6f} best={min(edps):.6f} -> {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the [EDP] optimisation trajectory from a run log.")
    parser.add_argument("path", help="Path to a run log, or a directory with -d")
    parser.add_argument("-d", "--directory", action="store_true",
                        help="Treat PATH as a directory and process all *.txt in it "
                             "(files without [EDP] lines are skipped)")
    args = parser.parse_args()

    path = os.path.abspath(args.path)

    if args.directory:
        if not os.path.isdir(path):
            sys.exit(f"[trajectory] not a directory: {path}")
        logs = sorted(glob.glob(os.path.join(path, "*.txt")))
        if not logs:
            sys.exit(f"[trajectory] no *.txt files in {path}")
        written = sum(process_log(p) for p in logs)
        print(f"[trajectory] done: {written} plotted, {len(logs) - written} skipped "
              f"of {len(logs)} *.txt files")
    else:
        if not os.path.isfile(path):
            sys.exit(f"[trajectory] not a file: {path} (use -d for a directory)")
        if not process_log(path):
            sys.exit(1)


if __name__ == "__main__":
    main()
