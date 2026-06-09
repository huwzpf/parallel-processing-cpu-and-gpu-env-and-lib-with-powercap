"""
SNR diagnostic for EDP gradient optimization.

This runs the library in EQUAL_SPLIT_EDP_MONITOR mode: power caps are held fixed
(exactly like EQUAL_SPLIT), but the power capping manager still executes so it
measures EDP and emits, once per optimizer step (i.e. every N=optimizer_step_interval
sync windows, on the N-window-averaged value), a minimal log line:

    [EDP] edp=<value>

This script parses *only* those [EDP] lines.  Each line is one EDP estimate the
optimizer would have consumed at the current cap.

For a fixed cap, the spread of these estimates is the measurement *noise*.  The
difference in mean EDP between two caps is the *signal* a gradient optimizer would
have to detect.  SNR = signal / noise tells us whether the EDP landscape is
distinguishable above noise — i.e. whether gradient optimization can work for a
given (batch_size, N).

N is the optimizer_step_interval (the number of windows averaged per estimate).
The dataset length scales with N so each run yields a comparable number of EDP
estimates regardless of N or batch size:

    iters = round(N * number_of_nodes * batch_size / 2)

Usage:
    python snr_test.py [--app {cnn,rnn}] [--out-dir OUTPUT_DIR] [--nodes N [N ...]]
    python snr_test.py --plots-only [--out-dir OUTPUT_DIR]   # replot saved CSVs

Each node count in --nodes runs the full experiment + plot pipeline into its own
directory named '<out-dir>_<n>_nodes' (e.g. snr_results_4_nodes, snr_results_8_nodes).

Everything is parameterised so you can also import the functions directly.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import single_app_run
from models import RunParameters


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def compute_iters(n: int, number_od_nodes: int, batch_size: int) -> int:
    """Dataset iteration count for a run: round(N * nodes * batch_size / 20.0). - treat 20 as default batch size to keep iters in a reasonable range across different batch sizes."""
    return int(round(n * number_od_nodes * batch_size / 20.0))


def parse_edp_estimates(log_file: str) -> pd.DataFrame | None:
    """
    Parse the minimal '[EDP] edp=<value>' lines from a cudampilib log file.

    Returns a DataFrame with columns (step, edp) — one row per optimizer step —
    or None if no [EDP] lines were found.
    """
    pattern = re.compile(r"\[EDP\]\s+edp=([\d.eE+\-]+)")
    edps = []
    try:
        with open(log_file, "r", errors="replace") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    edps.append(float(m.group(1)))
    except FileNotFoundError:
        print(f"[parse_edp_estimates] Log file not found: {log_file}")
        return None

    if not edps:
        return None
    return pd.DataFrame({"step": range(len(edps)), "edp": edps})


def _run_edp_monitor(app_name: str, batch_size: int, cap: float, step_interval: int,
                     n_runs: int, out_dir: str, number_od_nodes: int = 8) -> pd.DataFrame | None:
    """
    Run EQUAL_SPLIT_EDP_MONITOR at a fixed cap and collect every [EDP] estimate.
    Returns a DataFrame of per-step estimates tagged with rep / batch_size /
    base_cap / optimizer_step_interval, or None if nothing was captured.
    """
    iters = compute_iters(step_interval, number_od_nodes, batch_size)
    params = RunParameters(
        app_name=app_name,
        cpu_enabled=True,
        number_of_streams=2,
        number_od_nodes=number_od_nodes,
        batch_size=batch_size,
        strategy="EQUAL_SPLIT_EDP_MONITOR",
        cpu_power_scaling=0.0,
        initial_cpu_batch_size_scaling=0,
        start_powercap=cap,
        iters=iters,
        optimizer_step_interval=step_interval,
    )

    frames = []
    for rep in range(n_runs):
        log_path = os.path.abspath(os.path.join(
            out_dir, f"log_{app_name}_bs{batch_size}_cap{cap:.2f}_N{step_interval}_rep{rep}.txt"
        ))
        print(f"[SNR] MONITOR app={app_name} bs={batch_size} cap={cap:.3f} "
              f"N={step_interval} iters={iters} rep={rep + 1}/{n_runs}")
        single_app_run(params, log_save_path=log_path)
        df = parse_edp_estimates(log_path)
        if df is None or df.empty:
            print(f"[SNR]   no [EDP] estimates parsed from {log_path}")
            continue
        df = df.copy()
        df["rep"] = rep
        df["batch_size"] = batch_size
        df["base_cap"] = cap
        df["optimizer_step_interval"] = step_interval
        df["iters"] = iters
        frames.append(df)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def run_snr_experiments(
    app_name: str,
    batch_sizes: list[int],
    base_caps: list[float],
    n_values: list[int],
    n_runs: int = 3,
    out_dir: str = "snr_results",
    number_od_nodes: int = 8,
) -> dict:
    """
    Run the EDP-monitor SNR diagnostic over the grid (batch_size, N, base_cap).

    Collects the raw [EDP] estimates, saves them to edp_estimates_raw.csv, then
    hands them to process_snr_results() for statistics, SNR, and logging.

    Returns the dict produced by process_snr_results.
    """
    out_dir = os.path.abspath(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    caps = sorted(base_caps)
    frames = []
    for batch_size in batch_sizes:
        for N in n_values:
            for cap in caps:
                df = _run_edp_monitor(app_name, batch_size, cap, N, n_runs, out_dir, number_od_nodes)
                if df is not None:
                    frames.append(df)

    if not frames:
        print("[SNR] No [EDP] estimates collected from any run.")
        empty = pd.DataFrame()
        return {"estimates": empty, "per_cap": empty, "snr_grid": empty}

    estimates = pd.concat(frames, ignore_index=True)
    estimates.to_csv(os.path.join(out_dir, "edp_estimates_raw.csv"), index=False)
    print(f"[SNR] Collected {len(estimates)} EDP estimates across "
          f"{estimates[['batch_size', 'optimizer_step_interval', 'base_cap']].drop_duplicates().shape[0]} "
          f"(batch_size, N, cap) cells; raw data -> {out_dir}/edp_estimates_raw.csv")

    return process_snr_results(estimates, out_dir=out_dir)


# ---------------------------------------------------------------------------
# Data processing (separated from experiment execution)
# ---------------------------------------------------------------------------

def process_snr_results(estimates: pd.DataFrame, out_dir: str = "snr_results") -> dict:
    """
    Turn raw [EDP] estimates into per-cap noise statistics and SNR between
    adjacent caps, saving CSVs and printing a readable summary.

    Returns a dict with:
      "estimates" : the raw per-step estimates (unchanged)
      "per_cap"   : pd.DataFrame, noise stats per (batch_size, N, cap)
      "snr_grid"  : pd.DataFrame, SNR per (batch_size, N, adjacent cap pair)
    """
    out_dir = os.path.abspath(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if estimates is None or estimates.empty:
        print("[SNR] process_snr_results: no estimates to process.")
        empty = pd.DataFrame()
        return {"estimates": estimates, "per_cap": empty, "snr_grid": empty}

    caps = sorted(estimates["base_cap"].unique())

    # --- Per-cap noise statistics ---
    per_cap_rows = []
    for (batch_size, N, cap), grp in estimates.groupby(
        ["batch_size", "optimizer_step_interval", "base_cap"]
    ):
        edp = grp["edp"].values
        mean = float(np.mean(edp))
        std = float(np.std(edp, ddof=1)) if len(edp) >= 2 else float("nan")
        per_cap_rows.append({
            "batch_size": batch_size,
            "N": N,
            "cap": cap,
            "edp_mean": mean,
            "edp_std": std,
            "edp_cv": (std / mean) if mean else float("nan"),
            "n_estimates": len(edp),
        })
    per_cap = pd.DataFrame(per_cap_rows)
    per_cap.to_csv(os.path.join(out_dir, "per_cap_stats.csv"), index=False)

    # --- SNR between adjacent caps, per (batch_size, N) ---
    snr_rows = []
    for (batch_size, N), grp in per_cap.groupby(["batch_size", "N"]):
        sub = grp.set_index("cap")
        present = [c for c in caps if c in sub.index]
        for lo, hi in zip(present, present[1:]):
            mean_lo, std_lo = sub.loc[lo, "edp_mean"], sub.loc[lo, "edp_std"]
            mean_hi, std_hi = sub.loc[hi, "edp_mean"], sub.loc[hi, "edp_std"]
            signal = abs(mean_hi - mean_lo)
            # Pooled variance of edp estimates at the two caps
            noise = float(np.sqrt(np.nanmean([std_lo ** 2, std_hi ** 2])))
            snr = signal / noise if noise > 0 else float("nan")
            snr_rows.append({
                "batch_size": batch_size,
                "N": N,
                "cap_lo": lo,
                "cap_hi": hi,
                "signal": signal,
                "noise_std": noise,
                "snr": snr,
            })
    snr_grid = pd.DataFrame(snr_rows)
    snr_grid.to_csv(os.path.join(out_dir, "snr_grid.csv"), index=False)

    _log_summary(per_cap, snr_grid, out_dir)

    return {"estimates": estimates, "per_cap": per_cap, "snr_grid": snr_grid}


def _log_summary(per_cap: pd.DataFrame, snr_grid: pd.DataFrame, out_dir: str) -> None:
    """Print a readable summary of the per-cap noise stats and SNR grid."""
    pd.set_option("display.float_format", lambda v: f"{v:.6g}")
    print("\n" + "=" * 78)
    print("SNR DIAGNOSTIC SUMMARY")
    print("=" * 78)

    print("\nPer-cap EDP statistics (noise = within-cap spread of [EDP] estimates):")
    cols = ["batch_size", "N", "cap", "edp_mean", "edp_std", "edp_cv", "n_estimates"]
    print(per_cap[cols].sort_values(["batch_size", "N", "cap"]).to_string(index=False))

    if not snr_grid.empty:
        print("\nSNR between adjacent caps (signal = |Δ mean EDP|, noise = pooled std):")
        print(snr_grid.sort_values(["batch_size", "N", "cap_lo"]).to_string(index=False))

        detectable = snr_grid[snr_grid["snr"] >= 1.0]
        print(f"\nCap pairs with SNR >= 1 (signal rises above noise): "
              f"{len(detectable)}/{len(snr_grid)}")
        if not detectable.empty:
            for _, r in detectable.sort_values("snr", ascending=False).iterrows():
                print(f"  bs={int(r['batch_size'])} N={int(r['N'])} "
                      f"caps {r['cap_lo']:.2f}->{r['cap_hi']:.2f}: SNR={r['snr']:.2f}")
        best = snr_grid.loc[snr_grid["snr"].idxmax()]
        print(f"\nBest SNR overall: {best['snr']:.2f} "
              f"(bs={int(best['batch_size'])}, N={int(best['N'])}, "
              f"caps {best['cap_lo']:.2f}->{best['cap_hi']:.2f})")

    print("\nCSV outputs:")
    print(f"  {out_dir}/edp_estimates_raw.csv")
    print(f"  {out_dir}/per_cap_stats.csv")
    print(f"  {out_dir}/snr_grid.csv")
    print("=" * 78 + "\n")


def load_results(out_dir: str = "snr_results") -> dict:
    """
    Rebuild the results dict from CSVs written by a previous experiment run,
    so plots can be regenerated without re-running the (slow) experiments.

    Reads edp_estimates_raw.csv, per_cap_stats.csv and snr_grid.csv from
    out_dir. Missing files become empty DataFrames (the plotters skip them).
    """
    out_dir = os.path.abspath(out_dir)

    def _read(name: str) -> pd.DataFrame:
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            print(f"[SNR] load_results: {path} not found, skipping.")
            return pd.DataFrame()
        return pd.read_csv(path)

    return {
        "estimates": _read("edp_estimates_raw.csv"),
        "per_cap": _read("per_cap_stats.csv"),
        "snr_grid": _read("snr_grid.csv"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _all_pair_snr(per_cap: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SNR between *every* cap pair (lo < hi), not just adjacent ones,
    directly from the per-cap noise statistics.

    snr_grid.csv only stores adjacent ("1-step") pairs, but per_cap_stats.csv
    holds the per-cap edp_mean / edp_std, which is all that's needed to compute
    the SNR for any pair. Deriving it here keeps this purely a plotting concern,
    so it works in --plots-only mode without regenerating results.

    'eps' is the cap gap (cap_hi - cap_lo): the smallest gap is the adjacent
    pair (the original snr_grid), larger gaps = wider cap separations
    (e.g. 0.5->0.6 is eps=0.10). Returns columns:
    batch_size, N, cap_lo, cap_hi, eps, signal, noise_std, snr.
    """
    if per_cap is None or per_cap.empty:
        return pd.DataFrame()

    caps = sorted(per_cap["cap"].unique())

    rows = []
    for (batch_size, N), grp in per_cap.groupby(["batch_size", "N"]):
        sub = grp.set_index("cap")
        present = [c for c in caps if c in sub.index]
        for i, lo in enumerate(present):
            for hi in present[i + 1:]:
                mean_lo, std_lo = sub.loc[lo, "edp_mean"], sub.loc[lo, "edp_std"]
                mean_hi, std_hi = sub.loc[hi, "edp_mean"], sub.loc[hi, "edp_std"]
                signal = abs(mean_hi - mean_lo)
                # Pooled std of the two caps' EDP estimates (same noise model as
                # the adjacent-pair SNR in process_snr_results).
                noise = float(np.sqrt(np.nanmean([std_lo ** 2, std_hi ** 2])))
                snr = signal / noise if noise > 0 else float("nan")
                rows.append({
                    "batch_size": batch_size,
                    "N": N,
                    "cap_lo": lo,
                    "cap_hi": hi,
                    # round to absorb float-subtraction noise so equal gaps group
                    "eps": round(hi - lo, 6),
                    "signal": signal,
                    "noise_std": noise,
                    "snr": snr,
                })
    return pd.DataFrame(rows)


def plot_snr_results(results: dict, out_dir: str = "snr_results") -> None:
    """
    Generate diagnostic plots from the dict returned by run_snr_experiments.

    1. EDP estimate distribution per cap (one figure per N and batch_size)
    2. SNR vs N (one line per batch_size and adjacent cap pair)
    3. Per-step noise (EDP std) vs N, log-log (shows the ~1/sqrt(N) averaging gain)
    4. SNR vs batch_size (one line per N and adjacent cap pair)
    """
    out_dir = os.path.abspath(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    estimates = results.get("estimates", pd.DataFrame())
    per_cap = results.get("per_cap", pd.DataFrame())
    snr_grid = results.get("snr_grid", pd.DataFrame())

    # --- Plot 1: EDP estimate distributions (one figure per N and batch_size) ---
    if len(estimates):
        for N in sorted(estimates["optimizer_step_interval"].unique()):
            sub_N = estimates[estimates["optimizer_step_interval"] == N]
            for bs in sorted(sub_N["batch_size"].unique()):
                sub = sub_N[sub_N["batch_size"] == bs]
                fig, ax = plt.subplots(figsize=(8, 4))
                for cap in sorted(sub["base_cap"].unique()):
                    edp_vals = sub[sub["base_cap"] == cap]["edp"].values
                    ax.hist(edp_vals, bins=max(5, len(edp_vals) // 4 + 1),
                            alpha=0.55, label=f"cap={cap:.2f} (n={len(edp_vals)})")
                ax.set_xlabel("EDP estimate")
                ax.set_ylabel("Count")
                ax.set_title(f"EDP estimate distribution — bs={bs}, N={int(N)}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"edp_dist_bs{bs}_N{int(N)}.png"), dpi=120)
                plt.close(fig)

    # --- Plot 2: SNR vs N (adjacent caps + wider cap separations) ---
    # Computed from per_cap (not snr_grid) so every cap pair is available, not
    # just adjacent ones. One line per cap gap eps=cap_hi-cap_lo (mean over all
    # cap pairs with that gap and all batch sizes); wider gaps = larger signal.
    pair_snr = _all_pair_snr(per_cap)
    if len(pair_snr):
        fig, ax = plt.subplots(figsize=(7, 4))
        eps_values = sorted(pair_snr["eps"].unique())
        cmap = plt.get_cmap("viridis")
        for i, eps in enumerate(eps_values):
            sub = pair_snr[pair_snr["eps"] == eps]
            trend = sub.groupby("N")["snr"].mean().sort_index()
            n_pairs = sub[["batch_size", "cap_lo", "cap_hi"]].drop_duplicates().shape[0]
            color = cmap(i / max(1, len(eps_values) - 1))
            ax.plot(trend.index, trend.values, "o-", color=color,
                    label=f"eps = {eps:g} (mean of {n_pairs} pairs)")
        ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="SNR=1 threshold")
        ax.set_xlabel("N (optimizer_step_interval)")
        ax.set_ylabel("SNR (signal / noise)")
        ax.set_title("EDP SNR vs N-window averaging (by cap separation)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "snr_vs_N.png"), dpi=120)
        plt.close(fig)

    # --- Plot 3: per-step relative noise (CV) vs N (log-log, with 1/sqrt(N) reference) ---
    if len(per_cap):
        fig, ax = plt.subplots(figsize=(7, 4))
        for (bs, cap), group in per_cap.groupby(["batch_size", "cap"]):
            group = group.sort_values("N")
            ax.loglog(group["N"], group["edp_cv"], "o-", label=f"bs={bs}, cap={cap:.2f}")
            # We plot the coefficient of variation (std / mean) rather than the raw
            # std so curves are comparable across configs with very different EDP
            # magnitudes: absolute std grows with EDP, but CV is dimensionless
            # relative noise (CV = 1/SNR), which is what the optimizer actually
            # "sees" when deciding if one cap's EDP is reliably below another's.
            #
            # Each EDP estimate is the mean of N per-step measurements. If those
            # measurements were independent, identically-distributed samples, the
            # std of their mean would shrink as sigma/sqrt(N) (standard error of
            # the mean); since the mean EDP is ~constant in N, the CV shrinks the
            # same way -- a slope of -1/2 on this log-log plot. We overlay that
            # ideal 1/sqrt(N) curve, anchored to the measured CV at the smallest
            # N, as a baseline: if the measured noise tracks it, averaging behaves
            # as pure random-error reduction; if it flattens above the reference,
            # a noise floor / correlated (autocorrelated, drifting) noise dominates
            # and raising the optimizer_step_interval buys little extra precision.
            # 1/sqrt(N) reference anchored at the smallest N
            n0 = group["N"].iloc[0]
            s0 = group["edp_cv"].iloc[0]
            if np.isfinite(s0) and s0 > 0:
                ref = s0 * np.sqrt(n0 / group["N"].astype(float))
                ax.loglog(group["N"], ref, ":", color="gray", linewidth=0.8)
        ax.set_xlabel("N (optimizer_step_interval)")
        ax.set_ylabel("EDP estimate CV (std / mean)")
        ax.set_title("Per-step relative EDP noise (CV) vs N (dotted = 1/√N reference)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "noise_vs_N.png"), dpi=120)
        plt.close(fig)

    # --- Plot 4: SNR vs batch_size (one line per N and adjacent cap pair) ---
    if len(snr_grid):
        fig, ax = plt.subplots(figsize=(7, 4))
        for (N, lo, hi), group in snr_grid.groupby(["N", "cap_lo", "cap_hi"]):
            group = group.sort_values("batch_size")
            ax.plot(group["batch_size"], group["snr"], "o-",
                    label=f"N={int(N)}, {lo:.2f}->{hi:.2f}")
        # average trend across all (N, cap pair) curves
        mean_trend = snr_grid.groupby("batch_size")["snr"].mean().sort_index()
        ax.plot(mean_trend.index, mean_trend.values, "s--", color="black",
                linewidth=2.0, markersize=6, label="mean", zorder=5)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="SNR=1 threshold")
        ax.set_xlabel("batch_size")
        ax.set_ylabel("SNR (signal / noise)")
        ax.set_title("EDP SNR vs batch_size")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "snr_vs_batchsize.png"), dpi=120)
        plt.close(fig)

    print(f"[SNR] Plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNR diagnostic for EDP optimization (EQUAL_SPLIT_EDP_MONITOR)")
    parser.add_argument("--app", choices=["cnn", "rnn"], default="cnn")
    parser.add_argument("--out-dir", default="snr_results",
                        help="Base output dir; each node count gets its own '<out-dir>_<n>_nodes'")
    parser.add_argument("--nodes", nargs="+", type=int, default=[8, 4],
                        help="Node counts to sweep; experiments+plots are repeated for each")
    parser.add_argument("--n-runs", type=int, default=3, help="Repetitions per (batch_size, N, cap)")
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[50],
        help="Batch sizes to sweep"
    )
    parser.add_argument(
        "--base-caps", nargs="+", type=float, default=[0.4, 0.45, 0.5, 0.55, 0.6],
        help="Fixed cap fractions (of range) to compare"
    )
    parser.add_argument(
        "--n-values", nargs="+", type=int, default=[5, 10, 20],
        help="N (optimizer_step_interval) values; iters = round(N * nodes * batch_size / 2)"
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip experiments; regenerate plots from CSVs already in --out-dir"
    )
    args = parser.parse_args()

    # Run the identical experiment + plotting pipeline once per node count,
    # writing each into its own directory (e.g. snr_results_4_nodes).
    for nodes in args.nodes:
        node_out_dir = f"{args.out_dir}_{nodes}_nodes"
        print(f"\n[SNR] ===== node count = {nodes} -> {node_out_dir} =====")

        if args.plots_only:
            results = load_results(out_dir=node_out_dir)
        else:
            results = run_snr_experiments(
                app_name=args.app,
                batch_sizes=args.batch_sizes,
                base_caps=args.base_caps,
                n_values=args.n_values,
                n_runs=args.n_runs,
                out_dir=node_out_dir,
                number_od_nodes=nodes,
            )
        plot_snr_results(results, out_dir=node_out_dir)
