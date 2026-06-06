"""
SNR diagnostic for EDP gradient optimization.

Runs EQUAL_SPLIT experiments at (cap - eps), cap, (cap + eps) across a grid of
(batch_size, iters) combinations to quantify whether the EDP signal is detectable
above measurement noise.  Also sweeps the N-window accumulation parameter.

Usage:
    python snr_test.py [--app {cnn,rnn}] [--out-dir OUTPUT_DIR]

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

from experiments import single_app_run, write_powercap_conf
from models import RunParameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_equal_split(app_name: str, batch_size: int, iters: int, cap: float,
                     n_reps: int, number_od_nodes: int = 8) -> list[dict]:
    """Run EQUAL_SPLIT at a fixed cap and return raw (duration, energy, edp) per rep."""
    params = RunParameters(
        app_name=app_name,
        cpu_enabled=True,
        number_of_streams=2,
        number_od_nodes=number_od_nodes,
        batch_size=batch_size,
        strategy="EQUAL_SPLIT",
        cpu_power_scaling=0.0,
        initial_cpu_batch_size_scaling=0,
        start_powercap=cap,
        iters=iters,
    )
    records = []
    for _ in range(n_reps):
        result = single_app_run(params)
        if result == -1:
            continue
        records.append({
            "duration": result.execution_duration,
            "energy": result.energy_used,
            "edp": result.execution_duration * result.energy_used,
            "cap": cap,
            "batch_size": batch_size,
            "iters": iters,
        })
    return records


def _run_gradient(app_name: str, batch_size: int, iters: int,
                  optimizer_step_interval: int, n_reps: int,
                  strategy: str = "EDP_GRADIENT_SPSA",
                  number_od_nodes: int = 8) -> list[dict]:
    """Run a gradient strategy and collect [EDP_SAMPLE] lines from the log."""
    params = RunParameters(
        app_name=app_name,
        cpu_enabled=True,
        number_of_streams=2,
        number_od_nodes=number_od_nodes,
        batch_size=batch_size,
        strategy=strategy,
        cpu_power_scaling=0.0,
        initial_cpu_batch_size_scaling=0,
        start_powercap=0.5,
        start_alpha=0.2,
        alpha_decay=0.98,
        epsilon_decay=0.95,
        gradient_opt_eps=0.2,
        edp_optimization_steps=0,
        iters=iters,
        optimizer_step_interval=optimizer_step_interval,
    )
    records = []
    for _ in range(n_reps):
        single_app_run(params)
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_snr_experiments(
    app_name: str,
    batch_sizes: list[int],
    iters_values: list[int],
    n_values: list[int],
    base_caps: list[float],
    eps: float = 0.05,
    n_runs: int = 5,
    out_dir: str = "snr_results",
    number_od_nodes: int = 8,
) -> dict:
    """
    Run the full SNR diagnostic grid.

    For each (batch_size, iters) pair:
      - Run EQUAL_SPLIT at (base_cap - eps), base_cap, (base_cap + eps)
        with n_runs repetitions each to measure noise and signal.
    For each N in n_values at a fixed (batch_size=batch_sizes[0], iters=iters_values[-1]):
      - Run EDP_GRADIENT_SPSA once and parse [EDP_SAMPLE] lines from cudampilib/log.txt.

    Returns a results dict with keys:
      "equal_split_records" : list of dicts
      "snr_grid"            : pd.DataFrame
      "gradient_records"    : dict {N: list of [EDP_SAMPLE] dicts}
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    equal_split_records = []
    for batch_size in batch_sizes:
        for iters in iters_values:
            for cap in base_caps:
                for c in [max(0.0, cap - eps), cap, min(1.0, cap + eps)]:
                    print(f"[SNR] EQUAL_SPLIT app={app_name} bs={batch_size} iters={iters} cap={c:.3f} x{n_runs}")
                    records = _run_equal_split(
                        app_name, batch_size, iters, c, n_runs, number_od_nodes
                    )
                    equal_split_records.extend(records)

    df_es = pd.DataFrame(equal_split_records)
    df_es.to_csv(os.path.join(out_dir, "equal_split_raw.csv"), index=False)

    snr_rows = []
    for batch_size in batch_sizes:
        for iters in iters_values:
            sub = df_es[(df_es["batch_size"] == batch_size) & (df_es["iters"] == iters)]
            for cap in base_caps:
                lo = max(0.0, cap - eps)
                hi = min(1.0, cap + eps)
                edp_lo = sub[sub["cap"] == lo]["edp"].values
                edp_hi = sub[sub["cap"] == hi]["edp"].values
                edp_base = sub[sub["cap"] == cap]["edp"].values
                if len(edp_base) < 2:
                    continue
                noise_std = float(np.std(edp_base, ddof=1))
                signal = abs(np.mean(edp_hi) - np.mean(edp_lo)) / 2.0 if len(edp_lo) and len(edp_hi) else 0.0
                snr = signal / noise_std if noise_std > 0 else 0.0
                snr_rows.append({
                    "batch_size": batch_size,
                    "iters": iters,
                    "cap": cap,
                    "noise_std": noise_std,
                    "signal": signal,
                    "snr": snr,
                    "n_samples": len(edp_base),
                })

    snr_grid = pd.DataFrame(snr_rows)
    snr_grid.to_csv(os.path.join(out_dir, "snr_grid.csv"), index=False)

    # Gradient runs with varying N (fixed largest iters, first batch_size)
    gradient_records: dict[int, list] = {}
    fixed_bs = batch_sizes[0]
    fixed_iters = iters_values[-1]
    log_path = Path.home() / "parallel-processing-cpu-and-gpu-env-and-lib-with-powercap" / "cudampilib" / "log.txt"
    for N in n_values:
        print(f"[SNR] GRADIENT N={N} app={app_name} bs={fixed_bs} iters={fixed_iters}")
        _run_gradient(app_name, fixed_bs, fixed_iters, N, n_runs=1, number_od_nodes=number_od_nodes)
        samples = parse_edp_samples(str(log_path))
        gradient_records[N] = samples.to_dict(orient="records") if samples is not None else []

    return {
        "equal_split_records": equal_split_records,
        "snr_grid": snr_grid,
        "gradient_records": gradient_records,
    }


def parse_edp_samples(log_file: str) -> pd.DataFrame | None:
    """
    Parse [EDP_SAMPLE] lines from a cudampilib log file.

    Expected format:
        [EDP_SAMPLE] window=W interval=N energy_J=X period_s=Y batches=Z edp=Q cap=v0,v1,...

    Returns a DataFrame with columns:
        window, interval, energy_J, period_s, batches, edp, cap_mean, cap_values (list)
    or None if no samples found.
    """
    pattern = re.compile(
        r"\[EDP_SAMPLE\]\s+"
        r"window=(\d+)\s+"
        r"interval=(\d+)\s+"
        r"phase=(\d+)\s+"
        r"energy_J=([\d.]+)\s+"
        r"period_s=([\d.]+)\s+"
        r"batches=(\d+)\s+"
        r"edp=([\d.eE+\-]+)\s+"
        r"avg_power=([\d.]+)W\s+"
        r"energy_devices=(\d+)\s+"
        r"cap=([\d.,]+)"
    )
    rows = []
    try:
        with open(log_file, "r", errors="replace") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    caps = [float(x) for x in m.group(10).split(",") if x]
                    rows.append({
                        "window": int(m.group(1)),
                        "interval": int(m.group(2)),
                        "phase": int(m.group(3)),
                        "energy_J": float(m.group(4)),
                        "period_s": float(m.group(5)),
                        "batches": int(m.group(6)),
                        "edp": float(m.group(7)),
                        "avg_power_W": float(m.group(8)),
                        "energy_devices": int(m.group(9)),
                        "cap_mean": float(np.mean(caps)) if caps else float("nan"),
                        "cap_values": caps,
                    })
    except FileNotFoundError:
        print(f"[parse_edp_samples] Log file not found: {log_file}")
        return None

    if not rows:
        return None
    return pd.DataFrame(rows)


def compute_snr(df_base: pd.DataFrame, df_plus: pd.DataFrame, df_minus: pd.DataFrame) -> dict:
    """
    Compute SNR from three EDP sample DataFrames at cap, cap+eps, cap-eps.

    Returns dict with: noise_std, signal, snr, n_samples
    """
    base_edp = df_base["edp"].values if df_base is not None and len(df_base) else np.array([])
    plus_edp = df_plus["edp"].values if df_plus is not None and len(df_plus) else np.array([])
    minus_edp = df_minus["edp"].values if df_minus is not None and len(df_minus) else np.array([])

    noise_std = float(np.std(base_edp, ddof=1)) if len(base_edp) >= 2 else float("nan")
    signal = abs(np.mean(plus_edp) - np.mean(minus_edp)) / 2.0 if len(plus_edp) and len(minus_edp) else float("nan")
    snr = signal / noise_std if noise_std > 0 and not np.isnan(signal) else float("nan")

    return {
        "noise_std": noise_std,
        "signal": signal,
        "snr": snr,
        "n_samples": len(base_edp),
    }


def plot_snr_results(results: dict, out_dir: str = "snr_results") -> None:
    """
    Generate four diagnostic plots from the results dict returned by run_snr_experiments.

    1. EDP distributions at (cap-eps, cap, cap+eps) per batch_size/iters combo
    2. SNR heatmap: batch_size x iters
    3. Noise floor (std of EDP at base cap) vs batch_size (log-log)
    4. SNR vs N (optimizer_step_interval) at fixed batch_size/iters
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df_es = pd.DataFrame(results.get("equal_split_records", []))
    snr_grid = results.get("snr_grid", pd.DataFrame())
    gradient_records = results.get("gradient_records", {})

    # --- Plot 1: EDP distributions ---
    if len(df_es):
        combos = df_es[["batch_size", "iters"]].drop_duplicates()
        for _, row in combos.iterrows():
            bs, it = int(row["batch_size"]), int(row["iters"])
            sub = df_es[(df_es["batch_size"] == bs) & (df_es["iters"] == it)]
            caps = sorted(sub["cap"].unique())
            fig, ax = plt.subplots(figsize=(8, 4))
            for cap in caps:
                edp_vals = sub[sub["cap"] == cap]["edp"].values
                ax.hist(edp_vals, bins=max(3, len(edp_vals) // 2 + 1), alpha=0.6, label=f"cap={cap:.3f}")
            ax.set_xlabel("EDP (J·s)")
            ax.set_ylabel("Count")
            ax.set_title(f"EDP distribution — bs={bs} iters={it}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"edp_dist_bs{bs}_iters{it}.png"), dpi=120)
            plt.close(fig)

    # --- Plot 2: SNR heatmap ---
    if len(snr_grid):
        pivot = snr_grid.pivot_table(index="iters", columns="batch_size", values="snr", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(max(4, len(pivot.columns)), max(3, len(pivot))))
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=max(2.0, pivot.values.max()))
        plt.colorbar(im, ax=ax, label="SNR")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.astype(int), fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(int), fontsize=9)
        ax.set_xlabel("batch_size")
        ax.set_ylabel("iters")
        ax.set_title("SNR heatmap (signal/noise of EDP gradient)")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                ax.text(j, i, f"{v:.2f}" if not np.isnan(v) else "N/A", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "snr_heatmap.png"), dpi=120)
        plt.close(fig)

    # --- Plot 3: noise floor vs batch_size ---
    if len(snr_grid):
        noise_by_bs = snr_grid.groupby("batch_size")["noise_std"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(noise_by_bs["batch_size"], noise_by_bs["noise_std"], "o-")
        ax.set_xlabel("batch_size")
        ax.set_ylabel("EDP noise std (J·s)")
        ax.set_title("Noise floor vs batch_size (log-log)")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "noise_vs_batchsize.png"), dpi=120)
        plt.close(fig)

    # --- Plot 4: SNR vs N ---
    if gradient_records:
        ns = sorted(gradient_records.keys())
        snrs = []
        for N in ns:
            records = gradient_records[N]
            if not records:
                snrs.append(float("nan"))
                continue
            df_n = pd.DataFrame(records)
            edp_vals = df_n["edp"].values
            snrs.append(float(np.std(edp_vals, ddof=1)) if len(edp_vals) >= 2 else float("nan"))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, snrs, "s-")
        ax.set_xlabel("N (optimizer_step_interval)")
        ax.set_ylabel("EDP std across windows (J·s)")
        ax.set_title("EDP noise vs N-window averaging")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "snr_vs_N.png"), dpi=120)
        plt.close(fig)

    print(f"[SNR] Plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNR diagnostic for EDP gradient optimization")
    parser.add_argument("--app", choices=["cnn", "rnn"], default="cnn")
    parser.add_argument("--out-dir", default="snr_results")
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--n-runs", type=int, default=5, help="Repetitions per cap point")
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[20, 100],
        help="Batch sizes to sweep"
    )
    parser.add_argument(
        "--iters-values", nargs="+", type=int, default=[50, 200],
        help="ITERS values to sweep"
    )
    parser.add_argument(
        "--n-values", nargs="+", type=int, default=[1, 10],
        help="N (optimizer_step_interval) values"
    )
    parser.add_argument(
        "--base-caps", nargs="+", type=float, default=[0.5],
        help="Base power cap fractions to centre the signal measurement on"
    )
    parser.add_argument("--eps", type=float, default=0.1, help="Cap perturbation size")
    args = parser.parse_args()

    results = run_snr_experiments(
        app_name=args.app,
        batch_sizes=args.batch_sizes,
        iters_values=args.iters_values,
        n_values=args.n_values,
        base_caps=args.base_caps,
        eps=args.eps,
        n_runs=args.n_runs,
        out_dir=args.out_dir,
        number_od_nodes=args.nodes,
    )
    plot_snr_results(results, out_dir=args.out_dir)
