"""
Offline analysis + plots for the single-device perturbation SNR experiment.

Consumes the per-window CSV written by snr_perturbation_test.py
(perturbation_samples.csv) plus device_topology.csv, and reconstructs — purely
offline — what a dynamic EDP optimizer would have seen at any averaging window N.

Key idea (exact, not approximate): the collector logs one [OPTIMIZER STEP] edp
per sync window. The library's N-window estimate is just the mean of N
consecutive windows, so we recover the N=k estimates by re-binning each run's
per-window edp series into non-overlapping blocks of k and averaging. N is
therefore a free, post-hoc axis; eps and sign were physical caps and are real
runs.

Signals use the *realized* per-device cap (perturbed_cap_realized), so
library-side clamping at extreme caps (e.g. base 0.2 - eps floors at the device
minimum) is handled correctly rather than assuming the requested eps.

This file only runs the analysis from CSVs (like snr_test.py --plots-only); it
never launches the app. Plots produced here:

  1. snr_vs_N.png            SNR vs N (avg across device types), panel per power
                             cap + an aggregate-over-caps panel, line per eps
  2. min_N_to_detect.png     heatmap (cap x eps): smallest N reaching SNR>=1,
                             averaged across device types
  3. noise_cv_vs_N.png       relative noise (CV) vs N, log-log, 1/sqrt(N) ref
  4. edp_response_curves.png per-device EDP vs its own cap, panel per base_cap
  5. sanity_timeseries_*.png per-window EDP time series for one representative
                             (cap, device, eps): baseline / +eps / -eps overlaid
                             with N-window block means (intuition for signal vs
                             noise and any within-run drift)
  6. edp_distributions_N*.png per-device EDP distribution vs its own power cap at
                             a fixed N (row per device, column per base cap):
                             violins show the spread of the N-averaged EDP, so
                             adjacent-cap overlap = low SNR by eye

SNR is averaged *across device types* (master-GPU / slave-GPU / slave-CPU each
weighted equally) so the headline question -- "what N gives a detectable signal
on average" -- is answered per power cap and in aggregate.

Usage:
    python snr_perturbation_analysis.py [--out-dir DIR] [--n-values 5 10 20 40]
        [--snr-threshold 1.0] [--noise {global,lag1}]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_ORDER = ["master-GPU", "slave-GPU", "master-CPU", "slave-CPU"]
CLASS_COLORS = {
    "master-GPU": "#1f77b4", "slave-GPU": "#2ca02c",
    "master-CPU": "#9467bd", "slave-CPU": "#d62728",
}


# ---------------------------------------------------------------------------
# Loading + device classification
# ---------------------------------------------------------------------------
def device_class(type_: str, role: str) -> str:
    """Map (type, role) -> one of the CLASS_ORDER labels."""
    t = str(type_).upper()
    r = str(role).lower()
    if t == "GPU":
        return "master-GPU" if r == "master" else "slave-GPU"
    if t == "CPU":
        return "master-CPU" if r == "master" else "slave-CPU"
    return "unknown"


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Make the per-window frame robust to collector-schema drift.

    Some older collector runs wrote an extra `energy_devices` column whose value
    was actually the per-device cap vector, shifting `cap_vector` /
    `perturbed_cap_realized` by one (the latter ending up NaN). We don't trust
    the stored realized cap: we locate whichever column actually holds the
    comma-separated cap vector and recompute `perturbed_cap_realized` from it at
    the perturbed device's index. No-op for already-canonical files.
    """
    df = df.copy()
    vec_col = None
    for c in ("cap_vector", "energy_devices"):
        if c in df.columns and df[c].astype(str).str.contains(",", na=False).any():
            vec_col = c
            break
    if vec_col is None:
        return df  # nothing parseable; leave as-is

    df["cap_vector"] = df[vec_col].astype(str)
    if vec_col == "energy_devices":
        df = df.drop(columns=["energy_devices"])

    def _realized(row):
        d = int(row["perturbed_device"])
        if d < 0:
            return np.nan  # baseline rows have no single perturbed device
        parts = row["cap_vector"].split(",")
        try:
            return float(parts[d])
        except (IndexError, ValueError):
            return np.nan

    df["perturbed_cap_realized"] = df.apply(_realized, axis=1)
    return df


def load(out_dir: str):
    """Return (per-window DataFrame, {device_index: class}) from the CSVs."""
    out_dir = os.path.abspath(out_dir)
    df = pd.read_csv(os.path.join(out_dir, "perturbation_samples.csv"))
    df = normalize_schema(df)

    topo_path = os.path.join(out_dir, "device_topology.csv")
    dev_class: dict[int, str] = {}
    if os.path.exists(topo_path):
        topo = pd.read_csv(topo_path)
        for _, r in topo.iterrows():
            dev_class[int(r["index"])] = device_class(r["type"], r["role"])
    else:
        # Fall back to the per-row type/role columns (baseline rows excluded).
        for _, r in df[df["perturbed_device"] >= 0].iterrows():
            dev_class[int(r["perturbed_device"])] = device_class(
                r["perturbed_type"], r["perturbed_role"])
    return df, dev_class


# ---------------------------------------------------------------------------
# N-window reconstruction primitives
# ---------------------------------------------------------------------------
def block_means(arr: np.ndarray, n: int) -> np.ndarray:
    """Mean of every non-overlapping block of n consecutive samples (drops the
    final partial block). This is exactly the library's N-window EDP estimate."""
    arr = np.asarray(arr, dtype=float)
    nb = len(arr) // n
    if nb < 1:
        return np.array([])
    return arr[:nb * n].reshape(nb, n).mean(axis=1)


def lag1_std(blocks: np.ndarray) -> float:
    """Short-timescale noise from consecutive-block differences (Allan-like):
    std of x_{k+1}-x_k divided by sqrt(2). Isolates the noise the optimizer sees
    between adjacent estimates from slow drift that inflates the global std."""
    if len(blocks) < 2:
        return float("nan")
    d = np.diff(blocks)
    return float(np.sqrt(np.mean(d * d) / 2.0))


# ---------------------------------------------------------------------------
# Derivation layer: per-operating-point block stats, then paired SNR
# ---------------------------------------------------------------------------
def compute_point_stats(df: pd.DataFrame, dev_class: dict, n_values: list[int]) -> pd.DataFrame:
    """One row per (base_cap, device, class, eps, sign, rep, N): block-mean EDP,
    its global std and lag-1 std, relative noise (CV) both ways, block count, and
    the realized perturbed cap. Baseline runs (device=-1) get class 'baseline'."""
    rows = []
    keys = ["base_cap", "perturbed_device", "eps", "sign", "rep"]
    for (cap, dev, eps, sign, rep), grp in df.groupby(keys):
        grp = grp.sort_values("window")
        edp = grp["edp"].to_numpy()
        klass = "baseline" if dev < 0 else dev_class.get(int(dev), "unknown")
        realized = (float(grp["perturbed_cap_realized"].mean())
                    if dev >= 0 else float(cap))
        for n in n_values:
            b = block_means(edp, n)
            if len(b) < 2:
                continue
            mean = float(np.mean(b))
            std = float(np.std(b, ddof=1))
            l1 = lag1_std(b)
            rows.append({
                "base_cap": cap, "device": int(dev), "klass": klass,
                "eps": eps, "sign": int(sign), "rep": int(rep), "N": n,
                "n_blocks": len(b), "edp_mean": mean,
                "edp_std": std, "edp_lag1_std": l1,
                "cv_global": std / mean if mean else np.nan,
                "cv_lag1": l1 / mean if mean else np.nan,
                "realized_cap": realized,
            })
    return pd.DataFrame(rows)


def compute_pair_snr(point_stats: pd.DataFrame, snr_threshold: float) -> pd.DataFrame:
    """Pair the +eps and -eps runs to form the central-difference signal the
    optimizer estimates. One row per (base_cap, device, class, eps, rep, N) with
    signal, realized Delta-cap, gradient, pooled noise (global + lag-1), SNR."""
    ps = point_stats[point_stats["klass"] != "baseline"]
    plus = ps[ps["sign"] == 1]
    minus = ps[ps["sign"] == -1]
    merge_keys = ["base_cap", "device", "klass", "eps", "rep", "N"]
    m = plus.merge(minus, on=merge_keys, suffixes=("_p", "_m"))

    signal = (m["edp_mean_p"] - m["edp_mean_m"]).abs()
    dcap = (m["realized_cap_p"] - m["realized_cap_m"]).abs()
    noise_g = np.sqrt((m["edp_std_p"] ** 2 + m["edp_std_m"] ** 2) / 2.0)
    noise_l1 = np.sqrt((m["edp_lag1_std_p"] ** 2 + m["edp_lag1_std_m"] ** 2) / 2.0)
    mean_edp = (m["edp_mean_p"] + m["edp_mean_m"]) / 2.0

    out = m[merge_keys].copy()
    out["signal"] = signal
    out["realized_dcap"] = dcap
    out["gradient"] = signal / dcap.replace(0, np.nan)
    out["mean_edp"] = mean_edp
    out["noise_global"] = noise_g
    out["noise_lag1"] = noise_l1
    out["snr_global"] = signal / noise_g.replace(0, np.nan)
    out["snr_lag1"] = signal / noise_l1.replace(0, np.nan)
    # Clamped when the realized cap gap is well below the requested 2*eps span.
    out["clamped"] = out["realized_dcap"] < 1.9 * out["eps"]
    out["detectable_global"] = out["snr_global"] >= snr_threshold
    return out


def raw_run_means(df: pd.DataFrame, dev_class: dict) -> pd.DataFrame:
    """Full-series mean EDP + realized cap per run (N-independent; for the
    signal-vs-eps and EDP-response-curve plots). Baseline rows kept (device=-1)."""
    g = df.groupby(["base_cap", "perturbed_device", "eps", "sign"]).agg(
        edp_mean=("edp", "mean"),
        cap_real=("perturbed_cap_realized", "mean"),
    ).reset_index()
    g["klass"] = g["perturbed_device"].apply(
        lambda d: "baseline" if d < 0 else dev_class.get(int(d), "unknown"))
    return g


def present_classes(values) -> list[str]:
    have = set(values)
    return [c for c in CLASS_ORDER if c in have]


def aggregate_snr_table(pair: pd.DataFrame, snr_col: str) -> pd.DataFrame:
    """SNR averaged *across device types* per (base_cap, eps, N).

    Device types are weighted equally: we average over devices within each class
    first, then average the class means. This keeps the result a genuine
    'average device type' rather than letting the 3 GPUs + 3 CPUs outvote the
    single master GPU. Returns columns base_cap, eps, N, snr.
    """
    per_class = (pair.groupby(["base_cap", "eps", "N", "klass"])[snr_col]
                 .mean().reset_index())
    agg = (per_class.groupby(["base_cap", "eps", "N"])[snr_col]
           .mean().reset_index().rename(columns={snr_col: "snr"}))
    return agg


# ---------------------------------------------------------------------------
# Plot 1: SNR vs N (averaged across device types), panel per power cap + an
# aggregate-over-caps panel, line per eps.
# ---------------------------------------------------------------------------
def plot_snr_vs_N(pair: pd.DataFrame, out_dir: str, snr_col: str,
                  snr_threshold: float) -> None:
    agg = aggregate_snr_table(pair, snr_col)
    if agg.empty:
        return
    caps = sorted(agg["base_cap"].unique())
    epsv = sorted(agg["eps"].unique())
    cmap = plt.get_cmap("viridis")
    panels = [("cap", c) for c in caps] + [("agg", None)]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2),
                             sharey=True, squeeze=False)
    for ax, (kind, cap) in zip(axes[0], panels):
        if kind == "cap":
            sub = agg[agg["base_cap"] == cap]
            title = f"cap = {cap:g}"
        else:
            # aggregate over caps too (mean of the per-cap across-type SNR)
            sub = agg.groupby(["eps", "N"])["snr"].mean().reset_index()
            title = "aggregate (all caps)"
        for i, eps in enumerate(epsv):
            trend = (sub[sub["eps"] == eps].groupby("N")["snr"].mean().sort_index())
            if trend.empty:
                continue
            ax.plot(trend.index, trend.values, "o-",
                    color=cmap(i / max(1, len(epsv) - 1)), label=f"eps={eps:g}")
        ax.axhline(snr_threshold, color="red", ls="--", lw=0.8,
                   label=f"SNR={snr_threshold:g}")
        ax.set_xlabel("N (windows averaged per estimate)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel(f"SNR ({snr_col}), avg across device types")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("EDP perturbation SNR vs averaging window N "
                 "(averaged across device types)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "snr_vs_N.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: minimum N to reach SNR>=threshold (averaged across device types),
# single heatmap (cap x eps).
# ---------------------------------------------------------------------------
def plot_min_N_heatmap(pair: pd.DataFrame, out_dir: str, snr_col: str,
                       snr_threshold: float) -> None:
    agg = aggregate_snr_table(pair, snr_col)
    if agg.empty:
        return
    caps = sorted(agg["base_cap"].unique())
    epsv = sorted(agg["eps"].unique())
    n_sorted = sorted(agg["N"].unique())

    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#dddddd")  # 'never' cells: light gray so the label is readable

    grid = np.full((len(caps), len(epsv)), np.nan)
    for ci, cap in enumerate(caps):
        for ei, eps in enumerate(epsv):
            by_n = (agg[(agg["base_cap"] == cap) & (agg["eps"] == eps)]
                    .set_index("N")["snr"])
            hit = [n for n in n_sorted if n in by_n.index
                   and by_n[n] >= snr_threshold]
            grid[ci, ei] = hit[0] if hit else np.nan

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto",
                   cmap=cmap, vmin=min(n_sorted), vmax=max(n_sorted))
    ax.set_xticks(range(len(epsv)))
    ax.set_xticklabels([f"{e:g}" for e in epsv])
    ax.set_yticks(range(len(caps)))
    ax.set_yticklabels([f"{c:g}" for c in caps])
    ax.set_xlabel("eps")
    ax.set_ylabel("base cap")
    for ci in range(len(caps)):
        for ei in range(len(epsv)):
            v = grid[ci, ei]
            if np.isnan(v):
                txt, color = "never", "#333333"
            else:
                txt = f"{int(v)}"
                color = "white" if v > np.nanmean(n_sorted) else "black"
            ax.text(ei, ci, txt, ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, label="min N for SNR>=thresh")
    ax.set_title(f"Smallest N reaching SNR>={snr_threshold:g} ({snr_col}), "
                 f"avg across device types\n'never' = not detectable within tested N",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "min_N_to_detect.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: relative noise (CV) vs N, log-log, with 1/sqrt(N) reference
# ---------------------------------------------------------------------------
def plot_noise_vs_N(point_stats: pd.DataFrame, out_dir: str) -> None:
    ps = point_stats[point_stats["klass"] != "baseline"]
    classes = present_classes(ps["klass"].unique())
    if not classes:
        return
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4.2),
                             sharey=True, squeeze=False)
    for ax, klass in zip(axes[0], classes):
        sub = ps[ps["klass"] == klass]
        gcv = sub.groupby("N")["cv_global"].mean().sort_index()
        lcv = sub.groupby("N")["cv_lag1"].mean().sort_index()
        if not gcv.empty:
            ax.loglog(gcv.index, gcv.values, "o-", color="#1f77b4",
                      label="global CV")
        if not lcv.empty:
            ax.loglog(lcv.index, lcv.values, "s--", color="#ff7f0e",
                      label="lag-1 CV (short-timescale)")
            # 1/sqrt(N) reference anchored at the smallest N of the lag-1 curve
            n0, c0 = lcv.index[0], lcv.values[0]
            if np.isfinite(c0) and c0 > 0:
                ref = c0 * np.sqrt(n0 / np.asarray(lcv.index, dtype=float))
                ax.loglog(lcv.index, ref, ":", color="gray", lw=1.0,
                          label="1/sqrt(N) ref")
        ax.set_xlabel("N (windows averaged)")
        ax.set_title(klass)
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel("EDP estimate CV (std / mean)")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Relative EDP noise vs N (does averaging buy 1/sqrt(N)?)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "noise_cv_vs_N.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: per-device EDP vs its own cap, panel per base_cap
# ---------------------------------------------------------------------------
def plot_edp_response(raw: pd.DataFrame, dev_class: dict, out_dir: str) -> None:
    caps = sorted(raw["base_cap"].unique())
    if not caps:
        return
    devices = sorted([d for d in raw["perturbed_device"].unique() if d >= 0])
    fig, axes = plt.subplots(1, len(caps), figsize=(5 * len(caps), 4.4),
                             sharey=False, squeeze=False)
    for ax, cap in zip(axes[0], caps):
        # baseline EDP at this cap = the "all devices equal" point
        base_rows = raw[(raw["base_cap"] == cap) & (raw["perturbed_device"] < 0)]
        base_edp = float(base_rows["edp_mean"].mean()) if len(base_rows) else np.nan
        for dev in devices:
            klass = dev_class.get(int(dev), "unknown")
            s = raw[(raw["base_cap"] == cap) & (raw["perturbed_device"] == dev)]
            if s.empty:
                continue
            pts = [(float(cap), base_edp)] if np.isfinite(base_edp) else []
            pts += list(zip(s["cap_real"], s["edp_mean"]))
            pts = sorted(set(pts))
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "o-", color=CLASS_COLORS.get(klass, "gray"),
                    alpha=0.85, label=f"dev{dev} ({klass})")
        if np.isfinite(base_edp):
            ax.axvline(cap, color="black", ls=":", lw=0.8)
        ax.set_xlabel("device's own cap (realized fraction)")
        ax.set_title(f"base cap = {cap:g}")
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("mean system EDP")
    # de-duplicated legend (one entry per device)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[0][-1].legend(uniq.values(), uniq.keys(), fontsize=7, ncol=1)
    fig.suptitle("Per-device EDP response: bump one device's cap, hold the rest "
                 "(flat line = device barely moves system EDP)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "edp_response_curves.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: sanity / intuition -- per-window EDP time series + histogram for one
# representative (cap, device, eps), baseline / +eps / -eps overlaid, with the
# N-window block means marked. Justifies the noise model and exposes drift.
# ---------------------------------------------------------------------------
def _series(df: pd.DataFrame, cap, device, eps, sign) -> np.ndarray:
    """Per-window EDP series (ordered) for one run; empty if absent."""
    if device < 0:
        sub = df[(df["base_cap"] == cap) & (df["perturbed_device"] < 0)]
    else:
        sub = df[(df["base_cap"] == cap) & (df["perturbed_device"] == device)
                 & (df["eps"] == eps) & (df["sign"] == sign)]
    return sub.sort_values("window")["edp"].to_numpy()


def _pick_representative(df: pd.DataFrame, dev_class: dict, cap, eps):
    """Among perturbed devices with both signs present at (cap, eps), pick the one
    whose central signal |mean(+)-mean(-)| is the median -- a representative, not
    cherry-picked, example."""
    cand = []
    for dev in sorted(d for d in df["perturbed_device"].unique() if d >= 0):
        p = _series(df, cap, dev, eps, 1)
        m = _series(df, cap, dev, eps, -1)
        if len(p) and len(m):
            cand.append((abs(p.mean() - m.mean()), dev))
    if not cand:
        return None
    cand.sort()
    return cand[len(cand) // 2][1]  # median signal


def plot_sanity_timeseries(df: pd.DataFrame, dev_class: dict, out_dir: str,
                           n_mark: int, cap=None, device=None, eps=None) -> None:
    pert = df[df["perturbed_device"] >= 0]
    if pert.empty:
        return
    caps = sorted(df["base_cap"].unique())
    epsv = sorted(e for e in pert["eps"].unique() if e > 0)
    if cap is None:
        cap = caps[len(caps) // 2]
    if eps is None:
        eps = epsv[len(epsv) // 2] if epsv else None
    if device is None:
        device = _pick_representative(df, dev_class, cap, eps)
    if device is None:
        print("[analysis] sanity plot: no (cap,device,eps) with both signs; skipping")
        return
    klass = dev_class.get(int(device), "unknown")

    series = {
        f"-eps ({cap - eps:.2f})": (_series(df, cap, device, eps, -1), "#1f77b4"),
        f"baseline ({cap:.2f})":   (_series(df, cap, -1, 0.0, 0),      "#7f7f7f"),
        f"+eps ({cap + eps:.2f})": (_series(df, cap, device, eps, 1),  "#d62728"),
    }

    fig, axt = plt.subplots(figsize=(11, 4.6))

    means = {}
    for label, (s, color) in series.items():
        if not len(s):
            continue
        x = np.arange(len(s))
        # raw per-window EDP (faint) -- this is the noise the optimizer fights
        axt.plot(x, s, color=color, alpha=0.30, lw=0.8)
        # N-window block means (bold) -- what the optimizer actually consumes
        b = block_means(s, n_mark)
        if len(b):
            bx = (np.arange(len(b)) + 0.5) * n_mark
            axt.plot(bx, b, "o-", color=color, lw=1.8, ms=4, label=label)
        means[label] = float(s.mean())
        # horizontal mean line: the gap between +eps/-eps means is the signal
        axt.axhline(means[label], color=color, ls="--", lw=0.8, alpha=0.7)

    axt.set_xlabel("sync window index")
    axt.set_ylabel("per-window system EDP")
    axt.grid(True, alpha=0.3)
    axt.legend(fontsize=8, loc="upper right", title=f"bold = N={n_mark} block means")

    # Tie the picture to the numbers: signal vs raw noise vs N-block noise -> SNR.
    plus = _series(df, cap, device, eps, 1)
    minus = _series(df, cap, device, eps, -1)
    if len(plus) and len(minus):
        signal = abs(plus.mean() - minus.mean())
        raw_noise = np.sqrt((plus.std(ddof=1) ** 2 + minus.std(ddof=1) ** 2) / 2.0)
        bp, bm = block_means(plus, n_mark), block_means(minus, n_mark)
        if len(bp) >= 2 and len(bm) >= 2:
            blk_noise = np.sqrt((bp.std(ddof=1) ** 2 + bm.std(ddof=1) ** 2) / 2.0)
        else:
            blk_noise = float("nan")
        txt = (f"signal |Δmean EDP| = {signal:.4f}\n"
               f"per-window noise (N=1) = {raw_noise:.4f}  → SNR={signal / raw_noise:.2f}\n"
               f"N={n_mark} block noise = {blk_noise:.4f}  → SNR={signal / blk_noise:.2f}")
        axt.text(0.02, 0.02, txt, transform=axt.transAxes, fontsize=8,
                 va="bottom", ha="left",
                 bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    fig.suptitle(f"Per-window EDP sanity view — cap={cap:g}, device {device} "
                 f"({klass}), eps={eps:g}\n"
                 f"raw windows (faint) vs N={n_mark} averaged estimates (bold); "
                 f"gap between +eps/-eps means = signal")
    fig.tight_layout()
    fname = f"sanity_timeseries_cap{cap:g}_dev{device}_eps{eps:g}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=130)
    plt.close(fig)
    print(f"[analysis] sanity plot -> {fname}")


# ---------------------------------------------------------------------------
# Plot: per-device EDP distribution vs power cap, for a fixed N. Row per device,
# column per base_cap (others held at base). Violin = spread of the N-window EDP
# estimates at each realized cap; overlap of adjacent caps = low SNR by eye.
# ---------------------------------------------------------------------------
def _blocks_by_cap(df: pd.DataFrame, cap, devices, epsv, n: int) -> dict:
    """N-block EDP estimates keyed by realized cap, pooled over `devices`. Merges
    clamped duplicates landing on the same cap (e.g. base 0.2 minus-eps) and
    includes the shared baseline (all devices at base) at the base-cap point."""
    from collections import defaultdict
    by_cap: dict = defaultdict(list)
    base_blocks = block_means(_series(df, cap, -1, 0.0, 0), n)
    if len(base_blocks) >= 2:
        by_cap[round(float(cap), 3)].append(base_blocks)
    for dev in devices:
        for eps in epsv:
            for sign in (-1, 1):
                b = block_means(_series(df, cap, dev, eps, sign), n)
                if len(b) < 2:
                    continue
                sub = df[(df["base_cap"] == cap) & (df["perturbed_device"] == dev)
                         & (df["eps"] == eps) & (df["sign"] == sign)]
                xr = round(float(sub["perturbed_cap_realized"].mean()), 3)
                by_cap[xr].append(b)
    return by_cap


def _draw_violins(ax, by_cap: dict, color: str, cap) -> None:
    positions = sorted(by_cap)
    data = [np.concatenate(by_cap[p]) for p in positions]
    if data:
        vp = ax.violinplot(data, positions=positions, widths=0.035,
                           showmeans=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)
        meds = [float(np.median(d)) for d in data]
        ax.plot(positions, meds, "-o", color=color, lw=1.1, ms=3)
    ax.axvline(cap, color="black", ls=":", lw=0.6, alpha=0.6)
    ax.grid(True, alpha=0.25)


def plot_edp_distributions(df: pd.DataFrame, dev_class: dict, out_dir: str,
                           n: int = 20) -> None:
    caps = sorted(df["base_cap"].unique())
    devices = sorted(d for d in df["perturbed_device"].unique() if d >= 0)
    epsv = sorted(e for e in df[df["perturbed_device"] >= 0]["eps"].unique() if e > 0)
    if not caps or not devices:
        return

    # One row per device + a final aggregate row pooling all devices.
    row_specs = ([(dev, dev_class.get(int(dev), "unknown"),
                   CLASS_COLORS.get(dev_class.get(int(dev), "unknown"), "gray"),
                   [dev]) for dev in devices]
                 + [("agg", "aggregate\n(all devices)", "#333333", devices)])

    nrows, ncols = len(row_specs), len(caps)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 2.2 * nrows),
                             sharex="col", squeeze=False)
    for ri, (tag, label, color, dev_list) in enumerate(row_specs):
        for ci, cap in enumerate(caps):
            ax = axes[ri][ci]
            _draw_violins(ax, _blocks_by_cap(df, cap, dev_list, epsv, n), color, cap)
            if ri == 0:
                ax.set_title(f"base cap = {cap:g}")
            if ci == 0:
                ylab = label if tag == "agg" else f"dev{tag}\n{label}"
                ax.set_ylabel(ylab, fontsize=8)
    for ci in range(ncols):
        axes[-1][ci].set_xlabel("device's own cap (realized fraction)")
    fig.suptitle(f"Per-device EDP distribution vs power cap (N={n} block estimates)\n"
                 "violin = spread of N-averaged EDP; overlap of adjacent caps = low SNR; "
                 "line = median (response curve); bottom row pools all devices")
    fig.tight_layout()
    fname = f"edp_distributions_N{n}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=130)
    plt.close(fig)
    print(f"[analysis] distribution plot -> {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plots 1-5 for the single-device perturbation SNR experiment")
    parser.add_argument("--out-dir", default="montecarlo_gpu_snr_perturbation_results")
    parser.add_argument("--n-values", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--snr-threshold", type=float, default=1.0)
    parser.add_argument("--noise", choices=["global", "lag1"], default="global",
                        help="Which noise estimator drives the SNR plots")
    # Sanity time-series plot selectors (None = auto-pick a representative cell)
    parser.add_argument("--sanity-cap", type=float, default=None)
    parser.add_argument("--sanity-device", type=int, default=None)
    parser.add_argument("--sanity-eps", type=float, default=None)
    parser.add_argument("--sanity-N", type=int, default=20,
                        help="Block size N marked on the sanity time series")
    parser.add_argument("--dist-N", type=int, default=20,
                        help="Block size N for the per-device EDP distribution plot")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    snr_col = "snr_global" if args.noise == "global" else "snr_lag1"

    print(f"[analysis] loading {out_dir}/perturbation_samples.csv")
    df, dev_class = load(out_dir)
    print(f"[analysis] {len(df)} per-window rows; "
          f"{df['perturbed_device'].nunique()} perturbed-device groups; "
          f"classes: {sorted(set(dev_class.values()))}")

    point_stats = compute_point_stats(df, dev_class, args.n_values)
    pair = compute_pair_snr(point_stats, args.snr_threshold)
    raw = raw_run_means(df, dev_class)

    # Persist the derived tables so they can be reused / inspected.
    point_stats.to_csv(os.path.join(out_dir, "point_stats.csv"), index=False)
    pair.to_csv(os.path.join(out_dir, "pair_snr.csv"), index=False)
    n_clamped = int(pair["clamped"].sum())
    print(f"[analysis] derived {len(pair)} (cap,device,eps,N) SNR cells "
          f"({n_clamped} clamped); tables -> point_stats.csv, pair_snr.csv")

    plot_snr_vs_N(pair, out_dir, snr_col, args.snr_threshold)
    plot_min_N_heatmap(pair, out_dir, snr_col, args.snr_threshold)
    plot_noise_vs_N(point_stats, out_dir)
    plot_edp_response(raw, dev_class, out_dir)
    plot_sanity_timeseries(df, dev_class, out_dir, args.sanity_N,
                           cap=args.sanity_cap, device=args.sanity_device,
                           eps=args.sanity_eps)
    plot_edp_distributions(df, dev_class, out_dir, n=args.dist_N)
    print(f"[analysis] plots written to {out_dir}/")


if __name__ == "__main__":
    main()
