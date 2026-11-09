"""
Plot the fixed-cap sweep of run_search_optimizers.py as a GPU x CPU grid, with the
caps each grid-search run settled on overlaid.

Per app directory (e.g. results/search_optimizer_sweep/rnn) it reads:
  - monitor_search.csv : short EDP_MONITOR_SPLIT runs, one per (gpu_cap, cpu_cap)
                         cell; colour = mean per-step EDP of that cell.
  - results.csv        : full-length runs. Search rows give best_gpu/best_cpu/best_J
                         (where the search ended and the per-step EDP it saw there);
                         every row gives the end-to-end EDP, shown relative to DISABLED.

Both the cell colours and best_J are per-step EDP averaged over N windows, so they
are on the same scale. Monitor cells are a single short sample each, so expect noise.

GPU-only apps (empty cpu_cap, e.g. montecarlo) are drawn as a 1D curve instead.

Output: <app_dir>/search_grid.png

Usage:
    python plot_search_grid.py [--dir ../results/search_optimizer_sweep] [--apps rnn cnn]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGY_STYLE = {
    "EDP_SEARCH_SHARED": ("o", "SHARED"),
    "EDP_SEARCH_SPLIT": ("s", "SPLIT"),
}


def load(app_dir: str):
    mon = pd.read_csv(os.path.join(app_dir, "monitor_search.csv"))
    res = pd.read_csv(os.path.join(app_dir, "results.csv"))
    mon = mon[mon["status"] == "ok"]
    res = res[res["status"] == "ok"]
    return mon, res


def edp_vs_disabled(res: pd.DataFrame, edp: float) -> str:
    dis = res[res["strategy"] == "DISABLED"]["edp"]
    if dis.empty or not np.isfinite(edp):
        return ""
    return f"{(edp / dis.iloc[0] - 1) * 100:+.1f}%"


def search_rows(res: pd.DataFrame) -> pd.DataFrame:
    s = res[res["strategy"].isin(STRATEGY_STYLE)].copy()
    s = s.sort_values(["strategy", "grid_points", "depth"]).reset_index(drop=True)
    s["label"] = s.index + 1
    return s


def best_fixed_row(res: pd.DataFrame):
    b = res[res["mode"] == "best_config"]
    return b.iloc[0] if not b.empty else None


# Label offsets (points) for markers that sit close together: first one in the
# marker, the rest fanned out around it with a leader line.
_FAN = [(0, 0), (16, 14), (-16, 14), (16, -14), (-16, -14), (0, 20), (0, -20), (22, 0), (-22, 0)]


def annotate_labels(ax, s: pd.DataFrame, xcol: str, ycol: str, near: float, scale_y: bool = False) -> None:
    """Number each search marker; spread labels of markers closer than `near`
    (in axis units, y normalised to the axis range when scale_y)."""
    y0, y1 = ax.get_ylim()
    yspan = (y1 - y0) if scale_y else 1.0
    placed: list = []  # (x, y_norm, count)
    for _, r in s.iterrows():
        x, y = r[xcol], r[ycol]
        yn = (y - y0) / yspan if scale_y else y
        k = 0
        for i, (px, py, n) in enumerate(placed):
            if abs(px - x) < near and abs(py - yn) < near:
                k = n
                placed[i] = (px, py, n + 1)
                break
        else:
            placed.append((x, yn, 1))
        dx, dy = _FAN[k % len(_FAN)]
        ax.annotate(str(r["label"]), (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=7,
                    arrowprops=None if k == 0 else dict(arrowstyle="-", lw=0.6, color="black"),
                    bbox=None if k == 0 else dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def table_text(res: pd.DataFrame, s: pd.DataFrame, two_d: bool) -> str:
    """Side table: one line per search run plus references."""
    lines = []
    head = f"{'#':>2} {'mode':<6} {'gp':>2} {'d':>1}  {'gpu':>5} "
    head += f"{'cpu':>5} " if two_d else ""
    head += f"{'best_J':>8}  {'final EDP':>9}"
    lines.append(head)
    lines.append("-" * len(head))
    for _, r in s.iterrows():
        line = f"{r['label']:>2} {STRATEGY_STYLE[r['strategy']][1]:<6} {int(r['grid_points']):>2} {int(r['depth']):>1}  {r['best_gpu']:>5.3f} "
        line += f"{r['best_cpu']:>5.3f} " if two_d else ""
        line += f"{r['best_J']:>8.4f}  {edp_vs_disabled(res, r['edp']):>9}"
        lines.append(line)
    lines.append("")
    b = best_fixed_row(res)
    if b is not None:
        cpu = f", cpu {b['cpu_cap']:.1f}" if two_d else ""
        lines.append(f"* best fixed (gpu {b['gpu_cap']:.1f}{cpu}): {edp_vs_disabled(res, b['edp'])}")
    lines.append("final EDP = end-to-end run EDP vs DISABLED")
    lines.append("best_J    = per-step EDP at the held caps")
    return "\n".join(lines)


def plot_2d(app: str, mon: pd.DataFrame, res: pd.DataFrame, out_path: str) -> None:
    gpus = np.sort(mon["gpu_cap"].unique())
    cpus = np.sort(mon["cpu_cap"].unique())
    grid = np.full((len(cpus), len(gpus)), np.nan)
    for _, r in mon.iterrows():
        if pd.notna(r["mean_edp"]):
            grid[np.searchsorted(cpus, r["cpu_cap"]), np.searchsorted(gpus, r["gpu_cap"])] = r["mean_edp"]

    step_g = gpus[1] - gpus[0] if len(gpus) > 1 else 0.1
    step_c = cpus[1] - cpus[0] if len(cpus) > 1 else 0.1
    g_edges = np.append(gpus - step_g / 2, gpus[-1] + step_g / 2)
    c_edges = np.append(cpus - step_c / 2, cpus[-1] + step_c / 2)

    fig, (ax, tax) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1.35, 1]})
    ax.set_facecolor("#d9d9d9")  # cells without an EDP sample
    mesh = ax.pcolormesh(g_edges, c_edges, np.ma.masked_invalid(grid), cmap="viridis_r", shading="flat")
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("monitor cell: mean per-step EDP (lower = better)")

    finite = grid[np.isfinite(grid)]
    mid = np.nanmedian(finite) if finite.size else 0
    for i, c in enumerate(cpus):
        for j, g in enumerate(gpus):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(g, c, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="black" if v < mid else "white")
            else:
                ax.text(g, c, "n/a", ha="center", va="center", fontsize=6.5, color="#555")

    # Best monitor cell (lowest mean EDP) and the fixed config run at full length.
    if finite.size:
        i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
        ax.add_patch(plt.Rectangle((g_edges[j], c_edges[i]), step_g, step_c,
                                   fill=False, ec="red", lw=2.2, zorder=4))
    b = best_fixed_row(res)
    if b is not None:
        ax.scatter(b["gpu_cap"], b["cpu_cap"], marker="*", s=380, c="red", ec="white",
                   lw=1.2, zorder=6, label="best fixed config (full run)")

    s = search_rows(res)
    for strat, (marker, name) in STRATEGY_STYLE.items():
        sub = s[s["strategy"] == strat]
        if sub.empty:
            continue
        ax.scatter(sub["best_gpu"], sub["best_cpu"], marker=marker, s=170, c="white",
                   ec="black", lw=1.4, zorder=5, label=f"SEARCH_{name} result")
    annotate_labels(ax, s, "best_gpu", "best_cpu", near=0.04)

    ax.plot([0, 1.05], [0, 1.05], ls=":", c="white", lw=1, zorder=3)  # SHARED lies on this line
    ax.set_xlim(min(g_edges[0], 0) - 0.02, g_edges[-1])
    ax.set_ylim(min(c_edges[0], 0) - 0.02, c_edges[-1])
    ax.set_xlabel("GPU cap (fraction of range)")
    ax.set_ylabel("CPU cap (fraction of range)")
    ax.set_title(f"{app}: fixed-cap grid (red box = best cell) vs. search results")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8, frameon=False)

    tax.axis("off")
    tax.text(0, 1, table_text(res, s, two_d=True), family="monospace", fontsize=8.5, va="top")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_1d(app: str, mon: pd.DataFrame, res: pd.DataFrame, out_path: str) -> None:
    mon = mon.dropna(subset=["mean_edp"]).sort_values("gpu_cap")
    fig, (ax, tax) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax.plot(mon["gpu_cap"], mon["mean_edp"], "-o", c="#555", label="monitor cell: mean per-step EDP")
    if not mon.empty:
        best = mon.loc[mon["mean_edp"].idxmin()]
        ax.scatter(best["gpu_cap"], best["mean_edp"], s=260, facecolor="none", ec="red", lw=2,
                   zorder=4, label="best monitor cell")

    s = search_rows(res)
    for strat, (marker, name) in STRATEGY_STYLE.items():
        sub = s[s["strategy"] == strat]
        if not sub.empty:
            ax.scatter(sub["best_gpu"], sub["best_J"], marker=marker, s=150, c="white", ec="black",
                       lw=1.4, zorder=5, label=f"SEARCH_{name} result (cap, best_J)")
    b = best_fixed_row(res)
    if b is not None:
        ax.axvline(b["gpu_cap"], c="red", ls="--", lw=1, label="best fixed config (full run)")

    # Search results cluster near the minimum: show them in a zoomed inset.
    if not s.empty and not mon.empty:
        xs = pd.concat([s["best_gpu"], pd.Series([best["gpu_cap"]])])
        ys = pd.concat([s["best_J"], pd.Series([best["mean_edp"]])])
        xpad, ypad = 0.04, (ys.max() - ys.min()) * 0.4 + 1e-12
        ins = ax.inset_axes([0.33, 0.42, 0.36, 0.5])
        ins.plot(mon["gpu_cap"], mon["mean_edp"], "-o", c="#555")
        ins.scatter(best["gpu_cap"], best["mean_edp"], s=260, facecolor="none", ec="red", lw=2, zorder=4)
        if b is not None:
            ins.axvline(b["gpu_cap"], c="red", ls="--", lw=1)
        for strat, (marker, _) in STRATEGY_STYLE.items():
            sub = s[s["strategy"] == strat]
            ins.scatter(sub["best_gpu"], sub["best_J"], marker=marker, s=110, c="white", ec="black", lw=1.2, zorder=5)
        ins.set_xlim(xs.min() - xpad, xs.max() + xpad)
        ins.set_ylim(ys.min() - ypad, ys.max() + ypad)
        ins.tick_params(labelsize=7)
        ins.grid(alpha=0.3)
        annotate_labels(ins, s, "best_gpu", "best_J", near=0.06, scale_y=True)
        ax.indicate_inset_zoom(ins, edgecolor="gray")
    else:
        annotate_labels(ax, s, "best_gpu", "best_J", near=0.03, scale_y=True)

    ax.set_xlabel("GPU cap (fraction of range)")
    ax.set_ylabel("per-step EDP (lower = better)")
    ax.set_title(f"{app}: fixed-cap sweep vs. search results")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    tax.axis("off")
    tax.text(0, 1, table_text(res, s, two_d=False), family="monospace", fontsize=8.5, va="top")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=os.path.join(here, "..", "results", "search_optimizer_sweep"))
    parser.add_argument("--apps", nargs="+", default=None, help="Subdirectories to plot (default: all)")
    args = parser.parse_args()

    base = os.path.abspath(args.dir)
    apps = args.apps or sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    for app in apps:
        app_dir = os.path.join(base, app)
        try:
            mon, res = load(app_dir)
        except FileNotFoundError as e:
            print(f"[grid] skip {app}: {e}")
            continue
        out_path = os.path.join(app_dir, "search_grid.png")
        if mon["cpu_cap"].notna().any():
            plot_2d(app, mon, res, out_path)
        else:
            plot_1d(app, mon, res, out_path)
        print(f"[grid] {app} -> {out_path}")


if __name__ == "__main__":
    main()
