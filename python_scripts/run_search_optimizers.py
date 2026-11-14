"""
Focused benchmark of the grid-search EDP optimizers (EDP_SEARCH_SHARED /
EDP_SEARCH_SPLIT), swept over each app in turn: montecarlo (GPU-only), RNN
(CPU + GPU) and CNN (CPU + GPU). Per-app batch size, batches/window and the
CPU-execution setting are taken from the defaults defined below.

For every app the search optimizers are compared against two references:

  1. DISABLED       — no power capping (default caps), i.e. "do nothing". Run for
                      the same length (iters) as the search algorithms.
  2. best fixed cfg — the GPU x CPU power-cap space is actively searched: each cell
                      (gpu_frac, cpu_frac) over 0.1,0.2,...,1.0 (step configurable
                      via --search-step) is run briefly in EDP_MONITOR_SPLIT for
                      MONITOR_STEPS optimizer steps, the lowest mean-EDP cell wins,
                      and that winning fixed config is then run for the full iters.

So the final comparison is: search algorithms vs. best manually-found fixed config
vs. do-nothing default.

Search modes
  - EDP_SEARCH_SHARED : one power-cap fraction for all devices (1D grid).
  - EDP_SEARCH_SPLIT  : distinct GPU and CPU fractions (2D grid).

The search evaluates `search_grid_points` equally-spaced caps over the full range
(level 0: e.g. 0.1/0.3/0.5/0.7/0.9 for 5 points), picks the lowest-EDP cell, then
subdivides the interval between its two neighbours into the same number of points
for the next level, repeating for `search_depth` levels before holding the best
caps. One grid point is evaluated per optimizer step, and each optimizer step
averages N (= optimizer_step_interval) sync windows.

--iters is computed from first principles so the *largest* search config has
enough optimizer steps to complete all levels and settle on its minimum:

    steps   = search_depth * search_grid_points ** dims     (dims: 1 shared, 2 split)
    windows = N * steps * SAFETY_MARGIN
    iters   = ceil(windows * batches_per_window * devices * batch_size / VECTORSIZE)

The fixed-config search runs are sized the same way but for just MONITOR_STEPS
optimizer steps (a few iters), since each only needs to log ~MONITOR_STEPS EDP
samples for one fixed cap.

devices is the true device count (windows scale inversely with it). From the
[DEVICE TOPOLOGY] logs this cluster exposes one GPU per node on every node, plus
one CPU device per *slave* node when CPU is enabled (master/rank-0 has no CPU):

    devices = nodes * gpus_per_node + (nodes - 1 if cpu_enabled else 0)

Default batch size mirrors the SNR scripts: 50 for cnn/rnn, 240000 for others.

Crash-safe: the fixed-config search is appended to each app's monitor_search.csv
and the final comparison to results.csv (under a per-app subdirectory of
--out-dir), flushed after every run; a restart skips runs already present.

Usage:
    python run_search_optimizers.py [--nodes N] [--N WINDOWS] [--n-runs 3]
        [--out-dir DIR] [--iters ITERS] [--search-step 0.1] [--no-baseline]
"""

import argparse
import csv
import math
import os
import re
from pathlib import Path

from experiments import single_app_run
from models import RunParameters

# --- Default batch size per app (mirrors snr_perturbation_test.py) ---
DEFAULT_BATCH_SIZES = {"cnn": 50, "rnn": 50}
DEFAULT_BATCH_SIZE_FALLBACK = 240000

# Datapoints processed per dataset iteration (the app's VECTORSIZE / problem size,
# from the app sources). A run of `iters` covers iters * VECTORSIZE datapoints.
VECTORSIZE = {"montecarlo": 960_000_000, "cnn": 4000, "rnn": 4000}

# Conservative peak batches processed per sync window *per device*, read from the
# [OPTIMIZER STEP] logs (busiest window / device count), calibrated at each app's
# default batch size. Overestimating only lengthens the run, so the per-device peak
# guarantees at least the needed windows.
BATCHES_PER_WINDOW_PER_DEVICE = {"montecarlo": 80, "cnn": 4, "rnn": 2}
DEFAULT_BATCHES_PER_WINDOW_PER_DEVICE = 80

# Apps swept end-to-end, each with its CPU-execution setting. montecarlo runs
# GPU-only; the CNN/RNN apps run across CPU + GPU. Batch size and batches/window
# are taken from the per-app defaults above.
APPS = [
    {"name": "montecarlo", "cpu_enabled": False},
    # {"name": "rnn", "cpu_enabled": True},
    # {"name": "cnn", "cpu_enabled": True},
]

# GPUs exposed per node (this cluster: one GPU per node, on every node).
GPUS_PER_NODE = 1

# --- Swept search hyperparameters (sensible, manageable grid) ---
# The grid spans the full range with spacing derived from grid_points, so the only
# search knobs are the grid resolution and the refinement depth.
SEARCH_STRATEGIES = ["EDP_SEARCH_SHARED", "EDP_SEARCH_SPLIT"]
GRID_POINTS = [3, 5]             # points per axis (5 -> 0.1/0.3/0.5/0.7/0.9 at level 0)
DEPTHS = [2, 3, 4]               # refinement levels


def swept_strategies(cpu_enabled: bool) -> list[str]:
    """Search strategies to sweep. With CPU disabled there is no CPU axis, so
    EDP_SEARCH_SPLIT collapses to EDP_SEARCH_SHARED (see strategy_dims) and is
    dropped as a redundant duplicate."""
    if cpu_enabled:
        return SEARCH_STRATEGIES
    return [s for s in SEARCH_STRATEGIES if s != "EDP_SEARCH_SPLIT"]


# --- Fixed-config reference (best-cap search + do-nothing baseline) ---
DISABLED_STRATEGY = "DISABLED"          # do-nothing reference (default caps)
MONITOR_STRATEGY = "EDP_MONITOR_SPLIT"  # fixed independent GPU/CPU caps + EDP logging
MONITOR_STEPS = 1                      # optimizer steps per fixed-config search run
DEFAULT_SEARCH_STEP = 0.1               # cap-grid resolution (0.1 -> 0.1,0.2,...,1.0)

# Run length must cover the largest search config's optimizer steps with margin.
SAFETY_MARGIN = 1.5

CSV_COLUMNS = [
    "strategy", "mode", "gpu_cap", "cpu_cap", "grid_points", "depth",
    "N", "iters", "nodes", "devices", "batch_size", "rep",
    "time_s", "energy_j", "edp", "best_gpu", "best_cpu", "best_J", "status",
]

# Per-cell record of the fixed-config (best-cap) search.
MONITOR_CSV_COLUMNS = [
    "gpu_cap", "cpu_cap", "mean_edp", "n_steps", "time_s", "energy_j", "edp", "status",
]

# "SEARCH: max depth N reached, holding best caps (gpu=0.5000 cpu=0.4500 J=...)"
_BEST_RE = re.compile(
    r"holding best caps \(gpu=(?P<gpu>[\d.eE+\-]+)\s+cpu=(?P<cpu>[\d.eE+\-]+)\s+"
    r"J=(?P<J>[\d.eE+\-]+)\)"
)

# Per-step averaged EDP the EDP-monitor emits once per optimizer_step_interval windows.
_EDP_RE = re.compile(r"\[EDP\] edp=([\d.eE+\-]+)")


def default_batch_size(app: str) -> int:
    return DEFAULT_BATCH_SIZES.get(app, DEFAULT_BATCH_SIZE_FALLBACK)


def cap_key(x) -> str:
    """Canonical string key for a cap fraction (or '' for an absent CPU axis)."""
    if x is None or x == "":
        return ""
    return f"{float(x):.4f}"


def strategy_dims(strategy: str, cpu_enabled: bool = True) -> int:
    """Search dimensions: SPLIT searches a GPU axis and a CPU axis (2D); SHARED is
    1D. In a GPU-only run there is no CPU axis, so SPLIT collapses to 1D as well
    (matching __cudampi__initSearch, which drops the CPU axis when no CPU devices
    are present)."""
    if strategy == "EDP_SEARCH_SPLIT" and cpu_enabled:
        return 2
    return 1


def required_steps(strategy: str, depth: int, grid_points: int, cpu_enabled: bool = True) -> int:
    """Optimizer steps to fully enumerate every level of the search."""
    return depth * grid_points ** strategy_dims(strategy, cpu_enabled)


def total_device_count(nodes: int, cpu_enabled: bool, gpus_per_node: int = GPUS_PER_NODE) -> int:
    """Number of compute devices the library exposes — the true divisor in the
    window count (windows ~ iters / (batch_size * devices)).

    From the [DEVICE TOPOLOGY] logs: one GPU per node on every node, plus one CPU
    device per *slave* node when CPU is enabled (the master/rank-0 node exposes no
    CPU device). So devices = nodes*gpus_per_node + (nodes-1 if cpu_enabled)."""
    gpus = nodes * gpus_per_node
    cpus = (nodes - 1) if (cpu_enabled and nodes > 1) else 0
    return gpus + cpus


def app_vectorsize(app: str) -> int:
    if app not in VECTORSIZE:
        raise SystemExit(
            f"[search] no VECTORSIZE known for app '{app}'. Add its per-iteration "
            f"problem size to VECTORSIZE in run_search_optimizers.py (from the app source).")
    return VECTORSIZE[app]


def batches_per_window(app: str) -> int:
    """Conservative per-device peak batches per sync window for `app`."""
    return BATCHES_PER_WINDOW_PER_DEVICE.get(app, DEFAULT_BATCHES_PER_WINDOW_PER_DEVICE)


def iters_for_windows(windows: float, devices: int, batch_size: int, app: str, bpw: int) -> int:
    """Dataset iteration count yielding ~`windows` sync windows. A window processes
    ~bpw batches per device (bpw*devices*batch_size datapoints); a run of `iters`
    covers iters*VECTORSIZE datapoints, so windows = iters*VECTORSIZE / per-window."""
    datapoints_per_window = bpw * devices * batch_size
    return int(math.ceil(windows * datapoints_per_window / app_vectorsize(app)))


def sweep_iters(devices: int, batch_size: int, app: str, n_windows: int, bpw: int,
                cpu_enabled: bool) -> int:
    """Single iters for the whole sweep, sized for the most demanding search config so
    every run has enough time; smaller configs converge earlier and hold their best caps."""
    max_steps = max(
        required_steps(strat, depth, gp, cpu_enabled)
        for strat in swept_strategies(cpu_enabled)
        for gp in GRID_POINTS
        for depth in DEPTHS
    )
    windows = math.ceil(n_windows * max_steps * SAFETY_MARGIN)
    return iters_for_windows(windows, devices, batch_size, app, bpw)


def monitor_iters(devices: int, batch_size: int, app: str, n_windows: int, bpw: int) -> int:
    """iters for one fixed-config search run: just enough windows for MONITOR_STEPS
    optimizer steps (a few iters), since each cell only logs ~MONITOR_STEPS EDP samples."""
    windows = math.ceil(MONITOR_STEPS * n_windows)
    return iters_for_windows(windows, devices, batch_size, app, bpw)


def cap_grid(step: float) -> list[float]:
    """Cap fractions swept by the fixed-config search: step, 2*step, ..., 1.0."""
    n = int(round(1.0 / step))
    return [round(k * step, 4) for k in range(1, n + 1)]


def monitor_grid(cpu_enabled: bool, step: float) -> list[tuple]:
    """(gpu, cpu) cells for the fixed-config search. GPU-only apps sweep the GPU axis
    only (cpu = None); CPU+GPU apps sweep the full GPU x CPU product."""
    caps = cap_grid(step)
    if cpu_enabled:
        return [(g, c) for g in caps for c in caps]
    return [(g, None) for g in caps]


def search_mode(strategy: str) -> str:
    return "split" if strategy == "EDP_SEARCH_SPLIT" else "shared"


def make_params(cfg: dict, *, app: str, nodes: int, batch_size: int, n_windows: int,
                iters: int, cpu_enabled: bool) -> RunParameters:
    strat = cfg["strategy"]
    is_search = strat in SEARCH_STRATEGIES
    is_monitor = strat == MONITOR_STRATEGY
    gpu_cap = cfg.get("gpu_cap")
    cpu_cap = cfg.get("cpu_cap")
    return RunParameters(
        app_name=app,
        cpu_enabled=cpu_enabled,
        number_of_streams=2,
        number_od_nodes=nodes,
        batch_size=batch_size,
        strategy=strat,
        cpu_power_scaling=0.0,
        initial_cpu_batch_size_scaling=0,
        start_powercap=None,
        optimizer_step_interval=n_windows,
        edp_optimization_steps=0,  # unlimited; depth / run length bound the run instead
        iters=iters,
        search_depth=cfg["depth"] if is_search else None,
        search_grid_points=cfg["grid_points"] if is_search else None,
        gpu_powercap=(float(gpu_cap) if is_monitor and gpu_cap not in (None, "") else None),
        cpu_powercap=(float(cpu_cap) if is_monitor and cpu_cap not in (None, "") else None),
    )


# --------------------------------------------------------------------------- #
# Fixed-config (best-cap) search
# --------------------------------------------------------------------------- #

def parse_edp_steps(log_path: str) -> tuple:
    """Return (mean_edp, n_steps) from the monitor's per-step '[EDP] edp=' lines,
    dropping the first step as warmup. (None, 0) if none were emitted."""
    try:
        with open(log_path, "r", errors="replace") as f:
            text = f.read()
    except FileNotFoundError:
        return None, 0
    vals = [float(m.group(1)) for m in _EDP_RE.finditer(text)]
    if not vals:
        return None, 0
    kept = vals[1:] if len(vals) > 1 else vals
    return sum(kept) / len(kept), len(vals)


def load_monitor_results(csv_path: str) -> dict:
    """Map (gpu_key, cpu_key) -> recorded row for cells already searched (resume)."""
    results: dict = {}
    if not os.path.exists(csv_path):
        return results
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                results[(cap_key(row["gpu_cap"]), cap_key(row["cpu_cap"]))] = row
            except KeyError:
                continue
    return results


def _monitor_score(row: dict):
    """Selection score for a fixed-config cell: mean per-step EDP if available, else
    the end-to-end EDP of the short run. None if neither is a positive number."""
    for k in ("mean_edp", "edp"):
        try:
            v = float(row.get(k))
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v
    return None


def _row_caps(row: dict) -> tuple:
    g = float(row["gpu_cap"])
    c = row.get("cpu_cap")
    c = None if c in (None, "") else float(c)
    return g, c


def pick_best_monitor(results: dict):
    """Best (gpu, cpu) over the searched cells, or None if nothing succeeded."""
    best_row, best_score = None, None
    for row in results.values():
        if row.get("status") != "ok":
            continue
        score = _monitor_score(row)
        if score is None:
            continue
        if best_score is None or score < best_score:
            best_score, best_row = score, row
    return _row_caps(best_row) if best_row is not None else None


def run_monitor_search(app: str, args, out_dir: str, *, cpu_enabled: bool,
                       batch_size: int, devices: int, bpw: int):
    """Search the GPU x CPU cap space with short EDP_MONITOR_SPLIT runs; return the
    lowest-EDP (gpu, cpu) (or None). Crash-safe/resumable via monitor_search.csv."""
    m_iters = monitor_iters(devices, batch_size, app, args.n_windows, bpw)
    grid = monitor_grid(cpu_enabled, args.search_step)
    csv_path = os.path.join(out_dir, "monitor_search.csv")
    results = load_monitor_results(csv_path)
    todo = [(g, c) for (g, c) in grid if (cap_key(g), cap_key(c)) not in results]

    print(f"[search] {app}: fixed-config search — {len(grid)} cells "
          f"(step={args.search_step}, m_iters={m_iters}); {len(results)} done; {len(todo)} to run")

    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MONITOR_CSV_COLUMNS)
        if new_file:
            writer.writeheader()
            f.flush()
        for i, (g, c) in enumerate(todo, 1):
            cfg = {"strategy": MONITOR_STRATEGY, "mode": "monitor",
                   "gpu_cap": g, "cpu_cap": c, "grid_points": "", "depth": ""}
            params = make_params(cfg, app=app, nodes=args.nodes, batch_size=batch_size,
                                 n_windows=args.n_windows, iters=m_iters, cpu_enabled=cpu_enabled)
            log_path = os.path.join(
                out_dir, f"log_MONITOR_gpu{cap_key(g)}_cpu{cap_key(c) or 'NA'}.txt")
            print(f"\n[search] ({i}/{len(todo)}) fixed cfg gpu={cap_key(g)} "
                  f"cpu={cap_key(c) or '-'} -> {os.path.basename(log_path)}")
            res = single_app_run(params, log_save_path=log_path)
            mean_edp, n_steps = parse_edp_steps(log_path)
            row = {"gpu_cap": g, "cpu_cap": ("" if c is None else c), "n_steps": n_steps}
            if res is None or res == -1:
                row.update({"mean_edp": "", "time_s": "", "energy_j": "", "edp": "",
                            "status": "FAILED"})
                print(f"[search]     FAILED (see {log_path})")
            else:
                edp = res.execution_duration * res.energy_used
                row.update({"mean_edp": (round(mean_edp, 6) if mean_edp is not None else ""),
                            "time_s": round(res.execution_duration, 3),
                            "energy_j": round(res.energy_used, 3),
                            "edp": round(edp, 3), "status": "ok"})
                print(f"[search]     mean_edp={mean_edp} (steps={n_steps}) "
                      f"end-to-end EDP={edp:.1f}")
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            results[(cap_key(g), cap_key(c))] = row

    best = pick_best_monitor(results)
    if best is None:
        print(f"[search] {app}: fixed-config search found no successful cell; "
              f"skipping best-config reference.")
    else:
        bg, bc = best
        print(f"[search] {app}: best fixed config gpu={cap_key(bg)} "
              f"cpu={cap_key(bc) or '-'}")
    return best


# --------------------------------------------------------------------------- #
# Final comparison (references + search algorithms)
# --------------------------------------------------------------------------- #

def build_final_runs(cpu_enabled: bool, best, include_refs: bool) -> list[dict]:
    """Configs for results.csv: the DISABLED + best-fixed-config references (unless
    --no-baseline) followed by the search-strategy sweep."""
    runs: list[dict] = []
    if include_refs:
        runs.append({"strategy": DISABLED_STRATEGY, "mode": "disabled",
                     "grid_points": "", "depth": "", "gpu_cap": "", "cpu_cap": ""})
        if best is not None:
            bg, bc = best
            runs.append({"strategy": MONITOR_STRATEGY, "mode": "best_config",
                         "grid_points": "", "depth": "",
                         "gpu_cap": bg, "cpu_cap": ("" if bc is None else bc)})
    for strat in swept_strategies(cpu_enabled):
        for gp in GRID_POINTS:
            for depth in DEPTHS:
                runs.append({"strategy": strat, "mode": search_mode(strat),
                             "grid_points": gp, "depth": depth,
                             "gpu_cap": "", "cpu_cap": ""})
    return runs


def run_key(cfg: dict, rep: int) -> tuple:
    return (cfg["strategy"], cfg["mode"], cap_key(cfg["gpu_cap"]), cap_key(cfg["cpu_cap"]),
            str(cfg["grid_points"]), str(cfg["depth"]), rep)


def load_done(csv_path: str) -> set:
    """Set of (config, rep) keys already recorded, so a restart resumes per rep."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add((row["strategy"], row["mode"], cap_key(row["gpu_cap"]),
                          cap_key(row["cpu_cap"]), str(row["grid_points"]),
                          str(row["depth"]), int(row["rep"])))
            except (KeyError, ValueError):
                continue
    return done


def log_name(cfg: dict, rep: int) -> str:
    mode = cfg["mode"]
    if mode == "disabled":
        return f"log_DISABLED_rep{rep}.txt"
    if mode == "best_config":
        return (f"log_BEST_gpu{cap_key(cfg['gpu_cap'])}"
                f"_cpu{cap_key(cfg['cpu_cap']) or 'NA'}_rep{rep}.txt")
    short = cfg["strategy"].replace("EDP_SEARCH_", "")
    return f"log_SEARCH_{short}_gp{cfg['grid_points']}_d{cfg['depth']}_rep{rep}.txt"


def parse_best_caps(log_path: str) -> tuple:
    """Return (best_gpu, best_cpu, best_J) parsed from the search's 'holding best
    caps' line, or ('', '', '') if absent (e.g. reference runs or unfinished search)."""
    try:
        with open(log_path, "r", errors="replace") as f:
            text = f.read()
    except FileNotFoundError:
        return "", "", ""
    m = None
    for m in _BEST_RE.finditer(text):
        pass  # keep the last occurrence
    if not m:
        return "", "", ""
    return m.group("gpu"), m.group("cpu"), m.group("J")


def main():
    parser = argparse.ArgumentParser(
        description="Grid-search EDP optimizer benchmark (EDP_SEARCH_SHARED / SPLIT)")
    parser.add_argument("--nodes", type=int, default=4, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, default=GPUS_PER_NODE,
                        help=f"GPUs exposed per node (default: {GPUS_PER_NODE})")
    parser.add_argument("--devices", type=int, default=None,
                        help="Override the computed total device count used to size iters")
    parser.add_argument("--N", type=int, default=20, dest="n_windows",
                        help="optimizer_step_interval: sync windows averaged per step")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Repetitions per configuration (default: 1)")
    parser.add_argument("--out-dir", default="search_optimizer_sweep")
    parser.add_argument("--iters", type=int, default=None,
                        help="Override the auto-computed dataset iteration count")
    parser.add_argument("--search-step", type=float, default=DEFAULT_SEARCH_STEP,
                        help="Fixed-config cap-grid resolution (default: 0.1 -> 0.1..1.0)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the DISABLED + best-config references (and their search)")
    args = parser.parse_args()

    base_out_dir = os.path.abspath(args.out_dir)
    for app_cfg in APPS:
        run_app_sweep(app_cfg, args, base_out_dir)


def run_app_sweep(app_cfg: dict, args, base_out_dir: str) -> None:
    """Run the fixed-config search + final comparison for a single app, writing
    results and logs under a per-app subdirectory of base_out_dir."""
    app = app_cfg["name"]
    cpu_enabled = app_cfg["cpu_enabled"]
    batch_size = default_batch_size(app)
    bpw = batches_per_window(app)

    devices = (args.devices if args.devices is not None
               else total_device_count(args.nodes, cpu_enabled, args.gpus_per_node))
    iters = args.iters if args.iters is not None else sweep_iters(
        devices, batch_size, app, args.n_windows, bpw, cpu_enabled)

    out_dir = os.path.join(base_out_dir, app)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")

    print(f"[search] app={app} nodes={args.nodes} devices={devices} bs={batch_size} "
          f"N={args.n_windows} bpw={bpw} iters={iters} n_runs={args.n_runs}")

    # Phase 1: find the best fixed GPU/CPU cap config (unless references are skipped).
    best = None
    if not args.no_baseline:
        best = run_monitor_search(app, args, out_dir, cpu_enabled=cpu_enabled,
                                  batch_size=batch_size, devices=devices, bpw=bpw)

    # Phase 2: final comparison (references + search algorithms) at full iters.
    runs = build_final_runs(cpu_enabled, best, include_refs=not args.no_baseline)
    done = load_done(csv_path)
    units = [(cfg, rep) for cfg in runs for rep in range(args.n_runs)]
    todo = [(cfg, rep) for cfg, rep in units if run_key(cfg, rep) not in done]

    print(f"[search] {len(runs)} configs x {args.n_runs} reps = {len(units)} launches; "
          f"{len(done)} already done; {len(todo)} to run")

    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
            f.flush()
        for i, (cfg, rep) in enumerate(todo, 1):
            is_search = cfg["strategy"] in SEARCH_STRATEGIES
            steps = (required_steps(cfg["strategy"], cfg["depth"], cfg["grid_points"], cpu_enabled)
                     if is_search else 0)
            params = make_params(cfg, app=app, nodes=args.nodes, batch_size=batch_size,
                                 n_windows=args.n_windows, iters=iters, cpu_enabled=cpu_enabled)
            log_path = os.path.join(out_dir, log_name(cfg, rep))
            if is_search:
                desc = f"gp={cfg['grid_points']} depth={cfg['depth']} (~{steps} steps)"
            elif cfg["mode"] == "best_config":
                desc = f"(best fixed gpu={cap_key(cfg['gpu_cap'])} cpu={cap_key(cfg['cpu_cap']) or '-'})"
            else:
                desc = "(disabled / default caps)"
            print(f"\n[search] ({i}/{len(todo)}) {cfg['strategy']} {desc} "
                  f"rep {rep + 1}/{args.n_runs} -> {os.path.basename(log_path)}")
            res = single_app_run(params, log_save_path=log_path)
            best_gpu, best_cpu, best_J = parse_best_caps(log_path)
            row = {
                "strategy": cfg["strategy"], "mode": cfg["mode"],
                "gpu_cap": cfg["gpu_cap"], "cpu_cap": cfg["cpu_cap"],
                "grid_points": cfg["grid_points"], "depth": cfg["depth"],
                "N": args.n_windows, "iters": iters, "nodes": args.nodes,
                "devices": devices, "batch_size": batch_size, "rep": rep,
                "best_gpu": best_gpu, "best_cpu": best_cpu, "best_J": best_J,
            }
            if res is None or res == -1:
                row.update({"time_s": "", "energy_j": "", "edp": "", "status": "FAILED"})
                print(f"[search]     FAILED (see {log_path})")
            else:
                edp = res.execution_duration * res.energy_used
                row.update({"time_s": round(res.execution_duration, 3),
                            "energy_j": round(res.energy_used, 3),
                            "edp": round(edp, 3), "status": "ok"})
                extra = (f" best gpu={best_gpu} cpu={best_cpu}" if best_gpu else "")
                print(f"[search]     time={res.execution_duration:.1f}s "
                      f"energy={res.energy_used:.1f}J EDP={edp:.1f}{extra}")
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    summarize(csv_path)
    print(f"\n[search] results -> {csv_path}")


def summarize(csv_path: str) -> None:
    """Reference EDPs (DISABLED do-nothing, best fixed config), then every search
    config's mean end-to-end EDP sorted best-first as a percent of each reference."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            try:
                r["edp"] = float(r["edp"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    if not rows:
        print("\n[search] no successful runs to summarize yet.")
        return

    # Mean EDP per configuration (averaged over reps).
    by_cfg: dict[tuple, dict] = {}
    for r in rows:
        key = (r["strategy"], r["mode"], cap_key(r["gpu_cap"]), cap_key(r["cpu_cap"]),
               str(r["grid_points"]), str(r["depth"]))
        agg = by_cfg.setdefault(key, {"row": r, "edps": []})
        agg["edps"].append(r["edp"])

    configs = []
    for _key, agg in by_cfg.items():
        r = agg["row"]
        configs.append({"strategy": r["strategy"], "mode": r["mode"],
                        "gpu_cap": r["gpu_cap"], "cpu_cap": r["cpu_cap"],
                        "gp": r["grid_points"], "depth": r["depth"],
                        "edp": sum(agg["edps"]) / len(agg["edps"]),
                        "n": len(agg["edps"]), "row": r})

    disabled = next((c for c in configs if c["mode"] == "disabled"), None)
    best_cfg = next((c for c in configs if c["mode"] == "best_config"), None)
    disabled_edp = disabled["edp"] if disabled else None
    best_edp = best_cfg["edp"] if best_cfg else None

    print("\n" + "=" * 100)
    print("SEARCH OPTIMIZER BENCHMARK — end-to-end EDP = time * energy (lower is better)")
    print("=" * 100)

    print("\nReferences:")
    if disabled:
        print(f"  DISABLED (default caps)      EDP={disabled_edp:>12.1f}  (n={disabled['n']})")
    if best_cfg:
        print(f"  best fixed cfg gpu={cap_key(best_cfg['gpu_cap'])} "
              f"cpu={cap_key(best_cfg['cpu_cap']) or '-'}  "
              f"EDP={best_edp:>12.1f}  (n={best_cfg['n']})")
    if not disabled and not best_cfg:
        print("  (none — run without --no-baseline to include references)")

    def pct(edp, ref):
        return f"{100.0 * edp / ref:.1f}%" if ref else "-"

    print(f"\nSearch configs (sorted best-EDP-first):")
    print(f"  {'strategy':<20}{'gp':>4}{'depth':>6}{'EDP':>13}"
          f"{'vs best':>10}{'vs disabled':>13}{'best gpu/cpu':>16}")
    for c in sorted([c for c in configs if c["strategy"] in SEARCH_STRATEGIES],
                    key=lambda x: x["edp"]):
        bg, bc = c["row"].get("best_gpu", ""), c["row"].get("best_cpu", "")
        best = f"{bg}/{bc}" if bg else "-"
        print(f"  {c['strategy']:<20}{str(c['gp']):>4}{str(c['depth']):>6}{c['edp']:>13.1f}"
              f"{pct(c['edp'], best_edp):>10}{pct(c['edp'], disabled_edp):>13}{best:>16}")


if __name__ == "__main__":
    main()
