"""
Sweep of the dynamic EDP optimizers (SIMPLE / SPSA / CMA-ES) on CNN or RNN.

Every configuration below runs once per start power cap (0.2 / 0.5 / 0.8), together
with a fixed-cap reference (EQUAL_SPLIT_EDP_MONITOR) at the same cap:

  Fixed caps        EQUAL_SPLIT_EDP_MONITOR
  SIMPLE            alpha 0.15 / 0.3 / 0.6, eps 0.1, epsilon_decay 0.99
  SIMPLE            alpha 0.3, eps 0.1, epsilon_decay 0.92 (eps ends near 0.05-0.06)
  SPSA              alpha 0.05 / 0.1, eps 0.1, epsilon_decay 0.99
  CMA-ES            sigma (gradient_opt_eps) 0.1 / 0.2 / 0.3

That is 30 runs per app. All runs use N = optimizer_step_interval = 10,
alpha_decay 0.99 and edp_optimization_steps = 0 (optimize until the run ends). The
SNR measurements behind these values are in results/snr_after_estimator_fixes.

Repetitions: re-run the script. Each invocation completes the current repetition,
i.e. it runs every configuration that has fewer successful runs than the others,
or, if all are level, starts the next repetition of all of them. So an interrupted
sweep resumes where it stopped and a finished one is repeated, with no flags. Failed
runs are recorded but do not count, so they are retried.

Run order: fixed caps first (check their run time before leaving the sweep alone),
then the main setting of each algorithm at every cap, then the variations, so a
partial sweep is still usable.

Output: '<results-dir>/<app>_<out-dir>_<nodes>_nodes/' with results.csv (one row per
run) and one log per run. Each log holds a per-window '[OPTIMIZER STEP]' line (caps
of every device, EDP) and a per-step '[EDP]' line, which is what the trajectories
are built from.

Usage:
    python run_optimizers.py --app {cnn,rnn} [--nodes 4] [--iters N]
        [--results-dir DIR] [--out-dir STEM] [--dry-run] [--summary-only]
"""

import argparse
import csv
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from experiments import single_app_run
from models import RunParameters
from run_length import device_count, windows_per_iter

# --- Fixed experiment setup ---
APPS = ["cnn", "rnn"]
DEFAULT_NODES = 4
BATCH_SIZE = 50
NUMBER_OF_STREAMS = 2
N = 10                    # optimizer_step_interval: sync windows averaged per step

# Dataset iterations per run (--iters). Both give about 1,240 sync windows at 4 nodes.
# Measured run time per iteration (snr_after_estimator_fixes, 4 nodes):
#   CNN 9.6 s at cap 0.4, 7.6 s at 0.6. Cap 0.2 is not measured with the current code;
#       one CPU alone at 0.2 slowed CNN by 22%, so all devices at 0.2 may be up to
#       ~1.5x slower than 0.4 (130 iters: ~21 min at 0.4, up to ~30 min at 0.2).
#   RNN 7.25 s at any cap 0.4-0.6, about 5% more at 0.2 (220 iters: ~27-28 min).
ITERS = {"cnn": 130, "rnn": 220}
SECONDS_PER_ITER_AT_0_4 = {"cnn": 9.6, "rnn": 7.25}   # for the --dry-run estimate only

START_POWERCAPS = [0.2, 0.5, 0.8]
EPS = 0.1                 # SIMPLE / SPSA perturbation, normalized cap fraction
ALPHA_DECAY = 0.99
EPS_DECAY = 0.99          # eps = eps0 * epsilon_decay^iteration
EPS_DECAY_FAST = 0.92     # SIMPLE eps-schedule variant

FIXED = "EQUAL_SPLIT_EDP_MONITOR"
SIMPLE = "EDP_GRADIENT_SIMPLE"
SPSA = "EDP_GRADIENT_SPSA"
CMAES = "EDP_GRADIENT_CMAES"
SHORT_NAME = {FIXED: "FIXED", SIMPLE: "SIMPLE", SPSA: "SPSA", CMAES: "CMAES"}

CSV_COLUMNS = [
    "app", "strategy", "start_powercap", "alpha", "alpha_decay", "eps", "eps_decay",
    "N", "iters", "nodes", "batch_size", "rep",
    "time_s", "energy_j", "edp", "windows", "opt_steps", "final_caps", "status", "log",
]
# Columns that identify a configuration (a repetition shares all of them).
KEY_COLUMNS = ["app", "strategy", "start_powercap", "alpha", "alpha_decay", "eps",
               "eps_decay", "N", "iters", "nodes", "batch_size"]


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
def _cfg(strategy, cap, alpha=None, eps=None, eps_decay=None) -> dict:
    gradient = strategy in (SIMPLE, SPSA)
    return {
        "strategy": strategy,
        "start_powercap": cap,
        "alpha": alpha,
        "alpha_decay": ALPHA_DECAY if gradient else None,
        "eps": eps,                       # CMA-ES: initial sigma; fixed caps: unused
        "eps_decay": eps_decay,
    }


def build_configs() -> list[dict]:
    """The 30 configurations of one repetition, in run order."""
    fixed = [_cfg(FIXED, cap) for cap in START_POWERCAPS]
    main, variations = [], []
    for cap in START_POWERCAPS:
        main += [
            _cfg(SIMPLE, cap, alpha=0.3, eps=EPS, eps_decay=EPS_DECAY),
            _cfg(SPSA, cap, alpha=0.05, eps=EPS, eps_decay=EPS_DECAY),
            _cfg(CMAES, cap, eps=0.2),
        ]
        variations += [
            _cfg(SIMPLE, cap, alpha=0.15, eps=EPS, eps_decay=EPS_DECAY),
            _cfg(SIMPLE, cap, alpha=0.6, eps=EPS, eps_decay=EPS_DECAY),
            _cfg(SIMPLE, cap, alpha=0.3, eps=EPS, eps_decay=EPS_DECAY_FAST),
            _cfg(SPSA, cap, alpha=0.1, eps=EPS, eps_decay=EPS_DECAY),
            _cfg(CMAES, cap, eps=0.1),
            _cfg(CMAES, cap, eps=0.3),
        ]
    return fixed + main + variations


def fmt(v) -> str:
    """CSV / key form of a parameter: '' when unused, else a compact number."""
    if v is None or v == "":
        return ""
    if isinstance(v, str):
        v = float(v)
    return f"{v:g}"


def label(cfg: dict) -> str:
    name = SHORT_NAME.get(cfg["strategy"], cfg["strategy"])
    if cfg["strategy"] == FIXED:
        return name
    if cfg["strategy"] == CMAES:
        return f"{name} sigma={fmt(cfg['eps'])}"
    return f"{name} alpha={fmt(cfg['alpha'])} eps={fmt(cfg['eps'])} eps_decay={fmt(cfg['eps_decay'])}"


def make_params(cfg: dict, app: str, nodes: int, iters: int) -> RunParameters:
    # Unused parameters (None) fall back to the defaults in write_powercap_conf.
    return RunParameters(
        app_name=app,
        cpu_enabled=True,
        number_of_streams=NUMBER_OF_STREAMS,
        number_od_nodes=nodes,
        batch_size=BATCH_SIZE,
        strategy=cfg["strategy"],
        cpu_power_scaling=0.0,
        initial_cpu_batch_size_scaling=0,
        start_powercap=cfg["start_powercap"],
        start_alpha=cfg["alpha"],
        alpha_decay=cfg["alpha_decay"],
        epsilon_decay=cfg["eps_decay"],
        gradient_opt_eps=cfg["eps"],
        edp_optimization_steps=0,
        optimizer_step_interval=N,
        iters=iters,
    )


def config_key(cfg: dict, app: str, nodes: int, iters: int) -> tuple:
    row = {**cfg, "app": app, "N": N, "iters": iters, "nodes": nodes,
           "batch_size": BATCH_SIZE}
    return row_key(row)


def row_key(row: dict) -> tuple:
    """Configuration identity of a config dict or a CSV row (numbers normalized by fmt)."""
    return tuple(str(row[c]) if c in ("app", "strategy") else fmt(row[c])
                 for c in KEY_COLUMNS)


def log_name(cfg: dict, rep: int) -> str:
    parts = [f"log_{SHORT_NAME[cfg['strategy']]}", f"cap{cfg['start_powercap']:.2f}"]
    if cfg["alpha"] is not None:
        parts.append(f"a{fmt(cfg['alpha'])}")
    if cfg["eps"] is not None:
        parts.append(f"{'sigma' if cfg['strategy'] == CMAES else 'eps'}{fmt(cfg['eps'])}")
    if cfg["eps_decay"] is not None:
        parts.append(f"ed{fmt(cfg['eps_decay'])}")
    parts.append(f"rep{rep}")
    return "_".join(parts) + ".txt"


# ---------------------------------------------------------------------------
# Budget estimate (dry run)
# ---------------------------------------------------------------------------
def steps_per_iteration(strategy: str, devices: int) -> int | None:
    """Optimizer steps one iteration (SIMPLE/SPSA) or generation (CMA-ES) takes."""
    if strategy == SIMPLE:
        return 2 * devices + 1          # base + plus/minus probe per device
    if strategy == SPSA:
        return 3                        # plus probe, minus probe, descent
    if strategy == CMAES:
        lam = 4 + int(3.0 * math.log(devices + 1.0))
        return lam + 2                  # sample, lambda evaluations, update
    return None


def print_budget(configs: list[dict], app: str, nodes: int, iters: int) -> None:
    devices = device_count(nodes, cpu_enabled=True)
    windows = windows_per_iter(app, BATCH_SIZE, nodes, NUMBER_OF_STREAMS) * iters
    minutes = SECONDS_PER_ITER_AT_0_4[app] * iters / 60.0
    print(f"[sweep] estimate: ~{windows:.0f} sync windows per run, ~{minutes:.0f} min "
          f"per run at cap 0.4 ({devices} devices; each cap change costs N+1={N + 1} windows)")
    for strategy in (SIMPLE, SPSA, CMAES):
        spi = steps_per_iteration(strategy, devices)
        unit = "generations" if strategy == CMAES else "iterations"
        print(f"           {SHORT_NAME[strategy]:<7} {spi:>2} steps x {N + 1} windows = "
              f"{spi * (N + 1)} windows each -> ~{windows / (spi * (N + 1)):.1f} {unit}")
    print(f"           {len(configs)} runs -> ~{len(configs) * minutes / 60.0:.1f} h at cap 0.4")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
_EDP_RE = re.compile(r"\[EDP\]\s+edp=")
_CAP_RE = re.compile(r"\[OPTIMIZER STEP\].*\bcap=([-\d.,eE+]+)")


def parse_log(log_path: str) -> tuple[int, int, str]:
    """(sync windows, optimizer steps, cap vector of the last window) of a run log."""
    windows = steps = 0
    last_caps = ""
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                m = _CAP_RE.search(line)
                if m:
                    windows += 1
                    last_caps = m.group(1)
                elif _EDP_RE.search(line):
                    steps += 1
    except FileNotFoundError:
        pass
    return windows, steps, last_caps


def load_rows(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def completed_counts(rows: list[dict]) -> Counter:
    counts: Counter = Counter()
    for r in rows:
        if r.get("status") == "ok":
            try:
                counts[row_key(r)] += 1
            except (KeyError, ValueError):
                continue
    return counts


def run_one(cfg: dict, rep: int, app: str, nodes: int, iters: int, out_dir: str) -> dict:
    log_path = os.path.join(out_dir, log_name(cfg, rep))
    row = {
        "app": app, "strategy": cfg["strategy"],
        "start_powercap": fmt(cfg["start_powercap"]), "alpha": fmt(cfg["alpha"]),
        "alpha_decay": fmt(cfg["alpha_decay"]), "eps": fmt(cfg["eps"]),
        "eps_decay": fmt(cfg["eps_decay"]), "N": N, "iters": iters, "nodes": nodes,
        "batch_size": BATCH_SIZE, "rep": rep, "log": os.path.basename(log_path),
    }
    try:
        res = single_app_run(make_params(cfg, app, nodes, iters), log_save_path=log_path)
    except Exception as e:  # missing timing/energy lines in the output, etc.
        print(f"[sweep]   run raised {type(e).__name__}: {e}")
        res = None
    windows, steps, final_caps = parse_log(log_path)
    row.update({"windows": windows, "opt_steps": steps, "final_caps": final_caps})
    if res is None or res == -1:
        row.update({"time_s": "", "energy_j": "", "edp": "", "status": "FAILED"})
        print(f"[sweep]   FAILED (see {log_path})")
    else:
        edp = res.execution_duration * res.energy_used
        row.update({"time_s": round(res.execution_duration, 3),
                    "energy_j": round(res.energy_used, 3),
                    "edp": round(edp, 3), "status": "ok"})
        print(f"[sweep]   time={res.execution_duration:.1f}s energy={res.energy_used:.1f}J "
              f"EDP={edp:.4g} windows={windows} steps={steps} final caps={final_caps}")
    return row


def summarize(csv_path: str) -> None:
    """Per start cap: every configuration's mean end-to-end EDP over its successful
    repetitions, relative to the fixed-cap run at that cap (<100% = better)."""
    groups: dict = defaultdict(list)
    for r in load_rows(csv_path):
        if r.get("status") != "ok":
            continue
        try:
            groups[row_key(r)].append(r)
        except (KeyError, ValueError):
            continue
    if not groups:
        print("\n[sweep] no successful runs to summarize yet.")
        return

    def mean(rs, col):
        return sum(float(x[col]) for x in rs) / len(rs)

    print("\n" + "=" * 96)
    print("OPTIMIZER SWEEP: end-to-end EDP = time * energy, mean over repetitions (lower is better)")
    print("=" * 96)
    by_cap: dict = defaultdict(list)
    for rs in groups.values():
        by_cap[(rs[0]["app"], rs[0]["iters"], float(rs[0]["start_powercap"]))].append(rs)
    for (app, iters, cap) in sorted(by_cap):
        entries = by_cap[(app, iters, cap)]
        base = next((rs for rs in entries if rs[0]["strategy"] == FIXED), None)
        base_edp = mean(base, "edp") if base else None
        print(f"\n--- {app}, iters={iters}, start cap {cap:g}"
              + (f" (fixed-cap EDP {base_edp:.4g})" if base_edp else "") + " ---")
        print(f"  {'configuration':<46}{'reps':>5}{'time s':>9}{'energy kJ':>11}"
              f"{'EDP':>12}{'vs fixed':>10}")
        for rs in sorted(entries, key=lambda x: mean(x, "edp")):
            r0 = rs[0]
            cfg = {"strategy": r0["strategy"], "alpha": r0["alpha"] or None,
                   "eps": r0["eps"] or None, "eps_decay": r0["eps_decay"] or None}
            edp = mean(rs, "edp")
            vs = f"{100.0 * edp / base_edp:.1f}%" if base_edp else "-"
            print(f"  {label(cfg):<46}{len(rs):>5}{mean(rs, 'time_s'):>9.0f}"
                  f"{mean(rs, 'energy_j') / 1000.0:>11.1f}{edp:>12.4g}{vs:>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Dynamic EDP optimizer sweep (CNN / RNN)")
    parser.add_argument("--app", required=True, choices=APPS)
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    parser.add_argument("--iters", type=int, default=None,
                        help="Dataset iterations per run (default: per-app value in ITERS). "
                             "Part of the configuration key, so changing it starts a new set.")
    parser.add_argument("--results-dir", default=str(repo_root / "results" / "optimizer_sweep"))
    parser.add_argument("--out-dir", default="optimizer_sweep",
                        help="Name stem; output goes to '<results-dir>/<app>_<out-dir>_<nodes>_nodes'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the runs this invocation would do, and the budget, then exit")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print the summary of results.csv")
    args = parser.parse_args()

    app, nodes = args.app, args.nodes
    iters = args.iters if args.iters is not None else ITERS[app]
    # Absolute, because single_app_run() changes the working directory.
    out_dir = os.path.join(os.path.abspath(args.results_dir), f"{app}_{args.out_dir}_{nodes}_nodes")
    csv_path = os.path.join(out_dir, "results.csv")

    if args.summary_only:
        summarize(csv_path)
        return

    configs = build_configs()
    counts = completed_counts(load_rows(csv_path))
    done = [counts[config_key(c, app, nodes, iters)] for c in configs]
    target = min(done) + 1
    todo = [(c, n) for c, n in zip(configs, done) if n < target]

    print(f"[sweep] app={app} nodes={nodes} bs={BATCH_SIZE} iters={iters} N={N} "
          f"alpha_decay={ALPHA_DECAY} -> {out_dir}")
    print(f"[sweep] repetition {target}: {len(todo)} of {len(configs)} configurations to run")
    print_budget(todo, app, nodes, iters)
    if args.dry_run:
        for c, n in todo:
            print(f"           cap={c['start_powercap']:g}  {label(c)}  (rep {n})")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
            f.flush()
        for i, (cfg, rep) in enumerate(todo, 1):
            print(f"\n[sweep] ({i}/{len(todo)}) cap={cfg['start_powercap']:g} {label(cfg)} rep={rep}")
            writer.writerow(run_one(cfg, rep, app, nodes, iters, out_dir))
            f.flush()
            os.fsync(f.fileno())

    summarize(csv_path)
    print(f"\n[sweep] results -> {csv_path}")


if __name__ == "__main__":
    main()
