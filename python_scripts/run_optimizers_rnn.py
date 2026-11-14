"""
Hyperparameter sweep of the dynamic EDP optimizers on RNN.

Runs SIMPLE / SPSA / CMA-ES (plus a fixed-cap EQUAL_SPLIT baseline for
reference) over a grid of start power caps and gradient hyperparameters, and
records the end-to-end outcome (time, energy, EDP) of each run.

Fixed setup: app=rnn, 3 nodes, batch_size=50, iters=400.

Swept:
  - start_powercap   in START_POWERCAPS  (0.2 / 0.5 / 0.8)
  - start_alpha      in ALPHAS
  - alpha_decay      in ALPHA_DECAYS
  - epsilon_decay    in EPS_DECAYS

Only SIMPLE and SPSA consume alpha / alpha_decay / epsilon_decay, so the
alpha/decay grid is swept *only* for them. CMA-ES (which uses neither — eps is
its initial sigma and it adapts internally) and the EQUAL_SPLIT baseline are run
once per start cap, avoiding redundant runs. gradient_opt_eps and N are held
fixed (SNR-informed: RNN's single-device EDP signal is detectable at eps~0.05-0.1).

Crash-safe: results are appended to results.csv and flushed after every run; a
restart skips configs already present, so nothing already run is lost or
repeated. Each run's full log (optimizer trace) is saved too.

Comparison metric: end-to-end EDP = execution_time * energy (lower is better),
reported per start cap relative to that cap's EQUAL_SPLIT baseline.

Usage:
    python run_optimizers_rnn.py [--out-dir DIR] [--no-baseline]
"""

import argparse
import csv
import os
from pathlib import Path

from experiments import single_app_run
from models import RunParameters

# --- Fixed experiment setup ---
APP = "rnn"
NODES = 3
BATCH_SIZE = 50
ITERS = 400
GRADIENT_OPT_EPS = 0.1        # normalized cap fraction; held fixed (SNR-informed)
OPTIMIZER_STEP_INTERVAL = 20  # windows averaged per optimizer step; held fixed

# --- Swept hyperparameters ---
START_POWERCAPS = [0.2, 0.5, 0.8]
ALPHAS = [0.05, 0.1, 0.2]
ALPHA_DECAYS = [0.9, 0.99]
EPS_DECAYS = [0.9, 0.99]

# Canonical values recorded for strategies that don't use the alpha/decay grid
# (CMA-ES, EQUAL_SPLIT). They have no effect on those runs.
DEF_ALPHA, DEF_ALPHA_DECAY, DEF_EPS_DECAY = 0.1, 0.99, 0.99

SWEEP_STRATEGIES = ["EDP_GRADIENT_SIMPLE", "EDP_GRADIENT_SPSA"]  # use alpha/decays
SINGLE_STRATEGIES = ["EDP_GRADIENT_CMAES"]  # cap-only, no alpha/decay sweep
BASELINE = "EQUAL_SPLIT"

CSV_COLUMNS = [
    "strategy", "start_powercap", "alpha", "alpha_decay", "eps_decay",
    "eps", "N", "iters", "nodes", "batch_size",
    "time_s", "energy_j", "edp", "status",
]


def make_params(cfg: dict) -> RunParameters:
    return RunParameters(
        app_name=APP,
        cpu_enabled=True,
        number_of_streams=2,
        number_od_nodes=NODES,
        batch_size=BATCH_SIZE,
        strategy=cfg["strategy"],
        cpu_power_scaling=0.0,
        initial_cpu_batch_size_scaling=0,
        start_powercap=cfg["start_powercap"],
        start_alpha=cfg["alpha"],
        alpha_decay=cfg["alpha_decay"],
        epsilon_decay=cfg["eps_decay"],
        gradient_opt_eps=GRADIENT_OPT_EPS,
        edp_optimization_steps=0,  # unlimited: optimize for the whole run
        optimizer_step_interval=OPTIMIZER_STEP_INTERVAL,
        iters=ITERS,
    )


def build_runs(include_baseline: bool) -> list[dict]:
    """One dict per run. SIMPLE/SPSA get the full alpha/decay grid; CMA-ES and the
    baseline run once per cap with canonical (unused) alpha/decay values."""
    runs: list[dict] = []
    for cap in START_POWERCAPS:
        if include_baseline:
            runs.append({"strategy": BASELINE, "start_powercap": cap,
                         "alpha": DEF_ALPHA, "alpha_decay": DEF_ALPHA_DECAY,
                         "eps_decay": DEF_EPS_DECAY})
        for strat in SINGLE_STRATEGIES:
            runs.append({"strategy": strat, "start_powercap": cap,
                         "alpha": DEF_ALPHA, "alpha_decay": DEF_ALPHA_DECAY,
                         "eps_decay": DEF_EPS_DECAY})
        for strat in SWEEP_STRATEGIES:
            for a in ALPHAS:
                for ad in ALPHA_DECAYS:
                    for ed in EPS_DECAYS:
                        runs.append({"strategy": strat, "start_powercap": cap,
                                     "alpha": a, "alpha_decay": ad, "eps_decay": ed})
    return runs


def run_key(cfg: dict) -> tuple:
    return (cfg["strategy"], round(cfg["start_powercap"], 3), round(cfg["alpha"], 4),
            round(cfg["alpha_decay"], 4), round(cfg["eps_decay"], 4))


def load_done(csv_path: str) -> set:
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add((row["strategy"], round(float(row["start_powercap"]), 3),
                          round(float(row["alpha"]), 4), round(float(row["alpha_decay"]), 4),
                          round(float(row["eps_decay"]), 4)))
            except (KeyError, ValueError):
                continue
    return done


def log_name(cfg: dict) -> str:
    return (f"log_{cfg['strategy']}_cap{cfg['start_powercap']:.2f}"
            f"_a{cfg['alpha']:g}_ad{cfg['alpha_decay']:g}_ed{cfg['eps_decay']:g}.txt")


def main():
    parser = argparse.ArgumentParser(
        description="RNN dynamic-EDP optimizer hyperparameter sweep")
    parser.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results", "optimizer_sweep", "rnn_optimizer_sweep"))
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the per-cap EQUAL_SPLIT reference runs")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")

    runs = build_runs(include_baseline=not args.no_baseline)
    done = load_done(csv_path)
    todo = [c for c in runs if run_key(c) not in done]
    print(f"[sweep] app={APP} nodes={NODES} bs={BATCH_SIZE} iters={ITERS} "
          f"eps={GRADIENT_OPT_EPS} N={OPTIMIZER_STEP_INTERVAL}")
    print(f"[sweep] {len(runs)} configs total; {len(done)} already done; "
          f"{len(todo)} to run")

    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
            f.flush()
        for i, cfg in enumerate(todo, 1):
            log_path = os.path.join(out_dir, log_name(cfg))
            print(f"\n[sweep] ({i}/{len(todo)}) {cfg['strategy']} cap={cfg['start_powercap']} "
                  f"alpha={cfg['alpha']} a_decay={cfg['alpha_decay']} eps_decay={cfg['eps_decay']}")
            res = single_app_run(make_params(cfg), log_save_path=log_path)
            row = {
                "strategy": cfg["strategy"], "start_powercap": cfg["start_powercap"],
                "alpha": cfg["alpha"], "alpha_decay": cfg["alpha_decay"],
                "eps_decay": cfg["eps_decay"], "eps": GRADIENT_OPT_EPS,
                "N": OPTIMIZER_STEP_INTERVAL, "iters": ITERS, "nodes": NODES,
                "batch_size": BATCH_SIZE,
            }
            if res is None or res == -1:
                row.update({"time_s": "", "energy_j": "", "edp": "", "status": "FAILED"})
                print(f"[sweep]   FAILED (see {log_path})")
            else:
                edp = res.execution_duration * res.energy_used
                row.update({"time_s": round(res.execution_duration, 3),
                            "energy_j": round(res.energy_used, 3),
                            "edp": round(edp, 3), "status": "ok"})
                print(f"[sweep]   time={res.execution_duration:.1f}s "
                      f"energy={res.energy_used:.1f}J EDP={edp:.1f}")
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    summarize(csv_path)
    print(f"\n[sweep] results -> {csv_path}")


def summarize(csv_path: str) -> None:
    """Per start cap: baseline EDP, then every config sorted best-EDP-first with
    its EDP as a percent of that cap's baseline (<100% = beat the fixed cap)."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            try:
                r["start_powercap"] = float(r["start_powercap"])
                r["edp"] = float(r["edp"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    if not rows:
        print("\n[sweep] no successful runs to summarize yet.")
        return

    print("\n" + "=" * 80)
    print("RNN OPTIMIZER SWEEP — end-to-end EDP = time * energy (lower is better)")
    print("=" * 80)
    caps = sorted({r["start_powercap"] for r in rows})
    for cap in caps:
        sub = [r for r in rows if r["start_powercap"] == cap]
        base = next((r for r in sub if r["strategy"] == BASELINE), None)
        base_edp = base["edp"] if base else None
        print(f"\n--- start cap = {cap:g}"
              + (f"  (EQUAL_SPLIT baseline EDP = {base_edp:.1f})" if base_edp else "") + " ---")
        print(f"  {'strategy':<20}{'alpha':>7}{'a_dec':>7}{'e_dec':>7}"
              f"{'EDP':>12}{'vs base':>10}")
        for r in sorted(sub, key=lambda x: x["edp"]):
            vs = f"{100.0 * r['edp'] / base_edp:.1f}%" if base_edp else "-"
            tag = "  <- baseline" if r["strategy"] == BASELINE else ""
            print(f"  {r['strategy']:<20}{float(r['alpha']):>7g}{float(r['alpha_decay']):>7g}"
                  f"{float(r['eps_decay']):>7g}{r['edp']:>12.1f}{vs:>10}{tag}")


if __name__ == "__main__":
    main()
