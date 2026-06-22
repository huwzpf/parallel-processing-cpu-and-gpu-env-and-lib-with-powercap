"""
Single-device perturbation SNR collector.

This is the *collector* half of the per-device EDP-sensitivity experiment (the
analysis half lives elsewhere). It measures the quantity a dynamic EDP optimizer
actually relies on: how the system-wide EDP responds when ONE device's power cap
is perturbed by +/-eps while every other device is held at the baseline cap.

For each base power cap it:
  1. runs an equal-split baseline (all devices at base_cap), then
  2. for every device, every eps and both signs, runs the app with that single
     device bumped to base_cap +/- eps (via the new device_powercaps override).

All runs use strategy EQUAL_SPLIT_EDP_MONITOR, which holds caps fixed and logs a
per-window '[OPTIMIZER STEP] ... edp=... cap=...' line every sync window. We run
once at the highest N (optimizer_step_interval = N_max) and persist every
per-window record; lower-N behaviour is reconstructed offline by re-binning the
per-window series into blocks of N (this is exact: the library averages the same
non-overlapping windows). So N is NOT a run axis here.

eps and sign ARE physical caps, so they are real runs. Both signs are collected
so the analyzer can form one-sided (base vs +eps / base vs -eps) and central
(+eps vs -eps) signals, keeping the data strategy-agnostic.

Crash safety: the per-window CSV is appended and flushed after every run, and the
script resumes by skipping (base_cap, device, eps, sign, rep) cells already
present in the CSV. Nothing already collected is lost or recomputed.

Usage:
    python snr_perturbation_test.py [--app {cnn,rnn}] [--nodes N] [--out-dir DIR]
        [--base-caps ...] [--epsilons ...] [--n-max 40] [--n-runs 3]
        [--batch-size 50] [--devices i j ...]

Columns in perturbation_samples.csv (one row per sync window):
    app, nodes, batch_size, base_cap, perturbed_device, perturbed_type,
    perturbed_role, eps, sign, rep, n_max, iters,
    window, interval, phase, energy_J, period_s, batches, edp, avg_power,
    energy_devices, cap_vector, perturbed_cap_realized
"""

import argparse
import csv
import os
import re
from pathlib import Path

from experiments import single_app_run
from models import RunParameters


# ---------------------------------------------------------------------------
# Run length: identical formula to snr_test.compute_iters. iters is normalized
# by the app's default batch size so it stays in a reasonable range across batch
# sizes. CNN and RNN default to 20; the remaining apps default to 240000.
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZES = {"cnn": 20, "rnn": 20}
DEFAULT_BATCH_SIZE_FALLBACK = 240000


def compute_iters(n: int, number_od_nodes: int, batch_size: int, app_name: str) -> int:
    default_bs = DEFAULT_BATCH_SIZES.get(app_name, DEFAULT_BATCH_SIZE_FALLBACK)
    return int(round(n * number_od_nodes * batch_size / default_bs))


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
_OPT_STEP_RE = re.compile(
    r"\[OPTIMIZER STEP\]\s+"
    r"window=(?P<window>\d+)\s+"
    r"interval=(?P<interval>\d+)\s+"
    r"phase=(?P<phase>-?\d+)\s+"
    r"energy_J=(?P<energy_J>[-\d.eE+]+)\s+"
    r"period_s=(?P<period_s>[-\d.eE+]+)\s+"
    r"batches=(?P<batches>\d+)\s+"
    r"edp=(?P<edp>[-\d.eE+]+)\s+"
    r"avg_power=(?P<avg_power>[-\d.eE+]+)W\s+"
    r"energy_devices=(?P<energy_devices>\d+)\s+"
    r"cap=(?P<cap>[-\d.,eE+]+)"
)

_TOPOLOGY_RE = re.compile(
    r"\[DEVICE TOPOLOGY\]\s+index=(?P<index>\d+)\s+type=(?P<type>\w+)\s+"
    r"rank=(?P<rank>\d+)\s+role=(?P<role>\w+)"
)


def parse_topology(log_file: str) -> dict[int, dict]:
    """Parse '[DEVICE TOPOLOGY]' lines -> {index: {type, rank, role}}."""
    topo: dict[int, dict] = {}
    try:
        with open(log_file, "r", errors="replace") as f:
            for line in f:
                m = _TOPOLOGY_RE.search(line)
                if m:
                    idx = int(m.group("index"))
                    topo[idx] = {
                        "type": m.group("type"),
                        "rank": int(m.group("rank")),
                        "role": m.group("role"),
                    }
    except FileNotFoundError:
        pass
    return topo


def parse_optimizer_steps(log_file: str) -> list[dict]:
    """Parse every per-window '[OPTIMIZER STEP]' line into a list of dicts."""
    rows: list[dict] = []
    try:
        with open(log_file, "r", errors="replace") as f:
            for line in f:
                m = _OPT_STEP_RE.search(line)
                if not m:
                    continue
                rows.append({
                    "window": int(m.group("window")),
                    "interval": int(m.group("interval")),
                    "phase": int(m.group("phase")),
                    "energy_J": float(m.group("energy_J")),
                    "period_s": float(m.group("period_s")),
                    "batches": int(m.group("batches")),
                    "edp": float(m.group("edp")),
                    "avg_power": float(m.group("avg_power")),
                    "energy_devices": int(m.group("energy_devices")),
                    "cap_vector": m.group("cap"),
                })
    except FileNotFoundError:
        print(f"[perturb] Log file not found: {log_file}")
    return rows


# ---------------------------------------------------------------------------
# CSV (append-after-every-run, resumable)
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "app", "nodes", "batch_size", "base_cap", "perturbed_device",
    "perturbed_type", "perturbed_role", "eps", "sign", "rep", "n_max", "iters",
    "window", "interval", "phase", "energy_J", "period_s", "batches", "edp",
    "avg_power", "energy_devices", "cap_vector", "perturbed_cap_realized",
]


def _cell_key(base_cap, device, eps, sign, rep) -> tuple:
    """Identity of one run, used to skip already-collected cells on resume."""
    return (round(float(base_cap), 4), int(device), round(float(eps), 4),
            int(sign), int(rep))


def load_completed_cells(csv_path: str) -> set:
    """Return the set of (base_cap, device, eps, sign, rep) cells already in CSV."""
    completed: set = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(_cell_key(
                    row["base_cap"], row["perturbed_device"],
                    row["eps"], row["sign"], row["rep"]))
            except (KeyError, ValueError):
                continue
    return completed


def open_csv_appending(csv_path: str):
    """Open CSV for append, writing the header only if the file is new/empty."""
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    if new_file:
        writer.writeheader()
        f.flush()
    return f, writer


def _realized_perturbed_cap(cap_vector: str, device: int) -> float | None:
    """The normalized cap actually applied to `device`, read from the cap vector
    (captures library-side clamping at extreme base_cap +/- eps)."""
    if device < 0:
        return None
    parts = cap_vector.split(",")
    if 0 <= device < len(parts):
        try:
            return float(parts[device])
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------
def _make_params(app_name, batch_size, nodes, n_max, iters,
                 device_powercaps, cpu_enabled=True) -> RunParameters:
    return RunParameters(
        app_name=app_name,
        cpu_enabled=cpu_enabled,
        number_of_streams=2,
        number_od_nodes=nodes,
        batch_size=batch_size,
        strategy="EQUAL_SPLIT_EDP_MONITOR",
        cpu_power_scaling=0.0,
        initial_cpu_batch_size_scaling=0,
        start_powercap=device_powercaps[0] if device_powercaps else 0.5,
        iters=iters,
        optimizer_step_interval=n_max,
        device_powercaps=device_powercaps,
    )


def run_and_collect(params, log_path, meta, writer, csv_file) -> int:
    """Run one config, parse its per-window steps, append rows, flush. Returns
    the number of per-window rows written (0 on a failed/empty run)."""
    single_app_run(params, log_save_path=log_path)
    steps = parse_optimizer_steps(log_path)
    if not steps:
        print(f"[perturb]   no [OPTIMIZER STEP] rows parsed from {log_path}")
        return 0
    for s in steps:
        row = dict(meta)
        row.update(s)
        row["perturbed_cap_realized"] = _realized_perturbed_cap(
            s["cap_vector"], meta["perturbed_device"])
        writer.writerow(row)
    csv_file.flush()
    os.fsync(csv_file.fileno())
    return len(steps)


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------
def discover_devices(app_name, batch_size, nodes, n_max, iters, out_dir,
                     cpu_enabled=True):
    """Run one equal-split baseline to learn the device count + topology.

    Returns (num_devices, topology_dict). Falls back to parsing the cap-vector
    length if no [DEVICE TOPOLOGY] lines are present (e.g. old library build).
    """
    log_path = os.path.abspath(os.path.join(out_dir, "discover_topology.txt"))
    print("[perturb] Discovering device count/topology via one baseline run...")
    params = _make_params(app_name, batch_size, nodes, n_max, 1, None,
                          cpu_enabled=cpu_enabled)
    single_app_run(params, log_save_path=log_path)

    topo = parse_topology(log_path)
    if topo:
        num = max(topo) + 1
    else:
        steps = parse_optimizer_steps(log_path)
        if not steps:
            raise RuntimeError(
                "Could not discover devices: no [DEVICE TOPOLOGY] or "
                "[OPTIMIZER STEP] lines in the baseline run. Is the library "
                "rebuilt with the new logging?")
        num = len(steps[0]["cap_vector"].split(","))
        topo = {i: {"type": "UNKNOWN", "rank": -1, "role": "unknown"}
                for i in range(num)}
        print("[perturb]   no topology lines; inferred device count from cap vector")

    # Persist topology so the offline analyzer can classify device indices.
    topo_csv = os.path.join(out_dir, "device_topology.csv")
    with open(topo_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "type", "rank", "role"])
        w.writeheader()
        for idx in sorted(topo):
            w.writerow({"index": idx, **topo[idx]})
    print(f"[perturb] {num} devices discovered; topology -> {topo_csv}")
    for idx in sorted(topo):
        t = topo[idx]
        print(f"           device {idx}: {t['type']} rank={t['rank']} ({t['role']})")
    return num, topo


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def run_perturbation_experiments(
    app_name: str,
    batch_size: int,
    nodes: int,
    base_caps: list[float],
    epsilons: list[float],
    n_max: int,
    n_runs: int,
    out_dir: str,
    devices: list[int] | None = None,
    cpu_enabled: bool = True,
):
    out_dir = os.path.abspath(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    iters = compute_iters(n_max, nodes, batch_size, app_name)
    csv_path = os.path.join(out_dir, "perturbation_samples.csv")

    num_devices, _topo = discover_devices(
        app_name, batch_size, nodes, n_max, iters, out_dir, cpu_enabled=cpu_enabled)
    if devices is None:
        devices = list(range(num_devices))
    else:
        devices = [d for d in devices if 0 <= d < num_devices]

    completed = load_completed_cells(csv_path)
    if completed:
        print(f"[perturb] Resuming: {len(completed)} (cap,device,eps,sign,rep) "
              f"cells already in {csv_path} will be skipped")

    signs = [+1, -1]
    # Baseline cells are tagged perturbed_device=-1, eps=0, sign=0.
    total_cells = (len(base_caps) * n_runs
                   + len(base_caps) * len(devices) * len(epsilons)
                   * len(signs) * n_runs)
    print(f"[perturb] Grid: {len(base_caps)} caps x "
          f"({n_runs} baseline + {len(devices)} dev x {len(epsilons)} eps x "
          f"{len(signs)} signs x {n_runs} reps) = {total_cells} runs; "
          f"n_max={n_max} iters={iters} "
          f"cpu_enabled={cpu_enabled} ({'GPU+CPU' if cpu_enabled else 'GPU-only'})")

    csv_file, writer = open_csv_appending(csv_path)
    done = 0
    try:
        for base_cap in base_caps:
            # --- Baseline: all devices equal at base_cap ---
            for rep in range(n_runs):
                done += 1
                key = _cell_key(base_cap, -1, 0.0, 0, rep)
                if key in completed:
                    continue
                meta = {
                    "app": app_name, "nodes": nodes, "batch_size": batch_size,
                    "base_cap": base_cap, "perturbed_device": -1,
                    "perturbed_type": "baseline", "perturbed_role": "baseline",
                    "eps": 0.0, "sign": 0, "rep": rep,
                    "n_max": n_max, "iters": iters,
                }
                caps = [base_cap] * num_devices
                params = _make_params(app_name, batch_size, nodes, n_max, iters,
                                      caps, cpu_enabled=cpu_enabled)
                log_path = os.path.abspath(os.path.join(
                    out_dir, f"log_baseline_cap{base_cap:.2f}_rep{rep}.txt"))
                print(f"[perturb] ({done}/{total_cells}) BASELINE cap={base_cap:.2f} "
                      f"rep={rep + 1}/{n_runs}")
                run_and_collect(params, log_path, meta, writer, csv_file)

            # --- Single-device perturbations ---
            for device in devices:
                for eps in epsilons:
                    for sign in signs:
                        for rep in range(n_runs):
                            done += 1
                            key = _cell_key(base_cap, device, eps, sign, rep)
                            if key in completed:
                                continue
                            target = base_cap + sign * eps
                            target = min(1.0, max(0.0, target))  # C clamps to device lower bound too
                            caps = [base_cap] * num_devices
                            caps[device] = target
                            t = _topo.get(device, {"type": "UNKNOWN", "role": "unknown"})
                            meta = {
                                "app": app_name, "nodes": nodes,
                                "batch_size": batch_size, "base_cap": base_cap,
                                "perturbed_device": device,
                                "perturbed_type": t["type"],
                                "perturbed_role": t["role"],
                                "eps": eps, "sign": sign, "rep": rep,
                                "n_max": n_max, "iters": iters,
                            }
                            params = _make_params(
                                app_name, batch_size, nodes, n_max, iters, caps,
                                cpu_enabled=cpu_enabled)
                            log_path = os.path.abspath(os.path.join(
                                out_dir,
                                f"log_dev{device}_cap{base_cap:.2f}_eps{eps:.2f}"
                                f"_sgn{'p' if sign > 0 else 'm'}_rep{rep}.txt"))
                            print(f"[perturb] ({done}/{total_cells}) DEV={device} "
                                  f"({t['type']}/{t['role']}) cap={base_cap:.2f} "
                                  f"eps={eps:.2f} sign={'+' if sign > 0 else '-'} "
                                  f"target={target:.2f} rep={rep + 1}/{n_runs}")
                            run_and_collect(params, log_path, meta, writer, csv_file)
    finally:
        csv_file.close()
    print(f"[perturb] Done. Per-window samples -> {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-device perturbation SNR collector "
                    "(EQUAL_SPLIT_EDP_MONITOR + device_powercaps override)")
    # Apps that honour --iters (needed so run length scales with N). vecadd and
    # patternsearch are intentionally excluded (they don't support iters).
    parser.add_argument(
        "--app",
        choices=["cnn", "rnn", "collatz", "montecarlo", "twinprime", "vecmaxdiv"],
        default="cnn")
    parser.add_argument("--nodes", type=int, default=4,
                        help="Number of nodes in the cluster run")
    parser.add_argument("--out-dir", default="snr_perturbation_results")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--base-caps", nargs="+", type=float,
                        default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--epsilons", nargs="+", type=float,
                        default=[0.05, 0.1, 0.2])
    parser.add_argument("--n-max", type=int, default=40,
                        help="Highest N (optimizer_step_interval); run length is "
                             "iters=compute_iters(n_max,...). Lower N is derived "
                             "offline from the per-window logs.")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Repetitions per (cap, device, eps, sign) cell")
    parser.add_argument("--devices", nargs="+", type=int, default=None,
                        help="Device indices to perturb (default: all discovered)")
    parser.add_argument("--gpu-only", action="store_true",
                        help="Disable CPU devices (perturb GPUs only). Works with "
                             "a single master node holding many GPUs (--nodes 1).")
    args = parser.parse_args()

    run_perturbation_experiments(
        app_name=args.app,
        batch_size=args.batch_size,
        nodes=args.nodes,
        base_caps=args.base_caps,
        epsilons=args.epsilons,
        n_max=args.n_max,
        n_runs=args.n_runs,
        out_dir=args.out_dir,
        devices=args.devices,
        cpu_enabled=not args.gpu_only,
    )
