#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_experiment(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def compute_edp(run):
    # EDP = energy_used * execution_duration
    return float(run["energy_used"]) * float(run["execution_duration"])


def summarize_group(entries, exclude_keys=None):
    """
    entries: list of dicts with keys: parameters (dict), runs (list)
    Returns list of dicts with keys:
      params_changed: dict of only params that vary within this strategy
      edp_avg: float average EDP across runs
      edp_runs: list of per-run EDPs
      key_tuple: tuple used for stable sorting/reporting
    """
    # Collect all parameter keys and values per entry
    exclude_keys = set(exclude_keys or [])
    all_params = [e["parameters"] for e in entries]

    # Determine which keys change across this strategy
    varying_keys = set()
    if all_params:
        keys = all_params[0].keys()
        for k in keys:
            vals = tuple(p.get(k) for p in all_params)
            if any(v != vals[0] for v in vals[1:]):
                varying_keys.add(k)

    # We do not need to print the strategy itself in the changed set
    varying_keys.difference_update(exclude_keys)

    # Build summaries
    summaries = []
    for e in entries:
        params = e["parameters"]
        # Only include varying keys for this strategy
        changed = {k: params.get(k) for k in sorted(varying_keys)}
        runs = e.get("runs", [])
        edp_runs = [compute_edp(r) for r in runs]
        edp_avg = sum(edp_runs) / len(edp_runs) if edp_runs else float("inf")
        energy_runs = [float(r["energy_used"]) for r in runs if "energy_used" in r]
        energy_avg = sum(energy_runs) / len(energy_runs) if energy_runs else None
        time_runs = [float(r["execution_duration"]) for r in runs if "execution_duration" in r]
        time_avg = sum(time_runs) / len(time_runs) if time_runs else None
        # Key tuple for sorting: values of changed keys in order
        key_tuple = tuple(changed.get(k) for k in sorted(varying_keys))
        summaries.append({
            "params_changed": changed,
            "edp_avg": edp_avg,
            "edp_runs": edp_runs,
            "energy_avg": energy_avg,
            "energy_runs": energy_runs,
            "time_avg": time_avg,
            "time_runs": time_runs,
            "key_tuple": key_tuple,
        })
    return summaries


def format_params(params: dict) -> str:
    items = [f"{k}={params[k]}" for k in params]
    return ", ".join(items) if items else "(no varying params)"


def format_metric(label, runs, average, unit=None):
    if not runs:
        return f"{label}=N/A"

    display_label = f"{label}_avg" if len(runs) > 1 else label
    value = average if average is not None else sum(runs) / len(runs)
    suffix = f" {unit}" if unit else ""
    if len(runs) > 1:
        return f"{display_label}={value:.3f}{suffix} (n={len(runs)})"
    return f"{display_label}={value:.3f}{suffix}"


def main():
    parser = argparse.ArgumentParser(description="Parse dynamic powercap experiment results and report EDPs")
    parser.add_argument(
        "--file",
        default="../cudampilib/dynamic_cnn_powercap_8_nodes.json",
        help="Path to experiment results JSON (default: cudampilib/dynamic_cnn_powercap_8_nodes.json)",
    )
    args = parser.parse_args()

    path = Path(args.file)
    data = load_experiment(path)

    entries = data.get("experiment_result", [])
    if not entries:
        print("No entries found in experiment_result.")
        return

    # Group by strategy
    by_strategy = defaultdict(list)
    for e in entries:
        strat = e.get("parameters", {}).get("strategy", "<unknown>")
        by_strategy[strat].append(e)

    # Report per strategy
    for strategy in sorted(by_strategy.keys()):
        print(f"Strategy: {strategy}")
        # Group within strategy by start_powercap
        by_sp = defaultdict(list)
        for e in by_strategy[strategy]:
            sp = e.get("parameters", {}).get("start_powercap", None)
            by_sp[sp].append(e)

        for sp in sorted(by_sp.keys(), key=lambda x: (float(x) if isinstance(x, (int, float)) else x)):
            print(f"  Start powercap: {sp}")
            summaries = summarize_group(by_sp[sp], exclude_keys=["strategy", "start_powercap"]) 
            summaries.sort(key=lambda s: s["key_tuple"])  # stable order

            for s in summaries:
                params_str = format_params(s["params_changed"])
                if len(s["edp_runs"]) <= 1:
                    edp_str = f"EDP={s['edp_avg']:.3f}"
                else:
                    edp_str = f"EDP_avg={s['edp_avg']:.3f} (n={len(s['edp_runs'])})"
                energy_str = format_metric("Energy", s["energy_runs"], s["energy_avg"], unit="J")
                time_str = format_metric("Time", s["time_runs"], s["time_avg"], unit="s")
                print(f"    - {params_str} | {edp_str} | {energy_str} | {time_str}")

            best = min(summaries, key=lambda s: s["edp_avg"]) if summaries else None
            if best:
                best_params_str = format_params(best["params_changed"])
                print(f"    -> Lowest EDP: {best_params_str} | EDP={best['edp_avg']:.3f}")
        print()


if __name__ == "__main__":
    main()
