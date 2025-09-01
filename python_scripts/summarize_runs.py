#!/usr/bin/env python3
"""
summarize_runs.py
-----------------
Parse the experiment‑result JSON and report the average execution
time (seconds) and average energy used (joules) for each configuration.
Usage:
    python summarize_runs.py results.json
"""

import json
import sys
from statistics import mean


def summarize(data: dict) -> None:
    """Print one summary row per configuration."""
    header = f"{'CPU':<6} {'PowerCap':<9} {'Avg Exec [s]':>12} {'Avg Energy [J]':>15}"
    print(header)
    print("-" * len(header))

    for entry in data.get("experiment_result", []):
        params = entry["parameters"]
        runs   = entry["runs"]

        avg_time   = mean(r["execution_duration"] for r in runs)
        avg_energy = mean(r["energy_used"]        for r in runs)

        cpu_state  = "ON" if params["cpu_enabled"] else "OFF"
        powercap   = params["powercap"]

        print(f"{cpu_state:<6} {powercap:<9} {avg_time:>12.3f} {avg_energy:>15.2f}")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python summarize_runs.py <input_json_file>")

    input_path = sys.argv[1]
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summarize(data)


if __name__ == "__main__":
    main()