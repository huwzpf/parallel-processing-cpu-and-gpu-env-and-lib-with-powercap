#!/usr/bin/env python3
"""Compute the coefficient of variation (CV = std/mean) for the numeric
fields (energy, period, edp, power) parsed from OPTIMIZER STEP log lines."""
import re
import sys
import statistics

FIELDS = ["energy_J", "period_s", "edp", "avg_power"]
FIELD_RES = {f: re.compile(rf"{f}=([0-9.eE+-]+)") for f in FIELDS}
SMOOTHED_RE = re.compile(r"\[EDP\]\s*edp=([0-9.eE+-]+)")


def extract(path):
    data = {f: [] for f in FIELDS}
    data["smoothed_edp"] = []
    with open(path) as f:
        for line in f:
            m = SMOOTHED_RE.search(line)
            if m:
                data["smoothed_edp"].append(float(m.group(1)))
            if "OPTIMIZER STEP" not in line:
                continue
            for field, rgx in FIELD_RES.items():
                m = rgx.search(line)
                if m:
                    data[field].append(float(m.group(1)))
    return data


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "old_log_montecarlo_bs240000_cap0.40_N5_rep0.txt"
    data = extract(path)
    counts = {len(v) for v in data.values()}
    if counts == {0}:
        print(f"No OPTIMIZER STEP values found in {path}")
        return
    print(f"file:  {path}")
    print(f"{'field':<14}{'count':>7}{'mean':>16}{'std':>16}{'CV':>12}")
    for field in FIELDS + ["smoothed_edp"]:
        vals = data[field]
        if not vals:
            print(f"{field:<14}{'0':>7}{'(missing)':>16}")
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        cv = std / mean if mean else float("nan")
        print(f"{field:<14}{len(vals):>7}{mean:>16.6f}{std:>16.6f}"
              f"{cv * 100:>11.2f}%")


if __name__ == "__main__":
    main()
