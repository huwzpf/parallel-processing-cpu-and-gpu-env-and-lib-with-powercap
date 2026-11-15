"""
Run-length sizing: how many dataset iterations (--iters) a cudampilib app needs to
produce a wanted number of power-capping sync windows ([OPTIMIZER STEP] lines).

Per-app constants are hard-coded in APP_DATA_SIZE and APP_SYNC_PERIOD below, copied
from the app sources. Update them here when the sources change:

  data size    <APP>_VECTORSIZE / <APP>_PROBLEM_SIZE in apps/<app>/<app>_defines.h.
               One --iters is one pass over that many data points.
  sync period  how many main-loop rounds a device runs between two
               __cudampi__deviceSynchronize() calls ("privatecounter % K == 0" in
               apps/<app>/app-streams-<app>.c; 1 if the app synchronizes every round).

A sync window closes once every device has synchronized again. Per round a device
takes one batch per stream, so one window covers roughly

    batch_size * streams * sync_period * devices        data points

and therefore

    windows ~= iters * data_size / (batch_size * streams * sync_period * devices)

`devices` is the number of devices sharing one pass over the data. In most apps that
is all of them. Apps in PIPELINE_APPS run two stages instead: every data point goes
through a CPU device and then a GPU device, so each stage makes its own pass and the
stage with more devices (more data per round) sets the window length.

This is an estimate: devices do not run at the same speed. Against measured runs
(4 nodes, 2 streams) the real window count was 0.6x - 1.3x of it (rnn 0.87-1.3,
montecarlo GPU-only 0.9, cnn 0.6-0.97), so expect the requested count give or take
that much. Pass margin > 1 to iters_for_windows() when the count is a hard minimum.

Usage as a script:
    python run_length.py --app cnn --batch-size 50 --nodes 4 --windows 600
"""

import argparse
import math

# Data points in one dataset iteration (<APP>_VECTORSIZE / <APP>_PROBLEM_SIZE).
APP_DATA_SIZE = {
    "cnn": 4_000,
    "rnn": 4_000,
    "montecarlo": 960_000_000,
    "vecmaxdiv": 960_000_000,
    "collatz": 960_000_000,
    "twinprime": 960_000_000,
    "vecadd": 960_000_000,
    "patternsearch": 960_000_000,
}

# Main-loop rounds between two deviceSynchronize() calls.
APP_SYNC_PERIOD = {
    "cnn": 1,
    "rnn": 1,
    "montecarlo": 20,
    "vecmaxdiv": 20,
    "collatz": 8,
    "twinprime": 8,
    "vecadd": 2,
    "patternsearch": 2,
}

# Apps where CPU devices pre-process the data and GPU devices then consume it
# (app-streams-cnn.c: globalcounter1 for the CPU stage, globalcounter2 for the GPU one).
PIPELINE_APPS = {"cnn"}

# Multiplier on the computed iters. 1.0 aims at the requested window count; about 1.2
# makes even the slowest observed case (0.87x of the estimate) reach it.
DEFAULT_MARGIN = 1.0


def app_data_size(app_name: str) -> int:
    """Data points in one dataset iteration of the app."""
    return APP_DATA_SIZE[app_name]


def app_sync_period(app_name: str) -> int:
    """Main-loop rounds between two deviceSynchronize() calls of the app."""
    return APP_SYNC_PERIOD[app_name]


def device_count(number_of_nodes: int, cpu_enabled: bool) -> int:
    """Devices taking batches: one GPU per node, plus one CPU per worker node
    (the master's CPU runs the manager and is not a compute device)."""
    return number_of_nodes + (number_of_nodes - 1 if cpu_enabled else 0)


def pacing_device_count(app_name: str, number_of_nodes: int, cpu_enabled: bool) -> int:
    """Devices that share one pass over the data and so set the window length: all
    devices, or for a pipeline app the larger of its GPU and CPU stages."""
    if app_name in PIPELINE_APPS and cpu_enabled:
        gpus = device_count(number_of_nodes, cpu_enabled=False)
        return max(gpus, device_count(number_of_nodes, cpu_enabled=True) - gpus)
    return device_count(number_of_nodes, cpu_enabled)


def windows_per_iter(app_name: str, batch_size: int, number_of_nodes: int,
                     number_of_streams: int = 2, cpu_enabled: bool = True) -> float:
    """Estimated sync windows produced by one dataset iteration."""
    per_window = (batch_size * number_of_streams * app_sync_period(app_name)
                  * pacing_device_count(app_name, number_of_nodes, cpu_enabled))
    return app_data_size(app_name) / per_window


def iters_for_windows(app_name: str, batch_size: int, number_of_nodes: int, windows: int,
                      number_of_streams: int = 2, cpu_enabled: bool = True,
                      margin: float = DEFAULT_MARGIN) -> int:
    """Dataset iterations (--iters) expected to give about `windows` sync windows."""
    wpi = windows_per_iter(app_name, batch_size, number_of_nodes, number_of_streams, cpu_enabled)
    return max(1, math.ceil(margin * windows / wpi))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iters needed for a wanted number of sync windows")
    parser.add_argument("--app", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--windows", type=int, required=True)
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    a = parser.parse_args()
    wpi = windows_per_iter(a.app, a.batch_size, a.nodes, a.streams, not a.gpu_only)
    iters = iters_for_windows(a.app, a.batch_size, a.nodes, a.windows, a.streams, not a.gpu_only, a.margin)
    print(f"{a.app}: data_size={app_data_size(a.app)} sync_period={app_sync_period(a.app)} "
          f"pacing_devices={pacing_device_count(a.app, a.nodes, not a.gpu_only)} -> {wpi:.2f} windows/iter; "
          f"iters={iters} for ~{a.windows} windows")
