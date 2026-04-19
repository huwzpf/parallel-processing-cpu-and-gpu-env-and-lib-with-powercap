# CUDAMPILIB

CUDAMPILIB is a CPU+GPU cluster runtime for running OpenMP/CUDA-style programs
across multiple MPI nodes. Rank 0 runs the application manager, while remote MPI
processes act as GPU workers. The runtime supports CUDA streams, CPU execution,
dynamic batch sizing, and power-aware execution.

## Requirements

- Linux
- MPI compiler/runtime (`mpicc`, `mpirun`)
- OpenMP
- CUDA and NVML
- cuDNN for the CNN example (Optional)
- Intel MKL for the RNN example (Optional)
- `c-cmaes` sources for the CMA-ES power-capping optimizer (Optional)

The build script currently expects CUDA under `/usr/local/cuda-13.0` and looks
for `c-cmaes/src` either next to `cudampilib` or in the repository root.

## Build

Build from the library directory:

```bash
cd cudampilib
./compile
```

Compiled application and worker binaries are written to `cudampilib/build`.

## Run

The main launcher is `cudampilib/run_scripts/run-app`:

```bash
cd cudampilib
./run_scripts/run-app <app> <mode> <nodes-or-node-list> [app options]
```

Modes:

- `B`: use the first N machines from `hostfile`
- `C`: use selected hostfile line numbers, followed by `A`
- `H`: print launcher help

Example using the first 8 machines from `hostfile`:

```bash
./run_scripts/run-app cnn B 8 --cpu-enabled=1 --number-of-streams=2 --batch-size=100 --cpu-power-scaling=0 --initial-cpu-batch-size-scaling=0
```

Example using selected machines:

```bash
./run_scripts/run-app vecadd C 1 2 4 A --cpu-enabled=1 --number-of-streams=2 --batch-size=100
```

Available application names include `vecadd`, `vecmaxdiv`, `collatz`,
`patternsearch`, `twinprime`, `rnn`, `cnn`, and `montecarlo`.

Before running on a cluster, update `cudampilib/hostfile` for your machines.

## Power Capping

Power capping is configured with `cudampilib/powercap.conf`. The `strategy` key
selects the policy, and `global_powercap` sets a total wattage budget for the
global-budget strategies.

See [doc/POWER_CAPPING_STRATEGIES.md](doc/POWER_CAPPING_STRATEGIES.md)
for the full strategy documentation and configuration examples.

Power-cap writes usually require permission to set NVML GPU limits and write CPU
RAPL sysfs files.

## Repository Layout

- `cudampilib/`: runtime, build script, launch scripts, and examples
- `python_scripts/`: experiment and analysis utilities
- `doc/`: project notes and power-capping strategy documentation

## License

MIT License. See `cudampilib/LICENSE`.
