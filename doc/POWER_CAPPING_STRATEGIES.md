# Power Capping Strategies

This document describes the power capping strategies currently available in
CUDAMPILIB. Strategies are selected in `powercap.conf` with the `strategy` key.
If `powercap.conf` is missing, the library uses `DISABLED`.

Power cap values that are described as "range fractions" are normalized to the
device's reported power-cap range:

```text
physical_cap = range_min + fraction * (range_max - range_min)
```

For example, if a device supports 100 W to 250 W and a fraction is `0.25`, the
selected cap is `137.5 W`.

## Configuration Keys

The parser accepts these keys in `powercap.conf`:

| Key | Used by | Meaning | Default |
| --- | --- | --- | --- |
| `strategy` | all strategies | Strategy name. | `DISABLED` |
| `global_powercap` | greedy strategies | Total power budget in watts. `0` disables the global budget. | `0.0` |
| `cpu_min_powercap` | continuous greedy strategies | CPU minimum cap as a range fraction. | `0.0` |
| `gpu_min_powercap` | continuous greedy strategies | GPU minimum cap as a range fraction. | `0.0` |
| `start_powercap` | `EQUAL_SPLIT`, EDP strategies | Initial cap as a range fraction. | `0.5` |
| `start_alpha` | EDP strategies | Initial optimizer learning rate. | `2.0` |
| `alpha_decay` | EDP strategies | Learning-rate decay applied after optimizer updates. | `0.99` |
| `epsilon_decay` | EDP strategies | Perturbation-step decay applied after optimizer updates. | `0.99` |
| `gradient_opt_eps` | EDP strategies | Finite-difference, SPSA, or CMA-ES initial perturbation size in normalized cap space. | `5.0` |
| `edp_optimization_steps` | EDP strategies | Maximum number of completed optimization updates. `0` means unlimited. | `0` |
| `cpu_time_window_us` | all enabled strategies | CPU RAPL power cap time window in microseconds. | `1000000` |

Unknown keys are ignored.

## Available Strategies

### `DISABLED`

Power capping is disabled. The library does not change CPU or GPU power caps and
does not run any dynamic selection or optimization logic.

Use this when comparing against the platform defaults or when the runtime should
not require power-cap permissions.

Example:

```ini
strategy=DISABLED
```

### `BINARY_GREEDY`

Selects a subset of devices that fits under `global_powercap`. Each selected
device remains enabled, and each non-selected device is disabled for work
scheduling. The selection criterion is based on measured device performance per
power:

```text
computeDevPerformance(device_time) / current_power
```

The manager waits until every device has reported power measurements. It then
repeatedly selects the best remaining device whose current measured power fits
inside the remaining budget.

After the initial selection, the manager checks the total measured power of
enabled devices. If it rises above `global_powercap`, selection is recomputed.

Required configuration:

```ini
strategy=BINARY_GREEDY
global_powercap=450
```

### `EQUAL_SHARE_BINARY_GREEDY`

Uses the same measured-power subset selection as `BINARY_GREEDY`, but attempts to
alternate between GPU and CPU devices while selecting candidates. The goal is to
keep both CPU and GPU devices represented when the power budget allows it.

The strategy logs an error if it cannot select at least one CPU and one GPU under
the configured budget.

Required configuration:

```ini
strategy=EQUAL_SHARE_BINARY_GREEDY
global_powercap=450
```

### `CONTINUOUS_GREEDY`

Selects devices under `global_powercap` using configured minimum caps rather than
only current measured power. Selected devices are assigned at least their
minimum power cap:

```ini
cpu_min_powercap=0.10
gpu_min_powercap=0.10
```

The manager repeatedly selects the best candidate according to:

```text
computeDevPerformance(device_time) / current_power_cap
```

where `current_power_cap` is the cap assigned during the strategy. Remaining
budget is distributed proportionally across selected devices according to each
device's free capacity up to its maximum cap.

At startup, if the total configured minimum caps already meet or exceed
`global_powercap`, all devices are capped to their minimum. If the global budget
is above all devices' maximum capacity, all devices are capped to their maximum.

Required configuration:

```ini
strategy=CONTINUOUS_GREEDY
global_powercap=450
cpu_min_powercap=0.10
gpu_min_powercap=0.10
```

### `EQUAL_SHARE_CONTINUOUS_GREEDY`

Uses the continuous cap allocation behavior from `CONTINUOUS_GREEDY`, but
attempts to alternate GPU and CPU selection. Like the binary equal-share variant,
this is intended for runs where both CPU and GPU devices should participate when
the budget allows it.

Selected devices start from their configured minimum caps, and remaining power is
distributed proportionally by free power-cap capacity.

Required configuration:

```ini
strategy=EQUAL_SHARE_CONTINUOUS_GREEDY
global_powercap=450
cpu_min_powercap=0.10
gpu_min_powercap=0.10
```

### `EQUAL_SPLIT`

Applies the same normalized cap fraction to every device. This strategy does not
use `global_powercap`; the loader clears the global limit for this strategy.

The `start_powercap` value is interpreted independently for each device's range.
For example, `start_powercap=0.5` sets every device to the midpoint between its
own minimum and maximum supported power cap.

Required configuration:

```ini
strategy=EQUAL_SPLIT
start_powercap=0.5
```

### `EDP_GRADIENT_SIMPLE`

Dynamically optimizes device power caps to reduce the measured energy-delay
product (EDP). Each device starts at `start_powercap`, interpreted as a range
fraction.

This strategy uses central finite differences. For each optimization cycle it:

1. Captures the current normalized cap vector as the base point.
2. Probes `+epsilon` and `-epsilon` for each device dimension.
3. Estimates a log-objective gradient from the measured EDP values.
4. Applies a gradient descent step in normalized cap space.
5. Decays `alpha` and `epsilon`.

The optimizer only advances when all devices have completed their last batch and
current power measurements are available.

Example:

```ini
strategy=EDP_GRADIENT_SIMPLE
start_powercap=0.5
start_alpha=0.1
alpha_decay=0.99
epsilon_decay=0.99
gradient_opt_eps=0.05
edp_optimization_steps=100
```

### `EDP_GRADIENT_SPSA`

Dynamically optimizes EDP using Simultaneous Perturbation Stochastic
Approximation (SPSA). All device caps are perturbed together using a random
`+1/-1` vector, which makes each update cheaper than the simple finite-difference
strategy for large device counts.

Each update cycle has three phases:

1. Apply `base + epsilon * delta` to all devices.
2. Apply `base - epsilon * delta` to all devices.
3. Estimate the log-objective SPSA gradient and apply a descent step.

Like `EDP_GRADIENT_SIMPLE`, updates occur only after all devices complete their
last batch and report power. `alpha` and `epsilon` decay after completed descent
updates.

Example:

```ini
strategy=EDP_GRADIENT_SPSA
start_powercap=0.5
start_alpha=0.1
alpha_decay=0.99
epsilon_decay=0.99
gradient_opt_eps=0.05
edp_optimization_steps=100
```

### `EDP_GRADIENT_CMAES`

Dynamically optimizes EDP using CMA-ES. The optimizer works in normalized
`[0, 1]` power-cap space with boundary transformation and maps each candidate
back to physical CPU/GPU power caps.

Each CMA-ES generation:

1. Samples a population of candidate cap vectors.
2. Applies each candidate to the devices and records the measured EDP as fitness.
3. Updates the CMA-ES distribution from the population fitness values.
4. Applies the best-known candidate between generations.

The initial candidate center comes from `start_powercap`. `gradient_opt_eps` is
used as the initial normalized standard deviation, clamped internally to the
range accepted by the implementation.

This strategy requires the CMA-ES sources linked by the build script and uses
`cmaes_initials.par` for CMA-ES initialization parameters.

Example:

```ini
strategy=EDP_GRADIENT_CMAES
start_powercap=0.5
gradient_opt_eps=0.05
edp_optimization_steps=20
```

## EDP Measurement Notes

The EDP strategies measure an optimization period after all devices finish their
last assigned batch. The runtime combines measured device power, elapsed time,
and estimated processed data points to compute the objective used by the
optimizers. Power caps are clamped to each device's reported minimum and maximum
before they are applied.

Set `edp_optimization_steps` to a positive value for bounded experiments. When
the limit is reached, the runtime stops changing caps and records optimization
finish statistics while the application continues running.

## Operational Notes

- Power-cap ranges are discovered from NVML for GPUs and Linux RAPL sysfs for
  CPUs.
- CPU `cpu_time_window_us` is broadcast to MPI workers so all nodes use the same
  RAPL time window.
- On termination, enabled power capping strategies reset local CPU and GPU caps
  to their default values.
- Global-budget strategies require `global_powercap > 0`. Without a positive
  global cap, their selection functions return without applying a budget.
- CPU and GPU power-cap writes usually require permissions to write RAPL sysfs
  files and to set NVML GPU power limits.
