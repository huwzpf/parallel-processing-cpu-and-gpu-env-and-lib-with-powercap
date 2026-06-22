import os
import json
import statistics
import re

from dataclasses import dataclass, asdict, fields, MISSING
from typing import get_args, get_origin


def dataclass_from_dict(klass, d):
    try:
        fieldtypes = {f.name: f.type for f in fields(klass)}
        init_values = {}
        for field_name, field_type in fieldtypes.items():
            if isinstance(d[field_name], list):
                init_values[field_name] = [
                    dataclass_from_dict(field_type.__args__[0], item) if hasattr(field_type, '__args__') else item
                    for item in d[field_name]
                ]
            elif hasattr(field_type, '__dataclass_fields__'):
                init_values[field_name] = dataclass_from_dict(field_type, d[field_name])
            else:
                init_values[field_name] = d[field_name]
        return klass(**init_values)
    except Exception as e:
        print(f"Error while creating dataclass from dict: {e}")
        return d
    

@dataclass
class RunParameters:
    app_name: str
    cpu_enabled: bool
    number_of_streams: int
    number_od_nodes: int
    batch_size: int
    powercap: int | None = None
    cpu_power_scaling: float | None = None
    initial_cpu_batch_size_scaling: int = 0
    strategy: str | None  = None # "CONTINUOUS_GREEDY", "BINARY_GREEDY", "EDP_GRADIENT_SIMPLE", "EDP_GRADIENT_SPSA"
    # For CONTINUOUS_GREEDY: fractions of range; optional with sensible defaults
    cpu_min_powercap: float | None = None
    gpu_min_powercap: float | None = None
    # Optional CPU time window for power capping (microseconds)
    cpu_time_window_us: int | None = None
    # Optional starting power cap fraction (0..1) for gradient/EQUAL_SPLIT strategies
    start_powercap: float | None = None
    # Gradient optimisation parameters (EDP_GRADIENT_* strategies)
    start_alpha: float | None = None
    alpha_decay: float | None = None
    epsilon_decay: float | None = None
    gradient_opt_eps: float | None = None
    # Limit number of dynamic optimisation updates (0 = unlimited)
    edp_optimization_steps: int | None = None
    # Dataset iteration count passed as --iters (None = app compile-time default of 50)
    iters: int | None = None
    # Windows to aggregate per optimizer step (None = 1, i.e. every window)
    optimizer_step_interval: int | None = None
    # Optional per-device starting power caps (fractions of each device's range,
    # 0..1), one per device. Applied for EQUAL_SPLIT / EQUAL_SPLIT_EDP_MONITOR to
    # perturb a single device for SNR diagnostics. None = uniform start_powercap.
    device_powercaps: list[float] | None = None


@dataclass
class SingleRunResult:
    execution_duration: float # in seconds
    energy_used: float # in Watts

    @staticmethod
    def from_output(stdout: str, stderr: str) -> "SingleRunResult":
        return SingleRunResult(
            execution_duration=SingleRunResult.get_execution_duration_from_output(stdout=stdout, stderr=stderr), 
            energy_used=SingleRunResult.get_energy_used_from_output(stdout=stdout, stderr=stderr),
        )

    @staticmethod
    def get_execution_duration_from_output(stdout: str, stderr: str):
        main_time_match = re.search(r'Main elapsed time=([\d.]+)', stderr)
        ret = float(main_time_match.group(1))
        print(f"Execution time: {ret}")
        return ret

    @staticmethod
    def get_energy_used_from_output(stdout: str, stderr: str):
        energy_match = re.search(r'Total energy used ([\d.]+)', stderr)
        ret = float(energy_match.group(1))
        print(f"Energy used: {ret}")
        return ret
    

@dataclass
class MultipleRunResult:
    parameters: RunParameters
    runs: list[SingleRunResult]

    def min(self, single_run_result_property: str):
        return min([getattr(run, single_run_result_property) for run in self.runs])

    def max(self, single_run_result_property: str):
        return max([getattr(run, single_run_result_property) for run in self.runs])

    def average(self, single_run_result_property: str):
        return statistics.mean([getattr(run, single_run_result_property) for run in self.runs])

    def standard_deviation(self, single_run_result_property: str):
        return statistics.stdev([getattr(run, single_run_result_property) for run in self.runs])


@dataclass
class ExperimentResult:
    description: str
    experiment_result: list[MultipleRunResult]

    def to_file(self, file_path: str | os.PathLike):
        with open(file_path, "w") as file:
            file.write(json.dumps(asdict(self), indent=4))

    def default_metrics(self, baseline_powercap: int | float = 0) -> dict[str, float | None]:
        """Return mean execution time, energy, and EDP for runs with the given powercap."""
        metrics = {"execution_duration": None, "energy_used": None, "edp": None}

        baseline_runs = [
            multi for multi in self.experiment_result
            if getattr(multi.parameters, "powercap", None) == baseline_powercap
        ]
        if not baseline_runs:
            print(
                f"[ExperimentResult.default_metrics] No experiment entries with powercap={baseline_powercap}; returning empty metrics"
            )
            return metrics

        value_pairs: list[tuple[float, float]] = []
        for multi in baseline_runs:
            runs = getattr(multi, "runs", None)
            if not runs:
                print(
                    f"[ExperimentResult.default_metrics] Missing run data for configuration with powercap={baseline_powercap}; skipping configuration"
                )
                continue
            for idx, run in enumerate(runs):
                duration = getattr(run, "execution_duration", None)
                energy = getattr(run, "energy_used", None)
                if duration is None or energy is None:
                    print(
                        f"[ExperimentResult.default_metrics] Missing metrics for run #{idx} under powercap={baseline_powercap}; skipping run"
                    )
                    continue
                try:
                    duration_val = float(duration)
                    energy_val = float(energy)
                except (TypeError, ValueError):
                    print(
                        f"[ExperimentResult.default_metrics] Non-numeric metrics for run #{idx} under powercap={baseline_powercap}; skipping run"
                    )
                    continue
                value_pairs.append((duration_val, energy_val))

        if not value_pairs:
            print(
                f"[ExperimentResult.default_metrics] No usable metrics for powercap={baseline_powercap}; returning empty metrics"
            )
            return metrics

        durations = [pair[0] for pair in value_pairs]
        energies = [pair[1] for pair in value_pairs]
        edp_values = [d * e for d, e in value_pairs]

        metrics["execution_duration"] = statistics.mean(durations)
        metrics["energy_used"] = statistics.mean(energies)
        metrics["edp"] = statistics.mean(edp_values)

        return metrics

    def normalize(self, baseline_metrics: dict[str, float | None]):
        """Return new ExperimentResult with metrics normalized by provided baseline."""

        def safe_div(value, denom, label):
            if denom in (None, 0):
                print(
                    f"[ExperimentResult.normalize] Cannot normalize {label}; baseline is {denom}. Returning None."
                )
                return None
            try:
                return value / denom
            except TypeError:
                print(
                    f"[ExperimentResult.normalize] Non-numeric value encountered for {label}; returning None."
                )
                return None

        normalized_runs: list[MultipleRunResult] = []
        for idx, multi in enumerate(self.experiment_result):
            normalized_single_runs = []
            for run_idx, run in enumerate(multi.runs):
                duration = getattr(run, "execution_duration", None)
                energy = getattr(run, "energy_used", None)
                if duration is None or energy is None:
                    print(
                        f"[ExperimentResult.normalize] Missing metrics in run #{run_idx} of configuration #{idx}; skipping run"
                    )
                    normalized_single_runs.append(run)
                    continue

                normalized_duration = safe_div(duration, baseline_metrics.get("execution_duration"), "execution_duration")
                normalized_energy = safe_div(energy, baseline_metrics.get("energy_used"), "energy_used")
                normalized_edp = safe_div(duration * energy, baseline_metrics.get("edp"), "edp")

                normalized_single_runs.append(
                    SingleRunResult(
                        execution_duration=normalized_duration if normalized_duration is not None else duration,
                        energy_used=normalized_energy if normalized_energy is not None else energy,
                    )
                )

                if normalized_edp is None:
                    print(
                        f"[ExperimentResult.normalize] Unable to normalize EDP for run #{run_idx} in configuration #{idx}; leaving product implicit"
                    )

            normalized_multi = MultipleRunResult(
                parameters=multi.parameters,
                runs=normalized_single_runs,
            )
            normalized_runs.append(normalized_multi)

        return ExperimentResult(
            description=f"{self.description} (normalized)",
            experiment_result=normalized_runs,
        )

    @classmethod
    def from_file(cls, file_path: str | os.PathLike) -> "ExperimentResult":
        with open(file_path, "r") as file:
            data = json.load(file)

        def resolve_dataclass_type(tp):
            if hasattr(tp, "__dataclass_fields__"):
                return tp
            origin = get_origin(tp)
            if origin is None:
                return None
            for arg in get_args(tp):
                resolved = resolve_dataclass_type(arg)
                if resolved is not None:
                    return resolved
            return None

        def default_for_field(field):
            if field.default is not MISSING:
                return field.default
            if field.default_factory is not MISSING:  # type: ignore[attr-defined]
                return field.default_factory()  # type: ignore[attr-defined]
            origin = get_origin(field.type)
            if origin is list:
                return []
            return None

        def build_dataclass(klass, payload, context_path):
            if payload is None:
                print(f"[ExperimentResult.from_file] Missing object for {context_path}; defaulting to empty dict")
                payload = {}
            elif not isinstance(payload, dict):
                print(
                    f"[ExperimentResult.from_file] Unexpected payload type for {context_path}: {type(payload).__name__}; defaulting to empty dict"
                )
                payload = {}

            init_kwargs = {}
            for field in fields(klass):
                raw_value = payload.get(field.name, MISSING)
                if raw_value is MISSING or raw_value is None:
                    replacement = default_for_field(field)
                    # print(
                    #   f"[ExperimentResult.from_file] Missing value for {context_path}.{field.name}; defaulting to {repr(replacement)}"
                    # )
                else:
                    replacement = raw_value

                origin = get_origin(field.type)

                if origin is list:
                    elem_type = get_args(field.type)[0] if get_args(field.type) else None
                    if replacement is None or replacement is MISSING:
                        replacement = []
                    if not isinstance(replacement, list):
                        print(
                            f"[ExperimentResult.from_file] Expected list for {context_path}.{field.name}; defaulting to empty list"
                        )
                        replacement = []
                    if elem_type and resolve_dataclass_type(elem_type):
                        dataclass_elem = resolve_dataclass_type(elem_type)
                        replacement = [
                            build_dataclass(dataclass_elem, item, f"{context_path}.{field.name}[{idx}]")
                            for idx, item in enumerate(replacement)
                        ]
                else:
                    dataclass_type = resolve_dataclass_type(field.type)
                    if dataclass_type and replacement is not None:
                        replacement = build_dataclass(dataclass_type, replacement, f"{context_path}.{field.name}")

                init_kwargs[field.name] = replacement

            return klass(**init_kwargs)

        return build_dataclass(cls, data, cls.__name__)


@dataclass
class Experiment:
    description: str
    experiment_configurations: list[RunParameters]
    
