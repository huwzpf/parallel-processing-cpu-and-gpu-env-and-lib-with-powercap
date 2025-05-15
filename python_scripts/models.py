import os
import json
import statistics
import re

from dataclasses import dataclass, asdict, fields


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
    powercap: int | None
    cpu_power_scaling: float | None
    initial_cpu_batch_size_scaling: int


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

    @classmethod
    def from_file(cls, file_path: str | os.PathLike) -> "ExperimentResult":
        with open(file_path, "r") as file:
            data = json.load(file)
        return dataclass_from_dict(cls, data)


@dataclass
class Experiment:
    description: str
    experiment_configurations: list[RunParameters]
    