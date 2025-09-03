import os
import functools
from itertools import chain

from models import RunParameters, Experiment, ExperimentResult, MultipleRunResult, SingleRunResult
from charts import time_powercap_scatter, time_batch_size_scatter, time_number_of_nodes_bar, time_number_of_nodes_scatter, min_powercap_heatmap
from experiments import run_experiment

# For RNN: NUMBER_OF_RUNS = 5
NUMBER_OF_RUNS = 3

def experiment_time_nodes(description: str, app_name: str, file_path: str | os.PathLike, batch_size: int = 50000):
    common_run_parameters = functools.partial(
        RunParameters,
        app_name=app_name,
        batch_size=batch_size,
        powercap=None,
        cpu_power_scaling=None,
        initial_cpu_batch_size_scaling=100
    )
    experiment = Experiment(
        description=description,
        experiment_configurations=list(
            chain.from_iterable(
                [
                    [
                        common_run_parameters(cpu_enabled=False, number_of_streams=1, number_od_nodes=i),
                        common_run_parameters(cpu_enabled=True, number_of_streams=1, number_od_nodes=i),
                        common_run_parameters(cpu_enabled=False, number_of_streams=2, number_od_nodes=i),
                        common_run_parameters(cpu_enabled=True, number_of_streams=2, number_od_nodes=i),
                    ]
                for i in [1, 2, 4, 8, 16]
                ]
            )
        )
    )
    run_experiment(experiment_file_name=file_path, experiment=experiment, number_of_runs=NUMBER_OF_RUNS)

def experiment_time_powercap(description: str, app_name: str, file_path: str | os.PathLike, number_od_nodes: int = 16, batch_size: int = 480000, cpu_power_scaling: int | None = None, initial_cpu_batch_size_scaling=100):
    common_run_parameters = functools.partial(
        RunParameters,
        app_name=app_name,
        batch_size=batch_size,
        number_od_nodes=number_od_nodes,
        cpu_power_scaling=cpu_power_scaling,
        number_of_streams=2,
        initial_cpu_batch_size_scaling=initial_cpu_batch_size_scaling
    )
    experiment = Experiment(
        description=description,
        experiment_configurations=list(
            chain.from_iterable(
                [
                    [
                        common_run_parameters(cpu_enabled=True, powercap=powercap),
                        common_run_parameters(cpu_enabled=False, powercap=powercap),
                    ]
                for powercap in range(500, 4000, 500)
                ]
            )
        )
    )
    run_experiment(experiment_file_name=file_path, experiment=experiment, number_of_runs=NUMBER_OF_RUNS)


def experiment_powercap_opt(description: str, app_name: str, file_path: str | os.PathLike, number_od_nodes: int = 16, batch_size: int = 480000, cpu_power_scaling: int | None = None, initial_cpu_batch_size_scaling=100):
    common_run_parameters = functools.partial(
        RunParameters,
        app_name=app_name,
        batch_size=batch_size,
        number_od_nodes=number_od_nodes,
        cpu_power_scaling=cpu_power_scaling,
        number_of_streams=2,
        initial_cpu_batch_size_scaling=0,
        cpu_enabled=True
    )
    experiment = Experiment(
        description=description,
        experiment_configurations=[
            common_run_parameters(
                powercap=powercap,
                cpu_min_powercap=cpu_min,
                gpu_min_powercap=gpu_min,
            )
            for powercap in [500, 1000, 1500, 2000, 2500, 3000]
            for cpu_min in [0.1, 0.3, 0.5, 0.7, 0.9]
            for gpu_min in [0.1, 0.3, 0.5, 0.7, 0.9]
        ]
    )
    run_experiment(experiment_file_name=file_path, experiment=experiment, number_of_runs=NUMBER_OF_RUNS)

def experiment_time_batch_size(description: str, app_name: str, file_path: str | os.PathLike, number_of_nodes: int):
    common_run_parameters = functools.partial(
        RunParameters,
        app_name=app_name,
        number_od_nodes=number_of_nodes,
        number_of_streams=2,
        powercap=None,
        cpu_power_scaling=None,
        # For RNN: initial_cpu_batch_size_scaling=0
        initial_cpu_batch_size_scaling=100
    )
    experiment = Experiment(
        description=description,
        experiment_configurations=list(
            chain.from_iterable(
                [
                    [
                        common_run_parameters(cpu_enabled=True, batch_size=batch_size),
                        common_run_parameters(cpu_enabled=False, batch_size=batch_size),
                    ]
                for batch_size in [3_840_000, 960_000, 480_000, 120_000, 40_000, 12_800]
                # For RNN: for batch_size in [400, 200, 100, 50, 25]
                ]
            )
        )
    )
    run_experiment(experiment_file_name=file_path, experiment=experiment, number_of_runs=NUMBER_OF_RUNS)


if __name__ == "__main__":
    
    # experiment_time_powercap(description="time(powercap)", app_name="collatz", file_path="collatz_powercap_16_nodes.json", number_od_nodes=16, batch_size=480000, cpu_power_scaling=0)
    # experiment_time_powercap(description="time(powercap)", app_name="twinprime", file_path="twinprime_powercap_16_nodes.json", number_od_nodes=16, batch_size=480000, cpu_power_scaling=0)
    # experiment_time_powercap(description="time(powercap)", app_name="vecmaxdiv", file_path="vecmaxdiv_powercap_16_nodes.json", number_od_nodes=16, batch_size=960000, cpu_power_scaling=0)

    # experiment_time_powercap(description="time(powercap)", app_name="collatz", file_path="065-collatz_powercap_16_nodes.json", number_od_nodes=16, batch_size=480000, cpu_power_scaling=0.65)
    # experiment_time_powercap(description="time(powercap)", app_name="twinprime", file_path="064-twinprime_powercap_16_nodes.json", number_od_nodes=16, batch_size=480000, cpu_power_scaling=0.64)
    # experiment_time_powercap(description="time(powercap)", app_name="vecmaxdiv", file_path="056-vecmaxdiv_powercap_16_nodes.json", number_od_nodes=16, batch_size=960000, cpu_power_scaling=0.56)

    # experiment_time_batch_size(description="time(batch_size)", app_name="collatz", file_path="collatz_batch_size_16_nodes.json", number_of_nodes=16)
    # exp = ExperimentResult.from_file("../cudampilib/collatz_batch_size_16_nodes.json")
    # time_batch_size_scatter(exp)

    
    # experiment_time_nodes(description="time(number_of_nodes) and number of streams", app_name="collatz", file_path="collatz_time_nodes.json", batch_size=480000)
    # exp = ExperimentResult.from_file("../cudampilib/collatz_time_nodes.json")
    # time_number_of_nodes_bar(exp)

    # experiment_powercap_opt(description="time(powercap)", app_name="collatz", file_path="0_33_continous_collatz_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0)
    exp = ExperimentResult.from_file("/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/0_33_continous_collatz_powercap_8_nodes.json")
    min_powercap_heatmap(exp) 
    # experiment_powercap_opt(description="time(powercap)", app_name="twinprime", file_path="33_continous_twinprime_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0)
    # exp = ExperimentResult.from_file("../cudampilib/33_continous_twinprime_powercap_8_nodes.json")
    # time_powercap_scatter(exp) 
    # experiment_powercap_opt(description="time(powercap)", app_name="vecmaxdiv", file_path="33_continous_vecmaxdiv_powercap_8_nodes.json", number_od_nodes=8, batch_size=960000, cpu_power_scaling=0)
    # exp = ExperimentResult.from_file("../cudampilib/33_continous_vecmaxdiv_powercap_8_nodes.json")
    # time_powercap_scatter(exp) 
