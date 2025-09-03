import os
import subprocess
import sys
import time

from pathlib import Path

from models import RunParameters, SingleRunResult, MultipleRunResult, ExperimentResult, Experiment


def get_arguments(run_parameters: RunParameters):
    cpu_enabled_arg = "1" if run_parameters.cpu_enabled else "0"
    number_of_streams_arg = str(run_parameters.number_of_streams)
    batch_size_arg = str(run_parameters.batch_size)
    powercap_arg = str(run_parameters.powercap) if run_parameters.powercap else "0"
    cpu_power_scaling_arg = str(run_parameters.cpu_power_scaling) if run_parameters.cpu_power_scaling else "0"
    initial_cpu_batch_size_scaling = str(run_parameters.initial_cpu_batch_size_scaling) if run_parameters.initial_cpu_batch_size_scaling else "0"
    cpu_min_powercap = str(run_parameters.cpu_min_powercap) if run_parameters.cpu_min_powercap else "0"
    gpu_min_powercap = str(run_parameters.gpu_min_powercap) if run_parameters.gpu_min_powercap else "0"
    return f"--cpu-enabled={cpu_enabled_arg} --number-of-streams={number_of_streams_arg} --batch-size={batch_size_arg} --powercap={powercap_arg} --cpu-power-scaling={cpu_power_scaling_arg} --initial-cpu-batch-size-scaling={initial_cpu_batch_size_scaling} --cpu-min-powercap={cpu_min_powercap} --gpu-min-powercap={gpu_min_powercap}"

def single_app_run(run_parameters: RunParameters) -> SingleRunResult:
    os.chdir(Path.home() / Path("parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib"))
    arguments = get_arguments(run_parameters=run_parameters)
    command = f"./run_scripts/run-app {run_parameters.app_name} B {run_parameters.number_od_nodes} {arguments}"
    print(command)
    try:
        result = subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=5000)
    except Exception as e:
        result = None
        print(f"An error occurred while executing the command: {e}")

    if (result is None) or ("No devices found under the power limit" in result.stderr) or ("Main elapsed time" not in result.stderr):
        print("Error encountered when launching application")
        try:
            time.sleep(10)
            result = subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=5000)
        except Exception as e:
            result = None
            print(f"An error occurred while executing the command: {e}")
    
    if "No devices found under the power limit" in result.stderr:
        return -1

    return SingleRunResult.from_output(stdout=result.stdout, stderr=result.stderr)


def multiple_app_runs(run_parameters: RunParameters, numer_of_runs: int) -> MultipleRunResult:
    run_results = []
    for _ in range(numer_of_runs):
        run_results.append(single_app_run(run_parameters))
    return MultipleRunResult(
        parameters=run_parameters,
        runs=run_results,
    )

def run_experiment(experiment_file_name: str, experiment: Experiment, number_of_runs: int):
    experiment_result = ExperimentResult(
        description=experiment.description,
        experiment_result=[],
    )

    for experiment_configuration in experiment.experiment_configurations:
        experiment_result.experiment_result.append(multiple_app_runs(run_parameters=experiment_configuration, numer_of_runs=number_of_runs))
    
    experiment_result.to_file(file_path=experiment_file_name)
