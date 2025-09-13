import os
import functools
from itertools import chain

from models import RunParameters, Experiment, ExperimentResult, MultipleRunResult, SingleRunResult
from charts import time_powercap_scatter, time_batch_size_scatter, time_number_of_nodes_bar, time_number_of_nodes_scatter, min_powercap_heatmap_cpu_gpu, print_avg_edp_energy_per_configuration, min_powercap_heatmap_gpu, equal_split_start_powercap_plot
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

def experiment_time_powercap(description: str, app_name: str, file_path: str | os.PathLike, number_od_nodes: int = 16, batch_size: int = 480000, cpu_power_scaling: int | None = None, initial_cpu_batch_size_scaling=0, cpu_enabled=True):
    common_run_parameters = functools.partial(
        RunParameters,
        app_name=app_name,
        batch_size=batch_size,
        number_od_nodes=number_od_nodes,
        cpu_power_scaling=cpu_power_scaling,
        number_of_streams=2,
        initial_cpu_batch_size_scaling=initial_cpu_batch_size_scaling,
        strategy="BINARY_GREEDY"
    )
    
    experiment = Experiment(
        description=description,
        experiment_configurations=list(
            chain.from_iterable(
                [
                    [
                        # common_run_parameters(cpu_enabled=True, powercap=powercap),
                        common_run_parameters(cpu_enabled=cpu_enabled, powercap=powercap),
                    ]
                # for powercap in [0, 500, 1000, 1500, 2000, 2500]
                for powercap in [0, 500, 900, 1300, 1700, 2100, 2500]
                ]
            )
        )
    )
    run_experiment(experiment_file_name=file_path, experiment=experiment, number_of_runs=NUMBER_OF_RUNS)


def experiment_equal_split(description: str, app_name: str, file_path: str | os.PathLike, number_od_nodes: int = 16, batch_size: int = 480000, cpu_power_scaling: int | None = None, initial_cpu_batch_size_scaling=0, cpu_enabled=True):
    common_run_parameters = functools.partial(
        RunParameters,
        app_name=app_name,
        batch_size=batch_size,
        number_od_nodes=number_od_nodes,
        cpu_power_scaling=cpu_power_scaling,
        number_of_streams=2,
        initial_cpu_batch_size_scaling=0,
        # cpu_enabled=True,
        cpu_enabled=cpu_enabled,
        strategy = "EQUAL_SPLIT",
        powercap=0
    )
    experiment = Experiment(
        description=description,
        experiment_configurations=[
            common_run_parameters(
                start_powercap=start_pc,
            )
            for start_pc in [0.1, 0.3, 0.5, 0.7, 0.9]
        ]
    )
    run_experiment(experiment_file_name=file_path, experiment=experiment, number_of_runs=NUMBER_OF_RUNS)

def experiment_powercap_opt(description: str, app_name: str, file_path: str | os.PathLike, number_od_nodes: int = 16, batch_size: int = 480000, cpu_power_scaling: int | None = None, initial_cpu_batch_size_scaling=0, cpu_enabled=True):
    common_run_parameters = functools.partial(
        RunParameters,
        app_name=app_name,
        batch_size=batch_size,
        number_od_nodes=number_od_nodes,
        cpu_power_scaling=cpu_power_scaling,
        number_of_streams=2,
        initial_cpu_batch_size_scaling=0,
        # cpu_enabled=True,
        cpu_enabled=cpu_enabled,
        strategy = "CONTINOUS_EQUAL"
    )
    '''
    if cpu_enabled:
        cpu_pcs = [0.1, 0.3, 0.5, 0.7, 0.9]
        gpu_pcs = [0.1, 0.5, 0.9]
    else:
        cpu_pcs = [0.9]
        gpu_pcs = [0.1, 0.3, 0.5, 0.7, 0.9]
    '''
    cpu_pcs = [0.1, 0.3, 0.5, 0.7, 0.9]
    gpu_pcs = [0.1, 0.3, 0.5, 0.7, 0.9]
    experiment = Experiment(
        description=description,
        experiment_configurations=[
            common_run_parameters(
                powercap=powercap,
                cpu_min_powercap=cpu_min,
                gpu_min_powercap=gpu_min,
                cpu_time_window_us=500_000,
            )
            for powercap in [500, 900, 1300, 1700, 2100, 2500]
            for cpu_min in cpu_pcs
            for gpu_min in gpu_pcs
            # for powercap in [500, 1000, 1500, 2000, 2500]
            # for cpu_min in [0.1, 0.3, 0.5, 0.7]
            # for gpu_min in [0.1, 0.3, 0.5]
        ]
    )
    run_experiment(experiment_file_name=file_path, experiment=experiment, number_of_runs=NUMBER_OF_RUNS)

def experiment_powercap_dynamic(description: str, app_name: str, file_path: str | os.PathLike, number_od_nodes: int = 16, batch_size: int = 480000, cpu_power_scaling: int | None = None, initial_cpu_batch_size_scaling=0, cpu_enabled=True):
    common_run_parameters = functools.partial(
        RunParameters,
        app_name=app_name,
        batch_size=batch_size,
        number_od_nodes=number_od_nodes,
        cpu_power_scaling=cpu_power_scaling,
        number_of_streams=2,
        initial_cpu_batch_size_scaling=0,
        cpu_enabled=cpu_enabled,
    )

    start_pcs = [0.2, 0.5, 0.8]

    configs: list[RunParameters] = []

    # EDP_GRADIENT_SIMPLE
    for sp in start_pcs:
        for steps in [0, 8]:
            for start_alpha in [0.1, 0.25, 0.4]:
                for grad_eps in [0.1, 0.2]:
                    for eps_decay in [0.98, 0.9]:
                        for a_decay in [0.98, 0.9]:
                            configs.append(common_run_parameters(
                                strategy="EDP_GRADIENT_SIMPLE",
                                start_powercap=sp,
                                edp_optimization_steps=steps,
                                start_alpha=start_alpha,
                                gradient_opt_eps=grad_eps,
                                epsilon_decay=eps_decay,
                                alpha_decay=a_decay,
                            ))

    # EDP_GRADIENT_SPSA
    for sp in start_pcs:
        for steps in [0, 50, 100]:
            for start_alpha in [0.1, 0.2]:
                for grad_eps in [0.1, 0.2]:
                    for eps_decay in [0.98, 0.95]:
                        for a_decay in [0.98, 0.95]:
                            configs.append(common_run_parameters(
                                strategy="EDP_GRADIENT_SPSA",
                                start_powercap=sp,
                                edp_optimization_steps=steps,
                                start_alpha=start_alpha,
                                gradient_opt_eps=grad_eps,
                                epsilon_decay=eps_decay,
                                alpha_decay=a_decay,
                            ))

    # EDP_GRADIENT_CMAES
    for sp in start_pcs:
        for steps in [0, 16]:
            for grad_eps in [0.1, 0.2, 0.3]:
                configs.append(common_run_parameters(
                    strategy="EDP_GRADIENT_CMAES",
                    start_powercap=sp,
                    edp_optimization_steps=steps,
                    gradient_opt_eps=grad_eps,
                ))

    experiment = Experiment(
        description=description,
        experiment_configurations=configs,
    )

    # Run each configuration once
    run_experiment(experiment_file_name=file_path, experiment=experiment, number_of_runs=1)

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


    # experiment_powercap_opt(description="continous_powercap", app_name="montecarlo", file_path="continous_montecarlo_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0, cpu_enabled=False)
    # experiment_time_powercap(description="binary_powercap", app_name="montecarlo", file_path="binary_montecarlo_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0, cpu_enabled=False)
    # experiment_powercap_opt(description="continous_powercap", app_name="vecmaxdiv", file_path="continous_vecmaxdiv_powercap_8_nodes.json", number_od_nodes=8, batch_size=960000, cpu_power_scaling=0, cpu_enabled=False)
    # experiment_time_powercap(description="binary_powercap", app_name="vecmaxdiv", file_path="binary_vecmaxdiv_powercap_8_nodes.json", number_od_nodes=8, batch_size=960000, cpu_power_scaling=0, cpu_enabled=False)
    '''
    print("---------- Montecarlo experiments ----------")
    exp = ExperimentResult.from_file("../cudampilib/continous_montecarlo_powercap_8_nodes.json")
    min_powercap_heatmap_gpu(exp, "8_plots_montecarlo")
    exp = ExperimentResult.from_file("../cudampilib/binary_montecarlo_powercap_8_nodes.json")
    print_avg_edp_energy_per_configuration(exp)
    
    print("---------- Vecmaxdiv experiments ----------")
    exp = ExperimentResult.from_file("../cudampilib/continous_vecmaxdiv_powercap_8_nodes.json")
    min_powercap_heatmap_gpu(exp, "8_plots_vecmaxdiv")
    exp = ExperimentResult.from_file("../cudampilib/binary_vecmaxdiv_powercap_8_nodes.json")
    print_avg_edp_energy_per_configuration(exp)

    '''
    
    # experiment_powercap_opt(description="continous_powercap", app_name="collatz", file_path="continous_collatz_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0, cpu_enabled=True)
    # experiment_time_powercap(description="binary_powercap", app_name="collatz", file_path="binary_collatz_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0, cpu_enabled=True)
    # experiment_powercap_opt(description="continous_powercap", app_name="twinprime", file_path="continous_twinprime_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0, cpu_enabled=True)
    # experiment_time_powercap(description="binary_powercap", app_name="twinprime", file_path="binary_twinprime_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0, cpu_enabled=True)

    '''
    print("---------- Collatz experiments ----------")
    exp = ExperimentResult.from_file("../cudampilib/continous_collatz_powercap_8_nodes.json")
    min_powercap_heatmap_cpu(exp, "8_plots_collatz")
    exp = ExperimentResult.from_file("../cudampilib/binary_collatz_powercap_8_nodes.json")
    print_avg_edp_energy_per_configuration(exp)
    
    print("---------- Twinprime experiments ----------")
    exp = ExperimentResult.from_file("../cudampilib/continous_twinprime_powercap_8_nodes.json")
    min_powercap_heatmap_cpu(exp, "8_plots_twinprime")
    exp = ExperimentResult.from_file("../cudampilib/binary_twinprime_powercap_8_nodes.json")
    print_avg_edp_energy_per_configuration(exp)

    '''

    # experiment_equal_split(description="equal_powercap", app_name="cnn", file_path="equal_cnn_powercap_8_nodes.json", number_od_nodes=8, batch_size=100, cpu_power_scaling=0, cpu_enabled=True)
    # experiment_equal_split(description="equal_powercap", app_name="montecarlo", file_path="equal_montecarlo_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0, cpu_enabled=False)
    # experiment_equal_split(description="equal_powercap", app_name="vecmaxdiv", file_path="equal_vecmaxdiv_powercap_8_nodes.json", number_od_nodes=8, batch_size=960000, cpu_power_scaling=0, cpu_enabled=False)
    # experiment_equal_split(description="equal_powercap", app_name="collatz", file_path="equal_collatz_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0, cpu_enabled=True)
    # experiment_equal_split(description="equal_powercap", app_name="twinprime", file_path="equal_twinprime_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0, cpu_enabled=True)
    # experiment_equal_split(description="equal_powercap", app_name="rnn", file_path="equal_rnn_powercap_8_nodes.json", number_od_nodes=8, batch_size=100, cpu_power_scaling=0, cpu_enabled=True)

    # experiment_powercap_opt(description="continous_powercap", app_name="cnn", file_path="continous_cnn_powercap_8_nodes.json", number_od_nodes=8, batch_size=20, cpu_power_scaling=0, cpu_enabled=True)
    # experiment_time_powercap(description="binary_powercap", app_name="cnn", file_path="binary_cnn_powercap_8_nodes.json", number_od_nodes=8, batch_size=20, cpu_power_scaling=0, cpu_enabled=True)
    # experiment_powercap_opt(description="continous_powercap", app_name="rnn", file_path="continous_rnn_powercap_8_nodes.json", number_od_nodes=8, batch_size=100, cpu_power_scaling=0, cpu_enabled=True)
    # experiment_time_powercap(description="binary_powercap", app_name="rnn", file_path="binary_rnn_powercap_8_nodes.json", number_od_nodes=8, batch_size=100, cpu_power_scaling=0, cpu_enabled=True)

    '''
    exp = ExperimentResult.from_file("../cudampilib/equal_cnn_powercap_8_nodes.json")
    equal_split_start_powercap_plot(exp, "equal_plots_cnn", gpu_max_power=285, gpu_max_pc=285, cpu_max_pc=125)
    exp = ExperimentResult.from_file("../cudampilib/equal_montecarlo_powercap_8_nodes.json")
    equal_split_start_powercap_plot(exp, "equal_plots_montecarlo", gpu_max_power=285, gpu_max_pc=285, cpu_max_pc=0)
    exp = ExperimentResult.from_file("../cudampilib/equal_vecmaxdiv_powercap_8_nodes.json")
    equal_split_start_powercap_plot(exp, "equal_plots_vecmaxdiv", gpu_max_power=230, gpu_max_pc=285, cpu_max_pc=0)
    exp = ExperimentResult.from_file("../cudampilib/equal_collatz_powercap_8_nodes.json")
    equal_split_start_powercap_plot(exp, "equal_plots_collatz", gpu_max_power=100, gpu_max_pc=285, cpu_max_pc=125)
    exp = ExperimentResult.from_file("../cudampilib/equal_twinprime_powercap_8_nodes.json")
    equal_split_start_powercap_plot(exp, "equal_plots_twinprime", gpu_max_power=100, gpu_max_pc=285, cpu_max_pc=125)
    exp = ExperimentResult.from_file("../cudampilib/equal_rnn_powercap_8_nodes.json")
    equal_split_start_powercap_plot(exp, "equal_plots_rnn", gpu_max_power=170, gpu_max_pc=285, cpu_max_pc=125)
    '''





    # experiment_powercap_opt(description="time(powercap)", app_name="rnn", file_path="continous_rnn_powercap_8_nodes.json", number_od_nodes=8, batch_size=100, cpu_power_scaling=0)
    # exp = ExperimentResult.from_file("../cudampilib/continous_rnn_powercap_8_nodes.json")
    # min_powercap_heatmap_cpu_gpu(exp, "plots_rnn")
    # experiment_time_powercap(description="time(powercap)", app_name="rnn", file_path="binary_rnn_powercap_8_nodes.json", number_od_nodes=8, batch_size=100, cpu_power_scaling=0)
    # exp = ExperimentResult.from_file("../cudampilib/binary_rnn_powercap_8_nodes.json")
    # print_avg_edp_energy_per_configuration(exp)
    # experiment_time_nodes(description="time(number_of_nodes) and number of streams", app_name="collatz", file_path="collatz_time_nodes.json", batch_size=480000)
    # exp = ExperimentResult.from_file("../cudampilib/collatz_time_nodes.json")
    # time_number_of_nodes_bar(exp)
    # experiment_time_powercap(description="time(powercap)", app_name="collatz", file_path="binary_collatz_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0)
    # experiment_powercap_opt(description="time(powercap)", app_name="collatz", file_path="continous_collatz_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0)
    # exp = ExperimentResult.from_file("/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/continous_collatz_powercap_8_nodes.json")
    # min_powercap_heatmap_cpu_gpu(exp, "plots_new")
    # exp = ExperimentResult.from_file("/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/binary_collatz_powercap_8_nodes.json")
    # print_avg_edp_energy_per_configuration(exp)
    # experiment_powercap_opt(description="time(powercap)", app_name="twinprime", file_path="33_continous_twinprime_powercap_8_nodes.json", number_od_nodes=8, batch_size=480000, cpu_power_scaling=0)
    # exp = ExperimentResult.from_file("../cudampilib/33_continous_twinprime_powercap_8_nodes.json")
    # time_powercap_scatter(exp) 
    # experiment_powercap_opt(description="time(powercap)", app_name="vecmaxdiv", file_path="33_continous_vecmaxdiv_powercap_8_nodes.json", number_od_nodes=8, batch_size=960000, cpu_power_scaling=0)
    # exp = ExperimentResult.from_file("../cudampilib/33_continous_vecmaxdiv_powercap_8_nodes.json")
    # time_powercap_scatter(exp) 
