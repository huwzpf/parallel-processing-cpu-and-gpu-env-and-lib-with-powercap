import os
import functools
from itertools import chain

from models import RunParameters, Experiment, ExperimentResult, MultipleRunResult, SingleRunResult
from charts import time_powercap_scatter, time_batch_size_scatter, time_number_of_nodes_bar, time_number_of_nodes_scatter, min_powercap_heatmap_cpu_gpu, print_avg_edp_energy_per_configuration, min_powercap_heatmap_gpu, equal_split_start_powercap_plot, min_powercap_heatmap_cpu
from experiments import run_experiment

NUMBER_OF_RUNS = 5


BEST_DYNAMIC_CONFIGS = {
    "twinprime": [
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.2, "edp_optimization_steps": 16, "gradient_opt_eps": 0.1},
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.5, "edp_optimization_steps": 16, "gradient_opt_eps": 0.1},
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.8, "edp_optimization_steps": 16, "gradient_opt_eps": 0.1},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.2, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.2, "epsilon_decay": 0.9, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.5, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.2, "epsilon_decay": 0.9, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.8, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.2, "epsilon_decay": 0.9, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.2, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.2, "epsilon_decay": 0.95, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.5, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.2, "epsilon_decay": 0.95, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.8, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.2, "epsilon_decay": 0.95, "alpha_decay": 0.98},
    ],
    "montecarlo": [
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.2, "edp_optimization_steps": 16, "gradient_opt_eps": 0.2},
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.5, "edp_optimization_steps": 16, "gradient_opt_eps": 0.2},
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.8, "edp_optimization_steps": 16, "gradient_opt_eps": 0.2},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.2, "edp_optimization_steps": 8, "start_alpha": 0.1, "gradient_opt_eps": 0.1, "epsilon_decay": 0.9, "alpha_decay": 0.9},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.5, "edp_optimization_steps": 8, "start_alpha": 0.1, "gradient_opt_eps": 0.1, "epsilon_decay": 0.9, "alpha_decay": 0.9},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.8, "edp_optimization_steps": 8, "start_alpha": 0.1, "gradient_opt_eps": 0.1, "epsilon_decay": 0.9, "alpha_decay": 0.9},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.2, "edp_optimization_steps": 0, "start_alpha": 0.2, "gradient_opt_eps": 0.1, "epsilon_decay": 0.98, "alpha_decay": 0.95},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.5, "edp_optimization_steps": 0, "start_alpha": 0.2, "gradient_opt_eps": 0.1, "epsilon_decay": 0.98, "alpha_decay": 0.95},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.8, "edp_optimization_steps": 0, "start_alpha": 0.2, "gradient_opt_eps": 0.1, "epsilon_decay": 0.98, "alpha_decay": 0.95},
    ],
    "cnn": [
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.2, "edp_optimization_steps": 16, "gradient_opt_eps": 0.1},
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.5, "edp_optimization_steps": 16, "gradient_opt_eps": 0.1},
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.8, "edp_optimization_steps": 16, "gradient_opt_eps": 0.1},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.2, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.1, "epsilon_decay": 0.98, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.5, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.1, "epsilon_decay": 0.98, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.8, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.1, "epsilon_decay": 0.98, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.2, "edp_optimization_steps": 50, "start_alpha": 0.2, "gradient_opt_eps": 0.2, "epsilon_decay": 0.95, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.5, "edp_optimization_steps": 50, "start_alpha": 0.2, "gradient_opt_eps": 0.2, "epsilon_decay": 0.95, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.8, "edp_optimization_steps": 50, "start_alpha": 0.2, "gradient_opt_eps": 0.2, "epsilon_decay": 0.95, "alpha_decay": 0.98},
    ],
    "collatz": [
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.2, "edp_optimization_steps": 16, "gradient_opt_eps": 0.1},
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.5, "edp_optimization_steps": 16, "gradient_opt_eps": 0.1},
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.8, "edp_optimization_steps": 16, "gradient_opt_eps": 0.1},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.2, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.2, "epsilon_decay": 0.9, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.5, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.2, "epsilon_decay": 0.9, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.8, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.2, "epsilon_decay": 0.9, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.2, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.2, "epsilon_decay": 0.95, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.5, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.2, "epsilon_decay": 0.95, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.8, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.2, "epsilon_decay": 0.95, "alpha_decay": 0.98},
    ],
    "vecmaxdiv": [
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.2, "edp_optimization_steps": 20, "gradient_opt_eps": 0.2},
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.5, "edp_optimization_steps": 20, "gradient_opt_eps": 0.2},
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.8, "edp_optimization_steps": 20, "gradient_opt_eps": 0.2},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.2, "edp_optimization_steps": 12, "start_alpha": 0.1, "gradient_opt_eps": 0.1, "epsilon_decay": 0.9, "alpha_decay": 0.9},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.5, "edp_optimization_steps": 12, "start_alpha": 0.1, "gradient_opt_eps": 0.1, "epsilon_decay": 0.9, "alpha_decay": 0.9},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.8, "edp_optimization_steps": 12, "start_alpha": 0.1, "gradient_opt_eps": 0.1, "epsilon_decay": 0.9, "alpha_decay": 0.9},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.2, "edp_optimization_steps": 50, "start_alpha": 0.2, "gradient_opt_eps": 0.1, "epsilon_decay": 0.98, "alpha_decay": 0.95},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.5, "edp_optimization_steps": 50, "start_alpha": 0.2, "gradient_opt_eps": 0.1, "epsilon_decay": 0.98, "alpha_decay": 0.95},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.8, "edp_optimization_steps": 50, "start_alpha": 0.2, "gradient_opt_eps": 0.1, "epsilon_decay": 0.98, "alpha_decay": 0.95},
    ],
    "rnn": [
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.2, "edp_optimization_steps": 16, "gradient_opt_eps": 0.1},
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.5, "edp_optimization_steps": 16, "gradient_opt_eps": 0.1},
        {"strategy": "EDP_GRADIENT_CMAES", "start_powercap": 0.8, "edp_optimization_steps": 16, "gradient_opt_eps": 0.1},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.2, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.1, "epsilon_decay": 0.98, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.5, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.1, "epsilon_decay": 0.98, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SIMPLE", "start_powercap": 0.8, "edp_optimization_steps": 0, "start_alpha": 0.1, "gradient_opt_eps": 0.1, "epsilon_decay": 0.98, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.2, "edp_optimization_steps": 50, "start_alpha": 0.2, "gradient_opt_eps": 0.2, "epsilon_decay": 0.95, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.5, "edp_optimization_steps": 50, "start_alpha": 0.2, "gradient_opt_eps": 0.2, "epsilon_decay": 0.95, "alpha_decay": 0.98},
        {"strategy": "EDP_GRADIENT_SPSA", "start_powercap": 0.8, "edp_optimization_steps": 50, "start_alpha": 0.2, "gradient_opt_eps": 0.2, "epsilon_decay": 0.95, "alpha_decay": 0.98},
    ],
}

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
                for powercap in [0, 500, 1000, 1500, 2000, 2500]
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
            for start_pc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
        strategy = "CONTINUOUS_GREEDY"
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
        for steps in [0, 35]:
            for start_alpha in [0.1, 0.2, 0.4]:
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


def experiment_best_dynamic(description: str, app_name: str, file_path: str | os.PathLike, number_od_nodes: int = 16, batch_size: int = 480000, cpu_power_scaling: int | None = None, initial_cpu_batch_size_scaling=0, cpu_enabled=True):
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

    app_configs = BEST_DYNAMIC_CONFIGS.get(app_name.lower())
    if app_configs is None:
        raise ValueError(f"No best dynamic configuration defined for app '{app_name}'")

    experiment = Experiment(
        description=description,
        experiment_configurations=[common_run_parameters(**config) for config in app_configs],
    )

    run_experiment(experiment_file_name=file_path, experiment=experiment, number_of_runs=3)

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
    experiment_equal_split(description="equal_powercap", app_name="cnn", file_path="new_equal_cnn_powercap_8_nodes.json", number_od_nodes=8, batch_size=100, cpu_power_scaling=0, cpu_enabled=True)
    experiment_best_dynamic(description="best_dynamic_powercap", app_name="cnn", file_path="new_best_dynamic_cnn_powercap_8_nodes.json", number_od_nodes=8, batch_size=100, cpu_power_scaling=0, cpu_enabled=True)
    experiment_equal_split(description="equal_powercap", app_name="rnn", file_path="new_equal_rnn_powercap_8_nodes.json", number_od_nodes=8, batch_size=100, cpu_power_scaling=0, cpu_enabled=True)
    experiment_best_dynamic(description="best_dynamic_powercap", app_name="rnn", file_path="new_best_dynamic_rnn_powercap_8_nodes.json", number_od_nodes=8, batch_size=100, cpu_power_scaling=0, cpu_enabled=True)
    

    