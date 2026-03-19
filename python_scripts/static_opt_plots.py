import os
import functools
from itertools import chain

from models import RunParameters, Experiment, ExperimentResult, MultipleRunResult, SingleRunResult
from charts import (
    time_powercap_scatter,
    time_batch_size_scatter,
    time_number_of_nodes_bar,
    time_number_of_nodes_scatter,
    min_powercap_heatmap_cpu_gpu,
    print_avg_edp_energy_per_configuration,
    min_powercap_heatmap_gpu,
    equal_split_start_powercap_plot,
    min_powercap_heatmap_cpu,
    optimal_configuration_metric_trends,
)
from experiments import run_experiment

twinprime_binary_path = "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/WYKRESY_MAGISTERKA/STATIC/CPU/binary_twinprime_powercap_8_nodes.json"
twinprime_continous_path = "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/WYKRESY_MAGISTERKA/STATIC/CPU/continous_twinprime_powercap_8_nodes.json"

exp_twinprime_binary_raw = ExperimentResult.from_file(twinprime_binary_path)
twinprime_defaults = exp_twinprime_binary_raw.default_metrics()
exp_twinprime_binary = exp_twinprime_binary_raw.normalize(twinprime_defaults)
exp_twinptime_continous = ExperimentResult.from_file(twinprime_continous_path).normalize(twinprime_defaults)

collatz_binary_path = "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/WYKRESY_MAGISTERKA/STATIC/CPU/binary_collatz_powercap_8_nodes.json"
collatz_continous_path = "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/WYKRESY_MAGISTERKA/STATIC/CPU/continous_collatz_powercap_8_nodes.json"

exp_collatz_binary_raw = ExperimentResult.from_file(collatz_binary_path)
collatz_defaults = exp_collatz_binary_raw.default_metrics()
exp_collatz_binary = exp_collatz_binary_raw.normalize(collatz_defaults)
exp_collatz_continous = ExperimentResult.from_file(collatz_continous_path).normalize(collatz_defaults)

cnn_binary_path = "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/WYKRESY_MAGISTERKA/STATIC/CPU+GPU/binary_cnn_powercap_8_nodes.json"
cnn_continous_path = "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/WYKRESY_MAGISTERKA/STATIC/CPU+GPU/continous_cnn_powercap_8_nodes.json"

exp_cnn_binary_raw = ExperimentResult.from_file(cnn_binary_path)
cnn_defaults = exp_cnn_binary_raw.default_metrics()
exp_cnn_binary = exp_cnn_binary_raw.normalize(cnn_defaults)
exp_cnn_continous = ExperimentResult.from_file(cnn_continous_path).normalize(cnn_defaults)


rnn_binary_path = "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/WYKRESY_MAGISTERKA/STATIC/CPU+GPU/binary_rnn_powercap_8_nodes.json"
rnn_continous_path = "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/WYKRESY_MAGISTERKA/STATIC/CPU+GPU/continous_rnn_powercap_8_nodes.json"

exp_rnn_binary_raw = ExperimentResult.from_file(rnn_binary_path)
rnn_defaults = exp_rnn_binary_raw.default_metrics()
exp_rnn_binary = exp_rnn_binary_raw.normalize(rnn_defaults)
exp_rnn_continous = ExperimentResult.from_file(rnn_continous_path).normalize(rnn_defaults)

montecarlo_binary_path = "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/WYKRESY_MAGISTERKA/STATIC/GPU/binary_montecarlo_powercap_8_nodes.json"
montecarlo_continous_path = "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/WYKRESY_MAGISTERKA/STATIC/GPU/continous_montecarlo_powercap_8_nodes.json"

exp_montecarlo_binary_raw = ExperimentResult.from_file(montecarlo_binary_path)
monte_defaults = exp_montecarlo_binary_raw.default_metrics()
exp_montecarlo_binary = exp_montecarlo_binary_raw.normalize(monte_defaults)
exp_montecarlo_continous = ExperimentResult.from_file(montecarlo_continous_path).normalize(monte_defaults)

vecmaxdiv_binary_path = "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/WYKRESY_MAGISTERKA/STATIC/GPU/binary_vecmaxdiv_powercap_8_nodes.json"
vecmaxdiv_continous_path = "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/WYKRESY_MAGISTERKA/STATIC/GPU/continous_vecmaxdiv_powercap_8_nodes.json"

exp_vecmaxdiv_binary_raw = ExperimentResult.from_file(vecmaxdiv_binary_path)
vec_defaults = exp_vecmaxdiv_binary_raw.default_metrics()
exp_vecmaxdiv_binary = exp_vecmaxdiv_binary_raw.normalize(vec_defaults)
exp_vecmaxdiv_continous = ExperimentResult.from_file(vecmaxdiv_continous_path).normalize(vec_defaults)


min_powercap_heatmap_cpu(exp_twinptime_continous, "plots_twinprime_normalized")
optimal_configuration_metric_trends(exp_twinptime_continous, exp_twinprime_binary, "plots_twinprime_normalized")
min_powercap_heatmap_cpu(exp_collatz_continous, "plots_collatz_normalized")
optimal_configuration_metric_trends(exp_collatz_continous, exp_collatz_binary, "plots_collatz_normalized")
min_powercap_heatmap_gpu(exp_montecarlo_continous, "plots_montecarlo_normalized")
optimal_configuration_metric_trends(exp_montecarlo_continous, exp_montecarlo_binary, "plots_montecarlo_normalized")
min_powercap_heatmap_gpu(exp_vecmaxdiv_continous, "plots_vecmaxdiv_normalized")
optimal_configuration_metric_trends(exp_vecmaxdiv_continous, exp_vecmaxdiv_binary, "plots_vecmaxdiv_normalized")
min_powercap_heatmap_cpu_gpu(exp_cnn_continous, "plots_cnn_normalized")
optimal_configuration_metric_trends(exp_cnn_continous, exp_cnn_binary, "plots_cnn_normalized")
min_powercap_heatmap_cpu_gpu(exp_rnn_continous, "plots_rnn_normalized")
optimal_configuration_metric_trends(exp_rnn_continous, exp_rnn_binary, "plots_rnn_normalized")
