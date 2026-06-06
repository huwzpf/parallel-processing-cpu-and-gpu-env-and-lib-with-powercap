from models import ExperimentResult
from charts import (
    equal_split_dynamic_annotations_plot,
    dynamic_search_trajectories_plot,
    dynamic_trajectories_plot
)

'''

from models import ExperimentResult
from charts import (
    equal_split_dynamic_annotations_plot,
    dynamic_search_trajectories_plot,
    dynamic_trajectories_plot
)

vecmaxdiv_raw_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/4_test_vecmaxdiv_powercap_4_nodes.json"
vecmaxdiv_trajectories_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/4_equal_vecmaxdiv_powercap_4_nodes.json"
vecmaxdiv_equal_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/4_trajectories_best_dynamic_vecmaxdiv_powercap_4_nodes.json"


exp_vecmaxdiv_raw = ExperimentResult.from_file(vecmaxdiv_raw_path)
vecmaxdiv_defaults = exp_vecmaxdiv_raw.default_metrics()
exp_vecmaxdiv_equal = ExperimentResult.from_file(vecmaxdiv_equal_path).normalize(vecmaxdiv_defaults)
exp_vecmaxdiv_trajectories = ExperimentResult.from_file(vecmaxdiv_trajectories_path).normalize(vecmaxdiv_defaults)

    
dynamic_trajectories_plot (
    exp_vecmaxdiv_equal,
    exp_vecmaxdiv_trajectories,
    out_dir="../WYKRESY_MAGISTERKA/FINAL_DYNAMIC/4_vecmaxdiv_dynamic_best",
)


twinprime_equal_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/equal_twinprime_powercap_8_nodes.json"
twinprime_best_dynamic_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/best_dynamic_twinprime_powercap_8_nodes.json"
twinprime_trajectories_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/trajectories_best_dynamic_twinprime_powercap_8_nodes.json"
twinprime_search_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/DYNAMIC/dynamic_twinprime_powercap_8_nodes.json"
twinprime_raw_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/test_twinprime_powercap_8_nodes.json"

montecarlo_equal_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/equal_montecarlo_powercap_8_nodes.json"
montecarlo_best_dynamic_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/best_dynamic_montecarlo_powercap_8_nodes.json"
montecarlo_trajectories_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/trajectories_best_dynamic_montecarlo_powercap_8_nodes.json"
montecarlo_search_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/DYNAMIC/dynamic_montecarlo_powercap_8_nodes.json"
montecarlo_raw_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/test_montecarlo_powercap_8_nodes.json"

cnn_equal_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/equal_cnn_powercap_8_nodes.json"
cnn_best_dynamic_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/best_dynamic_cnn_powercap_8_nodes.json"
cnn_trajectories_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/trajectories_best_dynamic_cnn_powercap_8_nodes.json"
cnn_search_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/DYNAMIC/dynamic_cnn_powercap_8_nodes.json"
cnn_raw_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/test_cnn_powercap_8_nodes.json"

vecmaxdiv_raw_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/test_vecmaxdiv_powercap_8_nodes.json"
vecmaxdiv_equal_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/equal_vecmaxdiv_powercap_8_nodes.json"
vecmaxdiv_trajectories_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/trajectories_best_dynamic_vecmaxdiv_powercap_8_nodes.json"


collatz_raw_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/test_collatz_powercap_8_nodes.json"
collatz_equal_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/equal_collatz_powercap_8_nodes.json"
collatz_trajectories_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/trajectories_best_dynamic_collatz_powercap_8_nodes.json"

rnn_raw_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/test_rnn_8_nodes.json"
rnn_equal_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/equal_rnn_powercap_8_nodes.json"
rnn_trajectories_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/trajectories_best_dynamic_rnn_powercap_8_nodes.json"


exp_twinprime_raw = ExperimentResult.from_file(twinprime_raw_path)
twinprime_defaults = exp_twinprime_raw.default_metrics()
exp_twinprime_equal = ExperimentResult.from_file(twinprime_equal_path).normalize(twinprime_defaults)
exp_twinprime_best = ExperimentResult.from_file(twinprime_best_dynamic_path).normalize(twinprime_defaults)
exp_twinprime_trajectories = ExperimentResult.from_file(twinprime_trajectories_path).normalize(twinprime_defaults)
exp_twinprime_search = ExperimentResult.from_file(twinprime_search_path).normalize(twinprime_defaults)

exp_montecarlo_raw = ExperimentResult.from_file(montecarlo_raw_path)
montecarlo_defaults = exp_montecarlo_raw.default_metrics()
exp_montecarlo_equal = ExperimentResult.from_file(montecarlo_equal_path).normalize(montecarlo_defaults)
exp_montecarlo_best = ExperimentResult.from_file(montecarlo_best_dynamic_path).normalize(montecarlo_defaults)
exp_montecarlo_trajectories = ExperimentResult.from_file(montecarlo_trajectories_path).normalize(montecarlo_defaults)
exp_montecarlo_search = ExperimentResult.from_file(montecarlo_search_path).normalize(montecarlo_defaults)

exp_cnn_raw = ExperimentResult.from_file(cnn_raw_path)
cnn_defaults = exp_cnn_raw.default_metrics()
exp_cnn_equal = ExperimentResult.from_file(cnn_equal_path).normalize(cnn_defaults)
exp_cnn_best = ExperimentResult.from_file(cnn_best_dynamic_path).normalize(cnn_defaults)
exp_cnn_trajectories = ExperimentResult.from_file(cnn_trajectories_path).normalize(cnn_defaults)
exp_cnn_search = ExperimentResult.from_file(cnn_search_path).normalize(cnn_defaults)

exp_vecmaxdiv_raw = ExperimentResult.from_file(vecmaxdiv_raw_path)
vecmaxdiv_defaults = exp_vecmaxdiv_raw.default_metrics()
exp_vecmaxdiv_equal = ExperimentResult.from_file(vecmaxdiv_equal_path).normalize(vecmaxdiv_defaults)
exp_vecmaxdiv_trajectories = ExperimentResult.from_file(vecmaxdiv_trajectories_path).normalize(vecmaxdiv_defaults)

exp_collatz_raw = ExperimentResult.from_file(collatz_raw_path)
collatz_defaults = exp_collatz_raw.default_metrics()
exp_collatz_equal = ExperimentResult.from_file(collatz_equal_path).normalize(collatz_defaults)
exp_collatz_trajectories = ExperimentResult.from_file(collatz_trajectories_path).normalize(collatz_defaults)

exp_rnn_raw = ExperimentResult.from_file(rnn_raw_path)
rnn_defaults = exp_rnn_raw.default_metrics()
exp_rnn_equal = ExperimentResult.from_file(rnn_equal_path).normalize(rnn_defaults)
exp_rnn_trajectories = ExperimentResult.from_file(rnn_trajectories_path).normalize(rnn_defaults)



equal_split_dynamic_annotations_plot(
    exp_twinprime_equal,
    exp_twinprime_dynamic,
    out_dir="plots_twinprime_dynamic",
)

equal_split_dynamic_annotations_plot(
    exp_montecarlo_equal,
    exp_montecarlo_dynamic,
    out_dir="plots_montecarlo_dynamic",
)

equal_split_dynamic_annotations_plot(
    exp_cnn_equal,
    exp_cnn_dynamic,
    out_dir="plots_cnn_dynamic",
)

dynamic_search_trajectories_plot(
    exp_twinprime_equal,
    exp_twinprime_search,
    exp_twinprime_best,
    out_dir="../WYKRESY_MAGISTERKA/FINAL_DYNAMIC/twinprime_dynamic_search",
)

dynamic_search_trajectories_plot(
    exp_montecarlo_equal,
    exp_montecarlo_search,
    exp_montecarlo_best,
    out_dir="../WYKRESY_MAGISTERKA/FINAL_DYNAMIC/montecarlo_dynamic_search",
)

dynamic_search_trajectories_plot(
    exp_cnn_equal,
    exp_cnn_search,
    exp_cnn_best,
    out_dir="../WYKRESY_MAGISTERKA/FINAL_DYNAMIC/cnn_dynamic_search",
)

dynamic_trajectories_plot (
    exp_twinprime_equal,
    exp_twinprime_trajectories,
    out_dir="../WYKRESY_MAGISTERKA/FINAL_DYNAMIC/twinprime_dynamic_best",
)

dynamic_trajectories_plot (
    exp_montecarlo_equal,
    exp_montecarlo_trajectories,
    out_dir="../WYKRESY_MAGISTERKA/FINAL_DYNAMIC/montecarlo_dynamic_best",
)

dynamic_trajectories_plot (
    exp_cnn_equal,
    exp_cnn_trajectories,
    out_dir="../WYKRESY_MAGISTERKA/FINAL_DYNAMIC/cnn_dynamic_best",
)

dynamic_trajectories_plot (
    exp_vecmaxdiv_equal,
    exp_vecmaxdiv_trajectories,
    out_dir="../WYKRESY_MAGISTERKA/FINAL_DYNAMIC/vecmaxdiv_dynamic_best",
)

dynamic_trajectories_plot (
    exp_collatz_equal,
    exp_collatz_trajectories,
    out_dir="../WYKRESY_MAGISTERKA/FINAL_DYNAMIC/collatz_dynamic_best",
)

dynamic_trajectories_plot (
    exp_rnn_equal,
    exp_rnn_trajectories,
    out_dir="../WYKRESY_MAGISTERKA/FINAL_DYNAMIC/rnn_dynamic_best",
)

'''



exp_cnn_equal = ExperimentResult.from_file('/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/20_new_equal_cnn_powercap_8_nodes.json')
exp_cnn_best = ExperimentResult.from_file('/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/20_new_best_dynamic_cnn_powercap_8_nodes.json')


exp_rnn_equal = ExperimentResult.from_file('/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/20_new_equal_rnn_powercap_8_nodes.json')
exp_rnn_best = ExperimentResult.from_file('/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/20_new_best_dynamic_rnn_powercap_8_nodes.json')



dynamic_trajectories_plot (
    exp_cnn_equal,
    exp_cnn_best,
    out_dir="new_cnn_dynamic_best",
)

dynamic_trajectories_plot (
    exp_rnn_equal,
    exp_rnn_best,
    out_dir="new_rnn_dynamic_best",
)