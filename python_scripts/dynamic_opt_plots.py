from models import ExperimentResult
from charts import equal_split_dynamic_annotations_plot


twinprime_equal_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/DYNAMIC/equal_twinprime_powercap_8_nodes.json"
twinprime_dynamic_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/DYNAMIC/best_dynamic_twinprime_powercap_8_nodes.json"
twinprime_raw_path = \
     "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/test_twinprime_powercap_8_nodes.json"

montecarlo_equal_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/DYNAMIC/equal_montecarlo_powercap_8_nodes.json"
montecarlo_dynamic_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/DYNAMIC/best_dynamic_montecarlo_powercap_8_nodes.json"
montecarlo_raw_path = \
     "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/test_montecarlo_powercap_8_nodes.json"

cnn_equal_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/" \
    "WYKRESY_MAGISTERKA/DYNAMIC/equal_cnn_powercap_8_nodes.json"
cnn_dynamic_path = \
    "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/best_dynamic_cnn_powercap_8_nodes.json"
cnn_raw_path = \
     "/home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/test_cnn_powercap_8_nodes.json"

exp_twinprime_raw = ExperimentResult.from_file(twinprime_raw_path)
twinprime_defaults = exp_twinprime_raw.default_metrics()
exp_twinprime_equal = ExperimentResult.from_file(twinprime_equal_path).normalize(twinprime_defaults)
exp_twinprime_dynamic = ExperimentResult.from_file(twinprime_dynamic_path).normalize(twinprime_defaults)

exp_montecarlo_raw = ExperimentResult.from_file(montecarlo_raw_path)
montecarlo_defaults = exp_montecarlo_raw.default_metrics()
exp_montecarlo_equal = ExperimentResult.from_file(montecarlo_equal_path).normalize(montecarlo_defaults)
exp_montecarlo_dynamic = ExperimentResult.from_file(montecarlo_dynamic_path).normalize(montecarlo_defaults)

exp_cnn_raw = ExperimentResult.from_file(cnn_raw_path)
cnn_defaults = exp_cnn_raw.default_metrics()
exp_cnn_equal = ExperimentResult.from_file(cnn_equal_path).normalize(cnn_defaults)
exp_cnn_dynamic = ExperimentResult.from_file(cnn_dynamic_path).normalize(cnn_defaults)


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
