import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from models import ExperimentResult, MultipleRunResult, RunParameters
import numpy as np
from collections import defaultdict
from pathlib import Path
import statistics
from matplotlib.colors import to_rgb


MARKER_SIZE = 3
CAPSIZE = 3     # length of horizontal line of std deviation
CAPTHICK = 1    # thickness of horizontal line of std deviation
ELINEWIDTH = 1  # thickness of vertical line of std deviation

class ContinousPowerCappingResult:
    def __init__(self, run: MultipleRunResult):
        self.cpu_min_powercap = run.parameters.cpu_min_powercap
        self.gpu_min_powercap = run.parameters.gpu_min_powercap
        self.execution_duration = run.average("execution_duration")
        self.energy_used = run.average("energy_used")
        self.edp = self.execution_duration * self.energy_used 

def min_powercap_heatmap_cpu_gpu(experiment_result: ExperimentResult, out_dir = "plots"):
    """
    Create heatmaps per (powercap, cpu_time_window_us) combination.
    For each distinct pair in experiment_result.experiment_result:
      - gather results (converted to ContinousPowerCappingResult)
      - create heatmaps for energy_used, edp, execution_duration
      - annotate every grid cell with its exact value
      - highlight only the single best (minimum) value with a red dot
      - save as '<out_dir>/powercap_<PC>_cpu_time_window_us_<CTW>_<METRIC>.png'
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    app_name = "Application"
    if experiment_result.experiment_result:
        app_name = getattr(
            experiment_result.experiment_result[0].parameters,
            "app_name",
            app_name,
        )

    # Collect (powercap, cpu_time_window_us) pairs
    pc_ctw_pairs = [
        (r.parameters.powercap, r.parameters.cpu_time_window_us)
        for r in experiment_result.experiment_result
    ]

    # Helper for safe value formatting for filenames
    def fmt_val(val):
        try:
            if val is None:
                return "none"
            if isinstance(val, (int, np.integer)) or (isinstance(val, float) and float(val).is_integer()):
                return f"{int(val)}"
            return f"{val}".replace(".", "_").replace(" ", "_")
        except Exception:
            return str(val).replace(" ", "_")

    # Print best configuration (min EDP, min Energy) for each powercap across all cpu_time_window_us and min powercaps
    from math import isnan
    by_powercap = defaultdict(list)
    for r in experiment_result.experiment_result:
        pc = r.parameters.powercap
        if pc is None:
            continue
        avg_time = r.average("execution_duration")
        avg_energy = r.average("energy_used")
        try:
            edp_val = float(avg_time) * float(avg_energy)
            energy_val = float(avg_energy)
        except Exception:
            edp_val = float('nan')
            energy_val = float('nan')
        by_powercap[pc].append({
            "cpu_time_window_us": getattr(r.parameters, "cpu_time_window_us", None),
            "cpu_min_powercap": getattr(r.parameters, "cpu_min_powercap", None),
            "gpu_min_powercap": getattr(r.parameters, "gpu_min_powercap", None),
            "edp": edp_val,
            "energy": energy_val,
        })

    for pc in sorted(by_powercap.keys()):
        cands = [c for c in by_powercap[pc] if not isnan(c["edp"]) and not isnan(c["energy"])]
        if not cands:
            continue
        best_edp = min(cands, key=lambda c: c["edp"]) if cands else None
        best_energy = min(cands, key=lambda c: c["energy"]) if cands else None
        if best_edp:
            print(
                f"powercap={pc:<4} | min EDP = {best_edp['edp']:.2f}: (cpu_time_window_us={best_edp['cpu_time_window_us']}, cpu_min_powercap={best_edp['cpu_min_powercap']}, gpu_min_powercap={best_edp['gpu_min_powercap']})"
            )
        if best_energy:
            print(
                f"powercap={pc:<4} | min Energy = {best_energy['energy']:.2f}: (cpu_time_window_us={best_energy['cpu_time_window_us']}, cpu_min_powercap={best_energy['cpu_min_powercap']}, gpu_min_powercap={best_energy['gpu_min_powercap']})"
            )

    for powercap, cpu_time_window_us in sorted(set(pc_ctw_pairs)):
        results = [
            ContinousPowerCappingResult(r)
            for r in experiment_result.experiment_result
            if r.parameters.powercap == powercap
            and getattr(r.parameters, "cpu_time_window_us", None) == cpu_time_window_us
        ]
        
        # Extract rows (cpu, gpu, metrics)
        rows = []
        for res in results:
            rows.append({
                "cpu_min_powercap": getattr(res, "cpu_min_powercap"),
                "gpu_min_powercap": getattr(res, "gpu_min_powercap"),
                "energy_used": getattr(res, "energy_used"),
                "edp": getattr(res, "edp"),
                "execution_duration": getattr(res, "execution_duration"),
            })

        # Aggregate duplicates by (gpu, cpu) -> mean of metrics
        bucket = defaultdict(lambda: defaultdict(list))
        for row in rows:
            key = (row["gpu_min_powercap"], row["cpu_min_powercap"])
            for metric in ("energy_used", "edp", "execution_duration"):
                bucket[key][metric].append(row[metric])

        grid_means = {
            key: {m: float(np.mean(vals)) if len(vals) else np.nan
                  for m, vals in metric_map.items()}
            for key, metric_map in bucket.items()
        }

        # Sorted axes
        cpu_vals = sorted({cpu for (_, cpu) in grid_means.keys()})
        gpu_vals = sorted({gpu for (gpu, _) in grid_means.keys()})
        if not cpu_vals or not gpu_vals:
            continue

        # helper: make 2D matrix
        def to_grid(metric: str):
            grid = np.full((len(gpu_vals), len(cpu_vals)), np.nan, dtype=float)
            for i, g in enumerate(gpu_vals):
                for j, c in enumerate(cpu_vals):
                    val = grid_means.get((g, c), {}).get(metric, np.nan)
                    grid[i, j] = val
            return grid

        metrics = ("energy_used", "edp", "execution_duration")

        def fmt_powercap(val):
            # Backward-compatible alias that uses fmt_val
            return fmt_val(val)

        pc_str = fmt_val(powercap)
        ctw_str = fmt_val(cpu_time_window_us)

        for metric in metrics:
            grid = to_grid(metric)

            # Create figure
            plt.figure(figsize=(6, 4))
            im = plt.imshow(grid, aspect="auto", origin="upper")
            plt.title(
                f"{app_name}: {metric.replace('_', ' ').title()} @ powercap {powercap}"
            )
            plt.xlabel("CPU Min Powercap", fontsize=12)
            plt.ylabel("GPU Min Powercap", fontsize=12)
            plt.xticks(range(len(cpu_vals)), cpu_vals, rotation=45, ha="right", fontsize=12)
            plt.yticks(range(len(gpu_vals)), gpu_vals, fontsize=12)
            cbar = plt.colorbar(im)
            cbar.ax.tick_params(labelsize=12)

            # ---- Annotate all cells with exact values ----
            nrows, ncols = grid.shape
            for i in range(nrows):
                for j in range(ncols):
                    val = grid[i, j]
                    label = "-" if np.isnan(val) else f"{val:.2f}"
                    # Keep text above any markers for readability
                    plt.text(j, i, label, ha="center", va="center", color="black", fontsize=12, zorder=3)

            # ---- Highlight only the single best (minimum) value ----
            flat = [
                (grid[i, j], i, j)
                for i in range(nrows)
                for j in range(ncols)
                if not np.isnan(grid[i, j])
            ]
            if flat:
                best_val, bi, bj = min(flat, key=lambda x: x[0])
                # Red filled dot on the best cell
                plt.scatter(bj, bi, s=700, color='red', marker='o', zorder=2)

            # Save
            filename = (
                Path(out_dir)
                / f"powercap_{pc_str}_cpu_time_window_us_{ctw_str}_{metric}.png"
            )
            plt.tight_layout()
            plt.savefig(filename, dpi=800)
            plt.close()


def min_powercap_heatmap_gpu(experiment_result: ExperimentResult, out_dir: str = "plots"):
    """
    Create heatmaps over (powercap, gpu_min_powercap) pairs.
    For each metric (energy_used, edp, execution_duration):
      - aggregate duplicates by mean
      - annotate each cell with its value
      - highlight the single best (minimum) with a red dot
      - save as '<out_dir>/gpu_powercap_gpu_min_powercap_<METRIC>.png'
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    app_name = "Application"
    if experiment_result.experiment_result:
        app_name = getattr(
            experiment_result.experiment_result[0].parameters,
            "app_name",
            app_name,
        )

    rows = []
    for r in experiment_result.experiment_result:
        pc = getattr(r.parameters, "powercap", None)
        gmin = getattr(r.parameters, "gpu_min_powercap", None)
        if pc is None or gmin is None:
            continue
        avg_time = r.average("execution_duration")
        avg_energy = r.average("energy_used")
        rows.append({
            "powercap": pc,
            "gpu_min_powercap": gmin,
            "energy_used": float(avg_energy),
            "execution_duration": float(avg_time),
            "edp": float(avg_time) * float(avg_energy),
        })

    if not rows:
        return

    # Aggregate duplicates by (powercap, gpu_min_powercap) -> mean of metrics
    bucket = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["powercap"], row["gpu_min_powercap"])  # (pc, gmin)
        for metric in ("energy_used", "edp", "execution_duration"):
            bucket[key][metric].append(row[metric])

    grid_means = {
        key: {m: float(np.mean(vals)) if len(vals) else np.nan
              for m, vals in metric_map.items()}
        for key, metric_map in bucket.items()
    }

    # Sorted axes
    pc_vals = sorted({pc for (pc, _) in grid_means.keys()})
    gmin_vals = sorted({gmin for (_, gmin) in grid_means.keys()})
    if not pc_vals or not gmin_vals:
        return

    # Print best configuration (min EDP, min Energy) per powercap over gpu_min_powercap
    from math import isnan
    for pc in pc_vals:
        # Collect candidates for this powercap across all gpu_min_powercap values
        cands = []
        for g in gmin_vals:
            vals = grid_means.get((pc, g), {})
            edp_val = vals.get("edp", np.nan)
            energy_val = vals.get("energy_used", np.nan)
            if not (isinstance(edp_val, float) and isnan(edp_val)) and not (isinstance(energy_val, float) and isnan(energy_val)):
                cands.append({"gpu_min_powercap": g, "edp": edp_val, "energy": energy_val})
        if not cands:
            continue
        # Best by EDP and by Energy
        best_edp = min([c for c in cands if not isnan(c["edp"])], key=lambda c: c["edp"], default=None)
        best_energy = min([c for c in cands if not isnan(c["energy"])], key=lambda c: c["energy"], default=None)
        if best_edp is not None:
            print(f"powercap={pc:<4} | min EDP = {best_edp['edp']:.2f}: (gpu_min_powercap={best_edp['gpu_min_powercap']})")
        if best_energy is not None:
            print(f"powercap={pc:<4} | min Energy = {best_energy['energy']:.2f}: (gpu_min_powercap={best_energy['gpu_min_powercap']})")

    # helper: make 2D matrix with y=gmin, x=pc
    def to_grid(metric: str):
        grid = np.full((len(gmin_vals), len(pc_vals)), np.nan, dtype=float)
        for i, g in enumerate(gmin_vals):
            for j, p in enumerate(pc_vals):
                val = grid_means.get((p, g), {}).get(metric, np.nan)
                grid[i, j] = val
        return grid

    metrics = ("energy_used", "edp", "execution_duration")

    for metric in metrics:
        grid = to_grid(metric)

        plt.figure(figsize=(7, 4))
        im = plt.imshow(grid, aspect="auto", origin="upper")
        plt.title(f"{app_name}: {metric.replace('_', ' ').title()} vs Powercap and GPU Min PC")
        plt.xlabel("Powercap", fontsize=12)
        plt.ylabel("GPU Min Powercap", fontsize=12)
        plt.xticks(range(len(pc_vals)), pc_vals, rotation=45, ha="right", fontsize=12)
        plt.yticks(range(len(gmin_vals)), gmin_vals, fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=12)

        # Annotate cells
        nrows, ncols = grid.shape
        for i in range(nrows):
            for j in range(ncols):
                val = grid[i, j]
                label = "-" if np.isnan(val) else f"{val:.2f}"
                plt.text(j, i, label, ha="center", va="center", color="black", fontsize=12, zorder=3)

        # Highlight the single best (minimum)
        flat = [
            (grid[i, j], i, j)
            for i in range(nrows)
            for j in range(ncols)
            if not np.isnan(grid[i, j])
        ]
        if flat:
            _, bi, bj = min(flat, key=lambda x: x[0])
            plt.scatter(bj, bi, s=700, color='red', marker='o', zorder=2)

        filename = Path(out_dir) / f"gpu_powercap_gpu_min_powercap_{metric}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=800)
        plt.close()


def min_powercap_heatmap_cpu(experiment_result: ExperimentResult, out_dir: str = "plots"):
    """
    Create heatmaps over (powercap, cpu_min_powercap) pairs.
    For each metric (energy_used, edp, execution_duration):
      - aggregate duplicates by mean
      - annotate each cell with its value
      - highlight the single best (minimum) with a red dot
      - save as '<out_dir>/cpu_powercap_cpu_min_powercap_<METRIC>.png'
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    
    # Collect all rows (filter to entries that have both powercap and gpu_min_powercap)
    app_name = "Application"
    if experiment_result.experiment_result:
        app_name = getattr(
            experiment_result.experiment_result[0].parameters,
            "app_name",
            app_name,
        )

    rows = []
    for r in experiment_result.experiment_result:
        pc = getattr(r.parameters, "powercap", None)
        cmin = getattr(r.parameters, "cpu_min_powercap", None)
        if pc is None or cmin is None:
            continue
        avg_time = r.average("execution_duration")
        avg_energy = r.average("energy_used")
        rows.append({
            "powercap": pc,
            "cpu_min_powercap": cmin,
            "energy_used": float(avg_energy),
            "execution_duration": float(avg_time),
            "edp": float(avg_time) * float(avg_energy),
        })

    if not rows:
        return

    bucket = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["powercap"], row["cpu_min_powercap"])
        for metric in ("energy_used", "edp", "execution_duration"):
            bucket[key][metric].append(row[metric])

    grid_means = {
        key: {m: float(np.mean(vals)) if len(vals) else np.nan
              for m, vals in metric_map.items()}
        for key, metric_map in bucket.items()
    }

    pc_vals = sorted({pc for (pc, _) in grid_means.keys()})
    cmin_vals = sorted({cmin for (_, cmin) in grid_means.keys()})
    if not pc_vals or not cmin_vals:
        return

    from math import isnan
    for pc in pc_vals:
        cands = []
        for c in cmin_vals:
            vals = grid_means.get((pc, c), {})
            edp_val = vals.get("edp", np.nan)
            energy_val = vals.get("energy_used", np.nan)
            if not (isinstance(edp_val, float) and isnan(edp_val)) and not (isinstance(energy_val, float) and isnan(energy_val)):
                cands.append({"cpu_min_powercap": c, "edp": edp_val, "energy": energy_val})
        if not cands:
            continue
        best_edp = min([c for c in cands if not isnan(c["edp"])], key=lambda c: c["edp"], default=None)
        best_energy = min([c for c in cands if not isnan(c["energy"])], key=lambda c: c["energy"], default=None)
        if best_edp is not None:
            print(f"powercap={pc:<4} | min EDP = {best_edp['edp']:.2f}: (cpu_min_powercap={best_edp['cpu_min_powercap']})")
        if best_energy is not None:
            print(f"powercap={pc:<4} | min Energy = {best_energy['energy']:.2f}: (cpu_min_powercap={best_energy['cpu_min_powercap']})")

    def to_grid(metric: str):
        grid = np.full((len(cmin_vals), len(pc_vals)), np.nan, dtype=float)
        for i, c in enumerate(cmin_vals):
            for j, p in enumerate(pc_vals):
                val = grid_means.get((p, c), {}).get(metric, np.nan)
                grid[i, j] = val
        return grid

    metrics = ("energy_used", "edp", "execution_duration")

    for metric in metrics:
        grid = to_grid(metric)

        plt.figure(figsize=(7, 4))
        im = plt.imshow(grid, aspect="auto", origin="upper")
        plt.title(f"{app_name}: {metric.replace('_', ' ').title()} vs Powercap and CPU Min PC")
        plt.xlabel("Powercap", fontsize=12)
        plt.ylabel("CPU Min Powercap", fontsize=12)
        plt.xticks(range(len(pc_vals)), pc_vals, rotation=45, ha="right", fontsize=12)
        plt.yticks(range(len(cmin_vals)), cmin_vals, fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=12)

        nrows, ncols = grid.shape
        for i in range(nrows):
            for j in range(ncols):
                val = grid[i, j]
                label = "-" if np.isnan(val) else f"{val:.2f}"
                plt.text(j, i, label, ha="center", va="center", color="black", fontsize=12, zorder=3)

        flat = [
            (grid[i, j], i, j)
            for i in range(nrows)
            for j in range(ncols)
            if not np.isnan(grid[i, j])
        ]
        if flat:
            _, bi, bj = min(flat, key=lambda x: x[0])
            plt.scatter(bj, bi, s=700, color='red', marker='o', zorder=2)

        filename = Path(out_dir) / f"cpu_powercap_cpu_min_powercap_{metric}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=800)
        plt.close()


def optimal_configuration_metric_trends(
    continuous_results: ExperimentResult,
    binary_results: ExperimentResult | None = None,
    out_dir: str = "plots",
):
    """Plot execution time, energy, and EDP for best-EDP configs per powercap.

    If binary_results is provided, the averaged binary metrics (excluding powercap=0)
    are shown as an additional line on each plot.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    runs = continuous_results.experiment_result
    if not runs:
        return

    app_name = getattr(runs[0].parameters, "app_name", "Application")

    exclude_fields = {
        "app_name",
        "cpu_enabled",
        "number_of_streams",
        "number_od_nodes",
        "batch_size",
        "powercap",
        "strategy",
    }

    field_values = defaultdict(set)
    for multi in runs:
        params = vars(multi.parameters)
        for name, value in params.items():
            if name in exclude_fields:
                continue
            field_values[name].add(value)

    varying_fields = sorted(name for name, values in field_values.items() if len(values) > 1)

    def make_config_key(params: RunParameters):
        if not varying_fields:
            return ("__default__", "__default__")
        return tuple((field, getattr(params, field, None)) for field in varying_fields)

    def fmt_param_value(val):
        if isinstance(val, float):
            if np.isnan(val):
                return "nan"
            return (f"{val:.4f}".rstrip("0").rstrip(".") or "0")
        return str(val)

    def config_label(config_key):
        if config_key == ("__default__", "__default__"):
            return "default"
        parts = []
        for field, value in config_key:
            parts.append(f"{field}={fmt_param_value(value)}")
        return ", ".join(parts)

    metrics_by_pc_and_cfg = defaultdict(lambda: defaultdict(list))
    metadata_by_pc_cfg: dict[tuple[float, tuple], RunParameters] = {}
    for multi in runs:
        pc = getattr(multi.parameters, "powercap", None)
        if pc is None:
            continue
        cfg_key = make_config_key(multi.parameters)
        try:
            avg_time = float(multi.average("execution_duration"))
            avg_energy = float(multi.average("energy_used"))
        except (TypeError, ValueError, statistics.StatisticsError):
            print(
                f"[optimal_configuration_metric_trends] Unable to derive averages for powercap={pc}; skipping configuration"
            )
            continue
        edp_val = avg_time * avg_energy
        metrics_by_pc_and_cfg[(pc, cfg_key)]["execution_duration"].append(avg_time)
        metrics_by_pc_and_cfg[(pc, cfg_key)]["energy_used"].append(avg_energy)
        metrics_by_pc_and_cfg[(pc, cfg_key)]["edp"].append(edp_val)
        metadata_by_pc_cfg.setdefault((pc, cfg_key), multi.parameters)

    if not metrics_by_pc_and_cfg:
        return

    aggregated = {}
    for key, metric_lists in metrics_by_pc_and_cfg.items():
        aggregated[key] = {
            metric: float(np.mean(values)) if values else np.nan
            for metric, values in metric_lists.items()
        }

    def friendly_strategy(params: RunParameters | None) -> str:
        raw = getattr(params, "strategy", None)
        if not raw:
            return "Continuous"
        return str(raw).replace("_", " ").title()

    def describe_config(pc: float, cfg_key) -> str:
        params = metadata_by_pc_cfg.get((pc, cfg_key))
        strategy_name = friendly_strategy(params)
        label = config_label(cfg_key)
        if label == "default":
            return f"strategy={strategy_name}"
        return f"strategy={strategy_name}, params={label}"

    def fmt_metric(value: float | None) -> str:
        try:
            if value is None or np.isnan(value):
                return "nan"
            return f"{value:.4f}"
        except TypeError:
            return "nan"

    per_powercap = defaultdict(list)
    for (pc, cfg_key), metric_map in aggregated.items():
        per_powercap[pc].append((cfg_key, metric_map))

    best_per_powercap = {}
    configs_to_plot = set()
    tol = 1e-9
    for pc, candidates in per_powercap.items():
        valid = [c for c in candidates if not np.isnan(c[1].get("edp", np.nan))]
        if not valid:
            continue
        best_edp = min(c[1]["edp"] for c in valid)
        best_entries = sorted(
            [
                (cfg, metrics) for cfg, metrics in valid
                if abs(metrics["edp"] - best_edp) <= tol
            ],
            key=lambda item: config_label(item[0]),
        )
        best_per_powercap[pc] = best_entries
        for cfg_key, metrics in best_entries:
            configs_to_plot.add(cfg_key)
            print(
                f"[optimal_configuration_metric_trends] powercap={pc} best config {config_label(cfg_key)} -> EDP={metrics['edp']:.4f}"
            )

    if not configs_to_plot:
        return

    series_by_config = defaultdict(lambda: defaultdict(list))
    for (pc, cfg_key), metric_map in aggregated.items():
        if cfg_key not in configs_to_plot:
            continue
        series_by_config[cfg_key]["powercap"].append(pc)
        for metric, value in metric_map.items():
            series_by_config[cfg_key][metric].append(value)

    binary_series: dict[str, list[tuple[float, float]]] | None = None
    binary_means_by_pc: dict[float, dict[str, float]] = {}
    binary_cfg_by_pc: dict[float, tuple] = {}
    binary_params_by_pc: dict[float, RunParameters] = {}
    if binary_results is not None:
        binary_metrics_by_pc = defaultdict(lambda: defaultdict(list))
        for multi in binary_results.experiment_result:
            pc = getattr(multi.parameters, "powercap", None)
            if pc is None or pc == 0:
                continue
            try:
                avg_time = float(multi.average("execution_duration"))
                avg_energy = float(multi.average("energy_used"))
            except (TypeError, ValueError, statistics.StatisticsError):
                print(
                    f"[optimal_configuration_metric_trends] Unable to derive binary averages for powercap={pc}; skipping configuration"
                )
                continue
            binary_metrics_by_pc[pc]["execution_duration"].append(avg_time)
            binary_metrics_by_pc[pc]["energy_used"].append(avg_energy)
            binary_metrics_by_pc[pc]["edp"].append(avg_time * avg_energy)
            cfg_key = make_config_key(multi.parameters)
            binary_cfg_by_pc.setdefault(pc, cfg_key)
            binary_params_by_pc.setdefault(pc, multi.parameters)
            metadata_by_pc_cfg.setdefault((pc, cfg_key), multi.parameters)

        if binary_metrics_by_pc:
            binary_series = {}
            for pc, metrics in binary_metrics_by_pc.items():
                means = {}
                for metric in ("execution_duration", "energy_used", "edp"):
                    values = metrics.get(metric)
                    if values:
                        means[metric] = float(np.mean(values))
                if means:
                    binary_means_by_pc[pc] = means

            if binary_means_by_pc:
                for metric in ("execution_duration", "energy_used", "edp"):
                    points = []
                    for pc in sorted(binary_means_by_pc.keys()):
                        val = binary_means_by_pc[pc].get(metric)
                        if val is None or np.isnan(val):
                            continue
                        points.append((pc, val))
                    if points:
                        binary_series[metric] = points

    # ---- overall best including binary ----
    best_overall_edp: tuple[float, float, dict[str, float]] | None = None
    best_overall_energy: tuple[float, float, dict[str, float]] | None = None
    best_edp_desc = ""
    best_energy_desc = ""

    def update_best(pc: float, metrics_map: dict[str, float], desc: str):
        nonlocal best_overall_edp, best_overall_energy, best_edp_desc, best_energy_desc
        edp_val = metrics_map.get("edp")
        if edp_val is not None and not np.isnan(edp_val):
            if best_overall_edp is None or edp_val < best_overall_edp[0]:
                best_overall_edp = (edp_val, pc, metrics_map)
                best_edp_desc = desc
        energy_val = metrics_map.get("energy_used")
        if energy_val is not None and not np.isnan(energy_val):
            if best_overall_energy is None or energy_val < best_overall_energy[0]:
                best_overall_energy = (energy_val, pc, metrics_map)
                best_energy_desc = desc

    for (pc, cfg_key), metric_map in aggregated.items():
        update_best(pc, metric_map, describe_config(pc, cfg_key))

    for pc, metrics_map in binary_means_by_pc.items():
        cfg_key = binary_cfg_by_pc.get(pc)
        if cfg_key is not None:
            desc = describe_config(pc, cfg_key)
        else:
            desc = f"strategy={friendly_strategy(binary_params_by_pc.get(pc))}"
        desc += " (Binary Greedy)"
        update_best(pc, metrics_map, desc)

    if best_overall_edp:
        edp_val, pc_val, metrics_map = best_overall_edp
        print(
            f"[{app_name}] Overall min EDP: powercap={pc_val} | {best_edp_desc} | "
            f"EDP={fmt_metric(edp_val)}, Energy={fmt_metric(metrics_map.get('energy_used'))}, "
            f"Time={fmt_metric(metrics_map.get('execution_duration'))}"
        )

    if best_overall_energy:
        energy_val, pc_val, metrics_map = best_overall_energy
        print(
            f"[{app_name}] Overall min Energy: powercap={pc_val} | {best_energy_desc} | "
            f"Energy={fmt_metric(energy_val)}, Time={fmt_metric(metrics_map.get('execution_duration'))}"
        )

    metrics_to_plot = (
        ("execution_duration", "Execution Duration"),
        ("energy_used", "Energy Used"),
        ("edp", "Energy-Delay Product"),
    )

    for metric, ylabel in metrics_to_plot:
        plt.figure(figsize=(7, 4))
        has_data = False
        continuous_lines = []
        for cfg_key in sorted(series_by_config.keys(), key=config_label):
            data = series_by_config[cfg_key]
            pcs = data.get("powercap")
            values = data.get(metric)
            if not pcs or not values:
                continue
            combined = [pair for pair in sorted(zip(pcs, values), key=lambda pair: pair[0]) if not np.isnan(pair[1])]
            if not combined:
                continue
            pcs_sorted = [c[0] for c in combined]
            values_sorted = [c[1] for c in combined]
            legend_label = config_label(cfg_key)
            if legend_label == "default":
                legend_label = "Continuous Greedy"
            else:
                legend_label = f"Continuous Greedy: {legend_label}"

            line, = plt.plot(
                pcs_sorted,
                values_sorted,
                marker="o",
                label=legend_label,
            )
            continuous_lines.append((line, pcs_sorted, values_sorted))
            has_data = True

        if binary_series and metric in binary_series:
            pcs_sorted = [pair[0] for pair in binary_series[metric]]
            values_sorted = [pair[1] for pair in binary_series[metric]]
            filtered = [pair for pair in zip(pcs_sorted, values_sorted) if not np.isnan(pair[1])]
            if filtered:
                pcs_sorted = [pair[0] for pair in filtered]
                values_sorted = [pair[1] for pair in filtered]
                line, = plt.plot(
                    pcs_sorted,
                    values_sorted,
                    marker="s",
                    linestyle="--",
                    label="Binary Greedy",
                    color="black",
                )
                continuous_lines.append((line, pcs_sorted, values_sorted))
                has_data = True

        if not has_data:
            plt.close()
            continue

        global_min = None
        for line, pcs_sorted, values_sorted in continuous_lines:
            for x, y in zip(pcs_sorted, values_sorted):
                if np.isnan(y):
                    continue
                if global_min is None or y < global_min[2]:
                    global_min = (line, x, y)

        if global_min is not None:
            line, min_x, min_y = global_min
            line_color = line.get_color()
            plt.scatter(min_x, min_y, color=line_color, marker="s", s=70, zorder=5)
            plt.annotate(
                f"{min_y:.3f}",
                xy=(min_x, min_y),
                xytext=(6, -12),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=line_color, lw=0.9, alpha=0.95),
            )

        plt.xlabel("Powercap")
        plt.ylabel(ylabel)
        plt.title(
            f"{app_name}: Optimal Configurations vs Powercap ({metric.replace('_', ' ').title()})"
        )
        plt.legend(fontsize=9)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        filename = Path(out_dir) / f"optimal_configurations_{metric}.png"
        plt.savefig(filename, dpi=800)
        plt.close()


def time_powercap_scatter(experiment_result: ExperimentResult):
    # Extracting data
    power_cpu_gpu = [multiple_run_result.parameters.powercap for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    execution_duration_cpu_gpu_avg = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    std_deviation_cpu_gpu = [multiple_run_result.standard_deviation("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    # execution_duration_cpu_gpu_min = [multiple_run_result.min("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    # execution_duration_cpu_gpu_max = [multiple_run_result.max("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]

    power_gpu = [multiple_run_result.parameters.powercap for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]
    execution_duration_gpu_avg = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]
    std_deviation_gpu = [multiple_run_result.standard_deviation("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]
    # execution_duration_gpu_min = [multiple_run_result.min("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]
    # execution_duration_gpu_max = [multiple_run_result.max("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]

    # Compute differences for the line plot
    execution_duration_diff = [gpu / cpu for cpu, gpu in zip(execution_duration_cpu_gpu_avg, execution_duration_gpu_avg)]

    # Create figure and grid spec for stacked plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # --- Scatter and Line Plot (Main Plot) ---
    ax1.errorbar(power_cpu_gpu, execution_duration_cpu_gpu_avg, label="CPU+GPU avg", marker='^', color='blue', yerr=std_deviation_cpu_gpu, capsize=CAPSIZE, capthick=CAPTHICK, markersize=MARKER_SIZE, elinewidth=ELINEWIDTH)
    # ax1.scatter(power_cpu_gpu, execution_duration_cpu_gpu_min, label="CPU+GPU min", marker='^', color='red')
    # ax1.scatter(power_cpu_gpu, execution_duration_cpu_gpu_max, label="CPU+GPU max", marker='^', color='orange')

    ax1.errorbar(power_gpu, execution_duration_gpu_avg, label="GPU avg", marker='o', color='green', yerr=std_deviation_gpu, capsize=CAPSIZE, capthick=CAPTHICK, markersize=MARKER_SIZE, elinewidth=ELINEWIDTH)
    # ax1.scatter(power_gpu, execution_duration_gpu_min, label="GPU min", marker='o', color='purple')
    # ax1.scatter(power_gpu, execution_duration_gpu_max, label="GPU max", marker='o', color='brown')

    ax1.set_ylabel("Execution Duration [s]", fontsize=16)  # Increased font size
    ax1.legend(fontsize=16, ncol=2)  # Increased legend font size
    ax1.grid(True)
    plt.xticks(fontsize=14)  # Increased font size for x-axis ticks
    plt.yticks(fontsize=14)  # Increased font size for y-axis ticks

    # --- Line Plot for Execution Time Difference ---
    ax2.plot(power_cpu_gpu, execution_duration_diff, label="CPU+GPU speedup vs GPU", marker='s', linestyle='-', color='black')
    print(f'Avg speedup for {experiment_result.experiment_result[0].parameters.app_name} : {np.mean(execution_duration_diff)}')
    ax2.axhline(1, color='black', linewidth=0.8, linestyle="--")  # Reference line at y=0
    ax2.set_ylabel("Speedup", fontsize=16)  # Increased font size
    ax2.legend(fontsize=16, ncol=2)  # Increased legend font size
    ax2.grid(True)

    # Save the figure
    plt.xlabel("Powercap", fontsize=16)  # Increased font size
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    plt.savefig(f'{experiment_result.experiment_result[0].parameters.app_name}_time_power_cap_nodes_{experiment_result.experiment_result[0].parameters.number_od_nodes}.png')
    plt.close()


def print_avg_edp_energy_per_configuration(experiment_result: ExperimentResult):
    for r in experiment_result.experiment_result:
        avg_time = r.average("execution_duration")
        avg_energy = r.average("energy_used")
        edp = avg_time * avg_energy

        print(f"cpu_enabled={r.parameters.cpu_enabled} powercap={r.parameters.powercap:<4} | avg_energy={avg_energy:.2f}, avg_edp={edp:.2f}")


def time_batch_size_scatter(experiment_result: ExperimentResult):
    batch_size_cpu_gpu = [multiple_run_result.parameters.batch_size for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    execution_duration_cpu_gpu = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    std_deviation_cpu_gpu = [multiple_run_result.standard_deviation("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]

    batch_size_gpu = [multiple_run_result.parameters.batch_size for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]
    execution_duration_gpu = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]
    std_deviation_gpu = [multiple_run_result.standard_deviation("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]

    plt.figure()
    plt.errorbar(batch_size_cpu_gpu, execution_duration_cpu_gpu, yerr=std_deviation_cpu_gpu, label="CPU+GPU", capsize=CAPSIZE, capthick=CAPTHICK, marker='o', markersize=MARKER_SIZE, elinewidth=ELINEWIDTH)
    plt.errorbar(batch_size_gpu, execution_duration_gpu, yerr=std_deviation_gpu, label="GPU", capsize=CAPSIZE, capthick=CAPTHICK, marker='o', markersize=MARKER_SIZE, elinewidth=ELINEWIDTH)
    
    plt.xscale('log')
    plt.xlabel("Batch Size", fontsize=16)
    plt.ylabel("Time [s]", fontsize=16)
    
    plt.legend(fontsize=16)

    # Ensure y-tick labels are displayed as floats (e.g., 25.0 instead of 25)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Increase bottom and right margins by 10%
    if (execution_duration_gpu[1] > 1000):
        plt.subplots_adjust(bottom=0.15, left=0.2, right=0.95, top=0.95)  # Increase margins dynamically
    else:
        plt.subplots_adjust(bottom=0.15, left=0.15, right=0.95, top=0.95)  # Increase margins dynamically

    # Save figure
    plt.savefig(f'{experiment_result.experiment_result[0].parameters.app_name}_time_batch_size_nodes_{experiment_result.experiment_result[0].parameters.number_od_nodes}.png')
    plt.close()


def time_number_of_nodes_bar(experiment_result: ExperimentResult):
    number_of_nodes = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result]
    # "CPU+GPU one stream", "CPU+GPU two streams", "GPU one stream", "GPU two streams"
    configuration = [f'{"CPU+" if multiple_run_result.parameters.cpu_enabled else ""}GPU {"one stream" if multiple_run_result.parameters.number_of_streams == 1 else "two streams"}' for multiple_run_result in experiment_result.experiment_result]
    execution_duration = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result]
    std_deviation = [multiple_run_result.standard_deviation("execution_duration") for multiple_run_result in experiment_result.experiment_result]

    data = {
        'Number of Nodes': number_of_nodes,
        'Configuration': configuration,
        'Time (s)': execution_duration,
        'Std Deviation': std_deviation,
    }

    df = pd.DataFrame(data)
    # pivot the data so that each "Number of nodes" has its own column for each configuration
    df_pivot = df.pivot(index='Number of Nodes', columns='Configuration', values='Time (s)')
    std_pivot = df.pivot(index='Number of Nodes', columns='Configuration', values='Std Deviation')

    df_pivot.plot.bar(
        width=0.8,
        figsize=(10, 6),
        yerr=std_pivot,
        capsize=CAPSIZE
    )
    plt.xlabel('Number of nodes', fontsize=18)
    plt.ylabel('Time [s]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)
    plt.subplots_adjust(right=0.95, top=0.95)  # Increase margins dynamically
    plt.savefig(f'{experiment_result.experiment_result[0].parameters.app_name}_time_nodes.png')
    # Calculate average speedup for "CPU+GPU" vs "GPU"
    cpu_gpu_configurations = [col for col in df_pivot.columns if col.startswith("CPU+")]
    gpu_configurations = [col for col in df_pivot.columns if col.startswith("GPU") and not col.startswith("CPU+")]

    if cpu_gpu_configurations and gpu_configurations:
        avg_speedups = []
        for node in df_pivot.index:
            cpu_gpu_times = df_pivot.loc[node, cpu_gpu_configurations].mean()  # Average time for CPU+GPU
            gpu_times = df_pivot.loc[node, gpu_configurations].mean()  # Average time for GPU-only
            if gpu_times > 0:
                avg_speedup = gpu_times / cpu_gpu_times
                avg_speedups.append(avg_speedup)
        
        overall_avg_speedup = sum(avg_speedups) / len(avg_speedups) if avg_speedups else None
        print(f"Average speedup of CPU+GPU vs GPU-only: {overall_avg_speedup:.2f}")

def time_number_of_nodes_scatter(experiment_result: ExperimentResult):
    nodes_cpu_gpu_one_stream = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 1]
    execution_duration_cpu_gpu_one_stream = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 1] 
    nodes_cpu_gpu_two_streams = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 2]
    execution_duration_cpu_gpu_two_streams = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 2]
    
    nodes_gpu_one_stream = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 1]
    execution_duration_gpu_one_stream = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 1]
    nodes_gpu_two_streams = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 2]
    execution_duration_gpu_two_streams = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 2]
    
    plt.figure()
    plt.plot(nodes_cpu_gpu_one_stream, execution_duration_cpu_gpu_one_stream, label="CPU+GPU 1 stream", marker='o')
    plt.plot(nodes_cpu_gpu_two_streams, execution_duration_cpu_gpu_two_streams, label="CPU+GPU 2 streams", marker='o')
    plt.plot(nodes_gpu_one_stream, execution_duration_gpu_one_stream, label="GPU 1 stream", marker='o')
    plt.plot(nodes_gpu_two_streams, execution_duration_gpu_two_streams, label="GPU 2 streams", marker='o')
    plt.xlabel("number of nodes")
    plt.ylabel("time [s]")
    plt.legend()
    plt.savefig(f'{experiment_result.experiment_result[0].parameters.app_name}_time_nodes_nodes_{experiment_result.experiment_result[0].parameters.number_od_nodes}.png')
    plt.close()


def equal_split_start_powercap_plot(
    experiment_result: ExperimentResult,
    out_dir: str = "plots_equal_split",
):
    """
    Plot metrics vs start_powercap for EQUAL_SPLIT runs.

    X axis: start_powercap (0..1).
    Creates three PNGs in `out_dir` for metrics: energy_used, execution_duration, edp.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Gather rows that have start_powercap defined (EQUAL_SPLIT runs)
    rows = []
    for r in experiment_result.experiment_result:
        sp = getattr(r.parameters, "start_powercap", None)
        if sp is None:
            continue
        avg_time = float(r.average("execution_duration"))
        avg_energy = float(r.average("energy_used"))
        edp_val = avg_time * avg_energy
        rows.append({
            "start_powercap": float(sp),
            "nodes": int(getattr(r.parameters, "number_od_nodes", 1)),
            "energy_used": avg_energy,
            "execution_duration": avg_time,
            "edp": edp_val,
        })

    if not rows:
        return

    # Sort by start_powercap for nice lines/ordering
    rows.sort(key=lambda d: d["start_powercap"])

    xs = [d["start_powercap"] for d in rows]

    app_name = getattr(experiment_result.experiment_result[0].parameters, "app_name", "app")
    try:
        nodes_common = {d["nodes"] for d in rows}
        nodes_str = f"{next(iter(nodes_common))}nodes" if len(nodes_common) == 1 else "varnodes"
    except Exception:
        nodes_str = "nodes"

    print(f"EDP breakdown for {app_name} ({nodes_str}):")
    for row in rows:
        print(
            "  start_powercap={sp:.2f} | time={time:.3f}s | energy={energy:.3f}J | EDP={edp:.3f}".format(
                sp=row["start_powercap"],
                time=row["execution_duration"],
                energy=row["energy_used"],
                edp=row["edp"],
            )
        )

    # Prepare plotting helper for each metric
    def plot_metric(metric_key: str, ylabel: str, fname_suffix: str):
        ys = [d[metric_key] for d in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, marker='o', markersize=MARKER_SIZE)
        ax.set_xlabel("start_powercap")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric_key.replace('_', ' ').title()} vs start_powercap ({app_name}, {nodes_str})")
        ax.grid(True, linestyle='--', alpha=0.3)

        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:.2f}" for x in xs])

        plt.tight_layout()
        out_path = Path(out_dir) / f"{app_name}_equal_split_{nodes_str}_start_pc_{fname_suffix}.png"
        plt.savefig(out_path, dpi=800)
        plt.close(fig)

    # Generate plots for energy, time, and EDP
    plot_metric("energy_used", "Energy used [J]", "energy_used")
    plot_metric("execution_duration", "Time [s]", "time")
    plot_metric("edp", "EDP [J*s]", "edp")


def equal_split_dynamic_annotations_plot(
    equal_split_results: ExperimentResult,
    dynamic_best_results: ExperimentResult,
    out_dir: str = "plots_equal_dynamic",
):
    """Overlay best dynamic strategies on top of equal-split baseline plots."""

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def nice_strategy(name: str) -> str:
        return name.replace("_", " ").title()

    dynamic_param_fields = [
        "start_alpha",
        "alpha_decay",
        "epsilon_decay",
        "gradient_opt_eps",
        "edp_optimization_steps",
        "cpu_min_powercap",
        "gpu_min_powercap",
        "cpu_time_window_us",
    ]

    
    def fmt_value(val: float) -> str:
        if val == 0:
            return "0"
        abs_val = abs(val)
        if abs_val >= 1e5 or abs_val < 1e-2:
            return f"{val:.2e}"
        return f"{val:.3f}"

    def build_param_label(strategy_name: str, params: RunParameters) -> str:
        parts: list[str] = []
        for field in dynamic_param_fields:
            value = getattr(params, field, None)
            if value is None:
                continue
            parts.append(f"{field}={fmt_value(value)}")
        if not parts:
            return "default"
        return ", ".join(parts)

    def collect_equal_rows(exp: ExperimentResult) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for result in exp.experiment_result:
            sp = getattr(result.parameters, "start_powercap", None)
            if sp is None:
                continue
            durations = [run.execution_duration for run in result.runs]
            energies = [run.energy_used for run in result.runs]
            if not durations or not energies:
                continue
            avg_time = float(np.mean(durations))
            avg_energy = float(np.mean(energies))
            edp_samples = [d * e for d, e in zip(durations, energies)]
            avg_edp = float(np.mean(edp_samples))

            def stddev(values: list[float]) -> float:
                if len(values) <= 1:
                    return 0.0
                return float(np.std(values, ddof=1))

            rows.append({
                "start_powercap": float(sp),
                "execution_duration": avg_time,
                "energy_used": avg_energy,
                "edp": avg_edp,
                "execution_duration_std": stddev(durations),
                "energy_used_std": stddev(energies),
                "edp_std": stddev(edp_samples),
            })
        return rows

    equal_rows = collect_equal_rows(equal_split_results)
    if not equal_rows:
        return

    equal_rows.sort(key=lambda row: row["start_powercap"])
    xs = [row["start_powercap"] for row in equal_rows]

    app_name = getattr(equal_split_results.experiment_result[0].parameters, "app_name", "app")
    try:
        nodes_set = {
            int(getattr(result.parameters, "number_od_nodes", 0))
            for result in equal_split_results.experiment_result
            if getattr(result.parameters, "start_powercap", None) is not None
        }
        nodes_label = (
            f"{next(iter(nodes_set))}nodes" if len(nodes_set) == 1 and nodes_set else "nodes"
        )
    except Exception:
        nodes_label = "nodes"

    def collect_dynamic_points(exp: ExperimentResult) -> dict[str, dict[float, dict[str, float]]]:
        best: dict[str, dict[float, dict[str, float]]] = {}
        for result in exp.experiment_result:
            strategy = getattr(result.parameters, "strategy", "UNKNOWN") or "UNKNOWN"
            sp = getattr(result.parameters, "start_powercap", None)
            if sp is None:
                continue
            durations = [run.execution_duration for run in result.runs]
            energies = [run.energy_used for run in result.runs]
            if not durations or not energies:
                continue
            edp_samples = [d * e for d, e in zip(durations, energies)]

            def stddev(values: list[float]) -> float:
                if len(values) <= 1:
                    return 0.0
                return float(np.std(values, ddof=1))

            avg_time = float(np.mean(durations))
            avg_energy = float(np.mean(energies))
            avg_edp = float(np.mean(edp_samples))

            param_label = build_param_label(strategy, result.parameters)

            candidate = {
                "start_powercap": float(sp),
                "execution_duration": avg_time,
                "energy_used": avg_energy,
                "edp": avg_edp,
                "execution_duration_std": stddev(durations),
                "energy_used_std": stddev(energies),
                "edp_std": stddev(edp_samples),
                "param_label": param_label,
            }
            strategy_map = best.setdefault(strategy, {})
            current = strategy_map.get(float(sp))
            if current is None or candidate["edp"] < current["edp"]:
                strategy_map[float(sp)] = candidate
        return best

    dynamic_points = collect_dynamic_points(dynamic_best_results)

    equal_edp_by_sp = {row['start_powercap']: row['edp'] for row in equal_rows}
    min_equal_edp = min(equal_edp_by_sp.values()) if equal_edp_by_sp else None

    if dynamic_points:
        print(f"Dynamic best summary for {app_name} ({nodes_label}):")
        for strategy in sorted(dynamic_points.keys()):
            print(f"Strategy: {strategy}")
            for sp in sorted(dynamic_points[strategy].keys()):
                values = dynamic_points[strategy][sp]
                edp = values['edp']
                edp_std = values['edp_std']
                cfg = values.get('param_label', 'default')
                equal_edp = equal_edp_by_sp.get(sp)
                diff_equal = edp - equal_edp if equal_edp is not None else None
                diff_min = edp - min_equal_edp if min_equal_edp is not None else None
                print(
                    f"  start_pc={sp:.2f} | EDP={edp:.3f} ± {edp_std:.3f} | config: {cfg}"
                )
                if equal_edp is not None:
                    print(
                        f"      Delta vs equal @ {sp:.2f} = {diff_equal:+.3f} (equal EDP={equal_edp:.3f})"
                    )
                else:
                    print("      Delta vs equal @ start_pc: N/A (no equal data)")
                if diff_min is not None:
                    print(
                        f"      Delta vs equal min = {diff_min:+.3f} (min equal EDP={min_equal_edp:.3f})"
                    )


    metrics = (
        ("energy_used", "Energy Used", "energy_used", "Energy used"),
        ("execution_duration", "Execution Duration", "time", "Time"),
        ("edp", "Energy-Delay Product", "edp", "EDP"),
    )

    label_mapping = {
        "EDP_GRADIENT_CMAES": "CMA-ES",
        "EDP_GRADIENT_SPSA": "Gradient SPSA",
        "EDP_GRADIENT_SIMPLE": "Gradient Simple",
    }

    marker_options = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
    strategy_list = sorted(dynamic_points.keys())
    strategy_markers = {
        strategy: marker_options[idx % len(marker_options)]
        for idx, strategy in enumerate(strategy_list)
    }

    cmap = plt.get_cmap('tab10')
    strategy_colors = {}
    for idx, strategy in enumerate(strategy_list):
        if strategy == "EDP_GRADIENT_CMAES":
            strategy_colors[strategy] = (0.8, 0.2, 0.2)
        else:
            strategy_colors[strategy] = cmap(idx % cmap.N)

    for metric_key, ylabel, suffix, metric_title in metrics:
        ys = [row[metric_key] for row in equal_rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        if metric_key == "energy_used":
            errs = [row["energy_used_std"] for row in equal_rows]
        elif metric_key == "execution_duration":
            errs = [row["execution_duration_std"] for row in equal_rows]
        else:
            errs = [row["edp_std"] for row in equal_rows]

        ax.errorbar(
            xs,
            ys,
            yerr=errs,
            fmt='o-',
            markersize=MARKER_SIZE,
            color='tab:blue',
            ecolor='tab:blue',
            capsize=4,
            label="Equal Split",
        )

        if dynamic_points:
            seen: set[str] = set()
            for strategy in strategy_list:
                strategy_values = dynamic_points[strategy]
                for sp in sorted(strategy_values.keys()):
                    values = strategy_values[sp]
                    y_val = values.get(metric_key)
                    if y_val is None:
                        continue
                    if metric_key == "energy_used":
                        err = values.get("energy_used_std", 0.0)
                    elif metric_key == "execution_duration":
                        err = values.get("execution_duration_std", 0.0)
                    else:
                        err = values.get("edp_std", 0.0)

                    label = None
                    if strategy not in seen:
                        label = label_mapping.get(strategy, nice_strategy(strategy))
                        seen.add(strategy)

                    ax.errorbar(
                        sp,
                        y_val,
                        yerr=err,
                        fmt=strategy_markers[strategy],
                        markersize=7,
                        color=strategy_colors[strategy],
                        ecolor=strategy_colors[strategy],
                        capsize=4,
                        linestyle='None',
                        elinewidth=1,
                        markeredgecolor='black',
                        markeredgewidth=0.6,
                        label=label,
                        zorder=4,
                    )

        tick_positions = (
            sorted({*xs, *[sp for strategy_values in dynamic_points.values() for sp in strategy_values.keys()]})
            if dynamic_points
            else xs
        )
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{tick:.2f}" for tick in tick_positions], fontsize=9)
        ax.tick_params(axis='y', labelsize=9)

        ax.set_xlabel("start_powercap", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(
            f"{app_name}: Dynamic Optimization vs Equal Split on {metric_title} (Start Power Cap)",
            fontsize=10,
        )
        ax.grid(True, linestyle='--', alpha=0.3)

        ax.legend(fontsize=9, loc='best')

        candidates: list[tuple[float, float, str]] = []
        for x_val, y_val in zip(xs, ys):
            if not np.isnan(y_val):
                candidates.append((y_val, x_val, "Equal Split"))
        for strategy in strategy_list:
            for sp, values in dynamic_points[strategy].items():
                y_val = values.get(metric_key)
                if y_val is None or np.isnan(y_val):
                    continue
                label = label_mapping.get(strategy, strategy)
                candidates.append((y_val, sp, label))
        if candidates:
            best_val, best_x, best_label = min(candidates, key=lambda t: t[0])
            ax.annotate(
                f"{best_label}: {best_val:.3f}",
                xy=(best_x, best_val),
                xytext=(6, -12),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc='white', ec='red', lw=0.9, alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8),
            )

        plt.tight_layout()
        out_path = Path(out_dir) / f"{app_name}_equal_split_dynamic_{nodes_label}_{suffix}.png"
        plt.savefig(out_path, dpi=800)
        plt.close(fig)


def dynamic_search_trajectories_plot(
    equal_split_results: ExperimentResult,
    dynamic_search_results: ExperimentResult | None,
    dynamic_best_results: ExperimentResult | None,
    out_dir: str = "plots_dynamic_search",
):
    """Plot dynamic search trajectories for configurations that win at least once."""

    if (
        dynamic_search_results is None
        or not dynamic_search_results.experiment_result
        or dynamic_best_results is None
        or not dynamic_best_results.experiment_result
    ):
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def nice_strategy(name: str) -> str:
        return name.replace("_", " ").title()

    dynamic_param_fields = [
        "start_alpha",
        "alpha_decay",
        "epsilon_decay",
        "gradient_opt_eps",
        "edp_optimization_steps",
        "cpu_min_powercap",
        "gpu_min_powercap",
        "cpu_time_window_us",
    ]

    def fmt_value(val: float) -> str:
        if val == 0:
            return "0"
        abs_val = abs(val)
        if abs_val >= 1e5 or abs_val < 1e-2:
            return f"{val:.2e}"
        return f"{val:.3f}"

    def build_param_label(strategy_name: str, params: RunParameters) -> str:
        parts: list[str] = []
        for field in dynamic_param_fields:
            value = getattr(params, field, None)
            if value is None:
                continue
            parts.append(f"{field}={fmt_value(value)}")
        if not parts:
            return "default"
        return ", ".join(parts)

    def collect_equal_rows(exp: ExperimentResult) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for result in exp.experiment_result:
            sp = getattr(result.parameters, "start_powercap", None)
            if sp is None:
                continue
            durations = [run.execution_duration for run in result.runs]
            energies = [run.energy_used for run in result.runs]
            if not durations or not energies:
                continue
            avg_time = float(np.mean(durations))
            avg_energy = float(np.mean(energies))
            edp_samples = [d * e for d, e in zip(durations, energies)]
            avg_edp = float(np.mean(edp_samples))

            def stddev(values: list[float]) -> float:
                if len(values) <= 1:
                    return 0.0
                return float(np.std(values, ddof=1))

            rows.append({
                "start_powercap": float(sp),
                "execution_duration": avg_time,
                "energy_used": avg_energy,
                "edp": avg_edp,
                "execution_duration_std": stddev(durations),
                "energy_used_std": stddev(energies),
                "edp_std": stddev(edp_samples),
            })
        return rows

    equal_rows = collect_equal_rows(equal_split_results)
    if not equal_rows:
        return

    equal_rows.sort(key=lambda row: row["start_powercap"])
    xs = [row["start_powercap"] for row in equal_rows]

    app_name = getattr(equal_split_results.experiment_result[0].parameters, "app_name", "app")
    try:
        nodes_set = {
            int(getattr(result.parameters, "number_od_nodes", 0))
            for result in equal_split_results.experiment_result
            if getattr(result.parameters, "start_powercap", None) is not None
        }
        nodes_label = (
            f"{next(iter(nodes_set))}nodes" if len(nodes_set) == 1 and nodes_set else "nodes"
        )
    except Exception:
        nodes_label = "nodes"

    best_configs: dict[str, set[str]] = defaultdict(set)
    for result in dynamic_best_results.experiment_result:
        strategy = getattr(result.parameters, "strategy", "UNKNOWN") or "UNKNOWN"
        param_label = build_param_label(strategy, result.parameters)
        best_configs[strategy].add(param_label)

    def collect_search_configs(exp: ExperimentResult) -> dict[str, dict[str, list[dict[str, float]]]]:
        grouped: dict[str, dict[str, list[dict[str, float]]]] = {}
        for result in exp.experiment_result:
            strategy = getattr(result.parameters, "strategy", "UNKNOWN") or "UNKNOWN"
            sp = getattr(result.parameters, "start_powercap", None)
            if sp is None:
                continue

            durations = [run.execution_duration for run in result.runs]
            energies = [run.energy_used for run in result.runs]
            if not durations or not energies:
                continue

            edp_samples = [d * e for d, e in zip(durations, energies)]

            def stddev(values: list[float]) -> float:
                if len(values) <= 1:
                    return 0.0
                return float(np.std(values, ddof=1))

            avg_time = float(np.mean(durations))
            avg_energy = float(np.mean(energies))
            avg_edp = float(np.mean(edp_samples))

            param_label = build_param_label(strategy, result.parameters)

            if param_label not in best_configs.get(strategy, set()):
                continue

            entry = {
                "start_powercap": float(sp),
                "execution_duration": avg_time,
                "energy_used": avg_energy,
                "edp": avg_edp,
                "execution_duration_std": stddev(durations),
                "energy_used_std": stddev(energies),
                "edp_std": stddev(edp_samples),
                "param_label": param_label,
            }

            strategy_map = grouped.setdefault(strategy, {})
            strategy_map.setdefault(param_label, []).append(entry)
        for strategy_map in grouped.values():
            for entries in strategy_map.values():
                entries.sort(key=lambda item: item["start_powercap"])
        return grouped

    search_configs = collect_search_configs(dynamic_search_results)
    if not search_configs:
        return

    metrics = (
        ("energy_used", "Energy Used", "energy_used", "Energy used"),
        ("execution_duration", "Execution Duration", "time", "Time"),
        ("edp", "Energy-Delay Product", "edp", "EDP"),
    )

    label_mapping = {
        "EDP_GRADIENT_CMAES": "CMA-ES",
        "EDP_GRADIENT_SPSA": "Gradient SPSA",
        "EDP_GRADIENT_SIMPLE": "Gradient Simple",
    }

    cmap = plt.get_cmap('tab10')
    strategy_colors = {}
    strategy_list = sorted(search_configs.keys())
    base_palette = {
        "EDP_GRADIENT_CMAES": (0.8, 0.2, 0.2),
        "EDP_GRADIENT_SIMPLE": to_rgb("#1f77b4"),  # blue
        "EDP_GRADIENT_SPSA": to_rgb("#2ca02c"),   # green
    }
    for idx, strategy in enumerate(strategy_list):
        color = base_palette.get(strategy)
        if color is None:
            color = cmap(idx % cmap.N)
        strategy_colors[strategy] = color

    def lighten_color(color: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
        r, g, b = to_rgb(color)
        return tuple(min(1.0, c + (1.0 - c) * amount) for c in (r, g, b))

    search_powercaps = {
        entry["start_powercap"]
        for strategy_map in search_configs.values()
        for config_entries in strategy_map.values()
        for entry in config_entries
    }

    for metric_key, ylabel, suffix, metric_title in metrics:
        ys = [row[metric_key] for row in equal_rows]
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            xs,
            ys,
            linestyle='-',
            color='black',
            linewidth=2.0,
            marker='o',
            markersize=MARKER_SIZE,
            label="Equal Split",
        )

        for strategy in strategy_list:
            configs = search_configs[strategy]
            friendly_name = label_mapping.get(strategy, nice_strategy(strategy))
            if not configs:
                continue

            config_items = sorted(configs.items())
            shade_levels = (
                np.linspace(0.0, 0.3, num=len(config_items), endpoint=True)
                if len(config_items) > 1
                else [0.0]
            )
            line_styles = ['--', ':', '-.']
            for idx, ((_, entries), shade) in enumerate(zip(config_items, shade_levels), start=1):
                color = lighten_color(strategy_colors[strategy], shade)
                line_style = line_styles[(idx - 1) % len(line_styles)]
                x_vals = [entry["start_powercap"] for entry in entries if entry.get(metric_key) is not None]
                y_vals = [entry[metric_key] for entry in entries if entry.get(metric_key) is not None]
                if not x_vals or not y_vals:
                    continue

                legend_label = f"{friendly_name} #{idx}"

                ax.plot(
                    x_vals,
                    y_vals,
                    linestyle=line_style,
                    linewidth=1.2,
                    marker='o',
                    markersize=MARKER_SIZE + 1,
                    color=color,
                    label=legend_label,
                )

        tick_positions = sorted({*xs, *search_powercaps})
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{tick:.2f}" for tick in tick_positions], fontsize=9)
        ax.tick_params(axis='y', labelsize=9)

        ax.set_xlabel("start_powercap", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(
            f"{app_name}: Dynamic Search Trajectories on {metric_title} (Start Power Cap)",
            fontsize=10,
        )
        ax.grid(True, linestyle='--', alpha=0.3)

        ax.legend(fontsize=8, loc='best')

        plt.tight_layout()
        out_path = Path(out_dir) / f"{app_name}_dynamic_search_{nodes_label}_{suffix}.png"
        plt.savefig(out_path, dpi=800)
        plt.close(fig)


def dynamic_search_best_configurations_plot(
    equal_split_results: ExperimentResult,
    dynamic_search_results: ExperimentResult | None,
    out_dir: str = "plots_dynamic_search_best",
) -> None:
    """Plot best-per-strategy configurations plus the overall best against the baseline."""

    if (
        dynamic_search_results is None
        or not dynamic_search_results.experiment_result
        or equal_split_results is None
        or not equal_split_results.experiment_result
    ):
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def nice_strategy(name: str) -> str:
        return name.replace("_", " ").title()

    dynamic_param_fields = [
        "start_alpha",
        "alpha_decay",
        "epsilon_decay",
        "gradient_opt_eps",
        "edp_optimization_steps",
        "cpu_min_powercap",
        "gpu_min_powercap",
        "cpu_time_window_us",
    ]

    def fmt_value(val: float) -> str:
        if val == 0:
            return "0"
        abs_val = abs(val)
        if abs_val >= 1e5 or abs_val < 1e-2:
            return f"{val:.2e}"
        return f"{val:.3f}"

    def build_param_label(strategy_name: str, params: RunParameters) -> str:
        parts: list[str] = []
        for field in dynamic_param_fields:
            value = getattr(params, field, None)
            if value is None:
                continue
            parts.append(f"{field}={fmt_value(value)}")
        if not parts:
            return "default"
        return ", ".join(parts)

    def collect_equal_rows(exp: ExperimentResult) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for result in exp.experiment_result:
            sp = getattr(result.parameters, "start_powercap", None)
            if sp is None:
                continue
            durations = [run.execution_duration for run in result.runs]
            energies = [run.energy_used for run in result.runs]
            if not durations or not energies:
                continue
            edp_samples = [d * e for d, e in zip(durations, energies)]
            if not edp_samples:
                continue

            def stddev(values: list[float]) -> float:
                if len(values) <= 1:
                    return 0.0
                return float(np.std(values, ddof=1))

            rows.append({
                "start_powercap": float(sp),
                "edp": float(np.mean(edp_samples)),
                "edp_std": stddev(edp_samples),
            })
        return rows

    equal_rows = collect_equal_rows(equal_split_results)
    if not equal_rows:
        return

    equal_rows.sort(key=lambda row: row["start_powercap"])
    baseline_edp: dict[float, float] = {}
    for row in equal_rows:
        sp_val = round(row["start_powercap"], 6)
        baseline_edp[sp_val] = row["edp"]
    baseline_powercaps = sorted(baseline_edp.keys())

    app_name = getattr(equal_split_results.experiment_result[0].parameters, "app_name", "app")
    try:
        nodes_set = {
            int(getattr(result.parameters, "number_od_nodes", 0))
            for result in equal_split_results.experiment_result
            if getattr(result.parameters, "start_powercap", None) is not None
        }
        nodes_label = (
            f"{next(iter(nodes_set))}nodes" if len(nodes_set) == 1 and nodes_set else "nodes"
        )
    except Exception:
        nodes_label = "nodes"

    strategy_configs: dict[str, dict[str, dict[float, dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))
    for result in dynamic_search_results.experiment_result:
        strategy = getattr(result.parameters, "strategy", "UNKNOWN") or "UNKNOWN"
        sp = getattr(result.parameters, "start_powercap", None)
        if sp is None:
            continue

        durations = [run.execution_duration for run in result.runs]
        energies = [run.energy_used for run in result.runs]
        if not durations or not energies:
            continue

        edp_samples = [d * e for d, e in zip(durations, energies)]
        if not edp_samples:
            continue

        param_label = build_param_label(strategy, result.parameters)
        sp_val = round(float(sp), 6)

        def average(values: list[float]) -> float:
            return float(np.mean(values))

        strategy_configs[strategy][param_label][sp_val] = {
            "start_powercap": sp_val,
            "edp": average(edp_samples),
            "edp_std": float(np.std(edp_samples, ddof=1)) if len(edp_samples) > 1 else 0.0,
        }

    if not strategy_configs:
        return

    label_mapping = {
        "EDP_GRADIENT_CMAES": "CMA-ES",
        "EDP_GRADIENT_SPSA": "Gradient SPSA",
        "EDP_GRADIENT_SIMPLE": "Gradient Simple",
    }

    best_per_strategy: dict[str, tuple[str, list[tuple[float, float]], float]] = {}
    best_overall: tuple[str, str, list[tuple[float, float]], float] | None = None

    for strategy, config_map in strategy_configs.items():
        best_config_label: str | None = None
        best_config_points: list[tuple[float, float]] | None = None
        best_avg_improvement = float("-inf")
        best_avg_edp = float("inf")

        for param_label, entries_by_sp in config_map.items():
            common_powercaps = sorted(set(entries_by_sp.keys()) & set(baseline_powercaps))
            if not common_powercaps:
                continue

            points = [(sp, entries_by_sp[sp]["edp"]) for sp in common_powercaps]

            edps = [entries_by_sp[sp]["edp"] for sp in common_powercaps]
            improvements = [baseline_edp[sp] - entries_by_sp[sp]["edp"] for sp in common_powercaps]

            avg_improvement = float(np.mean(improvements))
            avg_edp = float(np.mean(edps))

            if avg_improvement > best_avg_improvement:
                best_avg_improvement = avg_improvement
                best_avg_edp = avg_edp
                best_config_label = param_label
                best_config_points = points

            if best_overall is None or avg_edp < best_overall[3]:
                best_overall = (strategy, param_label, points, avg_edp)

        if best_config_label and best_config_points is not None:
            best_per_strategy[strategy] = (
                best_config_label,
                best_config_points,
                best_avg_edp,
            )

    if not best_per_strategy:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        baseline_powercaps,
        [baseline_edp[sp] for sp in baseline_powercaps],
        linestyle='-',
        color='black',
        linewidth=2.0,
        marker='o',
        markersize=MARKER_SIZE,
        label="Equal Split",
    )

    cmap = plt.get_cmap('tab10')
    for idx, (strategy, (_, points, _)) in enumerate(sorted(best_per_strategy.items())):
        friendly_name = label_mapping.get(strategy, nice_strategy(strategy))
        color = cmap(idx % cmap.N)
        x_vals = [sp for sp, _ in points]
        y_vals = [edp for _, edp in points]

        ax.plot(
            x_vals,
            y_vals,
            linestyle='--',
            linewidth=1.5,
            marker='s',
            markersize=MARKER_SIZE + 1,
            color=color,
            label=f"{friendly_name} best avg ΔEDP",
        )

    if best_overall is not None:
        overall_strategy, _, overall_points, _ = best_overall
        overall_x = [sp for sp, _ in overall_points]
        overall_y = [edp for _, edp in overall_points]
        overall_label = label_mapping.get(overall_strategy, nice_strategy(overall_strategy))

        ax.plot(
            overall_x,
            overall_y,
            linestyle='-',
            linewidth=2.0,
            marker='D',
            markersize=MARKER_SIZE + 1,
            color='tab:red',
            label=f"Overall best ({overall_label})",
        )

    ax.set_xticks(baseline_powercaps)
    ax.set_xticklabels([f"{tick:.2f}" for tick in baseline_powercaps], fontsize=9)
    ax.tick_params(axis='y', labelsize=9)

    ax.set_xlabel("start_powercap", fontsize=9)
    ax.set_ylabel("EDP", fontsize=9)
    ax.set_title(
        f"{app_name}: Best Dynamic Trajectories vs Equal Split (EDP)",
        fontsize=10,
    )
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    out_path = Path(out_dir) / f"{app_name}_dynamic_search_best_{nodes_label}_edp.png"
    plt.savefig(out_path, dpi=800)
    plt.close(fig)
