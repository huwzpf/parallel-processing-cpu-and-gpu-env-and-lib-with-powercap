import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from models import ExperimentResult, MultipleRunResult
import numpy as np
from collections import defaultdict
from pathlib import Path


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

def min_powercap_heatmap(experiment_result: ExperimentResult):
    """
    For each distinct 'parameters.powercap' in experiment_result.experiment_result:
      - gather results (converted to ContinousPowerCappingResult)
      - create heatmaps for energy_used, edp, execution_duration
      - highlight 3 smallest values
      - save each heatmap as '<out_dir>/powercap_<VALUE>_<METRIC>.png'
    """
    out_dir = "plots"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    powercaps = [mr.parameters.powercap for mr in experiment_result.experiment_result]
    
    for powercap in set(powercaps):
        results = [
            ContinousPowerCappingResult(r)
            for r in experiment_result.experiment_result
            if r.parameters.powercap == powercap
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
            try:
                if isinstance(val, (int, np.integer)) or (isinstance(val, float) and val.is_integer()):
                    return f"{int(val)}"
                return f"{val}".replace(".", "_")
            except Exception:
                return str(val).replace(" ", "_")

        pc_str = fmt_powercap(powercap)

        for metric in metrics:
            grid = to_grid(metric)

            # Create figure
            plt.figure(figsize=(6, 4))
            im = plt.imshow(grid, aspect="auto", origin="upper")
            plt.title(f"{metric.replace('_', ' ').title()} @ powercap {powercap}")
            plt.xlabel("CPU Min Powercap")
            plt.ylabel("GPU Min Powercap")
            plt.xticks(range(len(cpu_vals)), cpu_vals, rotation=45, ha="right")
            plt.yticks(range(len(gpu_vals)), gpu_vals)
            plt.colorbar(im)

            # ---- Highlight 3 smallest values ----
            flat = [(grid[i, j], i, j)
                    for i in range(len(gpu_vals))
                    for j in range(len(cpu_vals))
                    if not np.isnan(grid[i, j])]
            flat.sort(key=lambda x: x[0])
            for val, i, j in flat[:3]:
                plt.scatter(j, i, s=120, facecolors='none', edgecolors='red', linewidths=2)
                plt.text(j, i, f"{val:.2f}",
                         ha="center", va="center",
                         color="white", fontsize=5, weight="bold")

            # Save
            filename = Path(out_dir) / f"powercap_{pc_str}_{metric}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=200)
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
