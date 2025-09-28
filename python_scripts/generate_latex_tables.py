"""Generate LaTeX table summarising dynamic search results."""

from __future__ import annotations

from pathlib import Path
import json
from statistics import mean

from models import ExperimentResult
import math
import re

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = Path(__file__).with_name("search_results.txt")
MAX_ROWS_PER_TABLE = 30

DYNAMIC_SEARCH_FILES = {
    "cnn": BASE_DIR / "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/dynamic_cnn_powercap_8_nodes.json",
    "montecarlo": BASE_DIR / "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/dynamic_montecarlo_powercap_8_nodes.json",
    "twinprime": BASE_DIR / "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/dynamic_twinprime_powercap_8_nodes.json",
}

RAW_BASELINE_FILES = {
    "cnn": BASE_DIR / "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/test_cnn_powercap_8_nodes.json",
    "montecarlo": BASE_DIR / "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/test_montecarlo_powercap_8_nodes.json",
    "twinprime": BASE_DIR / "WYKRESY_MAGISTERKA/FINAL_DYNAMIC/test_twinprime_powercap_8_nodes.json",
}

BASELINE_DEFAULTS: dict[str, dict[str, float | None] | None] = {}
for label, raw_path in RAW_BASELINE_FILES.items():
    if raw_path.exists():
        try:
            raw_exp = ExperimentResult.from_file(raw_path)
            BASELINE_DEFAULTS[label] = raw_exp.default_metrics()
        except Exception as exc:
            print(f"[generate_latex_tables] Failed to load baseline from {raw_path}: {exc}")
            BASELINE_DEFAULTS[label] = None
    else:
        BASELINE_DEFAULTS[label] = None

STRATEGY_DISPLAY = {
    "EDP_GRADIENT_SIMPLE": "Gradient Simple",
    "EDP_GRADIENT_CMAES": "CMA-ES",
    "EDP_GRADIENT_SPSA": "SPSA",
}

SEARCH_PARAM_HEADERS = {
    "EDP_GRADIENT_SIMPLE": [
        "alpha_decay",
        "edp_optimization_steps",
        "epsilon_decay",
        "gradient_opt_eps",
        "start_alpha",
    ],
    "EDP_GRADIENT_SPSA": [
        "alpha_decay",
        "edp_optimization_steps",
        "epsilon_decay",
        "gradient_opt_eps",
        "start_alpha",
    ],
    "EDP_GRADIENT_CMAES": [
        "edp_optimization_steps",
        "gradient_opt_eps",
    ],
}

SEARCH_PARAM_ORDER = [
    "start_powercap",
    "start_alpha",
    "alpha_decay",
    "epsilon_decay",
    "gradient_opt_eps",
    "edp_optimization_steps",
    "cpu_min_powercap",
    "gpu_min_powercap",
    "cpu_time_window_us",
]


def _parse_results(text: str):
    apps_in_order: list[str] = []
    strategies = {
        "Gradient Simple": {},
        "CMA-ES": {},
        "Gradient SPSA": {},
    }
    param_keys: dict[str, set[str]] = {name: set() for name in strategies}
    equal_split: dict[str, float] = {}

    block_pattern = re.compile(
        r"\[dynamic_best\]\s+(?P<app>\w+)[^\n]*\n(?P<body>.*?)(?=\n\[dynamic_best\]|\Z)",
        re.S,
    )

    for match in block_pattern.finditer(text):
        app = match.group("app")
        body = match.group("body")
        if app not in apps_in_order:
            apps_in_order.append(app)

        for line in (ln.strip() for ln in body.splitlines() if ln.strip()):
            if line.startswith("Equal Split best EDP="):
                equal_split[app] = float(line.split("=", 1)[1])
                continue

            if ":" not in line:
                continue

            strategy_name, rest = line.split(":", 1)
            strategy_name = strategy_name.strip()
            if strategy_name not in strategies:
                continue

            param_blob = rest.split("|", 1)[0].strip()
            param_map: dict[str, str] = {}
            for part in param_blob.split(','):
                part = part.strip()
                if '=' not in part:
                    continue
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                param_map[key] = value
                param_keys[strategy_name].add(key)

            metrics = {
                "avg_edp": math.nan,
                "avg_improvement": math.nan,
                "best_edp": math.nan,
                "best_improvement": math.nan,
                "params": param_blob,
                "param_map": param_map,
            }

            # Collect parameter key set for later column layout
            for part in param_blob.split(','):
                part = part.strip()
                if '=' in part:
                    key = part.split('=', 1)[0].strip()
                    param_keys[strategy_name].add(key)

            for part in rest.split("|"):
                part = part.strip()
                if part.startswith("avg EDP="):
                    metrics["avg_edp"] = float(part.split("=", 1)[1])
                elif part.startswith("avg dEDP="):
                    metrics["avg_improvement"] = float(part.split("=", 1)[1].rstrip("%"))
                elif part.startswith("best EDP="):
                    metrics["best_edp"] = float(part.split("=", 1)[1])
                elif part.startswith("best dEDP="):
                    metrics["best_improvement"] = float(part.split("=", 1)[1].rstrip("%"))

            strategies[strategy_name][app] = metrics

    return apps_in_order, strategies, equal_split, param_keys


def _fmt_value(value: float, decimals: int = 4) -> str:
    if value is None or math.isnan(value):
        return "--"
    return f"{value:.{decimals}f}"


def _fmt_percent(value: float) -> str:
    if value is None or math.isnan(value):
        return "--"
    return f"{value:.2f}\\%"


def _mean(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return mean(valid) if valid else math.nan


def _format_param_value(value) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _fmt_value(float(value))
    return str(value).replace("_", r"\_")


def _chunk_list(items: list, limit: int) -> list[list]:
    if limit <= 0 or len(items) <= limit:
        return [items]
    return [items[i : i + limit] for i in range(0, len(items), limit)]


def _continued_caption(caption: str) -> str:
    return caption if caption.strip().endswith("(continued)") else f"{caption} (continued)"


def _load_dynamic_search_data(path: Path, app_label: str):
    entries_by_strategy: dict[str, list[dict[str, object]]] = {}
    keys_by_strategy: dict[str, set[str]] = {}

    exp = ExperimentResult.from_file(path)
    defaults = BASELINE_DEFAULTS.get(app_label)
    if defaults is not None:
        normalized = exp.normalize(defaults)
    else:
        normalized = exp

    for multi in normalized.experiment_result:
        params = multi.parameters.__dict__ if hasattr(multi.parameters, "__dict__") else {}
        strategy = params.get("strategy")
        if not strategy:
            continue

        entry_params: dict[str, object] = {}
        for key in SEARCH_PARAM_ORDER:
            if key in params:
                entry_params[key] = params[key]

        runs = multi.runs
        durations = [getattr(run, "execution_duration", None) for run in runs]
        energies = [getattr(run, "energy_used", None) for run in runs]
        if durations and energies:
            avg_duration = sum(durations) / len(durations)
            avg_energy = sum(energies) / len(energies)
            edp_value = avg_duration * avg_energy
        else:
            edp_value = float("nan")

        allowed_keys = SEARCH_PARAM_HEADERS.get(strategy)
        if allowed_keys is not None:
            filtered_params = {k: entry_params.get(k) for k in allowed_keys}
            key_iterable = allowed_keys
        else:
            filtered_params = entry_params
            key_iterable = entry_params.keys()

        entry = {
            "param_map": filtered_params,
            "edp": edp_value,
        }

        entries_by_strategy.setdefault(strategy, []).append(entry)
        key_set = keys_by_strategy.setdefault(strategy, set())
        for key in key_iterable:
            key_set.add(key)

    for strategy, entries in entries_by_strategy.items():
        entries.sort(key=lambda e: e["param_map"].get("start_powercap", float("inf")))

    return entries_by_strategy, keys_by_strategy


def _build_rows(apps: list[str], strategies: dict, equal_split: dict[str, float]) -> list[tuple[str, str]]:
    header = r'Strategy & Application & Best EDP & Best $dEDP$ & Avg. EDP & Avg. $dEDP$ \\'
    rows: list[tuple[str, str]] = [(r"\hline", header)]

    display_names = {
        "Gradient Simple": "Gradient Simple",
        "CMA-ES": "CMAES",
        "Gradient SPSA": "SPSA",
        "Equal Split": "EQUAL\\_SPLIT",
    }
    strategy_order = ["Gradient Simple", "CMA-ES", "Gradient SPSA", "Equal Split"]

    for strategy in strategy_order:
        metrics_map = strategies.get(strategy, {})
        entries = apps + ["avg"]
        rowspan = len(entries)
        for idx, app in enumerate(entries):
            label = rf"\multirow{{{rowspan}}}{{*}}{{{display_names[strategy]}}}" if idx == 0 else ""

            if strategy == "Equal Split":
                if app == "avg":
                    best_values = [equal_split[a] for a in apps if a in equal_split]
                    best_edp = _mean(best_values)
                else:
                    best_edp = equal_split.get(app, math.nan)
                best_impr = 0.0
                avg_edp = best_edp
                avg_impr = 0.0
            else:
                if app == "avg":
                    best_vals = [metrics_map[a]["best_edp"] for a in apps if a in metrics_map]
                    best_imprs = [metrics_map[a]["best_improvement"] for a in apps if a in metrics_map]
                    avg_vals = [metrics_map[a]["avg_edp"] for a in apps if a in metrics_map]
                    avg_imprs = [metrics_map[a]["avg_improvement"] for a in apps if a in metrics_map]
                    best_edp = _mean(best_vals)
                    best_impr = _mean(best_imprs)
                    avg_edp = _mean(avg_vals)
                    avg_impr = _mean(avg_imprs)
                else:
                    data = metrics_map.get(app, {})
                    best_edp = data.get("best_edp", math.nan)
                    best_impr = data.get("best_improvement", math.nan)
                    avg_edp = data.get("avg_edp", math.nan)
                    avg_impr = data.get("avg_improvement", math.nan)

            if label:
                content = (
                    f"{label} & {app} & {_fmt_value(best_edp)} & {_fmt_percent(best_impr)} "
                    f"& {_fmt_value(avg_edp)} & {_fmt_percent(avg_impr)} "
                    + r"\\"
                )
                prefix = r"\hline\hline"
            else:
                content = (
                    f" & {app} & {_fmt_value(best_edp)} & {_fmt_percent(best_impr)} "
                    f"& {_fmt_value(avg_edp)} & {_fmt_percent(avg_impr)} "
                    + r"\\"
                )
                prefix = r"\cline{2-6}"

            rows.append((prefix, content))

    rows.append((r"\hline", ""))
    return rows


def _build_table(rows: list[tuple[str, str]], caption: str) -> str:
    if not rows:
        return ""

    header = rows[0]
    data_rows = rows[1:]
    footers: list[tuple[str, str]] = []
    while data_rows and not data_rows[-1][1]:
        footers.insert(0, data_rows.pop())

    if not data_rows:
        data_chunks = [[]]
    else:
        data_chunks = _chunk_list(data_rows, MAX_ROWS_PER_TABLE)

    tables: list[str] = []
    for idx, chunk in enumerate(data_chunks):
        current_caption = caption if idx == 0 else _continued_caption(caption)
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            rf"\caption{{{current_caption}}}",
            r"\begin{tabular}{|l|l|r|r|r|r|}",
        ]

        for prefix, content in [header] + chunk + (footers if idx == len(data_chunks) - 1 else []):
            if prefix:
                lines.append(prefix)
            if content:
                lines.append(content)

        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        tables.append("\n".join(lines))

    return "\n\n".join(tables)


def _build_param_table(
    strategy: str,
    apps: list[str],
    strategies: dict,
    param_keys: dict[str, set[str]],
    caption: str,
) -> str:
    data_map = strategies.get(strategy, {})
    keys = sorted(param_keys.get(strategy, []))
    if not keys:
        keys = []

    header_cols = [r"Application"] + [key.replace("_", r"\_") for key in keys]
    header = " & ".join(header_cols) + r" \\"

    column_spec = "|" + "|".join(["l"] + ["c"] * len(keys)) + "|"
    data_rows: list[str] = []
    for app in apps:
        param_map = data_map.get(app, {}).get("param_map", {})
        values = [app]
        for key in keys:
            raw = param_map.get(key, "--")
            values.append(raw.replace("_", r"\_"))
        row = " & ".join(values) + r" \\"
        data_rows.append(row)

    data_chunks = _chunk_list(data_rows, MAX_ROWS_PER_TABLE) if data_rows else [[]]
    tables: list[str] = []
    for idx, chunk in enumerate(data_chunks):
        current_caption = caption if idx == 0 else _continued_caption(caption)
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            rf"\caption{{{current_caption}}}",
            f"\\begin{{tabular}}{{{column_spec}}}",
            r"\hline",
            header,
            r"\hline",
        ]

        for row in chunk:
            lines.append(row)
            lines.append(r"\hline")

        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        tables.append("\n".join(lines))

    return "\n\n".join(tables)


def _build_search_table(
    app_label: str,
    strategy_key: str,
    entries: list[dict[str, object]],
    key_set: set[str],
) -> str | None:
    if not entries:
        return None

    preferred = SEARCH_PARAM_HEADERS.get(strategy_key)
    if preferred is not None:
        ordered_keys = [k for k in preferred if k in key_set]
    else:
        ordered_keys = [k for k in SEARCH_PARAM_ORDER if k in key_set]
        for key in sorted(key_set):
            if key not in ordered_keys:
                ordered_keys.append(key)

    header_cols = [key.replace("_", r"\_") for key in ordered_keys] + ["EDP"]
    column_spec = "|" + "|".join(["c"] * len(header_cols)) + "|"

    data_rows: list[str] = []
    for entry in entries:
        param_map = entry.get("param_map", {})
        row_values: list[str] = []
        for key in ordered_keys:
            row_values.append(_format_param_value(param_map.get(key)))
        row_values.append(_fmt_value(float(entry.get("edp", float("nan")))))
        data_rows.append(" & ".join(row_values) + r" \\")

    data_chunks = _chunk_list(data_rows, MAX_ROWS_PER_TABLE) if data_rows else [[]]
    tables: list[str] = []
    caption_base = f"Dynamic Search Parameters for {STRATEGY_DISPLAY.get(strategy_key, strategy_key)} on {app_label}"

    for idx, chunk in enumerate(data_chunks):
        current_caption = caption_base if idx == 0 else _continued_caption(caption_base)
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            rf"\caption{{{current_caption}}}",
            f"\\begin{{tabular}}{{{column_spec}}}",
            r"\hline",
            " & ".join(header_cols) + r" \\",
            r"\hline",
        ]

        for row in chunk:
            lines.append(row)
            lines.append(r"\hline")

        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        tables.append("\n".join(lines))

    return "\n\n".join(tables)


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Cannot locate results file at {RESULTS_PATH}")

    apps, strategy_data, equal_data, param_keys = _parse_results(RESULTS_PATH.read_text())

    preferred_order = ["collatz", "twinprime", "montecarlo", "vecmaxdiv", "cnn"]
    apps = [app for app in preferred_order if app in apps] + [app for app in apps if app not in preferred_order]

    rows = _build_rows(apps, strategy_data, equal_data)
    table = _build_table(rows, "Best and Average EDP per strategy and application")
    print(table)
    print()

    param_tables = [
        ("Gradient Simple", "Best Parameters for Gradient Simple"),
        ("CMA-ES", "Best Parameters for CMA-ES"),
        ("Gradient SPSA", "Best Parameters for SPSA"),
    ]

    for strategy_key, caption in param_tables:
        print(
            _build_param_table(
                strategy_key,
                apps,
                strategy_data,
                param_keys,
                caption,
            )
        )
        print()

    # Dynamic search parameter tables (per application & strategy)
    for app_label, file_path in DYNAMIC_SEARCH_FILES.items():
        if not file_path.exists():
            continue
        search_entries, search_keys = _load_dynamic_search_data(file_path, app_label)
        for strategy_code in STRATEGY_DISPLAY.keys():
            entries = search_entries.get(strategy_code, [])
            key_set = search_keys.get(strategy_code, set())
            table_text = _build_search_table(app_label, strategy_code, entries, key_set)
            if table_text:
                print(table_text)
                print()


if __name__ == "__main__":
    main()
