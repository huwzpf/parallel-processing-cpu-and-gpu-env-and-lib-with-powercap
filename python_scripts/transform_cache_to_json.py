from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Tuple


def load_cache_records(path: Path) -> Iterable[OrderedDict]:
    """Yield JSON objects parsed from each non-empty line of the cache file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line, object_pairs_hook=OrderedDict)


def build_group_key(parameters: OrderedDict) -> Tuple[Tuple[str, object], ...]:
    """Create a hashable key based on the ordered parameter mapping."""
    return tuple(parameters.items())


def transform_cache(input_path: Path, description: str) -> OrderedDict:
    """Convert the cache format into the aggregated experiment JSON structure."""
    experiment_groups: "OrderedDict[Tuple[Tuple[str, object], ...], OrderedDict]" = OrderedDict()

    for entry in load_cache_records(input_path):
        try:
            parameters = entry["parameters"]
            result = entry["result"]
        except KeyError as exc:  # pragma: no cover - guardrail for malformed rows
            raise ValueError(f"Missing expected key in cache entry: {exc}") from exc

        key = build_group_key(parameters)
        if key not in experiment_groups:
            experiment_groups[key] = OrderedDict([
                ("parameters", parameters),
                ("runs", []),
            ])

        experiment_groups[key]["runs"].append(result)

    return OrderedDict([
        ("description", description),
        ("experiment_result", list(experiment_groups.values())),
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform cache log entries into grouped experiment JSON",
    )
    parser.add_argument("input", type=Path, help="Path to cache.txt style input file")
    parser.add_argument(
        "output",
        type=Path,
        help="Destination path for the aggregated JSON output",
    )
    parser.add_argument(
        "--description",
        default="converted_experiment",
        help="Description string stored at the top level of the JSON output",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="Indentation level for the JSON output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = transform_cache(args.input, args.description)
    args.output.write_text(
        json.dumps(data, indent=args.indent),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
