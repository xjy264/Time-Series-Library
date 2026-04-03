import argparse
import csv
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


INPUT_TIMESTAMP_COLUMN = "timestamp_local_hour"
OUTPUT_TIMESTAMP_COLUMN = "date"

INPUT_COLUMNS = [
    "totaldemand_mw_avg",
    "uigf_mw_avg",
    "ss_solar_uigf_mw_avg",
    "ss_wind_uigf_mw_avg",
    "netinterchange_mw_avg",
]

OUTPUT_COLUMNS = [
    OUTPUT_TIMESTAMP_COLUMN,
    *INPUT_COLUMNS,
    "net_load",
]

DEFAULT_INPUT_PATH = Path(
    "/Users/xuejiayao/Desktop/paper/data/aemo_vic1_hourly_2022-08-25_2025-08-24.csv"
)
DEFAULT_OUTPUT_PATH = Path("./dataset/aemo_vic1/aemo_vic1_timexer_ms.csv")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a slim AEMO VIC1 dataset for TimeXer MS forecasting."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the raw AEMO hourly CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the processed TimeXer-ready CSV.",
    )
    return parser.parse_args()


def _require_columns(fieldnames: List[str], required_columns: List[str]) -> None:
    missing = [column for column in required_columns if column not in fieldnames]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _parse_float(value: str, column: str, timestamp: str) -> float:
    if value is None or value == "":
        raise ValueError(f"Empty value found in {column} at {timestamp}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value {value!r} in {column} at {timestamp}") from exc


def _count_missing_values(row: Dict[str, str], columns: List[str], counter: Counter) -> None:
    for column in columns:
        if row.get(column, "") == "":
            counter[column] += 1


def _build_output_row(row: Dict[str, str]) -> Dict[str, str]:
    timestamp = row[INPUT_TIMESTAMP_COLUMN]
    totaldemand = _parse_float(row["totaldemand_mw_avg"], "totaldemand_mw_avg", timestamp)
    uigf = _parse_float(row["uigf_mw_avg"], "uigf_mw_avg", timestamp)
    _parse_float(row["ss_solar_uigf_mw_avg"], "ss_solar_uigf_mw_avg", timestamp)
    _parse_float(row["ss_wind_uigf_mw_avg"], "ss_wind_uigf_mw_avg", timestamp)
    _parse_float(row["netinterchange_mw_avg"], "netinterchange_mw_avg", timestamp)

    return {
        OUTPUT_TIMESTAMP_COLUMN: timestamp,
        "totaldemand_mw_avg": row["totaldemand_mw_avg"],
        "uigf_mw_avg": row["uigf_mw_avg"],
        "ss_solar_uigf_mw_avg": row["ss_solar_uigf_mw_avg"],
        "ss_wind_uigf_mw_avg": row["ss_wind_uigf_mw_avg"],
        "netinterchange_mw_avg": row["netinterchange_mw_avg"],
        "net_load": f"{totaldemand - uigf:.6f}",
    }


def _summarize_timestamps(timestamps: List[datetime]) -> Tuple[int, int, List[Dict[str, str]]]:
    duplicate_timestamps = 0
    non_hourly_gaps = 0
    seen = Counter(timestamps)
    duplicate_timestamps = sum(count - 1 for count in seen.values() if count > 1)
    gap_examples: List[Dict[str, str]] = []

    unique_sorted = sorted(seen.keys())
    for previous, current in zip(unique_sorted, unique_sorted[1:]):
        delta = current - previous
        if delta != timedelta(hours=1):
            non_hourly_gaps += 1
            gap_examples.append(
                {
                    "previous": previous.strftime(TIMESTAMP_FORMAT),
                    "current": current.strftime(TIMESTAMP_FORMAT),
                    "delta_hours": f"{delta.total_seconds() / 3600:.1f}",
                }
            )

    return duplicate_timestamps, non_hourly_gaps, gap_examples


def prepare_aemo_timexer_dataset(input_path: Path, output_path: Path) -> Dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)

    required_columns = [INPUT_TIMESTAMP_COLUMN, *INPUT_COLUMNS]
    missing_value_counts: Counter = Counter()
    output_rows: List[Dict[str, str]] = []
    timestamps: List[datetime] = []

    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV does not contain a header row")
        _require_columns(reader.fieldnames, required_columns)

        for row in reader:
            _count_missing_values(row, INPUT_COLUMNS, missing_value_counts)
            timestamp_text = row.get(INPUT_TIMESTAMP_COLUMN, "")
            if not timestamp_text:
                raise ValueError("Empty timestamp_local_hour value found")
            try:
                timestamp = datetime.strptime(timestamp_text, TIMESTAMP_FORMAT)
            except ValueError as exc:
                raise ValueError(f"Invalid timestamp_local_hour value: {timestamp_text!r}") from exc

            output_rows.append(_build_output_row(row))
            timestamps.append(timestamp)

    output_rows.sort(key=lambda row: row[OUTPUT_TIMESTAMP_COLUMN])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    duplicate_timestamps, non_hourly_gaps, gap_examples = _summarize_timestamps(timestamps)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "row_count": len(output_rows),
        "duplicate_timestamps": duplicate_timestamps,
        "non_hourly_gaps": non_hourly_gaps,
        "gap_examples": gap_examples,
        "missing_value_counts": {column: missing_value_counts[column] for column in INPUT_COLUMNS},
    }


def main() -> None:
    args = _parse_args()
    summary = prepare_aemo_timexer_dataset(args.input, args.output)
    print("Prepared AEMO TimeXer dataset:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
