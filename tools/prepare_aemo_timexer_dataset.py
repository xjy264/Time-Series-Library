import argparse
import csv
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


INPUT_TIMESTAMP_COLUMN = "timestamp_local_hour"
OUTPUT_TIMESTAMP_COLUMN = "date"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

AEMO_REQUIRED_COLUMNS = [
    "totaldemand_mw_avg",
    "uigf_mw_avg",
    "ss_solar_uigf_mw_avg",
    "ss_wind_uigf_mw_avg",
    "netinterchange_mw_avg",
]

NOAA_FEATURE_COLUMNS = [
    "air_temperature_c",
    "dewpoint_c",
    "sea_level_pressure_hpa",
    "wind_speed_mps",
    "precip_1h_mm",
]

TIME_XER_EXOGENOUS_COLUMNS = [
    "ss_solar_uigf_mw_avg",
    "ss_wind_uigf_mw_avg",
    "netinterchange_mw_avg",
    *NOAA_FEATURE_COLUMNS,
]

OUTPUT_COLUMNS = [
    OUTPUT_TIMESTAMP_COLUMN,
    *TIME_XER_EXOGENOUS_COLUMNS,
    "net_load",
]

DEFAULT_AEMO_INPUT_PATH = Path(
    "./data/aemo_vic1_hourly_2022-08-25_2025-08-24.csv"
)
DEFAULT_NOAA_INPUT_PATH = Path(
    "./data/noaa_globalhourly_melbourne_olympic_park_hourly_2022-08-25_2025-08-24.csv"
)
DEFAULT_OUTPUT_PATH = Path("./dataset/aemo_vic1/aemo_vic1_timexer_weather_ms.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a TimeXer-specific AEMO VIC1 dataset with NOAA weather features."
    )
    parser.add_argument(
        "--aemo-input",
        type=Path,
        default=DEFAULT_AEMO_INPUT_PATH,
        help="Path to the raw AEMO hourly CSV.",
    )
    parser.add_argument(
        "--noaa-input",
        type=Path,
        default=DEFAULT_NOAA_INPUT_PATH,
        help="Path to the raw NOAA hourly CSV.",
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


def _parse_timestamp(timestamp_text: str) -> datetime:
    try:
        return datetime.strptime(timestamp_text, TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise ValueError(f"Invalid {INPUT_TIMESTAMP_COLUMN} value: {timestamp_text!r}") from exc


def _parse_float(value: str, column: str, timestamp: str) -> float:
    if value is None or value == "":
        raise ValueError(f"Empty value found in {column} at {timestamp}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value {value!r} in {column} at {timestamp}") from exc


def _summarize_timestamps(timestamps: List[datetime]) -> Tuple[int, int, List[Dict[str, str]]]:
    seen = Counter(timestamps)
    duplicate_timestamps = sum(count - 1 for count in seen.values() if count > 1)
    non_hourly_gaps = 0
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


def _load_noaa_rows(noaa_input_path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, object]]:
    rows_by_timestamp: Dict[str, Dict[str, str]] = {}
    timestamps: List[datetime] = []
    duplicate_counter = 0
    raw_missing_counts: Counter = Counter()

    with Path(noaa_input_path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("NOAA CSV does not contain a header row")
        _require_columns(reader.fieldnames, [INPUT_TIMESTAMP_COLUMN, *NOAA_FEATURE_COLUMNS])

        for row in reader:
            timestamp_text = row.get(INPUT_TIMESTAMP_COLUMN, "")
            if not timestamp_text:
                raise ValueError("Empty timestamp_local_hour value found in NOAA CSV")
            timestamp = _parse_timestamp(timestamp_text)
            timestamps.append(timestamp)

            if timestamp_text in rows_by_timestamp:
                duplicate_counter += 1

            compact_row = {column: row.get(column, "") for column in NOAA_FEATURE_COLUMNS}
            for column in NOAA_FEATURE_COLUMNS:
                if compact_row[column] == "":
                    raw_missing_counts[column] += 1
            rows_by_timestamp[timestamp_text] = compact_row

    duplicate_timestamps, non_hourly_gaps, gap_examples = _summarize_timestamps(timestamps)
    return rows_by_timestamp, {
        "row_count": len(timestamps),
        "duplicate_timestamps": duplicate_timestamps,
        "duplicate_overwrites": duplicate_counter,
        "non_hourly_gaps": non_hourly_gaps,
        "gap_examples": gap_examples,
        "raw_missing_value_counts": {column: raw_missing_counts[column] for column in NOAA_FEATURE_COLUMNS},
    }


def _impute_weather_rows(
    rows: List[Dict[str, str]]
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    missing_before_fill: Counter = Counter()
    zero_fill_counts: Counter = Counter()
    carry_fill_counts: Counter = Counter()

    for row in rows:
        for column in NOAA_FEATURE_COLUMNS:
            if row[column] == "":
                missing_before_fill[column] += 1

    for row in rows:
        if row["precip_1h_mm"] == "":
            row["precip_1h_mm"] = "0.0"
            zero_fill_counts["precip_1h_mm"] += 1

    for column in NOAA_FEATURE_COLUMNS:
        previous_value: Optional[str] = None
        for row in rows:
            if row[column] == "":
                if previous_value is not None:
                    row[column] = previous_value
                    carry_fill_counts[column] += 1
            else:
                previous_value = row[column]

        next_value: Optional[str] = None
        for row in reversed(rows):
            if row[column] == "":
                if next_value is not None:
                    row[column] = next_value
                    carry_fill_counts[column] += 1
            else:
                next_value = row[column]

        remaining = [row[OUTPUT_TIMESTAMP_COLUMN] for row in rows if row[column] == ""]
        if remaining:
            raise ValueError(
                f"Unable to impute NOAA column {column}; remaining missing timestamps: {remaining[:5]}"
            )

    return (
        {column: missing_before_fill[column] for column in NOAA_FEATURE_COLUMNS},
        {column: zero_fill_counts[column] for column in NOAA_FEATURE_COLUMNS},
        {column: carry_fill_counts[column] for column in NOAA_FEATURE_COLUMNS},
    )


def prepare_aemo_timexer_dataset(
    aemo_input_path: Path,
    noaa_input_path: Path,
    output_path: Path,
) -> Dict[str, object]:
    aemo_input_path = Path(aemo_input_path)
    noaa_input_path = Path(noaa_input_path)
    output_path = Path(output_path)

    required_columns = [INPUT_TIMESTAMP_COLUMN, *AEMO_REQUIRED_COLUMNS]
    noaa_rows_by_timestamp, noaa_summary = _load_noaa_rows(noaa_input_path)

    output_rows: List[Dict[str, str]] = []
    aemo_timestamps: List[datetime] = []
    aemo_missing_value_counts: Counter = Counter()
    exact_noaa_matches = 0

    with aemo_input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("AEMO CSV does not contain a header row")
        _require_columns(reader.fieldnames, required_columns)

        for row in reader:
            timestamp_text = row.get(INPUT_TIMESTAMP_COLUMN, "")
            if not timestamp_text:
                raise ValueError("Empty timestamp_local_hour value found in AEMO CSV")
            timestamp = _parse_timestamp(timestamp_text)
            aemo_timestamps.append(timestamp)

            for column in AEMO_REQUIRED_COLUMNS:
                if row.get(column, "") == "":
                    aemo_missing_value_counts[column] += 1

            totaldemand = _parse_float(row["totaldemand_mw_avg"], "totaldemand_mw_avg", timestamp_text)
            uigf = _parse_float(row["uigf_mw_avg"], "uigf_mw_avg", timestamp_text)
            _parse_float(row["ss_solar_uigf_mw_avg"], "ss_solar_uigf_mw_avg", timestamp_text)
            _parse_float(row["ss_wind_uigf_mw_avg"], "ss_wind_uigf_mw_avg", timestamp_text)
            _parse_float(row["netinterchange_mw_avg"], "netinterchange_mw_avg", timestamp_text)

            weather_row = noaa_rows_by_timestamp.get(timestamp_text)
            if weather_row is not None:
                exact_noaa_matches += 1
            else:
                weather_row = {column: "" for column in NOAA_FEATURE_COLUMNS}

            output_rows.append(
                {
                    OUTPUT_TIMESTAMP_COLUMN: timestamp_text,
                    "ss_solar_uigf_mw_avg": row["ss_solar_uigf_mw_avg"],
                    "ss_wind_uigf_mw_avg": row["ss_wind_uigf_mw_avg"],
                    "netinterchange_mw_avg": row["netinterchange_mw_avg"],
                    "air_temperature_c": weather_row["air_temperature_c"],
                    "dewpoint_c": weather_row["dewpoint_c"],
                    "sea_level_pressure_hpa": weather_row["sea_level_pressure_hpa"],
                    "wind_speed_mps": weather_row["wind_speed_mps"],
                    "precip_1h_mm": weather_row["precip_1h_mm"],
                    "net_load": f"{totaldemand - uigf:.6f}",
                }
            )

    output_rows.sort(key=lambda row: row[OUTPUT_TIMESTAMP_COLUMN])
    weather_missing_before_fill, zero_fill_counts, carry_fill_counts = _impute_weather_rows(output_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    duplicate_timestamps, non_hourly_gaps, gap_examples = _summarize_timestamps(aemo_timestamps)

    return {
        "aemo_input_path": str(aemo_input_path),
        "noaa_input_path": str(noaa_input_path),
        "output_path": str(output_path),
        "row_count": len(output_rows),
        "feature_column_count": len(OUTPUT_COLUMNS) - 1,
        "exact_noaa_match_count": exact_noaa_matches,
        "aemo_duplicate_timestamps": duplicate_timestamps,
        "aemo_non_hourly_gaps": non_hourly_gaps,
        "aemo_gap_examples": gap_examples,
        "aemo_missing_value_counts": {
            column: aemo_missing_value_counts[column] for column in AEMO_REQUIRED_COLUMNS
        },
        "weather_missing_before_fill": weather_missing_before_fill,
        "weather_zero_fill_counts": zero_fill_counts,
        "weather_carry_fill_counts": carry_fill_counts,
        "noaa_summary": noaa_summary,
    }


def main() -> None:
    args = _parse_args()
    summary = prepare_aemo_timexer_dataset(args.aemo_input, args.noaa_input, args.output)
    print("Prepared AEMO TimeXer dataset:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
