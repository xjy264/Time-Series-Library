import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


SOURCE_TZ = "Australia/Melbourne"
DEFAULT_INPUT_PATH = Path(
    "./data/aemo_vic1/aemo_vic1_dispatchis_vic1_5min_2022-08-25_2025-08-24.csv"
)
DEFAULT_OUTPUT_DIR = Path("./dataset/aemo_vic1")
DEFAULT_OUTPUT_PREFIX = "aemo_vic1_dispatchis_vic1_full"
DEFAULT_FREQS = ["5min", "15min", "30min", "1h"]
# 11 core system-side variables + the target = 12 data channels.
CORE_FEATURE_COLUMNS = [
    "totaldemand_mw_avg",
    "uigf_mw_avg",
    "netinterchange_mw_avg",
    "totalintermittentgeneration_mw_avg",
    "availablegeneration_mw_avg",
    "availableload_mw_avg",
    "dispatchablegeneration_mw_avg",
    "dispatchableload_mw_avg",
    "wdr_available_mw_avg",
    "ss_solar_uigf_mw_avg",
    "ss_wind_uigf_mw_avg",
]
OUTPUT_COLUMNS = CORE_FEATURE_COLUMNS + ["net_load"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare full AEMO DISPATCHREGIONSUM datasets at multiple frequencies."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--freqs",
        nargs="+",
        default=DEFAULT_FREQS,
        help="Target frequencies such as 5min 15min 30min 1h.",
    )
    return parser.parse_args()


def _parse_local_datetime(series: pd.Series) -> pd.DatetimeIndex:
    sample = series.dropna().astype(str).head(10)
    has_timezone = sample.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).any()
    if has_timezone:
        parsed = pd.to_datetime(series, utc=True, errors="raise")
        parsed = parsed.dt.tz_convert(SOURCE_TZ)
        return pd.DatetimeIndex(parsed)
    parsed = pd.to_datetime(series, errors="raise")
    if getattr(parsed.dt, "tz", None) is None:
        parsed = parsed.dt.tz_localize(
            SOURCE_TZ,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    else:
        parsed = parsed.dt.tz_convert(SOURCE_TZ)
    return pd.DatetimeIndex(parsed)


def _load_full_aemo_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError("Input AEMO CSV must contain a 'date' column")

    frame = frame.copy()
    frame.index = _parse_local_datetime(frame["date"])
    frame = frame.drop(columns=["date"])
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]

    for column in frame.columns:
        if column not in {"regionid", "source_month"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "net_load" not in frame.columns and {"totaldemand_mw_avg", "uigf_mw_avg"} <= set(frame.columns):
        frame["net_load"] = frame["totaldemand_mw_avg"] - frame["uigf_mw_avg"]

    return frame


def _select_core_columns(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Input AEMO CSV is missing required columns: {missing_columns}")
    return frame.loc[:, OUTPUT_COLUMNS].copy()


def _resample_full_frame(frame: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    if target_freq == "5min":
        return frame.copy()

    utc_frame = frame.tz_convert("UTC")
    numeric_cols = utc_frame.select_dtypes(include="number").columns.tolist()
    other_cols = [column for column in utc_frame.columns if column not in numeric_cols]

    parts: List[pd.DataFrame] = []
    if numeric_cols:
        parts.append(utc_frame[numeric_cols].resample(target_freq).mean())
    if other_cols:
        parts.append(utc_frame[other_cols].resample(target_freq).first())

    if not parts:
        return pd.DataFrame(index=utc_frame.resample(target_freq).mean().index)

    resampled = pd.concat(parts, axis=1)
    resampled = resampled.reindex(columns=frame.columns)

    if numeric_cols:
        resampled[numeric_cols] = resampled[numeric_cols].interpolate(method="time").ffill().bfill()
    if other_cols:
        resampled[other_cols] = resampled[other_cols].ffill().bfill()

    return resampled.tz_convert(SOURCE_TZ)


def _format_timestamp(index: pd.DatetimeIndex) -> List[str]:
    return [timestamp.isoformat(sep=" ") for timestamp in index]


def prepare_aemo_full_multifreq_dataset(
    input_path: Path,
    output_dir: Path,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
    freqs: Iterable[str] = DEFAULT_FREQS,
) -> Dict[str, object]:
    full_frame = _load_full_aemo_frame(input_path)
    full_frame = _select_core_columns(full_frame)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, object] = {
        "input_path": str(input_path),
        "row_count": len(full_frame),
        "start": full_frame.index[0].isoformat(sep=" ") if len(full_frame) else None,
        "end": full_frame.index[-1].isoformat(sep=" ") if len(full_frame) else None,
        "selected_columns": OUTPUT_COLUMNS,
        "outputs": {},
    }

    for freq in freqs:
        output_frame = _resample_full_frame(full_frame, freq)
        output_path = output_dir / f"{output_prefix}_{freq}.csv"
        export_frame = output_frame.reset_index().rename(columns={"index": "date"})
        export_frame["date"] = _format_timestamp(pd.DatetimeIndex(export_frame["date"]))
        export_frame.to_csv(output_path, index=False)
        summaries["outputs"][freq] = {
            "output_path": str(output_path),
            "row_count": len(export_frame),
            "start": export_frame["date"].iloc[0] if len(export_frame) else None,
            "end": export_frame["date"].iloc[-1] if len(export_frame) else None,
        }

    return summaries


def main() -> None:
    args = _parse_args()
    summary = prepare_aemo_full_multifreq_dataset(
        args.input,
        args.output_dir,
        output_prefix=args.output_prefix,
        freqs=args.freqs,
    )
    print("Prepared full AEMO datasets:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
