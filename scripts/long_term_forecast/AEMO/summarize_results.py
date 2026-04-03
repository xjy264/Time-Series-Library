import argparse
import csv
import re
from pathlib import Path


SETTING_PATTERN = re.compile(
    r"^long_term_forecast_(?P<model_id>aemo_[^_]+_[^_]+_[0-9]+)_(?P<model>[^_]+)_custom_"
)
METRIC_PATTERN = re.compile(r"mse:(?P<mse>[-+0-9.eE]+), mae:(?P<mae>[-+0-9.eE]+), dtw:(?P<dtw>.+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize AEMO experiment metrics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("result_long_term_forecast.txt"),
        help="Path to the raw long-term forecast metrics log.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aemo_results_summary.csv"),
        help="Path to the CSV summary output.",
    )
    return parser.parse_args()


def summarize_results(input_path: Path, output_path: Path) -> int:
    lines = [line.strip() for line in input_path.read_text().splitlines() if line.strip()]
    rows = []

    for setting, metrics in zip(lines[0::2], lines[1::2]):
        setting_match = SETTING_PATTERN.match(setting)
        metric_match = METRIC_PATTERN.match(metrics)
        if not setting_match or not metric_match:
            continue

        model_id = setting_match.group("model_id")
        parts = model_id.split("_")
        rows.append(
            {
                "setting": setting,
                "model_id": model_id,
                "model_family": parts[1],
                "seq_len": parts[2],
                "pred_len": parts[3],
                "model": setting_match.group("model"),
                "mse": metric_match.group("mse"),
                "mae": metric_match.group("mae"),
                "dtw": metric_match.group("dtw"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["setting", "model_id", "model_family", "seq_len", "pred_len", "model", "mse", "mae", "dtw"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    args = parse_args()
    row_count = summarize_results(args.input, args.output)
    print(f"Wrote {row_count} rows to {args.output}")


if __name__ == "__main__":
    main()
