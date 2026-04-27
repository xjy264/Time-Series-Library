import argparse
import csv
import re
from pathlib import Path


SETTING_PATTERN = re.compile(
    r"^long_term_forecast_(?P<model_id>aemo_.+?)_(?P<model>[^_]+)_custom_.*_sl(?P<seq_len>\d+)_ll(?P<label_len>\d+)_pl(?P<pred_len>\d+)_"
)
METRIC_PATTERN = re.compile(r"mse:(?P<mse>[-+0-9.eE]+), mae:(?P<mae>[-+0-9.eE]+), dtw:(?P<dtw>.+)")


def _infer_frequency(seq_len: str, pred_len: str) -> str:
    if (seq_len, pred_len) == ("2016", "24"):
        return "5min_24"
    if (seq_len, pred_len) == ("2016", "288"):
        return "5min_288"
    if (seq_len, pred_len) == ("672", "24"):
        return "15min_24"
    if (seq_len, pred_len) == ("672", "96"):
        return "15min_96"
    if (seq_len, pred_len) == ("336", "24"):
        return "30min_24"
    if (seq_len, pred_len) == ("336", "48"):
        return "30min_48"
    if (seq_len, pred_len) == ("168", "24"):
        return "1h_24"
    if (seq_len, pred_len) == ("168", "48"):
        return "1h_48"
    if (seq_len, pred_len) == ("168", "96"):
        return "1h_96"
    return f"{seq_len}_{pred_len}"


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
        rows.append(
            {
                "setting": setting,
                "model_id": model_id,
                "model_family": model_id.split("_")[1],
                "seq_len": setting_match.group("seq_len"),
                "label_len": setting_match.group("label_len"),
                "pred_len": setting_match.group("pred_len"),
                "frequency": _infer_frequency(setting_match.group("seq_len"), setting_match.group("pred_len")),
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
            fieldnames=[
                "setting",
                "model_id",
                "model_family",
                "seq_len",
                "label_len",
                "pred_len",
                "frequency",
                "model",
                "mse",
                "mae",
                "dtw",
            ],
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
