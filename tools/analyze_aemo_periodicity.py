import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = Path("./data/aemo_vic1_hourly_2022-08-25_2025-08-24.csv")
DEFAULT_OUTPUT_DIR = Path("./results/aemo_vic1/periodicity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze seasonal patterns in AEMO VIC1 net load."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _save_stats(df: pd.DataFrame, group_col: str, output_dir: Path, filename: str) -> pd.DataFrame:
    stats = df.groupby(group_col)["net_load"].agg(["mean", "std", "min", "max", "count"]).round(3)
    stats.to_csv(output_dir / filename)
    return stats


def _autocorrelation(values: np.ndarray, lags: list[int]) -> pd.DataFrame:
    rows = []
    for lag in lags:
        if lag >= len(values):
            continue
        a = values[:-lag]
        b = values[lag:]
        rows.append({"lag_hours": lag, "acf": float(np.corrcoef(a, b)[0, 1])})
    return pd.DataFrame(rows)


def _make_plots(df: pd.DataFrame, output_dir: Path, acf: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    hour_mean = df.groupby("hour")["net_load"].mean()
    dow_mean = df.groupby("dow")["net_load"].mean()
    month_mean = df.groupby("month")["net_load"].mean()
    hour_dow = df.pivot_table(index="hour", columns="dow", values="net_load", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 4))
    hour_mean.plot(ax=ax, marker="o")
    ax.set_title("AEMO VIC1 Net Load Mean by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Net Load (MW)")
    ax.set_xticks(range(24))
    fig.tight_layout()
    fig.savefig(output_dir / "hourly_mean_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    dow_mean.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_mean.plot(ax=ax, marker="o", color="#d97706")
    ax.set_title("AEMO VIC1 Net Load Mean by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Net Load (MW)")
    fig.tight_layout()
    fig.savefig(output_dir / "weekday_mean_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    month_mean.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_mean.plot(ax=ax, marker="o", color="#2563eb")
    ax.set_title("AEMO VIC1 Net Load Mean by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Net Load (MW)")
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_mean_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(hour_dow.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("Net Load Mean Heatmap: Hour x Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour")
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_yticks(range(0, 24, 2))
    fig.colorbar(im, ax=ax, label="Net Load (MW)")
    fig.tight_layout()
    fig.savefig(output_dir / "hour_by_dow_heatmap.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(acf["lag_hours"], acf["acf"], marker="o")
    ax.axvline(24, color="red", linestyle="--", alpha=0.5)
    ax.axvline(168, color="green", linestyle="--", alpha=0.5)
    ax.set_title("Net Load Autocorrelation by Lag")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "acf_lags.png", dpi=200)
    plt.close(fig)


def _seasonal_baselines(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    cutoff = pd.Timestamp("2024-08-25 00:00:00")
    train = df[df["timestamp_local_hour"] < cutoff].copy()
    test = df[df["timestamp_local_hour"] >= cutoff].copy()

    global_mean = train["net_load"].mean()
    hour_mean = train.groupby("hour")["net_load"].mean()
    hour_dow_mean = train.groupby(["hour", "dow"])["net_load"].mean()
    hour_dow_month_mean = train.groupby(["hour", "dow", "month"])["net_load"].mean()

    pred_global = np.full(len(test), global_mean)
    pred_hour = test["hour"].map(hour_mean).to_numpy(dtype=float)
    pred_hour_dow = test.apply(lambda r: hour_dow_mean.get((r["hour"], r["dow"]), np.nan), axis=1).to_numpy(dtype=float)
    pred_hour_dow_month = test.apply(
        lambda r: hour_dow_month_mean.get((r["hour"], r["dow"], r["month"]), np.nan), axis=1
    ).to_numpy(dtype=float)

    pred_hour_dow = np.where(np.isnan(pred_hour_dow), pred_hour, pred_hour_dow)
    pred_hour_dow_month = np.where(np.isnan(pred_hour_dow_month), pred_hour_dow, pred_hour_dow_month)

    y = test["net_load"].to_numpy(dtype=float)
    rows = []
    for name, pred in [
        ("global_mean", pred_global),
        ("hour_mean", pred_hour),
        ("hour_dow_mean", pred_hour_dow),
        ("hour_dow_month_mean", pred_hour_dow_month),
    ]:
        mse = float(np.mean((pred - y) ** 2))
        mae = float(np.mean(np.abs(pred - y)))
        rows.append({"baseline": name, "mse": mse, "mae": mae})

    baseline_df = pd.DataFrame(rows)
    baseline_df.to_csv(output_dir / "seasonal_baselines.csv", index=False)
    return baseline_df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, usecols=["timestamp_local_hour", "totaldemand_mw_avg", "uigf_mw_avg"])
    df["timestamp_local_hour"] = pd.to_datetime(df["timestamp_local_hour"])
    df["net_load"] = df["totaldemand_mw_avg"] - df["uigf_mw_avg"]
    df = df.sort_values("timestamp_local_hour").reset_index(drop=True)
    df["hour"] = df["timestamp_local_hour"].dt.hour
    df["dow"] = df["timestamp_local_hour"].dt.dayofweek
    df["month"] = df["timestamp_local_hour"].dt.month

    hour_stats = _save_stats(df, "hour", output_dir, "hour_stats.csv")
    dow_stats = _save_stats(df, "dow", output_dir, "dow_stats.csv")
    month_stats = _save_stats(df, "month", output_dir, "month_stats.csv")

    acf = _autocorrelation(
        df["net_load"].to_numpy(dtype=float),
        [1, 2, 3, 4, 6, 12, 24, 48, 72, 96, 120, 168, 336, 720, 8760],
    )
    acf.to_csv(output_dir / "acf_lags.csv", index=False)

    hour_dow = df.pivot_table(index="hour", columns="dow", values="net_load", aggfunc="mean").round(1)
    hour_dow.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hour_dow.to_csv(output_dir / "hour_by_dow_mean.csv")

    summary = pd.Series(
        {
            "rows": len(df),
            "start": str(df["timestamp_local_hour"].min()),
            "end": str(df["timestamp_local_hour"].max()),
            "hour_peak": int(hour_stats["mean"].idxmax()),
            "hour_trough": int(hour_stats["mean"].idxmin()),
            "dow_peak": int(dow_stats["mean"].idxmax()),
            "dow_trough": int(dow_stats["mean"].idxmin()),
            "month_peak": int(month_stats["mean"].idxmax()),
            "month_trough": int(month_stats["mean"].idxmin()),
            "daily_amplitude_mw": float(hour_stats["mean"].max() - hour_stats["mean"].min()),
            "weekday_amplitude_mw": float(dow_stats["mean"].max() - dow_stats["mean"].min()),
            "monthly_amplitude_mw": float(month_stats["mean"].max() - month_stats["mean"].min()),
        }
    )
    summary.to_csv(output_dir / "summary.csv")

    _make_plots(df, output_dir, acf)
    seasonal_baselines = _seasonal_baselines(df, output_dir)

    print("Wrote periodicity analysis to:", output_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nSeasonal baselines (last 12 months holdout):")
    print(seasonal_baselines.to_string(index=False))


if __name__ == "__main__":
    main()
