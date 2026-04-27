import csv
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone

from tools.prepare_aemo_full_multifreq_dataset import (
    CORE_FEATURE_COLUMNS,
    prepare_aemo_full_multifreq_dataset,
)


class PrepareAemoFullMultifreqDatasetTest(unittest.TestCase):
    def test_prepare_dataset_drops_empty_columns_and_keeps_sparse_columns(self):
        header = [
            "date",
            "empty_col",
        ]
        header.extend(CORE_FEATURE_COLUMNS)
        rows = []
        start = datetime(2022, 8, 25, 0, 0, tzinfo=timezone(timedelta(hours=10)))
        for idx in range(24):
            timestamp = start + timedelta(minutes=5 * idx)
            feature_values = []
            for column in CORE_FEATURE_COLUMNS:
                if column == "totaldemand_mw_avg":
                    feature_values.append(str(5000 + idx))
                elif column == "uigf_mw_avg":
                    feature_values.append(str(1200 + idx))
                elif column == "netinterchange_mw_avg":
                    feature_values.append(str(400 + idx))
                elif column == "totalintermittentgeneration_mw_avg":
                    feature_values.append(str(200 + idx))
                elif column == "availablegeneration_mw_avg":
                    feature_values.append(str(6000 + idx))
                elif column == "availableload_mw_avg":
                    feature_values.append(str(300 + idx))
                elif column == "dispatchablegeneration_mw_avg":
                    feature_values.append(str(2500 + idx))
                elif column == "dispatchableload_mw_avg":
                    feature_values.append(str(150 + idx))
                elif column == "wdr_available_mw_avg":
                    feature_values.append(str(55 + idx))
                elif column == "ss_solar_uigf_mw_avg":
                    feature_values.append(str(35 + idx))
                elif column == "ss_wind_uigf_mw_avg":
                    feature_values.append(str(45 + idx))
                else:
                    raise AssertionError(column)
            rows.append(
                [timestamp.isoformat(sep=" "), ""] + feature_values
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "aemo_raw.csv"
            output_dir = Path(tmpdir) / "out"
            with input_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

            summary = prepare_aemo_full_multifreq_dataset(
                input_path,
                output_dir,
                output_prefix="aemo_clean",
                freqs=["5min", "15min", "30min", "1h"],
            )

            outputs = {}
            for freq in ["5min", "15min", "30min", "1h"]:
                with (output_dir / f"aemo_clean_{freq}.csv").open(newline="") as f:
                    reader = csv.DictReader(f)
                    outputs[freq] = {
                        "fieldnames": reader.fieldnames,
                        "rows": list(reader),
                    }

        expected_columns = ["date"] + CORE_FEATURE_COLUMNS + ["net_load"]
        for freq, payload in outputs.items():
            self.assertEqual(payload["fieldnames"], expected_columns, freq)

        self.assertEqual(summary["selected_columns"], CORE_FEATURE_COLUMNS + ["net_load"])
        self.assertEqual(len(outputs["5min"]["rows"]), 24)
        self.assertEqual(len(outputs["15min"]["rows"]), 8)
        self.assertEqual(len(outputs["30min"]["rows"]), 4)
        self.assertEqual(len(outputs["1h"]["rows"]), 2)
        self.assertEqual(outputs["5min"]["rows"][0]["net_load"], "3800")
        self.assertEqual(len(outputs["5min"]["fieldnames"]), 13)


if __name__ == "__main__":
    unittest.main()
