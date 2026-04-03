import csv
import tempfile
import unittest
from pathlib import Path

from tools.prepare_aemo_timexer_dataset import prepare_aemo_timexer_dataset


class PrepareAemoTimeXerDatasetTest(unittest.TestCase):
    def test_prepare_dataset_outputs_expected_columns_and_net_load(self):
        header = [
            "timestamp_local_hour",
            "totaldemand_mw_avg",
            "uigf_mw_avg",
            "ss_solar_uigf_mw_avg",
            "ss_wind_uigf_mw_avg",
            "netinterchange_mw_avg",
        ]
        rows = [
            ["2022-08-25 00:00:00", "4878.766667", "1328.825147", "0.0", "1328.825147", "424.4125"],
            ["2022-08-25 01:00:00", "4598.655833", "1345.655399", "0.0", "1345.655399", "583.646667"],
            ["2022-08-25 02:00:00", "4319.8525", "1174.268276", "0.0", "1174.268276", "495.685"],
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "aemo_raw.csv"
            output_path = Path(tmpdir) / "aemo_timexer.csv"

            with input_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

            summary = prepare_aemo_timexer_dataset(input_path, output_path)

            with output_path.open(newline="") as f:
                reader = csv.DictReader(f)
                output_rows = list(reader)

        self.assertEqual(
            reader.fieldnames,
            [
                "date",
                "totaldemand_mw_avg",
                "uigf_mw_avg",
                "ss_solar_uigf_mw_avg",
                "ss_wind_uigf_mw_avg",
                "netinterchange_mw_avg",
                "net_load",
            ],
        )
        self.assertEqual(len(output_rows), 3)
        self.assertEqual(output_rows[0]["date"], "2022-08-25 00:00:00")
        self.assertAlmostEqual(float(output_rows[0]["net_load"]), 3549.94152, places=5)
        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["duplicate_timestamps"], 0)
        self.assertEqual(summary["non_hourly_gaps"], 0)
        self.assertEqual(summary["gap_examples"], [])
        self.assertEqual(summary["missing_value_counts"]["netinterchange_mw_avg"], 0)


if __name__ == "__main__":
    unittest.main()
