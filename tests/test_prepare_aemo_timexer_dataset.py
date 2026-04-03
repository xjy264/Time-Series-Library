import csv
import tempfile
import unittest
from pathlib import Path

from tools.prepare_aemo_timexer_dataset import prepare_aemo_timexer_dataset


class PrepareAemoTimeXerDatasetTest(unittest.TestCase):
    def test_prepare_dataset_outputs_expected_columns_and_weather_merge(self):
        aemo_header = [
            "timestamp_local_hour",
            "totaldemand_mw_avg",
            "uigf_mw_avg",
            "ss_solar_uigf_mw_avg",
            "ss_wind_uigf_mw_avg",
            "netinterchange_mw_avg",
        ]
        aemo_rows = [
            ["2022-08-25 00:00:00", "4878.766667", "1328.825147", "0.0", "1328.825147", "424.4125"],
            ["2022-08-25 01:00:00", "4598.655833", "1345.655399", "0.0", "1345.655399", "583.646667"],
            ["2022-08-25 02:00:00", "4319.8525", "1174.268276", "0.0", "1174.268276", "495.685"],
        ]
        noaa_header = [
            "timestamp_local_hour",
            "air_temperature_c",
            "dewpoint_c",
            "sea_level_pressure_hpa",
            "wind_speed_mps",
            "precip_1h_mm",
        ]
        noaa_rows = [
            ["2022-08-25 00:00:00", "12.3", "7.1", "1013.4", "3.5", ""],
            ["2022-08-25 02:00:00", "11.8", "", "1012.9", "3.2", ""],
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            aemo_input_path = Path(tmpdir) / "aemo_raw.csv"
            noaa_input_path = Path(tmpdir) / "noaa_raw.csv"
            output_path = Path(tmpdir) / "aemo_timexer_weather.csv"

            with aemo_input_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(aemo_header)
                writer.writerows(aemo_rows)

            with noaa_input_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(noaa_header)
                writer.writerows(noaa_rows)

            summary = prepare_aemo_timexer_dataset(aemo_input_path, noaa_input_path, output_path)

            with output_path.open(newline="") as f:
                reader = csv.DictReader(f)
                output_rows = list(reader)
                fieldnames = reader.fieldnames

        self.assertEqual(
            fieldnames,
            [
                "date",
                "ss_solar_uigf_mw_avg",
                "ss_wind_uigf_mw_avg",
                "netinterchange_mw_avg",
                "air_temperature_c",
                "dewpoint_c",
                "sea_level_pressure_hpa",
                "wind_speed_mps",
                "precip_1h_mm",
                "net_load",
            ],
        )
        self.assertEqual(len(output_rows), 3)
        self.assertEqual(output_rows[0]["date"], "2022-08-25 00:00:00")
        self.assertAlmostEqual(float(output_rows[0]["net_load"]), 3549.94152, places=5)
        self.assertEqual(output_rows[1]["air_temperature_c"], "12.3")
        self.assertEqual(output_rows[1]["dewpoint_c"], "7.1")
        self.assertEqual(output_rows[1]["precip_1h_mm"], "0.0")
        self.assertEqual(output_rows[2]["dewpoint_c"], "7.1")
        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["feature_column_count"], 9)
        self.assertEqual(summary["exact_noaa_match_count"], 2)
        self.assertEqual(summary["aemo_duplicate_timestamps"], 0)
        self.assertEqual(summary["aemo_non_hourly_gaps"], 0)
        self.assertEqual(summary["aemo_gap_examples"], [])
        self.assertEqual(summary["weather_missing_before_fill"]["air_temperature_c"], 1)
        self.assertEqual(summary["weather_missing_before_fill"]["dewpoint_c"], 2)
        self.assertEqual(summary["weather_zero_fill_counts"]["precip_1h_mm"], 3)
        self.assertEqual(summary["weather_carry_fill_counts"]["air_temperature_c"], 1)
        self.assertEqual(summary["weather_carry_fill_counts"]["dewpoint_c"], 2)


if __name__ == "__main__":
    unittest.main()
