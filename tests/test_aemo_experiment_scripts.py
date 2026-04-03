import unittest
from pathlib import Path


SCRIPT_DIR = Path("scripts/long_term_forecast/AEMO")
MODEL_SCRIPTS = [
    "Autoformer.sh",
    "DLinear.sh",
    "Informer.sh",
    "PatchTST.sh",
    "TimeMixer.sh",
    "TimeXer.sh",
    "iTransformer.sh",
]


class AemoExperimentScriptsTest(unittest.TestCase):
    def test_model_scripts_exist_and_share_common_aemo_settings(self):
        common_expectations = [
            "--root_path ./dataset/aemo_vic1/",
            "--data_path aemo_vic1_timexer_ms.csv",
            "--data custom",
            "--features MS",
            "--target net_load",
            "--freq h",
            "--seq_len 168",
            "--enc_in 6",
            "--dec_in 6",
        ]

        for script_name in MODEL_SCRIPTS:
            script_path = SCRIPT_DIR / script_name
            self.assertTrue(script_path.exists(), f"missing {script_path}")
            content = script_path.read_text()
            for expected in common_expectations:
                self.assertIn(expected, content, f"{script_name} missing {expected}")
            expected_c_out = "--c_out 6" if script_name == "TimeMixer.sh" else "--c_out 1"
            self.assertIn(expected_c_out, content, f"{script_name} missing {expected_c_out}")

    def test_batch_runner_and_summary_tools_exist(self):
        for relative_path in [
            SCRIPT_DIR / "run_smoke.sh",
            SCRIPT_DIR / "run_full.sh",
            SCRIPT_DIR / "summarize_results.py",
        ]:
            self.assertTrue(relative_path.exists(), f"missing {relative_path}")


if __name__ == "__main__":
    unittest.main()
