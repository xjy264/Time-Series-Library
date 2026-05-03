import unittest
from pathlib import Path


SCRIPT_DIR = Path("scripts/long_term_forecast/AEMO")
TIMEXER_SCRIPTS = [
    "TimeXer.sh",
    "TimeXer_5min.sh",
]

FIVE_MIN_BASELINE_SCRIPTS = [
    "DLinear_5min.sh",
    "PatchTST_5min.sh",
    "Informer_5min.sh",
    "Autoformer_5min.sh",
    "TimesNet_5min.sh",
    "TimeXer_5min.sh",
    "VPPGDFNet_5min.sh",
]


class AemoExperimentScriptsTest(unittest.TestCase):
    def test_timexer_scripts_exist_and_use_full_datasets(self):
        expected_snippets = {
            "TimeXer.sh": [
                'data_path="${DATA_PATH:-aemo_vic1_dispatchis_vic1_full_5min.csv}"',
                'pred_lens="${PRED_LENS:-24 48 96 288}"',
                'seq_len="${SEQ_LEN:-288}"',
                'label_len="${LABEL_LEN:-144}"',
                'patch_len="${PATCH_LEN:-12}"',
                'freq="${FREQ:-5min}"',
                'enc_in="${ENC_IN:-12}"',
                'dec_in="${DEC_IN:-12}"',
            ],
            "TimesNet_5min.sh": [
                'model_name=TimesNet',
                'data_path="${DATA_PATH:-aemo_vic1_dispatchis_vic1_full_5min.csv}"',
                'pred_lens="${PRED_LENS:-24 48 96 288}"',
                'seq_len="${SEQ_LEN:-288}"',
                'label_len="${LABEL_LEN:-144}"',
                'freq="${FREQ:-5min}"',
                'enc_in="${ENC_IN:-12}"',
                'dec_in="${DEC_IN:-12}"',
            ],
            "TimeXer_5min.sh": [
                'data_path="${DATA_PATH:-aemo_vic1_dispatchis_vic1_full_5min.csv}"',
                'pred_lens="${PRED_LENS:-24 48 96 288}"',
                'seq_len="${SEQ_LEN:-288}"',
                'label_len="${LABEL_LEN:-144}"',
                'patch_len="${PATCH_LEN:-12}"',
                'freq="${FREQ:-5min}"',
                'enc_in="${ENC_IN:-12}"',
                'dec_in="${DEC_IN:-12}"',
            ],
        }

        for script_name in TIMEXER_SCRIPTS:
            script_path = SCRIPT_DIR / script_name
            self.assertTrue(script_path.exists(), f"missing {script_path}")
            content = script_path.read_text()
            for expected in expected_snippets[script_name]:
                self.assertIn(expected, content, f"{script_name} missing {expected}")
            self.assertIn("--c_out 1", content, f"{script_name} missing --c_out 1")

    def test_batch_runner_and_summary_tools_exist(self):
        for relative_path in [
            SCRIPT_DIR / "run_smoke.sh",
            SCRIPT_DIR / "run_full.sh",
            SCRIPT_DIR / "run_timexer.sh",
            SCRIPT_DIR / "run_5min.sh",
            SCRIPT_DIR / "run_5min_matrix_26.sh",
            SCRIPT_DIR / "run_vppgdfnet_ablation_5min.sh",
            SCRIPT_DIR / "TimeXer_5min.sh",
            SCRIPT_DIR / "TimeXer.sh",
            SCRIPT_DIR / "summarize_results.py",
        ]:
            self.assertTrue(relative_path.exists(), f"missing {relative_path}")

    def test_run_timexer_is_5min_only(self):
        content = (SCRIPT_DIR / "run_timexer.sh").read_text()
        self.assertIn("tools/prepare_aemo_full_multifreq_dataset.py", content)
        self.assertIn("--freqs 5min", content)
        self.assertIn("TimeXer_5min.sh", content)
        self.assertNotIn("TimeXer_15min.sh", content)
        self.assertNotIn("TimeXer_30min.sh", content)
        self.assertNotIn("TimeXer_1h.sh", content)

    def test_five_min_baseline_scripts_exist_and_use_5min_dataset(self):
        expected_snippets = {
            "DLinear_5min.sh": [
                'data_path="${DATA_PATH:-aemo_vic1_dispatchis_vic1_full_5min.csv}"',
                'pred_lens="${PRED_LENS:-24 48 96 288}"',
                'seq_len="${SEQ_LEN:-288}"',
                'label_len="${LABEL_LEN:-144}"',
                'freq="${FREQ:-5min}"',
                'enc_in="${ENC_IN:-12}"',
                'dec_in="${DEC_IN:-12}"',
            ],
            "DLinearGlobalTimeXer_5min.sh": [
                'model_name=DLinearGlobalTimeXer',
                'data_path="${DATA_PATH:-aemo_vic1_dispatchis_vic1_full_5min.csv}"',
                'pred_lens="${PRED_LENS:-24 48 96 288}"',
                'seq_len="${SEQ_LEN:-288}"',
                'label_len="${LABEL_LEN:-144}"',
                'freq="${FREQ:-5min}"',
                'enc_in="${ENC_IN:-12}"',
                'dec_in="${DEC_IN:-12}"',
            ],
            "VPPGDFNet_5min.sh": [
                'model_name=VPPGDFNet',
                'ablation="${VPP_ABLATION:-full}"',
                'data_path="${DATA_PATH:-aemo_vic1_dispatchis_vic1_full_5min.csv}"',
                'pred_lens="${PRED_LENS:-24 48 96 288}"',
                'seq_len="${SEQ_LEN:-288}"',
                'label_len="${LABEL_LEN:-144}"',
                'freq="${FREQ:-5min}"',
                'enc_in="${ENC_IN:-12}"',
                'dec_in="${DEC_IN:-12}"',
            ],
            "Informer_5min.sh": [
                'data_path="${DATA_PATH:-aemo_vic1_dispatchis_vic1_full_5min.csv}"',
                'pred_lens="${PRED_LENS:-24 48 96 288}"',
                'seq_len="${SEQ_LEN:-288}"',
                'label_len="${LABEL_LEN:-144}"',
                'freq="${FREQ:-5min}"',
                'enc_in="${ENC_IN:-12}"',
                'dec_in="${DEC_IN:-12}"',
            ],
            "Autoformer_5min.sh": [
                'data_path="${DATA_PATH:-aemo_vic1_dispatchis_vic1_full_5min.csv}"',
                'pred_lens="${PRED_LENS:-24 48 96 288}"',
                'seq_len="${SEQ_LEN:-288}"',
                'label_len="${LABEL_LEN:-144}"',
                'freq="${FREQ:-5min}"',
                'enc_in="${ENC_IN:-12}"',
                'dec_in="${DEC_IN:-12}"',
            ],
            "PatchTST_5min.sh": [
                'data_path="${DATA_PATH:-aemo_vic1_dispatchis_vic1_full_5min.csv}"',
                'pred_lens="${PRED_LENS:-24 48 96 288}"',
                'seq_len="${SEQ_LEN:-288}"',
                'label_len="${LABEL_LEN:-144}"',
                'freq="${FREQ:-5min}"',
                'enc_in="${ENC_IN:-12}"',
                'dec_in="${DEC_IN:-12}"',
            ],
            "TimesNet_5min.sh": [
                'model_name=TimesNet',
                'data_path="${DATA_PATH:-aemo_vic1_dispatchis_vic1_full_5min.csv}"',
                'pred_lens="${PRED_LENS:-24 48 96 288}"',
                'seq_len="${SEQ_LEN:-288}"',
                'label_len="${LABEL_LEN:-144}"',
                'freq="${FREQ:-5min}"',
                'enc_in="${ENC_IN:-12}"',
                'dec_in="${DEC_IN:-12}"',
            ],
            "TimeXer_5min.sh": [
                'data_path="${DATA_PATH:-aemo_vic1_dispatchis_vic1_full_5min.csv}"',
                'pred_lens="${PRED_LENS:-24 48 96 288}"',
                'seq_len="${SEQ_LEN:-288}"',
                'label_len="${LABEL_LEN:-144}"',
                'freq="${FREQ:-5min}"',
                'enc_in="${ENC_IN:-12}"',
                'dec_in="${DEC_IN:-12}"',
            ],
        }

        for script_name in FIVE_MIN_BASELINE_SCRIPTS:
            script_path = SCRIPT_DIR / script_name
            self.assertTrue(script_path.exists(), f"missing {script_path}")
            content = script_path.read_text()
            for expected in expected_snippets[script_name]:
                self.assertIn(expected, content, f"{script_name} missing {expected}")
            self.assertIn("--c_out 1", content, f"{script_name} missing --c_out 1")

    def test_run_5min_runner_is_5min_only(self):
        content = (SCRIPT_DIR / "run_5min.sh").read_text()
        self.assertIn("DLinear_5min", content)
        self.assertIn("PatchTST_5min", content)
        self.assertIn("Informer_5min", content)
        self.assertIn("Autoformer_5min", content)
        self.assertIn("TimesNet_5min", content)
        self.assertIn("TimeXer_5min", content)
        self.assertIn("VPPGDFNet_5min", content)
        self.assertIn('PRED_LENS="${PRED_LENS:-24 48 96 288}"', content)
        self.assertIn('SEQ_LEN="${SEQ_LEN:-288}"', content)
        self.assertNotIn("TimeXer_15min.sh", content)
        self.assertNotIn("TimeXer_30min.sh", content)
        self.assertNotIn("TimeXer_1h.sh", content)

    def test_run_5min_matrix_26_skips_only_two_24_step_experiments(self):
        content = (SCRIPT_DIR / "run_5min_matrix_26.sh").read_text()
        for model in ["DLinear", "PatchTST", "Informer", "Autoformer", "TimesNet", "TimeXer", "VPPGDFNet"]:
            self.assertIn(model, content)
        self.assertIn("pred_lens=(24 48 96 288)", content)
        self.assertIn('[[ "${model}" == "DLinear" && "${pred_len}" == "24" ]]', content)
        self.assertIn('[[ "${model}" == "TimeXer" && "${pred_len}" == "24" ]]', content)
        self.assertIn("summary.csv", content)
        self.assertIn("status=\"failed\"", content)
        self.assertIn("continue", content)

    def test_dlinear_global_timexer_runner_exists(self):
        runner_path = SCRIPT_DIR / "run_dlinear_global_timexer.sh"
        self.assertTrue(runner_path.exists(), f"missing {runner_path}")
        content = runner_path.read_text()
        self.assertIn("DLinearGlobalTimeXer_5min.sh", content)

    def test_vpp_gdfnet_runner_exists(self):
        runner_path = SCRIPT_DIR / "run_vpp_gdfnet.sh"
        self.assertTrue(runner_path.exists(), f"missing {runner_path}")
        content = runner_path.read_text()
        self.assertIn("VPPGDFNet_5min.sh", content)

    def test_vpp_gdfnet_ablation_runner_is_serial_and_records_summary(self):
        runner_path = SCRIPT_DIR / "run_vppgdfnet_ablation_5min.sh"
        self.assertTrue(runner_path.exists(), f"missing {runner_path}")
        content = runner_path.read_text()

        for ablation in ["full", "no_exog", "unified_exog", "no_final_gate"]:
            self.assertIn(ablation, content)
        self.assertIn("pred_lens=(24 48 96 288)", content)
        self.assertIn("VPP_ABLATION", content)
        self.assertIn("summary.csv", content)
        self.assertIn("status=\"failed\"", content)
        self.assertNotIn("parallel", content)
        self.assertNotIn("xargs -P", content)


if __name__ == "__main__":
    unittest.main()
