import re
import unittest
from pathlib import Path


SCRIPT_PAIRS = [
    (Path("scripts/long_term_forecast/ETT_script/TimeXer_ETTh1.sh"), Path("scripts/long_term_forecast/ETT_script/VPPGDFNet_ETTh1.sh")),
    (Path("scripts/long_term_forecast/ETT_script/TimeXer_ETTh2.sh"), Path("scripts/long_term_forecast/ETT_script/VPPGDFNet_ETTh2.sh")),
    (Path("scripts/long_term_forecast/ETT_script/TimeXer_ETTm1.sh"), Path("scripts/long_term_forecast/ETT_script/VPPGDFNet_ETTm1.sh")),
    (Path("scripts/long_term_forecast/ETT_script/TimeXer_ETTm2.sh"), Path("scripts/long_term_forecast/ETT_script/VPPGDFNet_ETTm2.sh")),
    (Path("scripts/long_term_forecast/Weather_script/TimeXer.sh"), Path("scripts/long_term_forecast/Weather_script/VPPGDFNet.sh")),
    (Path("scripts/long_term_forecast/ECL_script/TimeXer.sh"), Path("scripts/long_term_forecast/ECL_script/VPPGDFNet.sh")),
    (Path("scripts/long_term_forecast/Traffic_script/TimeXer.sh"), Path("scripts/long_term_forecast/Traffic_script/VPPGDFNet.sh")),
]

ALLOWED_DIFFERENT_OPTIONS = {"--model_id", "--model", "--des"}


def command_options(content):
    options = []
    for line in content.splitlines():
        stripped = line.strip().rstrip("\\").strip()
        if stripped.startswith("--"):
            name, _, value = stripped.partition(" ")
            options.append((name, value.strip()))
    return options


class GeneralVPPGDFNetScriptsTest(unittest.TestCase):
    def test_vppgdfnet_scripts_keep_timexer_parameters(self):
        for timexer_path, vpp_path in SCRIPT_PAIRS:
            self.assertTrue(vpp_path.exists(), f"missing {vpp_path}")
            timexer_content = timexer_path.read_text()
            vpp_content = vpp_path.read_text()
            self.assertIn("model_name=VPPGDFNet", vpp_content)
            self.assertIn('pred_lens="${PRED_LENS:-96 192 336 720}"', vpp_content)
            self.assertNotIn("model_name=TimeXer", vpp_content)
            self.assertIn("--des 'VPPGDFNet-TimeXerParams'", vpp_content)

            timexer_options = command_options(timexer_content)
            vpp_options = command_options(vpp_content)
            self.assertEqual(len(timexer_options), len(vpp_options), f"option count mismatch for {vpp_path}")
            for (time_name, time_value), (vpp_name, vpp_value) in zip(timexer_options, vpp_options):
                self.assertEqual(time_name, vpp_name, f"option name mismatch for {vpp_path}")
                if time_name in ALLOWED_DIFFERENT_OPTIONS:
                    continue
                self.assertEqual(time_value, vpp_value, f"{vpp_path} changed {time_name}")

    def test_runner_is_serial_and_writes_average_outputs(self):
        runner = Path("scripts/long_term_forecast/run_vppgdfnet_timexer_params_matrix.sh")
        self.assertTrue(runner.exists(), f"missing {runner}")
        content = runner.read_text()
        self.assertIn("datasets=(ETTh1 ETTh2 ETTm1 ETTm2 Weather ECL Traffic)", content)
        self.assertIn("pred_lens=(96 192 336 720)", content)
        self.assertIn("summary.csv", content)
        self.assertIn("averages.csv", content)
        self.assertIn("avg_mse", content)
        self.assertIn("ALL", content)
        self.assertNotIn("parallel", content)
        self.assertNotIn("xargs -P", content)
        self.assertFalse(re.search(r"bash \"\$\{script_path\}\"[^\n]*(?<!2)\s&(?:\s|$)", content))


if __name__ == "__main__":
    unittest.main()
