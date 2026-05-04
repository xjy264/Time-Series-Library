import types
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - depends on local training environment
    torch = None


class VPPGDFNetTest(unittest.TestCase):
    def _configs(self):
        return types.SimpleNamespace(
            task_name="long_term_forecast",
            features="MS",
            seq_len=24,
            label_len=6,
            pred_len=4,
            enc_in=12,
            dec_in=12,
            c_out=1,
            d_model=16,
            embed="timeF",
            freq="5min",
            dropout=0.0,
            factor=3,
            n_heads=4,
            e_layers=1,
            d_ff=32,
            activation="gelu",
            moving_avg=3,
            use_norm=0,
        )

    def test_ms_forecast_returns_only_target_channel(self):
        if torch is None:
            self.skipTest("torch is required for model forward tests")

        from models.VPPGDFNet import Model

        configs = self._configs()
        model = Model(configs)
        x_enc = torch.randn(2, configs.seq_len, configs.enc_in)
        x_mark_enc = torch.randn(2, configs.seq_len, 5)
        x_dec = torch.randn(2, configs.label_len + configs.pred_len, configs.dec_in)
        x_mark_dec = torch.randn(2, configs.label_len + configs.pred_len, 5)

        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        self.assertEqual(output.shape, (2, configs.pred_len, 1))


    def test_m_forecast_returns_all_output_channels(self):
        if torch is None:
            self.skipTest("torch is required for model forward tests")

        from models.VPPGDFNet import Model

        configs = self._configs()
        configs.features = "M"
        configs.enc_in = 7
        configs.dec_in = 7
        configs.c_out = 7
        configs.freq = "h"
        model = Model(configs)
        x_enc = torch.randn(2, configs.seq_len, configs.enc_in)
        x_mark_enc = torch.randn(2, configs.seq_len, 4)
        x_dec = torch.randn(2, configs.label_len + configs.pred_len, configs.dec_in)
        x_mark_dec = torch.randn(2, configs.label_len + configs.pred_len, 4)

        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        self.assertEqual(output.shape, (2, configs.pred_len, configs.c_out))

    def test_weather_like_m_forecast_returns_all_output_channels(self):
        if torch is None:
            self.skipTest("torch is required for model forward tests")

        from models.VPPGDFNet import Model

        configs = self._configs()
        configs.features = "M"
        configs.enc_in = 21
        configs.dec_in = 21
        configs.c_out = 21
        configs.freq = "t"
        model = Model(configs)
        x_enc = torch.randn(2, configs.seq_len, configs.enc_in)
        x_mark_enc = torch.randn(2, configs.seq_len, 5)
        x_dec = torch.randn(2, configs.label_len + configs.pred_len, configs.dec_in)
        x_mark_dec = torch.randn(2, configs.label_len + configs.pred_len, 5)

        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        self.assertEqual(output.shape, (2, configs.pred_len, configs.c_out))

    def test_target_decomposition_uses_target_channel_only(self):
        if torch is None:
            self.skipTest("torch is required for model forward tests")

        from models.VPPGDFNet import Model

        configs = self._configs()
        model = Model(configs)
        model.eval()

        x_enc = torch.randn(2, configs.seq_len, configs.enc_in)
        changed_exog = x_enc.clone()
        changed_exog[:, :, :-1] = torch.randn_like(changed_exog[:, :, :-1]) * 100.0

        with torch.no_grad():
            trend_a, seasonal_a = model.decompose_target(x_enc)
            trend_b, seasonal_b = model.decompose_target(changed_exog)

        self.assertTrue(torch.allclose(trend_a, trend_b))
        self.assertTrue(torch.allclose(seasonal_a, seasonal_b))

    def test_model_source_uses_dual_cross_attention_and_dlinear_style_additive_fusion(self):
        content = Path("models/VPPGDFNet.py").read_text()

        self.assertIn("trend_query", content)
        self.assertIn("seasonal_query", content)
        self.assertIn("trend_fusion_layers", content)
        self.assertIn("seasonal_fusion_layers", content)
        self.assertIn("exog_tokens = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)", content)
        self.assertIn("dec_out = trend_pred + seasonal_pred", content)
        self.assertNotIn("branch_gate", content)
        self.assertNotIn("beta * trend_pred + (1 - beta) * seasonal_pred", content)
        self.assertNotIn("0.5 * trend_pred + 0.5 * seasonal_pred", content)
        self.assertNotIn("GVS", content)
        self.assertNotIn("sparse", content.lower())
        self.assertNotIn("variable_gate", content)

    def test_supported_ablation_modes_return_expected_shape(self):
        if torch is None:
            self.skipTest("torch is required for model forward tests")

        from models.VPPGDFNet import Model

        for ablation in ["full", "no_exog", "unified_exog"]:
            with self.subTest(ablation=ablation):
                configs = self._configs()
                configs.vpp_ablation = ablation
                model = Model(configs)
                x_enc = torch.randn(2, configs.seq_len, configs.enc_in)
                x_mark_enc = torch.randn(2, configs.seq_len, 5)
                x_dec = torch.randn(2, configs.label_len + configs.pred_len, configs.dec_in)
                x_mark_dec = torch.randn(2, configs.label_len + configs.pred_len, 5)

                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

                self.assertEqual(output.shape, (2, configs.pred_len, 1))

    def test_unknown_ablation_mode_is_rejected(self):
        if torch is None:
            self.skipTest("torch is required for model forward tests")

        from models.VPPGDFNet import Model

        configs = self._configs()
        configs.vpp_ablation = "no_final_gate"

        with self.assertRaisesRegex(ValueError, "Unsupported VPPGDFNet ablation mode"):
            Model(configs)

    def test_source_declares_all_ablation_modes(self):
        model_content = Path("models/VPPGDFNet.py").read_text()
        run_content = Path("run.py").read_text()

        self.assertIn("vpp_ablation", model_content)
        self.assertIn("--vpp_ablation", run_content)
        for ablation in ["full", "no_exog", "unified_exog"]:
            self.assertIn(ablation, model_content)
            self.assertIn(ablation, run_content)
        self.assertNotIn("no_final_gate", model_content)
        self.assertNotIn("'no_final_gate'", run_content)


if __name__ == "__main__":
    unittest.main()
