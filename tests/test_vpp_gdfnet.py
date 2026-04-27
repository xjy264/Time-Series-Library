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

    def test_model_source_uses_dual_cross_attention_and_final_branch_gate_only(self):
        content = Path("models/VPPGDFNet.py").read_text()

        self.assertIn("trend_query", content)
        self.assertIn("seasonal_query", content)
        self.assertIn("trend_fusion_layers", content)
        self.assertIn("seasonal_fusion_layers", content)
        self.assertIn("exog_tokens = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)", content)
        self.assertIn("branch_gate", content)
        self.assertIn("beta * trend_pred + (1 - beta) * seasonal_pred", content)
        self.assertNotIn("GVS", content)
        self.assertNotIn("sparse", content.lower())
        self.assertNotIn("variable_gate", content)


if __name__ == "__main__":
    unittest.main()
