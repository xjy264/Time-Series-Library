import types
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - depends on local training environment
    torch = None


class DLinearGlobalTimeXerTest(unittest.TestCase):
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

        from models.DLinearGlobalTimeXer import Model

        configs = self._configs()
        model = Model(configs)
        x_enc = torch.randn(2, configs.seq_len, configs.enc_in)
        x_mark_enc = torch.randn(2, configs.seq_len, 5)
        x_dec = torch.randn(2, configs.label_len + configs.pred_len, configs.dec_in)
        x_mark_dec = torch.randn(2, configs.label_len + configs.pred_len, 5)

        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        self.assertEqual(output.shape, (2, configs.pred_len, 1))

    def test_dlinear_global_token_uses_target_channel_only(self):
        if torch is None:
            self.skipTest("torch is required for model forward tests")

        from models.DLinearGlobalTimeXer import Model

        configs = self._configs()
        model = Model(configs)
        model.eval()

        x_enc = torch.randn(2, configs.seq_len, configs.enc_in)
        changed_exog = x_enc.clone()
        changed_exog[:, :, :-1] = torch.randn_like(changed_exog[:, :, :-1]) * 100.0

        with torch.no_grad():
            token_a = model.dlinear_global_token(x_enc)
            token_b = model.dlinear_global_token(changed_exog)

        self.assertTrue(torch.allclose(token_a, token_b))

    def test_model_source_wires_dlinear_target_token_to_cross_attention(self):
        content = Path("models/DLinearGlobalTimeXer.py").read_text()

        self.assertIn("class DLinearGlobalToken", content)
        self.assertIn("target_history = x_enc[:, :, -1:].contiguous()", content)
        self.assertIn("DataEmbedding_inverted", content)
        self.assertIn("global_token = layer(global_token, variable_tokens)", content)
        self.assertIn("return dec_out[:, -self.pred_len:, :]", content)


if __name__ == "__main__":
    unittest.main()
