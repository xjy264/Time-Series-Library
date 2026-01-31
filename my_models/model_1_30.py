"""my_models/model_1_30.py

Model_1_30: Autoformer-style decomposition + (TimesNet for seasonal) + (DLinear for trend).

What this model does
--------------------
Given an input time series x_enc (encoder input), we decompose it into:
  trend   = MovingAverage(x_enc)
  seasonal = x_enc - trend
This is the same *series decomposition* idea used in Autoformer.

Then we forecast each component separately:
  seasonal_pred = TimesNet(seasonal)
  trend_pred    = DLinear_trend_only(trend)

Finally we combine the forecasts:
  y_pred = seasonal_pred + trend_pred

Why this design
---------------
- Seasonal/periodic patterns: TimesNet is strong at periodicity discovery and modeling.
- Trend/slow dynamics: DLinear is a strong linear baseline for trend-dominant components.

Input/Output shapes (forecasting)
---------------------------------
- x_enc:       [B, seq_len, enc_in]
- x_mark_enc:  [B, seq_len, mark_dim] or None (depends on dataset)
- x_dec:       [B, label_len + pred_len, dec_in]
- x_mark_dec:  [B, label_len + pred_len, mark_dim] or None

Returns:
- dec_out: [B, pred_len, c_out]

Config fields used (from args/configs)
--------------------------------------
- task_name, seq_len, label_len, pred_len
- enc_in, c_out
- moving_avg (window size for decomposition)
- TimesNet-related: d_model, e_layers, d_ff, top_k, num_kernels, embed, freq, dropout, etc.
- DLinear-related: individual (optional)

How to select from training scripts
-----------------------------------
After registering in exp/exp_basic.py, use:
  --model Model_1_30

Notes
-----
- We reuse the repo's existing TimesNet implementation (`models/TimesNet.py`).
- For the trend branch, we reuse DLinear's Linear_Trend weights but *do not* re-decompose.
  We directly apply DLinear's trend linear projection to our trend component.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from models import TimesNet
from models import DLinear


class Model(nn.Module):
    """Hybrid model: decomposition + TimesNet(seasonal) + DLinear(trend-only)."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Autoformer-style series decomposition
        # returns (seasonal, trend)
        self.decomp = series_decomp(configs.moving_avg)

        # Seasonal predictor: TimesNet
        self.seasonal_model = TimesNet.Model(configs)

        # Trend predictor: reuse DLinear's trend projection weights
        # but apply them ONLY on our trend component.
        self._dlinear = DLinear.Model(configs, individual=getattr(configs, "individual", False))
        self.individual = getattr(configs, "individual", False)
        self.channels = configs.enc_in

        # If c_out != enc_in, we add a projection layer to match output dims.
        # Many models in this repo already project to c_out, but our trend path uses enc_in.
        self.out_proj = None
        if getattr(configs, "c_out", configs.enc_in) != configs.enc_in:
            self.out_proj = nn.Linear(configs.enc_in, configs.c_out, bias=True)

    def _trend_forecast(self, trend: torch.Tensor) -> torch.Tensor:
        """Forecast the trend component using DLinear's Linear_Trend weights.

        Args:
            trend: [B, seq_len, enc_in]

        Returns:
            trend_pred: [B, pred_len, enc_in]
        """
        # DLinear expects applying Linear_* on (B, C, L)
        trend_t = trend.permute(0, 2, 1)  # [B, C, L]

        if self.individual:
            # ModuleList per-channel
            outs = []
            for i in range(self.channels):
                out_i = self._dlinear.Linear_Trend[i](trend_t[:, i, :])  # [B, pred_len]
                outs.append(out_i.unsqueeze(1))  # [B, 1, pred_len]
            trend_out = torch.cat(outs, dim=1)  # [B, C, pred_len]
        else:
            trend_out = self._dlinear.Linear_Trend(trend_t)  # [B, C, pred_len]

        return trend_out.permute(0, 2, 1)  # [B, pred_len, C]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # We only define a forecasting behavior consistent with other models.
        # For non-forecast tasks, we fall back to using seasonal_model behavior.
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            seasonal, trend = self.decomp(x_enc)  # both [B, seq_len, C]

            # Seasonal forecast via TimesNet. TimesNet returns [B, pred_len, c_out] for forecast tasks.
            seasonal_pred = self.seasonal_model(seasonal, x_mark_enc, x_dec, x_mark_dec, mask)
            # Trend forecast via DLinear trend projection (returns [B, pred_len, enc_in])
            trend_pred = self._trend_forecast(trend)

            if self.out_proj is not None:
                trend_pred = self.out_proj(trend_pred)

            # Align dims and sum
            # seasonal_pred is [B, pred_len, c_out]
            # trend_pred is [B, pred_len, c_out]
            return seasonal_pred + trend_pred

        # Other tasks: keep it simple and reuse TimesNet behavior.
        return self.seasonal_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
