"""V2：Autoformer 分解 + PatchTST(周期/局部模式) + DLinear(趋势)

【拼接模块】
1) series_decomp（Autoformer 的分解） -> seasonal/trend
2) PatchTST（patch-based Transformer）
   - 适合捕捉局部重复形状、短周期模式
   - 来源：models/PatchTST.py
3) DLinear（趋势线性外推）
   - 来源：models/DLinear.py

【数据流】
- x_enc -> decomp -> seasonal, trend
- seasonal -> PatchTST -> seasonal_pred
- trend -> DLinear.Linear_Trend -> trend_pred
- y = seasonal_pred + trend_pred
"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from models import PatchTST, DLinear


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.decomp = series_decomp(configs.moving_avg)
        self.seasonal_model = PatchTST.Model(configs)
        self._dlinear = DLinear.Model(configs, individual=getattr(configs, "individual", False))
        self.individual = getattr(configs, "individual", False)
        self.channels = configs.enc_in

        self.out_proj = None
        if getattr(configs, "c_out", configs.enc_in) != configs.enc_in:
            self.out_proj = nn.Linear(configs.enc_in, configs.c_out, bias=True)

    def _trend_forecast(self, trend: torch.Tensor) -> torch.Tensor:
        trend_t = trend.permute(0, 2, 1)
        if self.individual:
            outs = []
            for i in range(self.channels):
                outs.append(self._dlinear.Linear_Trend[i](trend_t[:, i, :]).unsqueeze(1))
            trend_out = torch.cat(outs, dim=1)
        else:
            trend_out = self._dlinear.Linear_Trend(trend_t)
        return trend_out.permute(0, 2, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            seasonal, trend = self.decomp(x_enc)
            seasonal_pred = self.seasonal_model(seasonal, x_mark_enc, x_dec, x_mark_dec, mask)
            trend_pred = self._trend_forecast(trend)
            if self.out_proj is not None:
                trend_pred = self.out_proj(trend_pred)
            return seasonal_pred + trend_pred
        return self.seasonal_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
