"""V9：FEDformer 风格多尺度分解 + TimesNet(周期) + DLinear(趋势)

【拼接模块】
1) series_decomp_multi（来自 FEDformer 的多核分解，能处理多周期共存）
   - 来源：layers/Autoformer_EncDec.py
2) TimesNet（周期/多周期预测）
   - 来源：models/TimesNet.py
3) DLinear（趋势线性外推）
   - 来源：models/DLinear.py

【数据流】
- x_enc -> series_decomp_multi -> seasonal, trend
- seasonal -> TimesNet -> seasonal_pred
- trend -> DLinear.Linear_Trend -> trend_pred
- y = seasonal_pred + trend_pred

注意：configs.moving_avg 在本仓库是 int；这里我们自动构造一个 kernel 列表 [moving_avg//2, moving_avg, moving_avg*2]。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp_multi
from models import TimesNet, DLinear


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name

        k = int(getattr(configs, "moving_avg", 25))
        kernels = [max(3, k // 2), max(3, k), max(3, k * 2)]
        # 保证都是奇数（与 moving_avg padding 逻辑一致）
        kernels = [kk if kk % 2 == 1 else kk + 1 for kk in kernels]

        self.decomp = series_decomp_multi(kernels)
        self.seasonal_model = TimesNet.Model(configs)

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
