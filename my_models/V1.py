"""V1：Autoformer 分解 + TimesNet(周期) + DLinear(趋势)

【拼接模块】
1) series_decomp（来自 Autoformer 的时序分解模块）
   - 作用：把输入序列拆成 seasonal（周期/波动）与 trend（趋势）
   - 来源：layers/Autoformer_EncDec.py
2) TimesNet（来自 TimesNet 模型主体）
   - 作用：对周期性/多周期成分进行建模与预测
   - 来源：models/TimesNet.py
3) DLinear（来自 DLinear 的趋势线性投影）
   - 作用：对趋势部分做线性外推，稳定且可解释
   - 来源：models/DLinear.py

【数据流】
- x_enc -> series_decomp -> seasonal, trend
- seasonal -> TimesNet -> seasonal_pred
- trend -> DLinear.Linear_Trend(仅趋势头) -> trend_pred
- y = seasonal_pred + trend_pred

接口与其它模型保持一致：forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
输出：[B, pred_len, c_out]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from models import TimesNet, DLinear


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.decomp = series_decomp(configs.moving_avg)
        self.seasonal_model = TimesNet.Model(configs)

        # 用 DLinear 的趋势线性层，但不重复做分解
        self._dlinear = DLinear.Model(configs, individual=getattr(configs, "individual", False))
        self.individual = getattr(configs, "individual", False)
        self.channels = configs.enc_in

        self.out_proj = None
        if getattr(configs, "c_out", configs.enc_in) != configs.enc_in:
            self.out_proj = nn.Linear(configs.enc_in, configs.c_out, bias=True)

    def _trend_forecast(self, trend: torch.Tensor) -> torch.Tensor:
        # trend: [B, seq_len, enc_in] -> [B, pred_len, enc_in]
        trend_t = trend.permute(0, 2, 1)  # [B,C,L]
        if self.individual:
            outs = []
            for i in range(self.channels):
                out_i = self._dlinear.Linear_Trend[i](trend_t[:, i, :])
                outs.append(out_i.unsqueeze(1))
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
