"""V3：Autoformer 分解 + iTransformer(趋势/全局) + TimesNet(周期) + 门控融合

【拼接模块】
1) series_decomp（Autoformer）
2) iTransformer（全局依赖/趋势分支）
   - 来源：models/iTransformer.py
3) TimesNet（周期分支）
   - 来源：models/TimesNet.py
4) ScalarGate（简单门控融合胶水层，基于 x_enc 的均值生成权重）

【数据流】
- x_enc -> decomp -> seasonal, trend
- trend -> iTransformer -> pred_trend
- seasonal -> TimesNet -> pred_seasonal
- gate(x_enc) -> y = gate*pred_trend + (1-gate)*pred_seasonal
"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from models import iTransformer, TimesNet
from my_models.common import ScalarGate


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name

        self.decomp = series_decomp(configs.moving_avg)
        self.trend_model = iTransformer.Model(configs)
        self.seasonal_model = TimesNet.Model(configs)
        self.gate = ScalarGate(in_dim=configs.enc_in)

        # 这两条分支通常都输出 c_out；若不一致可再补投影
        self.proj_trend = None
        self.proj_seasonal = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            seasonal, trend = self.decomp(x_enc)
            pred_trend = self.trend_model(trend, x_mark_enc, x_dec, x_mark_dec, mask)
            pred_seasonal = self.seasonal_model(seasonal, x_mark_enc, x_dec, x_mark_dec, mask)
            g = self.gate(x_enc)
            return g * pred_trend + (1.0 - g) * pred_seasonal
        return self.seasonal_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
