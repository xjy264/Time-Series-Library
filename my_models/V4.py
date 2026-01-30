"""V4：DLinear(基线) + TimesNet(残差) + Autoformer 分解(可选输入增强)

【拼接模块】
1) DLinear（强基线，稳定）
   - 来源：models/DLinear.py
2) series_decomp（Autoformer 分解，用于给残差分支提供更“周期性”的输入）
   - 来源：layers/Autoformer_EncDec.py
3) TimesNet（拟合 DLinear 预测无法覆盖的残差/高频部分）
   - 来源：models/TimesNet.py

【数据流】
- base = DLinear(x_enc)
- seasonal = x_enc - moving_avg(x_enc)
- resid = TimesNet(seasonal)
- y = base + resid

意义：经典“基线 + 残差修正”结构，训练更稳，且让 TimesNet 专注于周期/复杂波动。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from models import DLinear, TimesNet


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.base = DLinear.Model(configs, individual=getattr(configs, "individual", False))
        self.decomp = series_decomp(configs.moving_avg)
        self.residual = TimesNet.Model(configs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            base_pred = self.base(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            seasonal, _trend = self.decomp(x_enc)
            resid_pred = self.residual(seasonal, x_mark_enc, x_dec, x_mark_dec, mask)
            return base_pred + resid_pred
        return self.base(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
