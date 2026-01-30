"""V6：MICN(局部/卷积多尺度) + iTransformer(全局依赖) + 门控融合

【拼接模块】
1) MICN（局部模式/多尺度卷积特征）
   - 来源：models/MICN.py
2) iTransformer（全局依赖建模）
   - 来源：models/iTransformer.py
3) ScalarGate（门控融合）

【数据流】
- pred_local = MICN(x_enc,...)
- pred_global = iTransformer(x_enc,...)
- y = g*pred_global + (1-g)*pred_local

意义：卷积分支抓局部形状，Transformer 分支抓长程依赖。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models import MICN, iTransformer
from my_models.common import ScalarGate


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.local = MICN.Model(configs)
        self.global_ = iTransformer.Model(configs)
        self.gate = ScalarGate(in_dim=configs.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            pl = self.local(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            pg = self.global_(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            g = self.gate(x_enc)
            return g * pg + (1.0 - g) * pl
        return self.global_(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
