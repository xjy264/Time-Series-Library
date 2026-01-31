"""V5：Autoformer(分解+自相关注意力) + DLinear(稳健趋势头) + 门控融合

【拼接模块】
1) Autoformer（完整模型，含分解与 AutoCorrelation 注意力）
   - 来源：models/Autoformer.py
2) DLinear（轻量线性预测，作为稳健分支）
   - 来源：models/DLinear.py
3) ScalarGate（门控融合，基于输入生成权重）

【数据流】
- pred_a = Autoformer(x_enc, ...)
- pred_b = DLinear(x_enc, ...)
- g = gate(x_enc)
- y = g*pred_a + (1-g)*pred_b

意义：Autoformer 擅长复杂依赖，DLinear 作为稳定后备；门控让不同数据自适应选择。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models import Autoformer, DLinear
from my_models.common import ScalarGate


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.a = Autoformer.Model(configs)
        self.b = DLinear.Model(configs, individual=getattr(configs, "individual", False))
        self.gate = ScalarGate(in_dim=configs.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            pa = self.a(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            pb = self.b(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            g = self.gate(x_enc)
            return g * pa + (1.0 - g) * pb
        return self.a(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
