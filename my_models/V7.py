"""V7：PatchTST(patch注意力) + TimeMixer/TSMixer(MLP混合) + 门控融合

【拼接模块】
1) PatchTST（patch-based Transformer）
   - 来源：models/PatchTST.py
2) TimeMixer（Mixer 结构，MLP 混合时间/通道）
   - 来源：models/TimeMixer.py
3) ScalarGate（门控融合）

【数据流】
- pred_attn = PatchTST(x_enc,...)
- pred_mlp  = TimeMixer(x_enc,...)
- y = g*pred_attn + (1-g)*pred_mlp

意义：Attention 与 MLP Mixer 两类归纳偏置互补。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models import PatchTST, TimeMixer
from my_models.common import ScalarGate


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.attn = PatchTST.Model(configs)
        self.mlp = TimeMixer.Model(configs)
        self.gate = ScalarGate(in_dim=configs.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            pa = self.attn(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            pm = self.mlp(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            g = self.gate(x_enc)
            return g * pa + (1.0 - g) * pm
        return self.attn(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
