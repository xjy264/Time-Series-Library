"""V10：iTransformer(teacher/全局) + DLinear(student/稳定) + TimesNet(residual/周期) 的三路融合

【拼接模块】
1) iTransformer（全局强模型分支）
   - 来源：models/iTransformer.py
2) DLinear（稳定轻量分支）
   - 来源：models/DLinear.py
3) TimesNet（残差/周期补偿分支）
   - 来源：models/TimesNet.py
4) ScalarGate（融合胶水层：根据输入决定更相信 iTransformer 还是 DLinear+TimesNet）

【数据流】
- p_teacher = iTransformer(x_enc,...)
- p_student = DLinear(x_enc,...)
- p_resid   = TimesNet(x_enc,...)
- p_student2 = p_student + p_resid
- y = g*p_teacher + (1-g)*p_student2

意义：在不同数据上自适应选择强模型 vs 轻模型+补偿，保持鲁棒与上限。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models import iTransformer, DLinear, TimesNet
from my_models.common import ScalarGate


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.teacher = iTransformer.Model(configs)
        self.student = DLinear.Model(configs, individual=getattr(configs, "individual", False))
        self.resid = TimesNet.Model(configs)
        self.gate = ScalarGate(in_dim=configs.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            pt = self.teacher(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            ps = self.student(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            pr = self.resid(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            g = self.gate(x_enc)
            return g * pt + (1.0 - g) * (ps + pr)
        return self.teacher(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
