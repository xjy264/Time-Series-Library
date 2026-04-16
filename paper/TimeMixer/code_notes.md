# TimeMixer 代码笔记

## 文件位置
`models/TimeMixer.py`（thuml/Time-Series-Library）

---

## 核心类结构

```python
class Model(nn.Module):
    # 主要超参数
    # configs.down_sampling_layers: 下采样层数 M（默认3）
    # configs.down_sampling_window: 下采样窗口大小（默认2）
    # configs.down_sampling_method: 'avg' / 'max' / 'conv'
    # configs.d_model, configs.d_ff
    # configs.channel_independence: 是否通道独立
```

---

## 关键组件

### 1. 多尺度输入生成
```python
# 平均池化下采样
self.down_pool = nn.AvgPool1d(
    kernel_size=configs.down_sampling_window,
    stride=configs.down_sampling_window,
    padding=0
)

def multi_scale_process_inputs(self, x_enc, x_mark_enc):
    # 生成多个尺度的输入
    down_inp_enc = []
    x_enc_ori = x_enc
    for i in range(configs.down_sampling_layers):
        x_enc_sampling = self.down_pool(x_enc_ori.permute(0, 2, 1)).permute(0, 2, 1)
        down_inp_enc.append(x_enc_sampling)
        x_enc_ori = x_enc_sampling
    # 返回: [x_full, x_half, x_quarter, ...]
    return [x_enc] + down_inp_enc
```

**对应论文**：Section 3.1 "Multiscale Input Strategy"

---

### 2. Past Decomposable Mixing（PDM）
```python
class PastDecomposableMixing(nn.Module):
    def forward(self, x_list):
        # x_list: [x_scale_0, x_scale_1, ..., x_scale_M]
        # 每个 x_scale_m: [B, L_m, C]

        # 分解：趋势 + 季节性
        x_trend_list = []
        x_seasonal_list = []
        for x in x_list:
            x_trend = self.moving_avg(x)           # 趋势（移动平均）
            x_seasonal = x - x_trend               # 季节性（残差）
            x_trend_list.append(x_trend)
            x_seasonal_list.append(x_seasonal)

        # Bottom-up Mixing（粗→细，传递趋势）
        for i in range(len(x_trend_list) - 1, 0, -1):
            # 将粗尺度趋势上采样后加到细尺度
            x_trend_list[i-1] = x_trend_list[i-1] + F.interpolate(
                x_trend_list[i].permute(0, 2, 1),
                size=x_trend_list[i-1].shape[1]
            ).permute(0, 2, 1)

        # Top-down Mixing（细→粗，传递季节性）
        for i in range(len(x_seasonal_list) - 1):
            # 将细尺度季节性下采样后加到粗尺度
            x_seasonal_list[i+1] = x_seasonal_list[i+1] + self.down_pool(
                x_seasonal_list[i].permute(0, 2, 1)
            ).permute(0, 2, 1)

        # 合并：趋势 + 季节性
        out_list = [t + s for t, s in zip(x_trend_list, x_seasonal_list)]
        return out_list
```

**对应论文**：Section 3.2 "Past Decomposable Mixing"

---

### 3. Future Multipredictor Mixing（FMM）
```python
class FutureMixing(nn.Module):
    def __init__(self):
        # 每个尺度一个线性预测器
        self.predictors = nn.ModuleList([
            nn.Linear(L_m, pred_len) for L_m in scale_lengths
        ])
        # 可学习融合权重
        self.mixing_weights = nn.Parameter(torch.ones(M+1) / (M+1))

    def forward(self, h_list):
        preds = []
        for m, (h, pred) in enumerate(zip(h_list, self.predictors)):
            # h: [B, L_m, C] → [B, pred_len, C]
            preds.append(pred(h.permute(0, 2, 1)).permute(0, 2, 1))

        # 加权融合
        weights = F.softmax(self.mixing_weights, dim=0)
        out = sum(w * p for w, p in zip(weights, preds))
        return out
```

**对应论文**：Section 3.3 "Future Multipredictor Mixing"

---

## Forward 总流程

```python
def forward(self, x_enc, ...):
    # 1. 生成多尺度输入
    x_list = self.multi_scale_process_inputs(x_enc)
    # x_list = [x@L, x@L/2, x@L/4, ...]

    # 2. Embedding（每个尺度独立）
    enc_list = [self.enc_embedding(x) for x in x_list]

    # 3. N 层 Past Decomposable Mixing
    for layer in self.pdm_blocks:
        enc_list = layer(enc_list)

    # 4. Future Multipredictor Mixing → 预测输出
    dec_out = self.future_mixing(enc_list)  # [B, H, C]
    return dec_out
```

---

## 与论文对应关系

| 论文模块 | 代码位置 |
|---------|---------|
| Multiscale Input | `multi_scale_process_inputs()` |
| Seasonal-Trend Decomposition | `PastDecomposableMixing` 中的 `moving_avg` |
| Bottom-up Mixing | PDM 中从高索引到低索引的趋势传递 |
| Top-down Mixing | PDM 中从低索引到高索引的季节性传递 |
| Future Mixing | `FutureMixing` 中的加权融合 |

---

## 复用建议

1. **多尺度输入**：可在任何时序模型前加 `multi_scale_process_inputs` 作为预处理
2. **PDM 模块**：作为独立的特征增强模块插入其他架构
3. **FMM 思想**：多预测器集成可用于提升电力负荷预测的鲁棒性
4. **下采样粒度**：对电力数据建议 window=4（对应 15min → 1h → 4h 粒度）
