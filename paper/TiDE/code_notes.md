# TiDE 代码笔记

## 文件位置

`models/TiDE.py`（thuml/Time-Series-Library）

---

## 1. 当前实现的整体结构

```python
class Model(nn.Module):
    def __init__(self, configs, bias=True, feature_encode_dim=2):
        self.feature_encoder = ResBlock(...)
        self.encoders = nn.Sequential(...)
        self.decoders = nn.Sequential(...)
        self.temporalDecoder = ResBlock(...)
        self.residual_proj = nn.Linear(...)
```

它的主线非常清楚：

1. 编码时间 / 动态特征
2. 编码历史序列
3. 解码未来表示
4. 与残差预测相加

---

## 2. 最关键的设计：动态特征路径

```python
feature = self.feature_encoder(batch_y_mark)
hidden = self.encoders(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
```

这里说明：

- `batch_y_mark` 是动态协变量路径
- 它会和历史序列一起进入编码器

在长预测任务里，代码会先把：

- 历史时间特征 `x_mark_enc`
- 未来时间特征 `x_mark_dec`

拼接起来，再送进 `feature_encoder`。

因此这份实现确实体现了：

- 未来已知特征可以进入模型

---

## 3. 当前实现的一个重要边界

`TiDE.py` 在当前仓库里的实现主要围绕：

- 时间特征
- 统一 forecasting 流程

它并没有天然为“任意未来天气协变量表”提供单独的数据管线接口。

也就是说：

- 如果你只用时间特征，它可以直接工作
- 如果你想把 NOAA 未来天气预报完整接进去，通常还需要改数据管线

---

## 4. 与当前净负荷任务的关系

### 4.1 可以直接借鉴的部分

- 未来动态协变量作为独立输入路径
- 轻量残差 MLP 结构
- 对多步预测友好的输出方式

### 4.2 不足之处

- 当前实现更偏“时间特征已知未来”
- 不是“天气 / 风光 /系统协变量分组建模”
- 也没有 `TimeXer` 那样明确的内生 / 外生 Cross-Attention

---

## 5. 对当前研究的直接启发

### 启发 1

如果你想做轻量对照，`TiDE` 很适合承担：

- “已知未来协变量是否有效”的验证任务

### 启发 2

如果后续要做更复杂方法，可以先用 `TiDE` 风格的协变量路径验证：

- 天气
- 时间特征
- 风光代理量

哪些真的有增益。
