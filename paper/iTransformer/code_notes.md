# iTransformer 代码笔记

## 文件位置

`models/iTransformer.py`（thuml/Time-Series-Library）

---

## 1. 代码主线很干净

```python
class Model(nn.Module):
    def __init__(self, configs):
        self.enc_embedding = DataEmbedding_inverted(...)
        self.encoder = Encoder([...])
        self.projection = nn.Linear(configs.d_model, configs.pred_len)
```

结构非常简单：

1. 倒置 embedding
2. 标准 Transformer Encoder
3. 线性预测头

这也是 iTransformer 的优点之一：

- 改动小
- 容易复用
- 很适合做强基线

---

## 2. 最关键的一步：倒置 embedding

```python
self.enc_embedding = DataEmbedding_inverted(
    configs.seq_len, configs.d_model, ...
)
```

这里的关键不是名字，而是输入维度：

- `c_in = configs.seq_len`

这说明在 `DataEmbedding_inverted` 里，模型把**时间长度**当成每个 token 的输入特征维度。

核心操作是：

```python
x = x.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
```

这一步之后：

- token 数量 = 变量数 `C`
- 每个 token 的特征 = 整段历史长度 `L`

---

## 3. 前向逻辑

```python
def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    means = x_enc.mean(1, keepdim=True).detach()
    x_enc = x_enc - means
    stdev = ...
    x_enc /= stdev

    enc_out = self.enc_embedding(x_enc, x_mark_enc)
    enc_out, attns = self.encoder(enc_out, attn_mask=None)

    dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
    dec_out = dec_out * ...
    dec_out = dec_out + ...
    return dec_out
```

要点：

- 先做归一化
- 再做倒置 embedding
- 再做变量维度的 Attention
- 最后线性映射到预测长度

---

## 4. 它与当前净负荷任务的关系

### 4.1 可以直接做什么

如果你把 AEMO + NOAA 合并成一个多变量序列矩阵，iTransformer 可以直接作为强基线使用。

它可以回答：

- 在不显式区分内生 / 外生的情况下
- 仅通过变量关系建模
- 能把净负荷预测做到什么水平

### 4.2 做不到什么

当前实现并没有专门区分：

- 历史目标
- 历史协变量
- 未来已知协变量

所以它不适合作为“已知未来外生变量机制”的直接表达。

---

## 5. 对当前研究最有用的代码启发

### 启发 1：倒置 token 化可作为变量关系模块

如果未来你想增强主方法，可以借鉴：

- 把系统侧协变量当作变量 token 做关系建模

### 启发 2：归一化策略值得保留

`iTransformer` 在前向里固定做了非平稳归一化，这对电力时序很常见，也对净负荷场景有实际价值。

### 启发 3：它适合做对照，而不是承载研究叙事

从代码上看，它非常适合做：

- 干净
- 易复现
- 强多变量

的 baseline。

但如果要写“为什么未来外生信息重要”，单靠这份实现不够。
