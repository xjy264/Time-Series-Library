# TimeXer 代码笔记

## 文件位置

`models/TimeXer.py`（thuml/Time-Series-Library）

---

## 1. 先看结论：论文设定与仓库实现并不完全相同

这点对后续研究非常重要。

### 论文层面

TimeXer 论文强调的是：

- 内生变量只有历史值
- 外生变量包含未来已知值
- 通过 Cross-Attention 利用未来外生信息

### 这个仓库中的实现

仓库里的 `models/TimeXer.py` 更像一个**兼容统一训练管线的落地版本**，并没有完全复刻“未来外生变量单独输入”的原始写法。

尤其要注意：

- 代码里真正进入 `ex_embedding` 的不是独立的 `x_mark_dec` 未来外生块
- 而是编码端输入中的协变量和时间特征

所以：

- **论文思想** 和 **当前仓库实现** 需要分开理解
- 如果后续你要针对 AEMO + NOAA 做研究，必须先明确你是复用当前实现，还是按照论文原意再做改造

---

## 2. 当前实现的核心结构

```python
class Model(nn.Module):
    def __init__(self, configs):
        self.en_embedding = EnEmbedding(...)
        self.ex_embedding = DataEmbedding_inverted(...)
        self.encoder = Encoder([...])
        self.head = FlattenHead(...)
```

模型可分为四块：

1. `en_embedding`
   - 对内生序列做 patch 化表示
2. `ex_embedding`
   - 对外部协变量表示做嵌入
3. `encoder`
   - 先做内生 self-attention，再做内生对外生的 cross-attention
4. `head`
   - 输出未来预测值

---

## 3. 两条前向路径

### 3.1 `features == 'M'`

对应：

- 多变量输入
- 多变量输出

调用：

```python
dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
```

在这条路径里：

- `x_enc` 的所有变量都被当作内生变量处理
- `ex_embed = self.ex_embedding(x_enc, x_mark_enc)`

这意味着这里并没有严格的“目标变量 vs 外生变量”切分，更接近标准多变量建模。

### 3.2 `features == 'MS'`

对应：

- 多变量输入
- 单变量输出

调用：

```python
dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
```

在这条路径里：

```python
en_embed = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)
```

这里才比较接近论文中的“内生 / 外生”分工：

- 最后一列：目标变量（内生变量）
- 前面各列：协变量（外生变量）

因此如果你做净负荷预测，最合适的入口是：

- 将 `net_load` 放在最后一列
- 使用 `features=MS`

---

## 4. 当前仓库实现的真正关键信号

### 4.1 `forecast()` 路径才是最贴近你任务的代码

```python
en_embed, n_vars = self.en_embedding(
    x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1)
)
ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)
```

这里说明：

- 目标变量必须放在最后一列
- 其余数值协变量进入外生路径
- 时间特征通过 `x_mark_enc` 一起进入外部表示

### 4.2 当前实现没有显式使用“未来外生变量块”

与很多直觉不同，这个仓库实现里：

- `x_mark_dec` 没有像论文示意那样直接被单独 token 化进外生路径

也就是说，如果你想严格按照论文里的“未来已知外生协变量”来做：

- 当前实现只能提供一个相近起点
- 不能直接等价看成论文原始版本

---

## 5. 对 AEMO + NOAA 任务意味着什么

### 5.1 可以直接复用的部分

- `TimeXer` 的“目标变量单独建模 + 协变量辅助”的总体思想
- `features=MS` 的最后一列目标设计
- 内生 self-attention + 外生 cross-attention 的基本结构

### 5.2 需要你自己额外确认或改造的部分

#### 目标列组织

需要保证数据表最终排列为：

```text
[协变量1, 协变量2, ..., 协变量N, net_load]
```

其中：

- `net_load` 必须在最后一列

#### 外生变量选择

当前更适合进入外生路径的变量是：

- 天气变量
- 风光相关量
- 时间特征

不适合直接进入首版模型的变量包括：

- 出清结果量
- 合规结果量
- 事后统计量
- 预测时点不可得字段

#### 若要真正使用“未来天气预报”

如果你希望严格复现论文那种：

- 未来 24 小时天气预报直接进入外生 token

那么很可能需要：

- 调整数据管线
- 调整 `TimeXer` 外生输入接口
- 明确区分历史协变量和未来已知协变量

---

## 6. 最值得借鉴的代码位置

### 6.1 目标与协变量切分

```python
x_enc[:, :, -1]      # 目标变量
x_enc[:, :, :-1]     # 其他协变量
```

### 6.2 外部表示构造

```python
ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)
```

### 6.3 内生对外生的 Cross-Attention

`EncoderLayer.forward()` 中：

```python
x_glb_attn = self.cross_attention(
    x_glb, cross, cross,
    ...
)
```

这里可以理解为：

- 内生全局表示去查询外生信息

---

## 7. 对当前研究的直接启发

### 启发 1

先用仓库现成 `MS` 路径做净负荷预测基线，是合理的。

### 启发 2

如果实验表明外生变量有效，下一步创新点不应该只是“换更深的 backbone”，而是：

- 显式区分历史协变量和未来已知协变量
- 对不同来源的外生变量分组编码
- 在外生路径上加入变量选择或质量控制

### 启发 3

当前仓库实现本身就提示了一个论文机会：

> 现有 `TimeXer` 落地实现并未完整体现论文中的未来外生变量优势，这给“面向真实净负荷任务的协变量接口重构”留下了研究空间。
