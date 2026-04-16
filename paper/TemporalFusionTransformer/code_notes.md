# Temporal Fusion Transformer 代码笔记

## 文件位置

`models/TemporalFusionTransformer.py`（thuml/Time-Series-Library）

---

## 1. 这份实现最关键的限制

文件开头就有：

```python
datatype_dict = {
    'ETTh1': TypePos([], [x for x in range(7)]),
    'ETTm1': TypePos([], [x for x in range(7)])
}
```

这意味着当前仓库实现只为极少数数据集写好了：

- 哪些列是静态变量
- 哪些列是历史观测变量

因此如果直接迁移到你的 AEMO 数据，第一步不是训练，而是：

- 先补自定义数据的变量索引映射

---

## 2. 输入类型划分

代码里核心定义是：

```python
TypePos = namedtuple('TypePos', ['static', 'observed'])
```

再配合：

```python
known_input = self.known_embedding(x_mark)
```

可以看出当前实现默认把输入分成三类：

- `static`
- `observed`
- `known`

其中：

- `known` 主要来自时间特征
- `observed` 来自 `x_enc`

---

## 3. 关键模块

### 3.1 `TFTEmbedding`

负责把不同类型的变量分别编码：

- 静态变量
- 历史观测变量
- 已知未来变量

### 3.2 `VariableSelectionNetwork`

```python
self.history_vsn = VariableSelectionNetwork(...)
self.future_vsn = VariableSelectionNetwork(...)
```

这是这份实现里最值得研究的部分之一：

- 历史输入有自己的变量选择网络
- 未来已知输入也有自己的变量选择网络

### 3.3 `TemporalFusionDecoder`

它把：

- 历史输入
- 未来输入
- 静态上下文

融合起来，输出未来多步预测。

---

## 4. 为什么它和净负荷任务高度相关

如果按你的任务重构变量类型：

- `observed`：
  - 历史净负荷
  - 历史系统观测变量
- `known`：
  - 时间特征
  - 天气预报
  - 预测时点已知的系统计划量

那么 TFT 的结构是非常贴切的。

---

## 5. 当前仓库实现为什么不能直接拿来用

### 5.1 `datatype_dict` 不支持你的自定义数据

必须先扩展。

### 5.2 `known` 路径当前更偏时间特征

如果想把完整 NOAA 天气预报作为未来已知协变量接入，还需要改数据流。

---

## 6. 对当前研究的直接启发

### 启发 1

你的方法设计里非常值得吸收 TFT 的“变量分类型”和“变量选择”思想。

### 启发 2

即使不直接用 TFT 做主模型，也可以把下面两个设计迁入主方法：

- 历史变量选择
- 未来已知变量选择

### 启发 3

如果后续论文强调解释性，TFT 会比单纯的黑盒对照更有写作价值。
