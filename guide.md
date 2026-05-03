# AEMO VIC1 净负荷研究核心指南

## 1. 研究目标

当前研究聚焦 AEMO VIC1 虚拟电网净负荷预测。

核心任务：

- 使用 AEMO VIC1 官方直采 `DISPATCHREGIONSUM` 5 分钟数据作为主数据。
- 使用 NOAA Melbourne Olympic Park 小时级天气数据作为外生天气数据。
- 构造并预测系统净负荷 `net_load`。
- 当前主线优先围绕 `VPP-GDFNet` 做趋势—季节/周期分解、双分支外生 cross-attention 与最终预测门控融合改造，同时保留 `DLinear`、`PatchTST`、`TimeXer` 等强基线对照。

固定目标定义：

```text
net_load = totaldemand_mw_avg - uigf_mw_avg
```

含义：

- `totaldemand_mw_avg`：系统总需求。
- `uigf_mw_avg`：系统层面的间歇性发电预测量。
- `net_load`：更接近系统净需求压力的预测目标。

## 2. 数据口径

### 2.1 原始数据

| 数据源 | 文件 | 时间键 |
| --- | --- | --- |
| AEMO VIC1 5min | `data/aemo_vic1/aemo_vic1_dispatchis_vic1_5min_2022-08-25_2025-08-24.csv` | `timestamp_local` |
| NOAA hourly | `data/noaa_globalhourly_melbourne_olympic_park_hourly_2022-08-25_2025-08-24.csv` | `timestamp_local_hour` |

统一成品时间列：`date`。

当前只保留 5 分钟主口径，不再维护 15 分钟、30 分钟或 1 小时派生数据集。

### 2.2 TimeXer 成品数据集

当前可直接用于 `TimeXer` 的数据集：

```text
dataset/aemo_vic1/aemo_vic1_dispatchis_vic1_full_5min.csv
```

当前核心列：

| 列名 | 角色 |
| --- | --- |
| `date` | 时间戳 |
| `ss_solar_uigf_mw_avg` | AEMO 安全外生变量 |
| `ss_wind_uigf_mw_avg` | AEMO 安全外生变量 |
| `netinterchange_mw_avg` | AEMO 安全外生变量 |
| `totalintermittentgeneration_mw_avg` | AEMO 安全外生变量 |
| `air_temperature_c` | NOAA 天气变量 |
| `wind_speed_mps` | NOAA 天气变量 |
| `net_load` | 预测目标与历史内生变量 |

推荐训练入口：

```bash
bash scripts/long_term_forecast/AEMO/run_timexer.sh
```

`TimeXer_5min.sh` 默认配置：

| 参数 | 值 |
| --- | ---: |
| `seq_len` | 288 |
| `label_len` | 144 |
| `pred_len` | 24 |
| `freq` | `5min` |
| `patch_len` | 12 |

含义：使用过去 1 天预测未来 2 小时。

## 3. 特征边界

### 3.1 首版推荐特征

| 类型 | 字段 |
| --- | --- |
| 目标 / 内生变量 | `net_load` |
| AEMO 外生变量 | `ss_solar_uigf_mw_avg`、`ss_wind_uigf_mw_avg`、`netinterchange_mw_avg`、`totalintermittentgeneration_mw_avg` |
| NOAA 外生变量 | `air_temperature_c`、`wind_speed_mps` |
| 时间特征 | 小时、星期几、月份、是否周末 |

### 3.2 高风险字段

以下字段不进入首版模型，避免信息泄漏：

- `demandforecast_mw_avg`
- `initialsupply_mw_avg`
- `clearedsupply_mw_avg`
- `semischedule_clearedmw_mw_avg`
- `semischedule_compliancemw_mw_avg`
- `ss_solar_clearedmw_mw_avg`
- `ss_wind_clearedmw_mw_avg`
- `ss_solar_compliancemw_mw_avg`
- `ss_wind_compliancemw_mw_avg`
- `wdr_dispatched_mw_avg`

原则：预测时点真实不可得、或更接近同周期调度 / 出清 / 事后统计结果的字段，不直接作为外生输入。

## 4. 待重新实验结果

### 4.1 周期性证据

已有分析表明，AEMO VIC1 净负荷存在明显日周期、周周期和月周期。后续如需重新统计周期性证据，应使用当前 5 分钟主数据口径，并在本节补充新的统计结果。

### 4.2 统一实验设置

新一轮模型对比统一使用以下设置，旧实验结果不再填入本轮对比表：

| 项目 | 设置 |
| --- | --- |
| 数据集 | `dataset/aemo_vic1/aemo_vic1_dispatchis_vic1_full_5min.csv` |
| 任务 | `long_term_forecast` |
| 特征模式 | `MS` |
| 预测目标 | `net_load` |
| `seq_len` | 288 |
| `label_len` | 144 |
| `pred_len` | 24、48、96、288 |
| `freq` | `5min` |
| `enc_in` / `dec_in` | 12 / 12 |

说明：本节只记录按上述统一口径重新跑出的结果；历史 `seq_len=2016`、旧 reset 实验或其他不一致口径结果不再混入。

### 4.3 统一实验矩阵

下表用于记录新一轮重跑结果。已完成实验后填写对应 `MSE / MAE`、结果来源和备注；未完成实验保持空白。

| 模型 | `seq_len` | `pred_len=24` MSE / MAE | `pred_len=48` MSE / MAE | `pred_len=96` MSE / MAE | `pred_len=288` MSE / MAE | 结果来源 | 状态 / 备注 |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `DLinear` | 288 | 0.060842 / 0.170233 | 0.162010 / 0.280279 | 0.349707 / 0.424036 | 0.648378 / 0.611372 | `result_long_term_forecast.txt`；`results/aemo_vic1/rerun_5min_matrix_aemo26_20260428_023850/summary.csv` | 已完成；`pred_len=24` 来自单独补跑，其余来自矩阵重跑 |
| `PatchTST` | 288 | 0.054667 / 0.163969 | 0.144515 / 0.262663 | 0.330549 / 0.414200 | 0.739699 / 0.640046 | `results/aemo_vic1/rerun_5min_matrix_aemo26_20260428_023850/summary.csv` | 已完成 |
| `Informer` | 288 | 0.071306 / 0.194340 | 0.164794 / 0.300190 | 0.360403 / 0.455223 | 0.904266 / 0.716028 | `results/aemo_vic1/rerun_5min_matrix_aemo26_20260428_023850/summary.csv` | 已完成 |
| `Autoformer` | 288 | 0.391551 / 0.498940 | 0.508135 / 0.564487 | 0.629772 / 0.619995 | 1.127724 / 0.832498 | `results/aemo_vic1/rerun_5min_matrix_aemo26_20260428_023850/summary.csv` | 已完成 |
| `TimesNet` | 288 | 0.054802 / 0.168520 | 0.137032 / 0.266420 |  |  | `results/aemo_vic1/rerun_5min_matrix_aemo26_20260428_023850/summary.csv` | `pred_len=24/48` 已完成；`pred_len=96/288` 失败，`exit_code=143` |
| `TimeXer` | 288 | 0.048770 / 0.153902 | 0.125251 / 0.245216 | 0.275121 / 0.370865 | 0.668956 / 0.603381 | `result_long_term_forecast.txt`；`results/aemo_vic1/rerun_5min_matrix_aemo26_20260428_023850/summary.csv` | 已完成；`pred_len=24` 来自单独补跑，其余来自矩阵重跑 |
| `VPP-GDFNet` | 288 | 0.052243 / 0.159371 | 0.133698 / 0.257003 | 0.293570 / 0.382305 | 0.690999 / 0.616395 | `results/aemo_vic1/rerun_5min_matrix_aemo26_20260428_023850/summary.csv` | 已完成 |

### 4.4 实验记录要求

- 每个结果必须记录来源文件，例如 `result_long_term_forecast.txt`、`results/.../metrics.npy` 或对应训练日志。
- 如果某个模型训练失败，不填指标，在“状态 / 备注”列说明失败原因。
- 不同 `seq_len`、不同数据文件或不同特征口径的结果不得填入本表。
- 完整重跑后优先比较平均表现、分 horizon 表现，以及高峰/低谷、工作日/周末、夏季/冬季等切片稳定性。

### 4.5 通用数据集泛化性实验矩阵

为判断 `VPP-GDFNet` 是否只适用于 AEMO 虚拟电厂净负荷场景，新增一组通用时序数据集验证。该实验不替代 AEMO 主实验，只用于说明模型结构在公开 benchmark 上是否具有基础泛化能力。

本组实验严格沿用仓库中 `TimeXer` 通用数据集脚本的参数，只将模型替换为 `VPP-GDFNet`：

| 项目 | 设置 |
| --- | --- |
| 数据集 | `ETTh1`、`ETTh2`、`ETTm1`、`ETTm2`、`Weather`、`ECL`、`Traffic` |
| 任务 | `long_term_forecast` |
| 模型 | `VPP-GDFNet` |
| 参数来源 | 对应数据集的 `TimeXer` 脚本 |
| 特征模式 | `M` |
| `seq_len` | 96 |
| `label_len` | 48 |
| `pred_len` | 96、192、336、720 |
| 指标 | `MSE` / `MAE` |
| 平均值 | 每个数据集 4 个 horizon 的成功结果算术平均；另计算全部成功实验总体平均 |
| 结果用途 | 验证跨数据集泛化性，不作为 AEMO 主结论的直接替代 |

#### TimeXer Table 3 同口径对比表

下表以 TimeXer 论文正文 Table 3 的多变量长期预测平均结果为基准，用于论文中的公开 benchmark 对比叙事。已有基线列沿用 TimeXer Table 3，`VPP-GDFNet` 列填入本文在相同通用数据集脚本参数下补跑的平均结果。

说明：TimeXer Table 3 记录的是 `pred_len ∈ {96, 192, 336, 720}` 的平均结果，不展开单个 horizon；原表不包含 `Stationary`，因此本表也不列该模型。

| 数据集 | TimeXer | iTransformer | RLinear | PatchTST | Crossformer | TiDE | TimesNet | DLinear | SCINet | Autoformer | VPP-GDFNet | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ECL` | 0.171 / 0.270 | 0.178 / 0.270 | 0.219 / 0.298 | 0.205 / 0.290 | 0.244 / 0.334 | 0.251 / 0.244 | 0.192 / 0.295 | 0.212 / 0.300 | 0.268 / 0.365 | 0.227 / 0.338 | 0.186484 / 0.287971 |  |
| `Weather` | 0.241 / 0.271 | 0.258 / 0.278 | 0.272 / 0.291 | 0.259 / 0.281 | 0.259 / 0.315 | 0.271 / 0.320 | 0.259 / 0.287 | 0.265 / 0.317 | 0.292 / 0.363 | 0.338 / 0.382 | 0.269793 / 0.292173 |  |
| `ETTh1` | 0.437 / 0.437 | 0.454 / 0.447 | 0.446 / 0.434 | 0.469 / 0.454 | 0.529 / 0.522 | 0.541 / 0.507 | 0.458 / 0.450 | 0.456 / 0.452 | 0.747 / 0.647 | 0.496 / 0.487 | 0.518306 / 0.481399 |  |
| `ETTh2` | 0.367 / 0.396 | 0.383 / 0.407 | 0.374 / 0.398 | 0.387 / 0.407 | 0.942 / 0.684 | 0.611 / 0.550 | 0.414 / 0.427 | 0.559 / 0.515 | 0.954 / 0.723 | 0.450 / 0.459 | 0.423900 / 0.427778 |  |
| `ETTm1` | 0.382 / 0.397 | 0.407 / 0.410 | 0.414 / 0.407 | 0.387 / 0.400 | 0.512 / 0.496 | 0.419 / 0.419 | 0.400 / 0.406 | 0.403 / 0.407 | 0.485 / 0.481 | 0.588 / 0.517 | 0.476054 / 0.446535 |  |
| `ETTm2` | 0.274 / 0.322 | 0.288 / 0.332 | 0.286 / 0.327 | 0.281 / 0.326 | 0.757 / 0.610 | 0.358 / 0.404 | 0.291 / 0.333 | 0.350 / 0.401 | 0.571 / 0.537 | 0.327 / 0.371 | 0.289253 / 0.330498 |  |
| `Traffic` | 0.466 / 0.287 | 0.428 / 0.282 | 0.626 / 0.378 | 0.481 / 0.304 | 0.550 / 0.304 | 0.760 / 0.473 | 0.620 / 0.336 | 0.625 / 0.383 | 0.804 / 0.509 | 0.628 / 0.379 | 0.619379 / 0.320857 | `VPP-GDFNet` 仅统计 `pred_len=96/192/336` 成功结果；`pred_len=720` 失败，`exit_code=1` |

#### VPP-GDFNet 原始实验记录表

下表不用于替代上面的 TimeXer Table 3 同口径对比表，只用于保存 `VPP-GDFNet` 在远程 `summary.csv` 和 `averages.csv` 中回填的原始实验结果、平均值和失败记录，便于后续追溯。

| 数据集 | `seq_len` | `pred_len=96` MSE / MAE | `pred_len=192` MSE / MAE | `pred_len=336` MSE / MAE | `pred_len=720` MSE / MAE | 平均 MSE / MAE | 结果来源 | 状态 / 备注 |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `ETTh1` | 96 | 0.469692 / 0.450498 | 0.502965 / 0.469367 | 0.539116 / 0.486969 | 0.561449 / 0.518762 | 0.518306 / 0.481399 | `results/general_vppgdfnet_timexer_params/vpp_general_20260428_150042/summary.csv`；`averages.csv` | 已完成 |
| `ETTh2` | 96 | 0.346841 / 0.378195 | 0.427705 / 0.423706 | 0.444531 / 0.441439 | 0.476524 / 0.467772 | 0.423900 / 0.427778 | `results/general_vppgdfnet_timexer_params/vpp_general_20260428_150042/summary.csv`；`averages.csv` | 已完成 |
| `ETTm1` | 96 | 0.415930 / 0.410336 | 0.454203 / 0.433047 | 0.487857 / 0.453956 | 0.546226 / 0.488801 | 0.476054 / 0.446535 | `results/general_vppgdfnet_timexer_params/vpp_general_20260428_150042/summary.csv`；`averages.csv` | 已完成 |
| `ETTm2` | 96 | 0.185142 / 0.267160 | 0.248007 / 0.306010 | 0.311628 / 0.345318 | 0.412233 / 0.403506 | 0.289253 / 0.330498 | `results/general_vppgdfnet_timexer_params/vpp_general_20260428_150042/summary.csv`；`averages.csv` | 已完成 |
| `Weather` | 96 | 0.186901 / 0.231593 | 0.236663 / 0.270874 | 0.292561 / 0.310344 | 0.363048 / 0.355881 | 0.269793 / 0.292173 | `results/general_vppgdfnet_timexer_params/vpp_general_20260428_150042/summary.csv`；`averages.csv` | 已完成 |
| `ECL` | 96 | 0.165116 / 0.268507 | 0.179926 / 0.281514 | 0.187202 / 0.289611 | 0.213693 / 0.312254 | 0.186484 / 0.287971 | `results/general_vppgdfnet_timexer_params/vpp_general_20260428_150042/summary.csv`；`averages.csv` | 已完成 |
| `Traffic` | 96 | 0.601038 / 0.313382 | 0.627238 / 0.325153 | 0.629862 / 0.324035 |  | 0.619379 / 0.320857 | `results/general_vppgdfnet_timexer_params/vpp_general_20260428_150042/summary.csv`；`averages.csv` | `pred_len=96/192/336` 已完成；`pred_len=720` 失败，`exit_code=1` |
| `ALL` | 96 |  |  |  |  | 0.389381 / 0.371407 | `results/general_vppgdfnet_timexer_params/vpp_general_20260428_150042/averages.csv` | 全部成功实验总体平均；`27 success, 1 failed` |

泛化性判断建议：

- 如果 `VPP-GDFNet` 在 AEMO 上有效，但在通用数据集上明显弱于常见基线，应将论文表述收敛为“面向虚拟电厂净负荷预测的结构设计”，避免泛化性过强的结论。
- 如果 `VPP-GDFNet` 在多个通用数据集和多个 horizon 上也能保持竞争力，可在论文中补充“公开数据集泛化验证”，但仍需说明模型核心动机来自净负荷趋势/周期分解与外生变量差异化影响。
- 仅跑 `VPP-GDFNet` 只能证明模型可运行和跨数据集稳定性；如果要证明相对优势，应补充 `DLinear`、`PatchTST` 或 `TimeXer` 的同口径对照。

## 5. 当前研究路线

当前确认的主方法为 `VPP-GDFNet`，核心结构如下：

```text
历史净负荷
   ↓
DLinear 趋势—季节/周期分解
   ↓
趋势分量表示        季节/周期分量表示
   ↓                    ↓
分别作为 Query       分别作为 Query
   ↓                    ↓
与外生变量 token 交叉注意力交互
   ↓                    ↓
趋势预测分支        季节/周期预测分支
   ↓                    ↓
        最后通过门控融合
              ↓
        最终净负荷预测
```

核心表述固定为：`VPP-GDFNet` 不是先把趋势项和季节项融合后再统一利用外生变量，而是让趋势分支和季节/周期分支分别与外生条件变量交互，使模型自适应学习外生变量对不同动态成分的差异化影响，最后通过门控融合机制整合两个分支的预测结果。

研究假设：不同外生变量对净负荷低频变化和高频/周期波动的作用强度可能不同。因此，先将趋势分量和季节/周期分量合并后再统一利用外生变量，可能削弱这种差异化作用；让两个分支分别与同一组外生变量 token 做普通 cross-attention，可以让模型分别学习外生变量对不同动态成分的影响。

优先顺序：

1. 保持 5 分钟单一主口径。
2. 先复现周期性分析和周期基线。
3. 用 `DLinear`、`PatchTST` 作为强对照基线。
4. 围绕 `VPP-GDFNet` 做趋势/季节双分支外生 cross-attention 与消融。
5. 对比先合并趋势/季节后统一外生融合与趋势/季节分别外生融合，验证双分支设计是否必要。
6. 如果新方法不能稳定优于强基线，不作为主结果。

当前推荐创新主线：

```text
趋势—季节/周期分解 + 双分支外生 cross-attention + 最终预测门控融合
```

核心判断：当前任务不是盲目增加变量或模型复杂度，而是在强周期净负荷轨迹基础上，判断外生变量对低频趋势和高频/周期波动是否具有不同作用。外生变量交互阶段不做门控筛选或门控 cross-attention；门控只用于最终整合趋势预测分支与季节/周期预测分支。

必须保留的关键消融包括：

- `DLinear-only`：验证趋势—季节/周期分解目标轨迹基线。
- `DLinear + unified exogenous fusion`：验证统一外生变量融合的收益。
- `DLinear + branch exogenous fusion`：验证趋势/季节分支分别融合是否优于统一融合。
- `VPP-GDFNet without final gate`：验证最终预测门控融合是否必要。
- `VPP-GDFNet full`：完整方法。

解释分析应重点比较趋势预测分支与季节/周期预测分支的贡献差异。例如，最终融合门在高温/普通日、白天/夜间、高风速/低风速、峰谷/爬坡等场景下是否呈现有意义变化。当前主方法不包含变量选择模块，也不在与外生变量交互时对外生变量 token 做门控处理。

## 6. 保留标准

新实验或新方法至少满足以下一项，才值得继续推进：

- 总体 `MSE` / `MAE` 更低。
- 在不同 horizon 上更稳定。
- 在工作日 / 周末、夏季 / 冬季、高峰 / 低谷切片上更稳定。
- 对周期残差的解释更合理。
- 能证明趋势预测分支与季节/周期预测分支的最终门控贡献存在有意义差异。
- 不引入信息泄漏风险。

## 7. 文档分工

- `guide.md`：研究目标、数据口径、特征边界、核心结果、研究路线。
- `AGENTS.md`：代码实现、脚本参数、数据路径、训练流程、产物位置。

更新规则：

- 研究假设、结论解释、特征取舍、实验路线变化时，优先更新 `guide.md`。
- 代码、脚本、路径、数据处理、训练流程变化时，优先更新 `AGENTS.md`。
- 同时影响研究判断和实现方式时，两份文档都要更新，但不要重复写同一层级内容。
