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

## 4. 已确认结果

### 4.1 周期性证据

分析范围：

| 项目 | 值 |
| --- | --- |
| 样本数 | 26301 |
| 时间范围 | `2022-08-25 00:00:00` 到 `2025-08-24 23:00:00` |
| 夏令时缺口 | 3 个 2 小时跳跃 |
| 重复小时 | 未发现 |

周期结果：

| 指标 | 结果 |
| --- | ---: |
| 日内峰值小时 | 19 |
| 日内低谷小时 | 13 |
| 日内振幅 | 2432.425 MW |
| 周内峰值 | 周一 |
| 周内低谷 | 周日 |
| 周振幅 | 648.125 MW |
| 月内峰值 | 6 月 |
| 月内低谷 | 12 月 |
| 月振幅 | 1518.0 MW |
| `lag=24h` ACF | 0.6163 |
| `lag=168h` ACF | 0.4369 |

结论：净负荷存在强日周期、周周期和月周期，周期建模是当前任务的核心基线。

### 4.2 周期基线

留出集：最后 12 个月。

| Baseline | MSE | MAE |
| --- | ---: | ---: |
| `global_mean` | 1.4342 | 0.9508 |
| `hour_mean` | 1.1116 | 0.8418 |
| `hour_dow_mean` | 1.0647 | 0.8300 |
| `hour_dow_month_mean` | 0.8938 | 0.7452 |

结论：小时、星期、月份等周期索引已经能显著降低误差。

### 4.3 5min 模型基线

当前 5min 主线固定：`pred_len=24`。

| 模型 | 状态 | MSE / MAE | 结果来源 |
| --- | --- | --- | --- |
| `DLinear` | 已完成 | 0.0486493855714798 / 0.1521093249320984 | `result_long_term_forecast.txt` |
| `PatchTST` | 已完成 | 0.04928891733288765 / 0.15639345347881317 | `results/.../metrics.npy` |
| `TimeXer` | 已完成 | 0.0599 / 0.1808 | 历史 5min 结果 |
| `iTransformer` | 已完成 | 0.08044913411140442 / 0.2077513337135315 | `result_long_term_forecast.txt` |
| `Informer` | 已完成 | 0.1519191712141037 / 0.2762392461299896 | `result_long_term_forecast.txt` |
| `Autoformer` | 旧配置 | 0.8475703597068787 / 0.735338032245636 | `result_long_term_forecast.txt` |
| `Autoformer` | 重设重跑 | 0.6184316277503967 / 0.6219154596328735 | `run_5min_reset_3.log` |
| `TimeMixer` | 旧配置 | 12.89955997467041 / 1.8169910907745361 | `result_long_term_forecast.txt` |
| `TimeMixer` | 重设重跑 | 30.53180503845215 / 3.1489510536193848 | `run_5min_reset_3.log` |

阶段结论：

- 当前 5min 口径下，`DLinear` 和 `PatchTST` 是最强对照。
- `TimeXer` 表现不差，但不是当前最强基线。
- `Autoformer` 重设后有改善，但仍明显弱于 `DLinear` / `PatchTST`。
- `TimeMixer` 在当前口径下不稳定，不作为优先方向。
- NOAA 天气特征目前没有带来稳定提升。

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
