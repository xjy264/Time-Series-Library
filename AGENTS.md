# 虚拟电网净负荷预测工作说明（TimeXer）

## 1. 当前任务

本仓库后续围绕「虚拟电网净负荷预测」开展工作，当前只使用 `TimeXer`，暂时不涉及 `iTransformer`、模型串联或多模型融合。

本任务的核心目标是：

- 以 AEMO 的 VIC1 小时级数据作为电力系统主数据。
- 以 NOAA 的 Melbourne Olympic Park 小时级天气数据作为外生天气数据。
- 构造净负荷（Net Load）预测任务。
- 使用 `TimeXer` 建模，其中：
  - 内生变量（endogenous variable）是历史净负荷序列。
  - 外生变量（exogenous variables）是天气变量、时间特征，以及经过筛选后在预测时点真实可获得的系统侧协变量。

当前固定的净负荷定义为：

```text
net_load = totaldemand_mw_avg - uigf_mw_avg
```

这里的 `uigf_mw_avg` 可理解为系统层。         面的间歇性发电预测量（当前主要覆盖光伏和风电）。这个定义更贴近系统层面的净需求压力。
这里的 `uigf_mw_avg` 可理解为系统层面的间歇性发电预测量（当前主要覆盖光伏和风电）。这个定义更贴近系统层面的净需求压力。

## 2. 数据源

当前使用以下两份原始数据，二者都不在仓库内，而是在本机桌面目录：

- AEMO 电力系统数据：
  `/Users/xuejiayao/Desktop/paper/data/aemo_vic1_hourly_2022-08-25_2025-08-24.csv`
- NOAA 天气数据：
  `/Users/xuejiayao/Desktop/paper/data/noaa_globalhourly_melbourne_olympic_park_hourly_2022-08-25_2025-08-24.csv`

时间对齐键统一使用：

- `timestamp_local_hour`

统一粒度：

- 小时级（hourly）

## 3. 建模约定

### 3.1 TimeXer 的变量分工

- 预测目标：`net_load`
- 内生变量：历史 `net_load`
- 外生变量：
  - NOAA 天气字段
  - 时间特征（小时、星期几、月份、是否周末）
  - 经过筛选的 AEMO 系统侧协变量

### 3.2 当前不做的事

- 不引入 `iTransformer`
- 不做 `TimeXer + iTransformer` 串联
- 不直接把所有 AEMO 字段无差别喂给模型
- 不把未来时点才能知道的调度结果字段直接作为外生变量

### 3.3 关键风险

- 信息泄漏（leakage）比模型结构更重要。
- 一些 AEMO 字段与目标高度相关，但可能在预测时点不可得，不能直接用作特征。
- 当前 NOAA 数据没有直接的太阳辐照度（irradiance / GHI）字段，因此它可以帮助描述天气，但不能完整替代光伏专用气象驱动。

## 4. 字段使用标记

本文档用以下标记说明字段对当前任务的价值：

- `必用`：当前方案必须使用。
- `推荐`：建议纳入首版特征集。
- `可选`：可做消融实验或增强实验。
- `暂不使用`：当前阶段不建议纳入。
- `高风险`：可能造成信息泄漏，或口径过于接近目标，不应直接进入首版模型。

## 5. AEMO 数据字段说明

数据文件：
`aemo_vic1_hourly_2022-08-25_2025-08-24.csv`

字段总数：34

| 字段名 | 含义 | 当前标记 | 备注 |
| --- | --- | --- | --- |
| `timestamp_local_hour` | 本地小时级时间戳 | 必用 | 两份数据按此字段对齐 |
| `regionid` | AEMO 区域 ID | 暂不使用 | 当前文件固定为 `VIC1`，信息量有限 |
| `interval_count` | 该小时内包含的调度间隔数量 | 暂不使用 | 更像数据完整性字段 |
| `totaldemand_mw_avg` | 该小时平均总需求（MW） | 必用 | 净负荷标签构造基准 |
| `totaldemand_mw_min` | 该小时最小总需求（MW） | 可选 | 可作为历史波动辅助特征，不进入首版 |
| `totaldemand_mw_max` | 该小时最大总需求（MW） | 可选 | 可作为历史波动辅助特征，不进入首版 |
| `demand_and_nonschedgen_mw_avg` | AEMO 口径的需求与非调度发电相关量（平均） | 暂不使用 | 含义接近系统平衡量，先不用作特征 |
| `demand_and_nonschedgen_mw_min` | 上述量的最小值 | 暂不使用 | 与目标关系过近，收益未必稳健 |
| `demand_and_nonschedgen_mw_max` | 上述量的最大值 | 暂不使用 | 与目标关系过近，收益未必稳健 |
| `uigf_mw_avg` | 平均 UIGF（无约束间歇性发电预测） | 必用 | 当前净负荷定义直接使用 |
| `uigf_mw_min` | UIGF 最小值 | 可选 | 可用于波动描述，但不是首版必须项 |
| `uigf_mw_max` | UIGF 最大值 | 可选 | 可用于波动描述，但不是首版必须项 |
| `availablegeneration_mw_avg` | 可用发电能力平均值 | 暂不使用 | 更偏系统供给约束，不是当前重点 |
| `availableload_mw_avg` | 可用负荷平均值 | 暂不使用 | 字段口径需后续确认 |
| `demandforecast_mw_avg` | 需求预测平均值 | 高风险 | 很可能是系统已有预测结果，容易泄漏 |
| `dispatchablegeneration_mw_avg` | 可调度发电平均值 | 暂不使用 | 供给侧量，业务意义强于直接预测价值 |
| `dispatchableload_mw_avg` | 可调度负荷平均值 | 可选 | 可做后续增强实验 |
| `netinterchange_mw_avg` | 净联络线交换功率平均值 | 必用 | 反映跨区交换，对净负荷有辅助价值 |
| `initialsupply_mw_avg` | 初始供给平均值 | 高风险 | 与调度结果过近，不进入首版 |
| `clearedsupply_mw_avg` | 出清供给平均值 | 高风险 | 与目标高度同周期相关，容易泄漏 |
| `lorsurplus_mw_avg` | 低运营备用盈余平均值 | 暂不使用 | 更适合系统安全评估 |
| `lrcsurplus_mw_avg` | 低可靠性备用盈余平均值 | 暂不使用 | 更适合系统安全评估 |
| `totalintermittentgeneration_mw_avg` | 间歇性发电总量平均值 | 可选 | 和 `uigf_mw_avg` 有联系，可做对照 |
| `semischedule_clearedmw_mw_avg` | 半调度机组出清功率平均值 | 高风险 | 属于实际出清结果，不直接用于预测 |
| `semischedule_compliancemw_mw_avg` | 半调度机组合规功率平均值 | 高风险 | 偏结果量，不建议直接使用 |
| `ss_solar_uigf_mw_avg` | 半调度光伏 UIGF 平均值 | 必用 | 对解释光伏影响非常重要 |
| `ss_wind_uigf_mw_avg` | 半调度风电 UIGF 平均值 | 必用 | 对解释风电影响非常重要 |
| `ss_solar_clearedmw_mw_avg` | 半调度光伏出清功率平均值 | 高风险 | 出清结果量，不进入首版 |
| `ss_wind_clearedmw_mw_avg` | 半调度风电出清功率平均值 | 高风险 | 出清结果量，不进入首版 |
| `ss_solar_compliancemw_mw_avg` | 半调度光伏合规功率平均值 | 高风险 | 更接近实际结果，不进入首版 |
| `ss_wind_compliancemw_mw_avg` | 半调度风电合规功率平均值 | 高风险 | 更接近实际结果，不进入首版 |
| `wdr_initialmw_mw_avg` | 需求响应初始功率平均值 | 可选 | 可能对高峰时段有帮助 |
| `wdr_available_mw_avg` | 需求响应可用功率平均值 | 可选 | 可做增强实验 |
| `wdr_dispatched_mw_avg` | 需求响应已调度功率平均值 | 高风险 | 结果量，不进入首版 |

### 5.1 AEMO 中对当前任务最重要的字段

首要关注：

- `totaldemand_mw_avg`
- `uigf_mw_avg`
- `ss_solar_uigf_mw_avg`
- `ss_wind_uigf_mw_avg`
- `netinterchange_mw_avg`

原因：

- `totaldemand_mw_avg` 是净负荷构造的负荷端基准。
- `uigf_mw_avg` 是当前净负荷定义的直接扣减项。
- `ss_solar_uigf_mw_avg` 和 `ss_wind_uigf_mw_avg` 可以帮助解释净负荷波动来自光伏还是风电。
- `netinterchange_mw_avg` 可以反映跨区交换对系统净需求的影响。

### 5.2 AEMO 中当前明确高风险的字段

以下字段暂不进入首版 `TimeXer` 特征集：

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

这些字段要么本身就是系统已有预测结果，要么更接近同周期调度结果或事后统计结果，容易造成离线评估偏乐观。

## 6. NOAA 天气数据字段说明

数据文件：
`noaa_globalhourly_melbourne_olympic_park_hourly_2022-08-25_2025-08-24.csv`

字段总数：42

| 字段名 | 含义 | 当前标记 | 备注 |
| --- | --- | --- | --- |
| `timestamp_local_hour` | 本地小时级时间戳 | 必用 | 与 AEMO 做时间对齐 |
| `date_utc` | UTC 时间 | 暂不使用 | 可用于校验时区转换 |
| `station_id` | 站点 ID | 暂不使用 | 元数据 |
| `station_name` | 站点名称 | 暂不使用 | 元数据 |
| `latitude` | 纬度 | 暂不使用 | 站点固定元数据 |
| `longitude` | 经度 | 暂不使用 | 站点固定元数据 |
| `elevation_m` | 海拔（m） | 暂不使用 | 站点固定元数据 |
| `source` | 数据来源 | 暂不使用 | 元数据 |
| `report_type` | 报文类型 | 暂不使用 | 元数据 |
| `call_sign` | 呼号 | 暂不使用 | 元数据 |
| `quality_control` | 质量控制标记 | 可选 | 可用于过滤异常记录 |
| `air_temperature_c` | 气温（°C） | 推荐 | 负荷与天气关系中的核心变量 |
| `dewpoint_c` | 露点温度（°C） | 推荐 | 与湿热感和天气状态有关 |
| `sea_level_pressure_hpa` | 海平面气压（hPa） | 推荐 | 可辅助表征天气系统变化 |
| `wind_direction_deg` | 风向（度） | 可选 | 可做循环编码后使用 |
| `wind_speed_mps` | 风速（m/s） | 推荐 | 与风电和体感环境均相关 |
| `visibility_m` | 能见度（m） | 可选 | 可作为天气状态代理变量 |
| `ceiling_ft` | 云底高（ft） | 可选 | 对光照条件有间接帮助 |
| `precip_1h_mm` | 1 小时降水量（mm） | 推荐 | 有助于刻画天气扰动 |
| `precip_2h_mm` | 2 小时降水量（mm） | 可选 | 与短时降水信息部分重复 |
| `precip_3h_mm` | 3 小时降水量（mm） | 可选 | 与短时降水信息部分重复 |
| `precip_6h_mm` | 6 小时降水量（mm） | 可选 | 可做累积湿润度代理 |
| `precip_24h_mm` | 24 小时降水量（mm） | 可选 | 可做背景天气状态代理 |
| `wnd_raw` | 原始风场编码字符串 | 暂不使用 | 已有拆解后的风向、风速字段 |
| `cig_raw` | 原始云高编码字符串 | 暂不使用 | 已有 `ceiling_ft` |
| `vis_raw` | 原始能见度编码字符串 | 暂不使用 | 已有 `visibility_m` |
| `tmp_raw` | 原始气温编码字符串 | 暂不使用 | 已有 `air_temperature_c` |
| `dew_raw` | 原始露点编码字符串 | 暂不使用 | 已有 `dewpoint_c` |
| `slp_raw` | 原始海平面气压编码字符串 | 暂不使用 | 已有 `sea_level_pressure_hpa` |
| `aa1_raw` | 原始降水编码字段 1 | 暂不使用 | 结构化字段已拆出 |
| `aa2_raw` | 原始降水编码字段 2 | 暂不使用 | 结构化字段已拆出 |
| `aw1_raw` | 原始天气现象编码字段 1 | 暂不使用 | 需要额外解码，不做首版 |
| `ay1_raw` | 原始天气现象编码字段 2 | 暂不使用 | 需要额外解码，不做首版 |
| `ay2_raw` | 原始天气现象编码字段 3 | 暂不使用 | 需要额外解码，不做首版 |
| `az1_raw` | 原始天气现象编码字段 4 | 暂不使用 | 需要额外解码，不做首版 |
| `az2_raw` | 原始天气现象编码字段 5 | 暂不使用 | 需要额外解码，不做首版 |
| `ka1_raw` | 原始天气附加编码字段 1 | 暂不使用 | 含义不直观，首版不使用 |
| `ma1_raw` | 原始气压/高度相关编码字段 | 暂不使用 | 含义不直观，首版不使用 |
| `md1_raw` | 原始云量/云层相关编码字段 | 暂不使用 | 如需云量特征，后续再解码 |
| `mw1_raw` | 原始天气现象编码字段 6 | 暂不使用 | 含义不直观，首版不使用 |
| `rem_raw` | 原始备注字段 | 暂不使用 | 文本型原始信息 |
| `eqd_raw` | 原始质量或事件字段 | 暂不使用 | 文本型原始信息 |

### 6.1 NOAA 中对当前任务最有用的字段

首版推荐优先使用：

- `air_temperature_c`
- `dewpoint_c`
- `sea_level_pressure_hpa`
- `wind_speed_mps`
- `precip_1h_mm`

第二优先级可尝试：

- `wind_direction_deg`
- `visibility_m`
- `ceiling_ft`
- `precip_6h_mm`
- `precip_24h_mm`

### 6.2 NOAA 的当前局限

当前天气文件没有直接给出以下典型光伏驱动量：

- 全球水平辐照度（GHI）
- 太阳辐射
- 云量百分比
- 直射辐照度或散射辐照度

因此：

- 这份天气数据可以帮助预测负荷和部分天气驱动下的净负荷变化。
- 这份天气数据不能完整替代光伏专用气象建模。
- 如果后续想更强地刻画光伏效率，优先补充辐照度或云量数据。

## 7. 当前推荐的首版特征集

### 7.1 目标变量

- `net_load = totaldemand_mw_avg - uigf_mw_avg`

### 7.2 内生变量

- 历史 `net_load`

### 7.3 外生变量

推荐第一版只使用以下字段：

- AEMO：
  - `ss_solar_uigf_mw_avg`
  - `ss_wind_uigf_mw_avg`
  - `netinterchange_mw_avg`
- NOAA：
  - `air_temperature_c`
  - `dewpoint_c`
  - `sea_level_pressure_hpa`
  - `wind_speed_mps`
  - `precip_1h_mm`
- 时间特征：
  - 小时
  - 星期几
  - 月份
  - 是否周末

### 7.4 当前 `TimeXer` 专用成品数据集

当前仓库内给 `TimeXer` 使用的专用成品数据集为：

- `dataset/aemo_vic1/aemo_vic1_timexer_weather_ms.csv`

该文件当前列结构为：

- `date`
- `ss_solar_uigf_mw_avg`
- `ss_wind_uigf_mw_avg`
- `netinterchange_mw_avg`
- `air_temperature_c`
- `dewpoint_c`
- `sea_level_pressure_hpa`
- `wind_speed_mps`
- `precip_1h_mm`
- `net_load`

说明：

- 这是一个面向 `TimeXer` 的 `MS` 数据集。
- `net_load` 是预测目标，也作为历史内生变量保留在输入通道中。
- NOAA 的 `precip_1h_mm` 原始缺失很多，当前按 `0.0` 填补。
- 其他 NOAA 缺失位置当前按时间顺序做前向/后向补齐，以保证模型训练时不出现空值。
- 旧文件 `dataset/aemo_vic1/aemo_vic1_timexer_ms.csv` 仍保留给之前的多模型基准，不作为当前 `TimeXer` 主实验输入。

## 8. 不建议直接进入首版模型的字段

以下字段优先排除：

- 所有明显属于系统已有预测结果的字段
- 所有明显属于出清结果、调度结果、合规结果的字段
- 所有 NOAA 的原始编码字段（`*_raw`）
- 与当前业务问题关系弱、但会增加噪声和清洗成本的元数据字段

## 9. 数据处理要求

- 必须先按 `timestamp_local_hour` 合并 AEMO 和 NOAA。
- 必须检查时间对齐后的缺失值比例。
- 必须显式记录是否存在夏令时导致的缺口或重复小时。
- 必须区分历史观测值和预测时点可获得值。
- 任何未来时点真实观测到的结果量都不能直接作为该时点的外生输入。

## 10. 研究执行顺序

建议按以下顺序推进：

1. 先只用历史 `net_load` 做纯内生基线。
2. 加入 NOAA 天气外生变量。
3. 加入 AEMO 中安全的系统侧外生变量。
4. 使用 `TimeXer` 做净负荷预测实验。
5. 做特征消融实验，比较：
   - 只有内生变量
   - 内生变量 + 天气
   - 内生变量 + 天气 + AEMO 安全外生变量

## 11. 后续代理工作要求

如果后续代理继续基于本仓库工作，默认遵守以下约束：

- 以本文件作为当前任务说明和字段口径说明。
- 不引入 `iTransformer`，除非用户明确重新指定。
- 优先保证特征可用性和防止信息泄漏，而不是优先追求更复杂的模型。
- 新增特征前，先判断该字段在预测时点是否真实可得。
- 如果净负荷定义要改，先更新本文件，再改代码或实验脚本。

## 12. 已完成的多模型基准实验（2026-04-03）

虽然当前后续研究默认仍以 `TimeXer` 为主，但已经基于同一份 AEMO 净负荷数据跑完一轮多模型基准实验，用于判断基线强弱与后续优化方向。

### 12.1 实验范围

- 数据集：`dataset/aemo_vic1/aemo_vic1_timexer_ms.csv`
- 任务：`long_term_forecast`
- 目标变量：`net_load`
- 输入长度：`seq_len = 168`
- 预测长度：`pred_len ∈ {24, 48, 96}`
- 特征模式：`MS`
- 对比模型：
  - `DLinear`
  - `TimeXer`
  - `iTransformer`
  - `PatchTST`
  - `Informer`
  - `Autoformer`
  - `TimeMixer`

### 12.2 最终结果文件

最终结果只保存在本机桌面 `paper` 目录下：

- `/Users/xuejiayao/Desktop/paper/aemo_full_summary.csv`
- `/Users/xuejiayao/Desktop/paper/result_long_term_forecast.txt`

### 12.3 模型总体排名（按 3 个 horizon 的平均 MSE 排序）

| 排名 | 模型 | 平均 MSE | 平均 MAE |
| --- | --- | ---: | ---: |
| 1 | `DLinear` | 0.8044 | 0.7017 |
| 2 | `PatchTST` | 0.8579 | 0.7194 |
| 3 | `TimeMixer` | 0.8618 | 0.7187 |
| 4 | `TimeXer` | 0.8750 | 0.7240 |
| 5 | `iTransformer` | 0.9264 | 0.7524 |
| 6 | `Informer` | 1.1212 | 0.8143 |
| 7 | `Autoformer` | 1.1306 | 0.8529 |

### 12.4 结论

- 按总体平均表现看，当前最好的模型是 `DLinear`。
- 按总体平均表现看，当前最差的模型是 `Autoformer`。
- 按单个实验配置看，最好的单次结果也是 `DLinear`：
  - `pred_len = 24`
  - `MSE = 0.6232`
  - `MAE = 0.6091`
- 按单个实验配置看，最差的单次结果是 `Informer`：
  - `pred_len = 96`
  - `MSE = 1.2842`
  - `MAE = 0.8959`

### 12.5 分 horizon 最优模型

| `pred_len` | 最优模型 | MSE | MAE |
| --- | --- | ---: | ---: |
| 24 | `DLinear` | 0.6232 | 0.6091 |
| 48 | `DLinear` | 0.8238 | 0.7123 |
| 96 | `DLinear` | 0.9661 | 0.7837 |

### 12.6 各模型完整结果

| 模型 | `pred_len=24` MSE / MAE | `pred_len=48` MSE / MAE | `pred_len=96` MSE / MAE |
| --- | --- | --- | --- |
| `DLinear` | 0.6232 / 0.6091 | 0.8238 / 0.7123 | 0.9661 / 0.7837 |
| `TimeXer` | 0.6608 / 0.6209 | 0.9159 / 0.7430 | 1.0484 / 0.8080 |
| `iTransformer` | 0.7228 / 0.6517 | 0.9582 / 0.7708 | 1.0982 / 0.8349 |
| `PatchTST` | 0.6435 / 0.6165 | 0.8929 / 0.7381 | 1.0373 / 0.8035 |
| `Informer` | 0.8841 / 0.6990 | 1.1954 / 0.8479 | 1.2842 / 0.8959 |
| `Autoformer` | 0.9802 / 0.7863 | 1.1990 / 0.8798 | 1.2124 / 0.8925 |
| `TimeMixer` | 0.6461 / 0.6210 | 0.8902 / 0.7336 | 1.0492 / 0.8015 |

### 12.7 原因推断

#### `DLinear` 为什么最好

- 当前数据是小时级净负荷，日周期、周周期和相对平滑的趋势都比较强，线性分解模型更容易直接抓住主导结构。
- 当前首版特征集是“少量高质量特征”，不是大规模高维变量。对这种低维、强季节性的任务，简单模型往往更稳。
- 数据量大约覆盖 3 年，足够训练中小模型，但未必足够支撑更复杂的注意力结构稳定学到泛化规律。
- 当前 NOAA 数据缺少辐照度、云量等更强的光伏驱动变量，复杂模型可利用的外生信息其实有限，因此复杂结构的优势发挥不出来。
- `DLinear` 参数更少、优化更稳定，对这类中短期 horizon 更不容易过拟合。

#### `Autoformer` / `Informer` 为什么较差

- 这类模型更依赖复杂的时序分解或注意力机制，在当前特征规模下，模型复杂度相对偏高。
- 当前任务的主要可预测性可能已经被基础季节性和少量安全协变量解释掉，复杂模型增加了优化难度，但没有换来足够的信息增益。
- 对 `pred_len = 96` 这类较长 horizon，编码器-解码器式结构更容易累积误差，性能下降更明显。
- 缺少辐照度、云量等关键天气驱动后，模型即使有更强表示能力，也难以补足真正缺失的信息。

#### 如何理解 `TimeXer`

- `TimeXer` 没有拿到最优，但总体并不差，平均排名第 4，和 `PatchTST`、`TimeMixer` 的差距不大。
- 这说明当前 `TimeXer` 的瓶颈大概率不在“模型完全不适合”，而更可能在特征质量、外生变量表达方式和任务设定。
- 如果后续继续只做 `TimeXer`，优先级应该是：
  1. 继续做防泄漏前提下的特征增强。
  2. 补充更有效的天气代理变量，尤其是辐照度、云量或太阳位置相关信息。
  3. 做特征消融与 horizon 分层调参，而不是先切到更复杂模型。

### 12.8 `TimeXer` 天气增强版重跑结果（2026-04-03）

在上述多模型基准完成后，又单独为 `TimeXer` 构造了一版带 NOAA 天气特征的专用数据集，并重新跑了 `pred_len ∈ {24, 48, 96}` 三组实验。

#### 数据口径

- 数据文件：`dataset/aemo_vic1/aemo_vic1_timexer_weather_ms.csv`
- 输入列：
  - `ss_solar_uigf_mw_avg`
  - `ss_wind_uigf_mw_avg`
  - `netinterchange_mw_avg`
  - `air_temperature_c`
  - `dewpoint_c`
  - `sea_level_pressure_hpa`
  - `wind_speed_mps`
  - `precip_1h_mm`
  - `net_load`
- `enc_in = 9`
- `dec_in = 9`

#### 结果

| 模型版本 | `pred_len=24` MSE / MAE | `pred_len=48` MSE / MAE | `pred_len=96` MSE / MAE | 平均 MSE | 平均 MAE |
| --- | --- | --- | --- | ---: | ---: |
| 旧版 `TimeXer`（无 NOAA） | 0.6608 / 0.6209 | 0.9159 / 0.7430 | 1.0484 / 0.8080 | 0.8750 | 0.7240 |
| 新版 `TimeXer`（含 NOAA） | 0.6886 / 0.6385 | 0.9156 / 0.7460 | 1.0814 / 0.8267 | 0.8952 | 0.7371 |

#### 结论

- 当前这版 NOAA 天气增强后的 `TimeXer`，整体没有优于旧版。
- 具体看：
  - `pred_len = 24` 变差，`MSE` 增加约 0.0278。
  - `pred_len = 48` 的 `MSE` 基本持平，略好约 0.0003，但 `MAE` 略差。
  - `pred_len = 96` 变差，`MSE` 增加约 0.0330。
- 因此，当前不能得出“加入这版 NOAA 天气特征后，`TimeXer` 性能提升”的结论。

#### 原因推断

- 当前 NOAA 字段里没有辐照度、云量等更强的光伏驱动特征，天气信息对净负荷的增益有限。
- `precip_1h_mm` 原始缺失极多，当前按 `0.0` 填补，它更像弱代理变量，未必提供了有效新增信息。
- NOAA 站点数据本身存在较多时间缺口，虽然已做前向/后向补齐，但这种填补方式只能保证可训练，未必能提升预测能力。
- 对 `TimeXer` 来说，增加外生变量维度后，优化难度也会上升；如果新增信号不够强，就可能只引入噪声。
- 这说明“天气要不要加”不能只看业务直觉，还要看天气变量本身是否足够有效、是否与目标机制真正对齐。

## 13. 当前可执行结论

- 现阶段如果目标是“先拿到更强基线”，优先使用 `DLinear` 作为对照基线。
- 现阶段如果目标是“围绕当前研究主线继续推进”，仍以 `TimeXer` 为主，但要把工作重点放在特征工程和数据口径，而不是先继续堆叠模型复杂度。
- 后续任何代理如果继续更新结论，必须同步更新本文件中的实验结果与结论部分，避免代码、结果和任务说明脱节。
