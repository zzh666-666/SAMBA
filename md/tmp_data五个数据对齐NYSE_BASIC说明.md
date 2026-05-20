# SAMBA tmp_data 五个数据对齐 NYSE_BASIC 说明

## 1. 目标

本次处理的目标是把 `SAMBA/tmp_data` 中的五个 Excel 行情文件，对齐到
`SAMBA/Dataset/combined_dataframe_NYSE_basic_indicators.csv` 的基础指标数据结构，
并在 `SAMBA/Dataset` 中生成可被现有 SAMBA 训练流程直接读取的 CSV 文件。

参考文件为：

```text
SAMBA/Dataset/combined_dataframe_NYSE_basic_indicators.csv
```

该参考文件包含 17 列：

```text
Date, Price, Vol., weekday, mom, mom1, mom2, mom3,
ROC_5, ROC_10, ROC_15, ROC_20,
EMA_10, EMA_20, EMA_50, EMA_200, Name
```

生成文件为：

| 源文件 | 输出文件 | 行数 | 日期范围 |
| --- | --- | ---: | --- |
| `000016.SH.xlsx` | `combined_dataframe_000016SH_basic_indicators.csv` | 3347 | 2010-01-04 至 2023-10-16 |
| `000100.SZ.xlsx` | `combined_dataframe_000100SZ_basic_indicators.csv` | 3105 | 2010-01-04 至 2023-10-16 |
| `000300.SH.xlsx` | `combined_dataframe_000300SH_basic_indicators.csv` | 3347 | 2010-01-04 至 2023-10-16 |
| `002129.SZ.xlsx` | `combined_dataframe_002129SZ_basic_indicators.csv` | 2843 | 2010-01-04 至 2023-10-16 |
| `300015.SZ.xlsx` | `combined_dataframe_300015SZ_basic_indicators.csv` | 3311 | 2010-01-04 至 2023-10-16 |

## 2. 为什么选择 NYSE_BASIC 作为对齐基准

现有训练代码 `SAMBA/utils/data_utils.py` 的数据读取逻辑是：

1. 使用 `Date` 作为时间索引读取 CSV。
2. 如果存在 `Name` 列，则删除 `Name`。
3. 将其余列转成数值。
4. 删除缺失值。
5. 使用第一列数值列 `Price` 作为预测目标。
6. 使用除 `Price` 外的其他数值列作为输入特征。

因此，新数据要能直接进入当前 SAMBA 训练流程，需要满足以下条件：

- 第一列必须是 `Date`。
- `Price` 必须是第一个数值字段，并作为预测目标。
- `Name` 可以保留，训练前会自动删除。
- 其他字段必须是可转成数值的特征列。
- 尽量不要包含缺失值，否则 `dropna()` 会删除样本。

`combined_dataframe_NYSE_basic_indicators.csv` 是当前项目中最适合这五个 Excel
文件的对齐对象，因为它只使用基础行情和技术指标，不依赖美股宏观、商品、外汇、
利率等外部因子。五个 Excel 文件本身只包含本地行情字段，无法可靠补齐 82 维完整
美股数据集中的外部因子，所以采用 17 列基础指标版本更加稳妥。

## 3. 原始数据读取方式

五个源文件是 `.xlsx` 格式，字段大体如下：

```text
代码, 名称, 日期, 开盘价(元), 最高价(元), 最低价(元),
收盘价(元), 涨跌幅, 成交额(百万), 成交量 或 成交量(股)
```

读取时需要注意两点：

1. Excel 文件中的日期是 Excel 序列号，例如 `37986`，需要转换为真实日期。
2. 部分 Excel 文件的样式元数据会导致 `openpyxl` 直接读取报错，因此使用底层
   xlsx XML 数据读取工作表内容，避开样式解析问题。

日期转换口径为：

```text
Excel serial date -> pandas datetime, origin = 1899-12-30
```

这样可以得到真实交易日期，如 `2010-01-04`。

## 4. 字段映射

原始 Excel 字段与 SAMBA 基础指标字段的映射如下：

| 输出字段 | 来源或计算方式 | 原因 |
| --- | --- | --- |
| `Date` | Excel `日期` 转换为 `YYYY-MM-DD` | 与参考 CSV 和训练代码一致 |
| `Price` | Excel `收盘价(元)` | SAMBA 当前流程将 `Price` 作为预测目标 |
| `Vol.` | `成交量` 或 `成交量(股)` 的变化率 | 参考文件中的 `Vol.` 是小数形式，更接近量能变化而非原始成交量 |
| `weekday` | `Date.dt.weekday` | 保留星期效应，口径为周一 0 到周日 6 |
| `mom` 到 `mom3` | 由 `Price` 计算的连续动量特征 | 与参考文件的技术指标结构一致 |
| `ROC_5` 到 `ROC_20` | 由 `Price` 计算的不同窗口变化率 | 表示多个尺度的价格变化 |
| `EMA_10` 到 `EMA_200` | 由 `Price` 计算的指数移动平均 | 表示短中长期趋势 |
| `Name` | 原始证券代码 | 保留数据来源标识，训练前自动删除 |

## 5. 技术指标计算口径

为了与 `combined_dataframe_NYSE_basic_indicators.csv` 保持完全一致，本次不是简单
按常见 A 股技术分析口径重新设计指标，而是先检查 NYSE_BASIC 中已有字段的数值规律，
再复刻其计算方向。

### 5.1 量能字段

`Vol.` 使用成交量的相邻交易日变化率：

```text
Vol. = volume / volume.shift(-1) - 1
```

这样生成的小数形式更接近参考文件中的 `Vol.` 数值分布。若直接使用原始成交量，
数值尺度会与参考文件差异很大，不利于保持实验口径一致。

### 5.2 动量字段

基础动量为：

```text
mom = Price / Price.shift(-1) - 1
```

随后构造：

```text
mom1 = mom.shift(-1)
mom2 = mom.shift(-2)
mom3 = mom.shift(-3)
```

采用该口径的原因是：经校验，NYSE_BASIC 中的 `mom/mom1/mom2/mom3` 与这种方向的
计算方式一致。对齐工作的优先目标是保证新数据与基准数据集特征工程口径相同，
便于横向比较模型实验。

### 5.3 ROC 字段

不同窗口的变化率计算为：

```text
ROC_n = (Price / Price.shift(-n) - 1) * 100
```

其中 `n` 分别为：

```text
5, 10, 15, 20
```

原因同样是保持与 NYSE_BASIC 的既有字段数值口径一致。

### 5.4 EMA 字段

EMA 字段使用反向序列上的指数移动平均，再恢复为正常日期顺序：

```text
EMA_n = Price.iloc[::-1].ewm(span=n, adjust=False).mean().iloc[::-1]
```

其中 `n` 分别为：

```text
10, 20, 50, 200
```

这样可以复现 NYSE_BASIC 中的 EMA 数值方向。若直接在升序日期上计算 EMA，
会得到与参考文件不同的结果。

## 6. 时间范围对齐

参考文件的日期范围为：

```text
2010-01-04 至 2023-10-16
```

五个源文件的原始时间范围不完全相同，且有的覆盖到 2026-05-19。为了与
NYSE_BASIC 保持实验可比性，本次统一裁剪到参考文件日期范围内：

```text
Date >= 2010-01-04
Date <= 2023-10-16
```

这样做的原因有三点：

1. 保持和基准实验相同的历史窗口。
2. 避免新数据包含 NYSE_BASIC 没有覆盖的未来区间。
3. 便于后续将多个数据集放在同一训练配置下比较。

需要注意的是，裁剪后并没有强制要求每个 A 股数据文件都拥有与 NYSE 完全相同的每一个交易日。
原因是中美市场节假日不同，如果强制按 NYSE 日期逐日内连接，会额外损失大量 A 股有效交易日。
因此本次采用“同一日期范围内保留各证券自身交易日”的策略。

## 7. 缺失值处理

技术指标计算会在序列尾部产生少量缺失值，例如：

- `shift(-1)` 会让最后一天没有可比较的下一条记录。
- `shift(-20)` 会让最后 20 条记录无法计算 `ROC_20`。
- 成交量为 0 时，变化率可能产生无穷值。

处理方式为：

1. 将无穷值替换为缺失值。
2. 删除任一字段存在缺失的行。
3. 输出前确认总缺失值为 0。

这样做是为了适配 `prepare_data()` 中的训练前 `dropna()` 逻辑，避免训练时再发生不可控的样本删除。

## 8. 输出校验结果

生成后对五个 CSV 做了统一校验：

| 文件 | 维度 | 列顺序是否一致 | 缺失值数量 |
| --- | ---: | --- | ---: |
| `combined_dataframe_000016SH_basic_indicators.csv` | 3347 × 17 | 是 | 0 |
| `combined_dataframe_000100SZ_basic_indicators.csv` | 3105 × 17 | 是 | 0 |
| `combined_dataframe_000300SH_basic_indicators.csv` | 3347 × 17 | 是 | 0 |
| `combined_dataframe_002129SZ_basic_indicators.csv` | 2843 × 17 | 是 | 0 |
| `combined_dataframe_300015SZ_basic_indicators.csv` | 3311 × 17 | 是 | 0 |

其中“列顺序是否一致”是指与：

```text
combined_dataframe_NYSE_basic_indicators.csv
```

的 17 个字段完全同名、同顺序。

## 9. 方法选择的总体原因

本次对齐遵循三个原则：

### 9.1 优先保证训练兼容性

现有 SAMBA 代码没有为新数据单独写读取逻辑，因此最稳妥的方式是让新 CSV 直接符合
现有 `prepare_data()` 的输入要求。

### 9.2 优先保证与基准数据可比较

NYSE_BASIC 是当前项目已有的基础指标数据集。五个新数据文件采用相同列结构、相同日期范围、
相同指标方向，有利于后续比较不同指数或个股在同一模型结构下的结果。

### 9.3 避免伪造无法从源文件得到的外部因子

五个 Excel 文件只包含本地行情字段，无法自然生成美股完整 82 维数据中的利率、外汇、
商品、全球股指、美股龙头股等外部因子。强行填充这些字段会引入较大主观性。

因此，本次选择生成 17 列基础指标版，而不是生成 82 列完整版。

## 10. 后续可扩展方向

如果后续需要更完整的多因子数据集，可以继续做两类扩展：

1. 使用 Wind 或其他数据源补齐外部因子，构造 82 列完整版。
2. 在 `paper_config.py` 中加入五个新数据集的 mapping，使其可以在 `main.py` 中直接选择训练。

在没有额外外部因子数据源之前，当前 17 列基础指标版是最稳妥、最可复现的对齐结果。
