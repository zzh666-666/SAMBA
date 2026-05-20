# SAMBA Dataset 指标构成分析

## 1. 数据集概况

`SAMBA/Dataset` 目录下共有 3 个市场数据文件：

- `combined_dataframe_NYSE.csv`
- `combined_dataframe_IXIC.csv`
- `combined_dataframe_DJI.csv`

三者的整体结构基本一致：

- 每个文件原始维度均为 `3470 行 × 84 列`
- 时间范围均为 `2010-01-04` 到 `2023-10-16`
- `Name` 列分别固定为 `NYSE`、`IXIC`、`DJI`

从代码 [`SAMBA/utils/data_utils.py`](/J:/lunwen/AI+finance/code/1/SAMBA/utils/data_utils.py:141) 可以确认，预处理逻辑为：

1. `Date` 作为时间索引读入
2. 删除 `Name` 列
3. 对全表执行 `dropna()`
4. 以 `Price` 作为预测目标
5. 以除 `Price` 之外的其余列作为模型输入特征

因此可以区分出三层“字段数”：

- 原始 CSV 列数：`84`
- 去掉 `Date` 与 `Name` 后的有效数值列：`82`
- 真正输入模型的特征数：`81`

原因是 `Price` 被单独拿去做预测目标，不作为输入特征。

## 2. 缺失值情况

每个原始 CSV 都有 `1517` 个缺失值。代码在训练前会直接删除含缺失值的样本行，因此：

- 原始行数：`3470`
- `dropna()` 后行数：`2945`
- 被删除行数：`525`

缺失值主要集中在以下列：

- `SSEC`
- `KOSPI-F`
- `HSI`
- `HSI-F`
- `Gold-F`
- `FTSE`
- `GDAXI`
- `FCHI`
- 部分利率利差列，如 `TE*`、`DE*`、`DGS*`、`CTB*`、`DTB*`

这说明该数据集是多市场拼接后的日频数据，不同国家/资产的交易日并不完全一致。

## 3. 指标构成总览

去掉 `Date` 和 `Name` 后，数据集共包含 `82` 个数值字段。按含义可分为 6 类：

| 类别 | 数量 | 说明 |
| --- | ---: | --- |
| 目标市场价格与技术指标 | 15 | 价格、成交量、动量、ROC、EMA 等 |
| 利率、国债与信用利差 | 20 | 美债收益率、票据利率、期限利差、违约利差 |
| 大宗商品与商品期货 | 10 | 原油、布伦特、黄金、白银、铜、天然气、小麦等 |
| 外汇与美元指数 | 10 | 主要汇率和美元指数 |
| 全球股指与股指期货 | 19 | 美股、欧股、亚洲市场及股指期货 |
| 美股代表性个股 | 8 | 科技、金融、能源、医疗龙头股 |

合计：`15 + 20 + 10 + 10 + 19 + 8 = 82`

## 4. 各类指标明细

### 4.1 目标市场价格与技术指标（15 个）

- `Price`
- `Vol.`
- `weekday`
- `mom`
- `mom1`
- `mom2`
- `mom3`
- `ROC_5`
- `ROC_10`
- `ROC_15`
- `ROC_20`
- `EMA_10`
- `EMA_20`
- `EMA_50`
- `EMA_200`

说明：

- `Price` 是当前市场指数/价格水平，也是模型预测目标。
- `Vol.` 表示成交量相关指标，但当前文件中数值已是小数形式，更像是做过缩放或收益化后的量。
- `weekday` 是星期编码。
- `mom`、`mom1`、`mom2`、`mom3` 为动量或滞后收益类特征。
- `ROC_*` 为不同窗口的变化率指标。
- `EMA_*` 为不同窗口的指数移动平均线。

### 4.2 利率、国债与信用利差（20 个）

- `DGS10`
- `DGS5`
- `DBAA`
- `DAAA`
- `CTB6M`
- `CTB1Y`
- `DTB3`
- `DTB4WK`
- `DTB6`
- `CTB3M`
- `TE1`
- `TE2`
- `TE3`
- `TE5`
- `TE6`
- `DE1`
- `DE2`
- `DE4`
- `DE5`
- `DE6`

说明：

- `DGS*` 常见于美国国债固定期限收益率序列。
- `DTB*`、`CTB*` 对应不同期限的短端票据/国债利率。
- `DAAA`、`DBAA` 为信用评级债券收益率序列。
- `TE*` 一般可理解为期限利差类特征（Term Spread）。
- `DE*` 一般可理解为违约利差类特征（Default Spread）。

这些变量反映了宏观利率环境、流动性状态和信用风险变化。

### 4.3 大宗商品与商品期货（10 个）

- `WTI-oil`
- `Brent`
- `Gold-F`
- `silver-F`
- `copper-F`
- `GAS-F`
- `wheat-F`
- `oil`
- `XAU`
- `XAG`

说明：

- `WTI-oil`、`Brent`、`oil` 与能源价格相关。
- `Gold-F`、`silver-F`、`copper-F`、`GAS-F`、`wheat-F` 为商品期货。
- `XAU`、`XAG` 通常表示黄金、白银现货或对应报价代码。

这类指标反映通胀预期、避险情绪和全球商品周期。

### 4.4 外汇与美元指数（10 个）

- `NZD`
- `AUD`
- `CNY`
- `CAD`
- `EUR`
- `JPY`
- `GBP`
- `CHF`
- `Dollar Index-F`
- `Dollar Index`

说明：

- 前 8 个是主要货币或汇率相关序列。
- `Dollar Index` 与 `Dollar Index-F` 分别对应美元指数及其期货/衍生序列。

这类指标可以刻画美元强弱、风险偏好和跨国资本流动。

### 4.5 全球股指与股指期货（19 个）

- `FTSE-F`
- `HSI-F`
- `FCHI`
- `S&P-F`
- `RUSSELL-F`
- `IXIC`
- `DAX-F`
- `DJI-F`
- `HSI`
- `DJI`
- `RUT`
- `GSPC`
- `SSEC`
- `Nikkei-F`
- `KOSPI-F`
- `GDAXI`
- `CAC-F`
- `NASDAQ-F`
- `FTSE`

说明：

- 包含美国、欧洲、香港、中国、日本、韩国等主要市场指数及其期货。
- 其中既有现货指数，也有期货合约。
- 该组是跨市场联动信息的核心来源。

### 4.6 美股代表性个股（8 个）

- `AMZN`
- `MSFT`
- `XOM`
- `WFC`
- `GE`
- `JPM`
- `AAPL`
- `JNJ`

说明：

- 这些是美股中具有代表性的龙头公司，覆盖科技、金融、能源、工业、医疗等板块。
- 它们为指数预测提供了行业风格和龙头股走势信息。

## 5. 三个文件之间的差异

三份数据并不是完全逐列相同，而是“本市场自身标识列”位置不同：

- `combined_dataframe_NYSE.csv` 中用 `Name` 占据了 `NYSE` 所在位置
- `combined_dataframe_IXIC.csv` 中用 `Name` 占据了 `IXIC` 所在位置
- `combined_dataframe_DJI.csv` 中用 `Name` 占据了 `DJI` 所在位置

这意味着：

- 对 NYSE 数据集，模型不会额外看到 `NYSE` 这一列，因为该市场本身已经由 `Price` 表示
- 对 IXIC 数据集，`IXIC` 列同理被替换为 `Name`
- 对 DJI 数据集，`DJI` 列同理被替换为 `Name`

换句话说，每个数据文件都在用：

- 当前市场自己的 `Price` 作为目标主序列
- 其余市场、宏观变量、商品、汇率、个股与技术指标作为辅助解释变量

## 6. 模型实际使用方式

从 [`SAMBA/utils/data_utils.py`](/J:/lunwen/AI+finance/code/1/SAMBA/utils/data_utils.py:196) 可以看到，样本构造方式是：

- 输入 `X`：历史 `window` 天、除 `Price` 外的全部列，即 `81` 维输入特征
- 输出 `Y`：未来 `predict` 天的 `Price`

也就是说，SAMBA 在这里做的不是“涨跌分类”，而是基于多变量时间序列的价格预测，之后再把预测价格转成收益率并计算 `IC`、`RIC`、`MAE`、`RMSE`。

## 7. 基于 Wind 构建 A 股或沪深300数据集的方法

如果要把本项目从现有美股数据扩展到 A 股个股或沪深300市场指数，核心不是只下载一条价格序列，而是要构造一个能被 SAMBA 直接读取的多因子日频表。最终 CSV 至少需要满足以下约束：

- 第一列为 `Date`
- 数值列第一列为 `Price`，训练时会被当作预测目标
- 可保留 `Name`，但训练前会被 [`prepare_data`](/J:/lunwen/AI+finance/code/1/SAMBA/utils/data_utils.py:54) 删除
- 除 `Price` 外的全部数值列都会成为模型输入特征
- 行频率建议统一为交易日频率，缺失值处理后再进入训练，否则 `dropna()` 会删除大量样本

### 7.1 构建沪深300市场指数数据

沪深300数据建议以 `000300.SH` 为主序列，对齐当前项目中的 `000300SH` 数据集。

基础字段可直接由 Wind 下载：

- `open`
- `high`
- `low`
- `close`
- `pre_close`
- `volume`
- `amt`
- `chg`
- `pct_chg`

其中：

- `Price` 使用 `close`
- `Vol.` 使用 `volume` 或标准化后的成交量
- `weekday` 由 `Date` 计算
- `mom`、`ROC_*`、`EMA_*`、`ret_*`、`rolling_std_*`、`rsi_14`、`macd_hist` 等技术指标由 `Price`、`high`、`low`、`volume` 再计算
- `Name` 固定为 `000300.SH`

如果目标是复刻美股 `82` 列结构，则沪深300主序列只负责本地价格与技术指标，剩余外部因子继续从 Wind 补齐全球股指、利率、商品、外汇和美股龙头股。

### 7.2 构建 A 股个股数据

个股数据可使用类似 `000001.SZ`、`600519.SH` 这类 Wind 证券代码构建。和指数数据相比，个股可以额外补充基本面和交易微观结构字段：

- 行情字段：`open`、`high`、`low`、`close`、`pre_close`、`volume`、`amt`、`pct_chg`
- 估值字段：`pe_ttm`、`pb_lf`、`ps_ttm`、`pcf_ocf_ttm`
- 市值字段：`mkt_cap_ard`、`free_float_shares`、`total_shares`
- 交易字段：`turn`、`free_turn`
- 复权口径：建议使用后复权或不复权口径固定一种，不要混用

个股版的 `Price` 可以有两种选择：

- 价格预测口径：`Price = close`
- 收益预测口径：`Price = ret_1` 或当日收益率

当前项目里的 `combined_dataframe_000300SH_return_opt.csv` 和 `combined_dataframe_000001SZ_return_opt.csv` 实际更接近“收益预测口径”，输入特征数为 `26`，比原始美股数据的 `81` 维输入更轻。

## 8. Wind 指标与本项目字段的对应关系

### 8.1 本地行情与技术指标

| SAMBA 字段 | Wind 来源 | 构造方式 |
| --- | --- | --- |
| `Date` | 交易日期 | 作为索引或首列 |
| `Price` | `close` | 指数/个股收盘价，或收益优化版中的目标收益 |
| `Vol.` | `volume` | 成交量，可直接使用或缩放 |
| `weekday` | `Date` | `0~4` 星期编码 |
| `mom` | `close` | 一阶动量或滞后收益 |
| `mom1`、`mom2`、`mom3` | `close` | 多阶滞后动量 |
| `ROC_5`、`ROC_10`、`ROC_15`、`ROC_20` | `close` | 不同窗口变化率 |
| `EMA_10`、`EMA_20`、`EMA_50`、`EMA_200` | `close` | 指数移动平均 |

### 8.2 当前沪深300收益优化版字段

当前项目已使用的 `combined_dataframe_000300SH_return_opt.csv` 为轻量优化版，字段可以按下表从 Wind 行情中重建：

| 字段组 | 项目字段 | Wind 基础来源 |
| --- | --- | --- |
| 目标与基础量价 | `Price`、`Vol.`、`weekday` | `close`、`volume`、`Date` |
| 动量与收益 | `mom`、`ret_1`、`ret_2`、`ret_5`、`ret_10`、`ret_20` | `close` |
| 变化率 | `ROC_5`、`ROC_10`、`ROC_20` | `close` |
| 波动率 | `rolling_std_5`、`rolling_std_10`、`rolling_std_20` | `close` 或收益率 |
| 日内结构 | `high_low_spread`、`intraday_return` | `high`、`low`、`open`、`close` |
| 量能变化 | `volume_change`、`amount_change` | `volume`、`amt` |
| 技术指标 | `rsi_14`、`macd_hist`、`atr_14_norm` | `close`、`high`、`low` |
| 均线偏离 | `close_ma_*_gap`、`close_ema_*_gap` | `close` |
| 标识 | `Name` | 固定证券代码，如 `000300.SH` |

### 8.3 复刻美股 82 列结构时的外部因子

| SAMBA 类别 | 项目字段 | Wind 可取或可重建的数据 |
| --- | --- | --- |
| 利率、国债与信用利差 | `DGS10`、`DGS5`、`DTB*`、`CTB*`、`DAAA`、`DBAA`、`TE*`、`DE*` | 美国国债收益率、短端票据利率、信用债收益率，`TE*` 和 `DE*` 可由底层利率/信用序列相减得到 |
| 大宗商品与商品期货 | `WTI-oil`、`Brent`、`Gold-F`、`silver-F`、`copper-F`、`GAS-F`、`wheat-F`、`oil`、`XAU`、`XAG` | 原油、布伦特、黄金、白银、铜、天然气、小麦、贵金属现货或连续合约 |
| 外汇与美元指数 | `NZD`、`AUD`、`CNY`、`CAD`、`EUR`、`JPY`、`GBP`、`CHF`、`Dollar Index`、`Dollar Index-F` | 主要汇率、美元指数、美元指数期货 |
| 全球股指与股指期货 | `FTSE-F`、`HSI-F`、`FCHI`、`S&P-F`、`RUSSELL-F`、`IXIC`、`DAX-F`、`DJI-F`、`HSI`、`DJI`、`RUT`、`GSPC`、`SSEC`、`Nikkei-F`、`KOSPI-F`、`GDAXI`、`CAC-F`、`NASDAQ-F`、`FTSE` | 全球主要现货指数与指数期货连续合约 |
| 美股代表性个股 | `AMZN`、`MSFT`、`XOM`、`WFC`、`GE`、`JPM`、`AAPL`、`JNJ` | 美股个股日收盘价或收益率 |

注意：Wind 中不同资产的正式代码可能随终端权限和市场口径略有差异，建议先在 WindApp 中确认代码，再固化到 API 脚本中。对于期货类字段，优先使用连续合约或主力连续口径，避免手动换月造成断点。

## 9. WindApp 与 Wind API 两种落地方法

### 9.1 WindApp / Excel 插件方法

这种方式适合先手工验证字段口径，尤其适合确认 Wind 代码、字段名和缺失情况。

推荐流程：

1. 在 Wind 金融终端中搜索目标证券，如 `000300.SH` 或具体 A 股代码
2. 打开 Excel 插件，使用 `WSD` 批量提取日频行情字段
3. 每类资产单独导出一个 sheet，如主行情、全球指数、商品、外汇、利率
4. 将所有 sheet 按 `Date` 左连接或内连接合并
5. 在 Excel 或 Python 中计算 `ROC_*`、`EMA_*`、`ret_*`、`rolling_std_*` 等派生指标
6. 统一列名为本项目 schema
7. 导出为 `combined_dataframe_000300SH_wind.csv` 或对应个股文件

Excel 公式示例：

```text
=WSD("000300.SH","open,high,low,close,pre_close,volume,amt,pct_chg","2010-01-01","2023-12-31","Period=D;PriceAdj=F")
```

优点：

- 上手快，便于人工检查
- 适合核对代码、字段、单位和缺失值
- 适合先做 15 维或 26 维基础数据

缺点：

- 可复现性弱
- 多资产批量更新麻烦
- 82 列完整版容易出现手工合并错误

### 9.2 WindPy API 方法

这种方式适合正式构建可复现的数据管道。建议先用 WindApp 确认代码，再把代码列表写入脚本。

基本流程：

1. `w.start()` 启动 WindPy
2. 用 `w.wsd()` 下载沪深300或个股主行情
3. 用 `w.wsd()` 分组下载外部因子
4. 将 Wind 返回结果转换为 `DataFrame`
5. 统一日期索引、字段命名和频率
6. 计算技术指标与派生利差
7. 对齐交易日历并处理缺失值
8. 输出为 SAMBA 可直接读取的 CSV

API 伪代码结构：

```python
from WindPy import w
import pandas as pd

w.start()

start_date = "2010-01-01"
end_date = "2023-12-31"

main_code = "000300.SH"
main_fields = "open,high,low,close,pre_close,volume,amt,pct_chg"
raw = w.wsd(main_code, main_fields, start_date, end_date, "Period=D;PriceAdj=F")

# 1. 转换为 DataFrame
# 2. close -> Price, volume -> Vol.
# 3. Date -> weekday
# 4. 计算 mom / ROC / EMA / ret / volatility / RSI / MACD / ATR
# 5. 下载并合并 global_equity / rates_fx / commodities / us_stocks
# 6. 添加 Name = main_code
# 7. 输出 CSV
```

优点：

- 可复现，适合论文实验
- 可以批量构建多个股票或多个指数
- 更容易固定列顺序、字段口径和缺失处理策略

缺点：

- 需要本机 Wind 终端和 API 权限
- 首次整理代码表和字段表耗时
- 国际期货、信用利差、连续合约的口径需要反复核对

### 9.3 推荐落地顺序

最稳妥的顺序是：

1. 用 WindApp 先下载 `000300.SH` 主行情，做出 15 维基础版
2. 按当前收益优化版 schema 扩展到 26 维，确认训练能跑通
3. 用 WindApp 核对外部因子代码和缺失值
4. 用 WindPy API 固化自动化脚本
5. 最后再冲 40~60 维增强版或 82 列完整复刻版

## 10. 结论

`SAMBA/Dataset` 的数据不是单纯的指数价格表，而是一个融合了：

- 目标市场自身技术指标
- 全球主要股指与股指期货
- 外汇与美元指数
- 大宗商品与商品期货
- 宏观利率、期限利差、信用利差
- 美股代表性个股

的多因子日频市场数据集。

如果从建模角度概括，可以把它理解为：

`目标市场价格序列 + 技术面因子 + 跨市场联动因子 + 宏观金融因子 + 商品与汇率因子`

这正是 SAMBA 这类“时序建模 + 图结构建模”方法适合发挥作用的数据形式。

对沪深300或 A 股个股来说，最推荐的 Wind 构建路径是：

`主行情字段 + 本地技术指标 + 可解释的交易/估值因子 + Wind 外部跨市场因子`

其中主行情和技术指标负责跑通模型，外部因子负责让 SAMBA 的特征图真正具备跨市场联动信息。

## 11. 完成难度评估

| 目标版本 | 维度目标 | 完成难度 | 主要难点 | 建议优先级 |
| --- | ---: | --- | --- | --- |
| 沪深300 15 维基础版 | `15` 数值列 | 低 | 只需主行情和技术指标计算 | 最高 |
| 沪深300/个股 26 维收益优化版 | `26` 输入特征左右 | 中低 | 技术指标、收益目标和缺失处理要统一 | 最高 |
| A 股个股增强版 | `30~50` 数值列 | 中 | 估值、市值、换手率字段口径和复权口径需要固定 | 中高 |
| 沪深300 40~60 维跨市场版 | `40~60` 数值列 | 中 | 多资产日历对齐、外部因子缺失值处理 | 中高 |
| 复刻美股 82 列完整版 | `82` 数值列 / `81` 输入特征 | 高 | 国际期货连续合约、利率/信用利差、`TE*`/`DE*` 定义复原 | 中 |

总体判断：

- 如果目标是尽快得到可训练数据集，难度为 **中低**，优先做 `26` 维收益优化版。
- 如果目标是最大程度复刻原始美股 `82` 列数据，难度为 **高**，主要工作量在 Wind 代码确认、跨市场日历对齐和利差字段复原。
- 如果目标是论文实验中更贴合中国市场，推荐做 **40~60 维增强版**，难度适中，解释性和可实现性最好。
