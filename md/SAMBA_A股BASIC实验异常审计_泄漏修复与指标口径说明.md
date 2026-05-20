# SAMBA A股 BASIC 实验异常审计、泄漏修复与指标口径说明

## 1. 审计结论

本次检查对象为 `SAMBA/saved_models` 中五个 A 股或指数 BASIC 数据集的训练结果：

| 数据集 | 旧 IC | 旧 RIC | 旧 MAE | 旧 RMSE | 评价 |
| --- | ---: | ---: | ---: | ---: | --- |
| `000300SH_BASIC` | 0.9222 | 0.9056 | 0.0032 | 0.0042 | 异常偏高 |
| `000016SH_BASIC` | 0.9201 | 0.9028 | 0.0033 | 0.0044 | 异常偏高 |
| `300015SZ_BASIC` | 0.8689 | 0.8085 | 0.0121 | 0.0165 | 异常偏高 |
| `002129SZ_BASIC` | 0.5908 | 0.5277 | 0.0168 | 0.0221 | 明显偏高 |
| `000100SZ_BASIC` | 0.2016 | 0.2037 | 0.0732 | 0.0976 | 相对正常，但仍需重跑 |

结论：上述结果不建议作为论文有效实验结果使用。`IC/RIC` 高到 `0.9` 的主要原因不是模型显著优于论文和主流方法，而是数据构造中存在未来函数，导致模型输入已经包含了未来价格信息。

已完成修复：重新从 `SAMBA/tmp_data` 原始 Excel 文件构造了 `SAMBA/Dataset` 下五个 `*_basic_indicators.csv`，并新增可复现脚本：

```text
SAMBA/scripts/rebuild_basic_indicators_no_leakage.py
```

旧的 `saved_models` 结果是在泄漏数据上得到的，数据集重建后应重新训练。

## 2. 为什么原构造会泄漏

被检查文档：

```text
SAMBA/md/tmp_data五个数据对齐NYSE_BASIC说明.md
```

该文档中原始构造为了对齐 `NYSE_BASIC` 的数值方向，采用了如下公式：

```text
Vol. = volume / volume.shift(-1) - 1
mom = Price / Price.shift(-1) - 1
mom1 = mom.shift(-1)
mom2 = mom.shift(-2)
mom3 = mom.shift(-3)
ROC_n = (Price / Price.shift(-n) - 1) * 100
EMA_n = Price.iloc[::-1].ewm(span=n, adjust=False).mean().iloc[::-1]
```

五个 CSV 的日期都是升序排列，即 `2010-01-04 -> 2023-10-16`。在升序时间序列里，`shift(-1)` 指向下一条交易日记录，因此：

```text
mom_t = Price_t / Price_{t+1} - 1
```

这等价于把 `t+1` 的未来价格写入了 `t` 时刻的输入特征。`ROC_5/10/15/20` 同理会看到未来 5、10、15、20 个交易日价格；反向 EMA 则使用未来序列计算当前时刻均线。

重构前的泄漏验证结果如下：

| 数据集 | `corr(-mom, 下一日收益)` |
| --- | ---: |
| `000016SH` | 0.999290 |
| `000100SZ` | 0.949679 |
| `000300SH` | 0.999370 |
| `002129SZ` | 0.955635 |
| `300015SZ` | 0.961021 |

这说明 `mom` 几乎直接等价于下一日收益的变体，模型在训练时事实上可以“看到答案”。因此 `IC/RIC=0.9` 是泄漏结果，而不是有效预测能力。

## 3. 无泄漏重构方案

修复原则：所有输入特征只能使用当前时刻及过去时刻的信息。

新的构造口径如下：

| 字段 | 新公式 | 是否使用未来 |
| --- | --- | --- |
| `Price` | 当日收盘价 `close_t` | 否 |
| `Vol.` | `volume_t / volume_{t-1} - 1` | 否 |
| `weekday` | `Date.dt.weekday` | 否 |
| `mom` | `Price_t / Price_{t-1} - 1` | 否 |
| `mom1` | `mom_{t-1}` | 否 |
| `mom2` | `mom_{t-2}` | 否 |
| `mom3` | `mom_{t-3}` | 否 |
| `ROC_n` | `(Price_t / Price_{t-n} - 1) * 100` | 否 |
| `EMA_n` | 升序日期上正常计算 `ewm(span=n)` | 否 |
| `Name` | 证券代码 | 否 |

实现细节：

1. 读取 `SAMBA/tmp_data/*.xlsx` 原始 Wind 文件。
2. 由于这些 Excel 文件的样式元数据会使 `openpyxl` 报错，脚本直接解析 `.xlsx` 内部 `sheet1.xml` 和 `sharedStrings.xml`，只读取单元格值。
3. 按日期升序排序、去重。
4. 在完整历史上计算过去信息指标。
5. 裁剪到 `2010-01-04` 至 `2023-10-16`。
6. 删除缺失或无穷值。
7. 输出到原有五个数据集文件名，保持 `main.py` 和 `paper_config.py` 的路径兼容。

## 4. 已重构的数据集

重构命令：

```powershell
python scripts\rebuild_basic_indicators_no_leakage.py
```

输出文件：

| 输出文件 | 行数 | 列数 | 缺失值 | 日期范围 |
| --- | ---: | ---: | ---: | --- |
| `Dataset/combined_dataframe_000016SH_basic_indicators.csv` | 3347 | 17 | 0 | 2010-01-04 至 2023-10-16 |
| `Dataset/combined_dataframe_000100SZ_basic_indicators.csv` | 3105 | 17 | 0 | 2010-01-04 至 2023-10-16 |
| `Dataset/combined_dataframe_000300SH_basic_indicators.csv` | 3347 | 17 | 0 | 2010-01-04 至 2023-10-16 |
| `Dataset/combined_dataframe_002129SZ_basic_indicators.csv` | 2843 | 17 | 0 | 2010-01-04 至 2023-10-16 |
| `Dataset/combined_dataframe_300015SZ_basic_indicators.csv` | 3311 | 17 | 0 | 2010-01-04 至 2023-10-16 |

重构后的泄漏校验：

| 数据集 | `corr(mom, 下一日收益)` |
| --- | ---: |
| `000016SH` | 0.005912 |
| `000100SZ` | -0.013301 |
| `000300SH` | 0.020328 |
| `002129SZ` | 0.011228 |
| `300015SZ` | -0.037774 |

重构后，`mom` 与下一日收益不再接近完全相关，泄漏信号已消失。

## 5. SAMBA 论文指标口径

本项目对应论文为：

```text
SAMBA/md/2025_ICASSP_Mamba遇见金融市场：面向股价预测的图-Mamba方法_Ali Mehrabian.pdf
```

论文题目为 `Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction`。arXiv 页面说明该工作是面向 stock return prediction 的 Graph-Mamba 方法，并提供代码和数据集链接。

论文实验口径要点：

| 项目 | 论文口径 |
| --- | --- |
| 数据集 | NASDAQ、NYSE、DJIA |
| 时间范围 | 2010-01 至 2023-11 |
| 特征数 | `N = 82` 个 daily stock features |
| 历史窗口 | `L = 5` |
| 训练/验证/测试 | 80% / 5% / 15% |
| 预测目标 | 一步收益率 `o_{L+1} = (c_{L+1} - c_L) / c_L` |
| 归一化 | Min-max 到 `[0, 1]` |
| 指标 | RMSE、IC、RIC、训练时间、MACs |

论文表 1 中 SAMBA 的结果：

| 数据集 | RMSE | IC | RIC |
| --- | ---: | ---: | ---: |
| NASDAQ | 0.0128 | 0.5046 | 0.4767 |
| NYSE | 0.0125 | 0.5044 | 0.4950 |
| DJIA | 0.0108 | 0.4483 | 0.4703 |

论文对 IC/RIC 的描述是：IC 为平均 Pearson 相关系数，RIC 为平均 Spearman 系数，用来衡量预测收益与真实收益的相关性和排序能力。

### 5.1 SAMBA 论文的任务定义

SAMBA 原论文的任务不是传统意义上的“沪深300成分股横截面选股”，而是**单一市场指数/市场序列的多变量时间序列收益预测任务**。论文把每个交易日的多个 daily stock features 视为图节点，输入最近 `L=5` 天的特征矩阵：

```text
X = (x_1, ..., x_L)^T ∈ R^{L×N}
```

其中 `N=82`，包含目标市场自身技术指标、其他市场指数、商品、期货、汇率等外部日频变量。模型预测下一日收益率：

```text
o_{L+1} = (c_{L+1} - c_L) / c_L
```

因此，SAMBA 论文中的图节点是“日频特征/市场变量”，不是“股票池中的每一只股票”。它更接近“给定一个目标市场 NASDAQ/NYSE/DJIA，利用多变量特征图预测该市场下一日收益”，而不是“每天对 300 只股票排序”。

### 5.2 SAMBA 论文同任务 SOTA 对比

在 SAMBA 论文表 1 中，作者将 SAMBA 与 LSTM、Transformer、FreTS、StockMixer、AGCRN、FourierGNN、MambaStock 等方法在同一 NASDAQ/NYSE/DJIA 任务上比较。该表是当前最适合与 SAMBA 论文任务直接比较的结果。

先列出参与同任务对比的代表性论文/方法来源：

| 方法 | 论文或来源 | 会议/期刊/年份 | 与 SAMBA 对比中的角色 |
| --- | --- | --- | --- |
| LSTM | Stock market prediction using LSTM recurrent neural network | Procedia Computer Science, 2020 | 经典循环神经网络基线 |
| Transformer | Stock market index prediction using deep Transformer model | Expert Systems with Applications, 2022 | Transformer 时间序列预测基线 |
| FreTS | Frequency-domain MLPs are more effective learners in time series forecasting | NeurIPS, 2023 | 频域 MLP 时间序列基线 |
| StockMixer | StockMixer: A simple yet strong MLP-based architecture for stock price forecasting | AAAI, 2024 | 股票预测 MLP-mixer 类强基线 |
| AGCRN | Adaptive Graph Convolutional Recurrent Network for traffic forecasting | NeurIPS, 2020 | 自适应图卷积循环网络基线 |
| FourierGNN | FourierGNN: Rethinking multivariate time series forecasting from a pure graph perspective | NeurIPS, 2023 | 多变量时间序列图模型基线 |
| MambaStock | MambaStock: Selective state space model for stock prediction | arXiv, 2024 | 单向 Mamba 股票预测基线 |
| SAMBA | Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction | ICASSP, 2025 | 本文方法，BI-Mamba + AGC |

在上述同任务方法中，SAMBA 论文报告的指标对比如下：

| 数据集 | SAMBA RMSE | SAMBA IC | SAMBA RIC | 同任务第二强 IC | 同任务第二强 RIC | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NASDAQ | 0.0128 | 0.5046 | 0.4767 | FreTS 0.2722 | FreTS 0.2644 | SAMBA 明显领先 |
| NYSE | 0.0125 | 0.5044 | 0.4950 | MambaStock 0.3697 | Transformer 0.3293 | SAMBA 明显领先 |
| DJIA | 0.0108 | 0.4483 | 0.4703 | MambaStock 0.3355 | Transformer 0.3063 | SAMBA 明显领先 |

这说明，在原论文定义的**单市场多变量收益预测任务**中，SAMBA 的 `IC/RIC≈0.45-0.50` 已经是论文报告的 SOTA 水平。也正因为如此，本项目五个 A 股 BASIC 泄漏实验中出现 `IC/RIC≈0.9` 明显不合理，远超同任务论文上限，应优先怀疑数据泄漏或指标口径错误。

### 5.3 与本项目五个 BASIC 数据集的关系

本项目五个 BASIC 数据集目前是：

```text
000016SH、000100SZ、000300SH、002129SZ、300015SZ
```

它们每个文件都是一条独立的单资产或指数时间序列，且只有 17 列基础技术指标。它们更接近 SAMBA 原论文的“单目标市场时间序列预测”形式，但仍有两点差异：

1. 原论文使用 82 个跨市场/宏观/商品/汇率/技术变量，本项目 BASIC 版只有本地基础技术指标。
2. 原论文问题定义直接预测下一日收益率，本项目当前代码先预测未来 `Price`，再将预测价格差分为收益率计算 IC/RIC。

因此，本项目重跑后的合理预期不应超过 SAMBA 原论文 `0.45-0.50` 太多。若重建无泄漏数据后仍出现 `0.8+`，应继续检查标签、反归一化、测试集切分和指标计算。

## 6. 本项目当前指标口径是否与论文对齐

代码位置：

```text
SAMBA/utils/data_utils.py
SAMBA/main.py
SAMBA/utils/metrics.py
```

当前 `prepare_data()` 的训练目标是未来 `Price`，不是直接构造未来收益率标签：

```text
X_seq = data[i:i+window, 1:]
Y_seq = data[i+window:i+window+predict, 0]
```

训练结束后，`main.py` 将预测价格和真实价格反归一化，再用相邻预测价格计算收益率：

```text
return_p = (y_p[1:] - y_p[:-1]) / y_p[:-1]
return_t = (y_t[1:] - y_t[:-1]) / y_t[:-1]
IC = pearson_correlation(return_t, return_p)
RIC = rank_information_coefficient(return_t[:, 0], return_p[:, 0])
```

判断：

1. 与 SAMBA 论文基本对齐的部分：最终 IC/RIC 是在收益率空间计算，使用 Pearson/Spearman；数据切分比例、窗口长度、MinMax 归一化方向也接近论文设置。
2. 与论文不完全一致的部分：论文问题定义直接预测下一日收益率；当前代码先预测价格，再事后转收益率。
3. 与主流 A 股横截面实验不一致的部分：当前五个 BASIC 文件是五条独立单资产或指数时间序列；`RIC` 实际是测试时间维度上的排序相关，不是每天在股票池横截面上对所有股票排序。

因此，如果目标是复现 SAMBA 原论文风格，当前代码指标口径可以作为近似复现，但建议把训练标签改为收益率以更严格对齐论文。若目标是做沪深300股票池预测论文，当前口径不够，应改为横截面日频 IC/RankIC。

## 7. 主流股票池实验任务与 SOTA 对比

### 7.1 主流股票池实验任务定义

主流股票池实验的任务与 SAMBA 原论文不同。它通常是**横截面股票收益预测/选股排序任务**：

```text
给定股票池 S，例如 CSI300/CSI500/CSI800，
在每个交易日 t，对池内每只股票 u ∈ S 预测未来 d 日收益 r_{u,t+d}，
再按预测分数对所有股票排序。
```

典型输入是每只股票过去若干天的量价、技术指标、基本面或关系图特征；典型输出是未来 1 日或 5 日收益率。评价时不是对单条时间序列整体算一次相关系数，而是每天在股票池横截面上计算：

```text
IC_t = corr(predicted_scores_t, future_returns_t)
RankIC_t = corr(rank(predicted_scores_t), rank(future_returns_t))
```

最后对所有测试日的 `IC_t / RankIC_t` 求均值、标准差和信息比率：

```text
ICIR = mean(IC_t) / std(IC_t)
RankICIR = mean(RankIC_t) / std(RankIC_t)
```

同时，很多论文还会构造投资组合，例如每天买入预测分数最高的 top 30 股票，报告年化收益、信息比率和最大回撤。

### 7.2 主流股票池任务的代表性 SOTA 结果

下表中的结果属于**股票池横截面选股/收益预测任务**，不能与 SAMBA 原论文的 NASDAQ/NYSE/DJIA 单市场时间序列任务直接比较，但可以作为“沪深300股票池实验”的合理指标区间参考。

| 方法/来源 | 会议或平台 | 股票池/数据 | 任务口径 | 代表结果 |
| --- | --- | --- | --- | --- |
| MASTER | AAAI 2024 | CSI300、CSI800，2008-2022，Alpha158 | 预测未来 5 日 normalized return ratio；日频横截面 IC/RankIC；top30 组合 | CSI300: `IC=0.064±0.006`，`RankIC=0.076±0.005` |
| COGRASP | IJCAI 2025 | CSI300，价格特征 + 雪球共现图 | CSI300 股票池横截面预测；IC、RankIC、ICIR、RankICIR | `IC=0.0546`，`RankIC=0.0647` |
| Qlib DoubleEnsemble | Qlib benchmark | 中国 A 股 + CSI300，Alpha158，20 seeds | 标准 Qlib workflow，日频横截面信号评价 + 组合评价 | `IC=0.0521±0.00`，`RankIC=0.0502±0.00` |
| Qlib HIST | Qlib benchmark | 中国 A 股 + CSI300，Alpha360，20 seeds | 标准 Qlib workflow，日频横截面信号评价 + 组合评价 | `IC=0.0522±0.00`，`RankIC=0.0667±0.00` |
| StockMixer | AAAI 2024 | NASDAQ、NYSE、S&P500 股票池 | 股票池历史多指标输入，预测下一日收盘价并计算 1 日收益排序指标 | 是股票池任务代表，但数据集与 CSI300 不同，不宜直接当作 A 股 SOTA |

从这些公开结果看，主流股票池任务里 `IC≈0.05-0.07`、`RankIC≈0.06-0.08` 已经是很强的水平。若在 CSI300 横截面任务上出现 `IC/RIC≈0.9`，基本可以直接判定为泄漏、标签错位、测试集污染或指标实现错误。

### 7.3 两类任务不能直接混比

| 对比项 | SAMBA 原论文任务 | 主流股票池任务 |
| --- | --- | --- |
| 预测对象 | 单一市场/指数序列，如 NASDAQ、NYSE、DJIA | 股票池内每只股票，如 CSI300 的 300 只成分股 |
| 图节点含义 | daily stock features / 市场变量 | 股票、行业、概念、关系边或股票-时间 token |
| 输出 | 下一日市场收益率 | 每只股票未来收益或排序分数 |
| IC/RIC 计算 | 收益率预测与真实收益的相关性，论文称平均 Pearson/Spearman | 每日横截面 IC/RankIC，再跨天求均值 |
| 代表强结果 | SAMBA 约 `IC/RIC=0.45-0.50` | CSI300 常见强结果约 `IC=0.05-0.07`、`RankIC=0.06-0.08` |
| 是否适合比较本项目五个 BASIC 文件 | 较接近，但特征数和标签口径仍不同 | 不直接适合，除非改造成 `date × stock × feature` 面板 |

因此，本文档建议将后续实验明确分成两条线：

1. **SAMBA 复现/扩展线**：继续使用单资产或单指数时间序列，重点与 SAMBA 原论文的 NASDAQ/NYSE/DJIA 任务对齐。
2. **沪深300股票池线**：构造完整 `date × stock × feature` 面板，按照 MASTER、Qlib、COGRASP 等主流口径报告日频横截面 IC/RankIC 和组合指标。

## 8. 主流股票池实验口径摘要

当前主流 A 股股票预测论文和 Qlib benchmark 通常不是挑几只股票单独训练后报告结果，而是：

1. 选定股票池，如 CSI300、CSI500、CSI800。
2. 每个交易日对股票池中所有股票输出预测分数或未来收益预测。
3. 每天按横截面计算 IC 和 RankIC。
4. 对日频 IC/RankIC 求均值和标准差，并报告 ICIR、RankICIR。
5. 构造组合，例如每天买入预测分数最高的 top 30 股票，再报告年化收益、信息比率、最大回撤。

代表性公开结果：

| 来源 | 股票池/数据 | 指标口径 | 代表结果 |
| --- | --- | --- | --- |
| MASTER, AAAI 2024 | CSI300、CSI800，2008-2022，Alpha158 | 日频 IC/RankIC、ICIR、RankICIR、top30 组合 | CSI300 上 MASTER `IC=0.064`、`RankIC=0.076` |
| Microsoft Qlib benchmark | 中国 A 股 + CSI300，Alpha158/Alpha360 | 每日全股票预测，20 个随机种子均值/标准差 | CSI300 Alpha158 中常见 IC 大约 `0.02-0.05`，RankIC 大约 `0.03-0.05`；Alpha360 中强方法 RankIC 可到约 `0.06` |

所以，对于无泄漏的 A 股横截面预测：

| 指标水平 | 大致评价 |
| --- | --- |
| `IC > 0` 且稳定 | 有正向信号 |
| `IC = 0.02-0.04` | 有研究价值 |
| `IC = 0.05-0.08` | 已经较强 |
| `RankIC = 0.06-0.10` | 接近顶会论文强结果区间 |
| `IC/RIC = 0.9` | 极大概率是泄漏或口径错误 |

## 9. 后续实验建议

1. 使用重构后的五个 CSV 重新训练，旧 `saved_models` 结果全部视为无效。
2. 若继续做 SAMBA 原论文复现，建议将训练标签从 `Price` 改为下一日收益率，减少“价格预测后再差分”的口径偏差。
3. 若论文目标是“沪深300股票池预测”，建议构造 `date × stock × feature` 面板数据，而不是五个单资产文件。
4. 横截面版本至少应报告 `IC_mean`、`IC_std`、`ICIR`、`RankIC_mean`、`RankIC_std`、`RankICIR`、topK 组合收益、IR、最大回撤。
5. 构造沪深300股票池时需注意成分股动态变化、停牌、退市、生存者偏差、复权价格、涨跌停和交易成本。

## 10. 参考资料

- SAMBA arXiv: https://arxiv.org/abs/2410.03707
- 本地 SAMBA 论文 PDF: `SAMBA/md/2025_ICASSP_Mamba遇见金融市场：面向股价预测的图-Mamba方法_Ali Mehrabian.pdf`
- MASTER, AAAI 2024: https://ojs.aaai.org/index.php/AAAI/article/download/27767/27575
- StockMixer, AAAI 2024: https://ojs.aaai.org/index.php/AAAI/article/view/28681/29322
- COGRASP, IJCAI 2025: https://www.ijcai.org/proceedings/2025/0837.pdf
- Qlib benchmark: https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md
