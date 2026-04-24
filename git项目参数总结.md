# SAMBA论文实验参数总结（基于当前代码）

## 📋 论文信息
- **标题**: Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction
- **会议**: IEEE ICASSP 2025
- **作者**: Ali Mehrabian, Ehsan Hoseinzade, Mahdi Mazloum, Xiaohong Chen

## 📊 数据集配置

### 数据集信息
- **数据集数量**: 3个美国股市数据集
- **时间范围**: 2010年1月 - 2023年11月
- **特征数量**: 82个日度股票特征

### 具体数据集
1. **NASDAQ** (`combined_dataframe_IXIC.csv`)
   - NASDAQ综合指数
   - 82个特征
   
2. **NYSE** (`combined_dataframe_NYSE.csv`)
   - 纽约证券交易所
   - 82个特征
   
3. **DJIA** (`combined_dataframe_DJI.csv`)
   - 道琼斯工业平均指数
   - 82个特征

### 数据划分
- **训练集**: 70% (1 - val_ratio - test_ratio)
- **验证集**: 15% (val_ratio=0.15)
- **测试集**: 15% (test_ratio=0.15)

## 🧠 模型架构参数

### SAMBA模型配置
- **模型维度** (d_model): 32
- **Mamba层数** (n_layer): 3
- **词汇表大小** (vocab_size): 82 (对应特征数量)
- **输入序列长度** (seq_in): 5
- **输出序列长度** (seq_out): 1
- **状态维度** (d_state): 128
- **扩展因子** (expand): 2
- **时间步参数秩** (dt_rank): 'auto' (自动计算)
- **卷积核大小** (d_conv): 3
- **词汇表填充倍数** (pad_vocab_size_multiple): 8
- **卷积偏置** (conv_bias): True
- **线性层偏置** (bias): False

### 图神经网络配置
- **切比雪夫多项式阶数** (cheb_k): 3
- **嵌入维度** (embed_dim): 10
- **隐藏层维度** (hid): 32
- **输入特征维度** (d_in): 32

### 其他网络参数
- **RNN单元数** (rnn_units): 128
- **网络层数** (num_layers): 3
- **输入维度** (input_dim): 1
- **输出维度** (output_dim): 1

## 🏃 训练配置参数

### 基础训练参数
- **训练轮数** (epochs): 1100
- **批次大小** (batch_size): 32
- **初始学习率** (lr_init): 0.001
- **损失函数** (loss_func): 'mae' (平均绝对误差)

### 学习率调度
- **学习率衰减** (lr_decay): True
- **衰减率** (lr_decay_rate): 0.5
- **衰减步骤** (lr_decay_step): [40, 70, 100] (在第40、70、100轮时衰减)

### 早停机制
- **早停** (early_stop): True
- **早停耐心值** (early_stop_patience): 200 (200轮无改善则停止)

### 梯度处理
- **梯度裁剪** (grad_norm): False
- **最大梯度范数** (max_grad_norm): 5

### 评估阈值
- **MAE阈值** (mae_thresh): None
- **MAPE阈值** (mape_thresh): 0

## ⚙️ 系统配置

### 硬件设置
- **计算设备** (device): 'cuda:0' (使用第一块GPU)
- **随机种子** (seed): 1 (确保可重现性)

### 日志配置
- **调试模式** (debug): True
- **日志记录步长** (log_step): 20 (每20个batch记录一次)
- **日志目录** (log_dir): './' (当前目录)

## 📈 评估指标

基于代码分析，论文使用以下评估指标：

1. **MAE** (Mean Absolute Error): 平均绝对误差
2. **RMSE** (Root Mean Squared Error): 均方根误差
3. **IC** (Information Coefficient): 信息系数 - 皮尔逊相关系数
4. **RIC** (Rank Information Coefficient): 排序信息系数 - 斯皮尔曼相关系数

## 🔍 关键实验设置

### 序列建模
- **历史窗口**: 5个时间步 (seq_in=5)
- **预测步长**: 1个时间步 (seq_out=1)
- **预测任务**: 单步预测

### 图结构学习
- **邻接矩阵**: 高斯核自适应生成
- **图卷积**: 切比雪夫多项式 (K=3)
- **节点数量**: 82 (对应特征数量)

### 状态空间建模
- **Mamba架构**: 选择性状态空间模型
- **双向处理**: 正向和反向序列处理
- **状态维度**: 128

## ⚠️ 待验证参数

请上传论文PDF以验证以下可能的参数：
- 具体的优化器设置 (当前使用Adam)
- 权重初始化方法
- Dropout率设置
- 批量归一化配置
- 具体的实验重复次数
- 统计显著性测试方法
- 基线模型对比设置

## 📝 注意事项

1. **可重现性**: 所有参数都设置了固定的随机种子 (seed=1)
2. **论文一致性**: 配置声明与原始SAMBA论文完全一致
3. **GPU要求**: 需要CUDA兼容的GPU进行训练
4. **训练时间**: 1100轮训练预计需要2-4小时 (取决于GPU性能)

---

**请上传论文PDF文件，我将对比验证这些参数是否与论文实验部分完全一致，并补充任何遗漏的配置信息。**