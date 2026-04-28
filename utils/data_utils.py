# -*- coding: utf-8 -*-
"""
数据加载和预处理工具

该模块提供股价数据的加载、预处理和数据集构建功能，
包括数据归一化、时间序列窗口生成、训练/验证/测试集划分等。
"""

import torch
import torch.utils.data
import pandas as pd
import numpy as np


class MinMaxNorm01:
    """
    最小-最大归一化器，将数据缩放到[0, 1]范围
    
    该归一化器用于将股价数据标准化，便于模型训练。
    支持拟合、变换和逆变换操作。
    """
    
    def __init__(self):
        """初始化归一化器"""
        pass
    
    def fit(self, x):
        """
        拟合归一化参数
        
        参数:
            x: 输入数据，用于计算最小值和最大值
        """
        self.min = x.min()  # 数据最小值
        self.max = x.max()  # 数据最大值
    
    def transform(self, x):
        """
        应用归一化变换
        
        参数:
            x: 待变换的数据
            
        返回:
            归一化后的数据，范围在[0, 1]
        """
        diff = self.max - self.min
        # 处理最大值等于最小值的情况（常数特征）
        if diff == 0:
            return np.zeros_like(x)
        x = 1.0 * (x - self.min) / diff
        return x
    
    def fit_transform(self, x):
        """
        拟合并变换数据
        
        参数:
            x: 输入数据
            
        返回:
            归一化后的数据
        """
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self, x):
        """
        逆变换，将归一化数据恢复到原始尺度
        
        参数:
            x: 归一化后的数据
            
        返回:
            恢复到原始尺度的数据
        """
        x = x * (self.max - self.min) + self.min
        return x


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    """
    创建PyTorch数据加载器
    
    参数:
        X: 输入特征张量
        Y: 目标标签张量
        batch_size: 批次大小
        shuffle: 是否打乱数据
        drop_last: 是否丢弃最后一个不完整的批次
        
    返回:
        dataloader: PyTorch数据加载器
    """
    # 检查CUDA可用性
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # 转换为PyTorch张量
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last
    )
    return dataloader


def prepare_data(csv_file, window=5, predict=1, test_ratio=0.15, val_ratio=0.05, batch_size=32):
    """
    从CSV文件准备训练数据
    
    该函数执行完整的数据预处理流程：
    1. 加载CSV数据
    2. 基础预处理（删除名称列，创建目标变量）
    3. 数据集划分
    4. 归一化处理
    5. 时间序列窗口生成
    6. 创建数据加载器
    
    参数:
        csv_file: CSV文件路径
        window: 输入序列长度（历史时间步数）
        predict: 输出序列长度（预测时间步数）
        test_ratio: 测试集比例
        val_ratio: 验证集比例
        batch_size: 批次大小
    
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        mmn: 归一化器（用于逆变换）
        num_features: 特征数量
    """
    # 加载数据
    X = pd.read_csv(csv_file, index_col="Date", parse_dates=True)
    
    # 基础预处理
    if "Name" in X.columns:
        del X["Name"]        # 删除名称列

    # 不将 Target 类标签加入输入特征，避免信息泄露
    X.dropna(inplace=True)  # 删除缺失值
    
    # 转换为numpy数组
    a = X.to_numpy()
    
    # 计算划分大小（基于序列数量）
    ran = a.shape[0]           # 总数据点数
    n_seq = ran - window       # 总序列数
    test_len = int(test_ratio * n_seq)    # 测试序列数
    val_len = int(val_ratio * n_seq)      # 验证序列数
    train_len = n_seq - test_len - val_len # 训练序列数
    
    # 确定原始数据的划分点（按时间正向切分）
    # 训练集：最早一段时间
    # 验证集：中间一段时间
    # 测试集：最后一段时间
    train_start_idx = 0
    train_end_idx = train_len + window
    val_start_idx = train_len
    val_end_idx = train_len + val_len + window
    test_start_idx = train_len + val_len
    
    # 划分原始数据
    a_train = a[train_start_idx:train_end_idx]
    a_val = a[val_start_idx:val_end_idx]
    a_test = a[test_start_idx:]
    
    # 仅在训练数据上拟合归一化器
    mmn = MinMaxNorm01()
    mmn.fit(a_train)
    
    # 使用相同的归一化器分别变换各个数据集
    a_test_norm = mmn.transform(a_test)
    a_train_norm = mmn.transform(a_train)
    a_val_norm = mmn.transform(a_val)
    
    # 从每个归一化数据集创建序列
    def create_sequences(data, start_offset=0):
        """
        从归一化数据创建时间序列
        
        参数:
            data: 归一化后的数据
            start_offset: 起始偏移量
            
        返回:
            XX: 输入序列张量
            YY: 目标序列张量
        """
        X_seq = []
        Y_seq = []
        data_len = data.shape[0]
        i = start_offset
        
        while i + window < data_len:
            # 输入序列：除价格外的所有特征
            X_seq.append(torch.Tensor(data[i:i+window, 1:]))
            # 目标序列：价格
            Y_seq.append(torch.Tensor(data[i+window:i+window+predict, 0]))
            i += 1
            
        if len(X_seq) > 0:
            XX = torch.stack(X_seq, dim=0)
            YY = torch.stack(Y_seq, dim=0)
            YY = YY[:, :, None]  # 添加特征维度
            return XX, YY
        return None, None
    
    # 为每个数据集创建序列
    X_test, Y_test = create_sequences(a_test_norm, start_offset=0)
    X_train, Y_train = create_sequences(a_train_norm, start_offset=0)
    X_val, Y_val = create_sequences(a_val_norm, start_offset=0)
    
    # 转换为CUDA张量
    X_test = torch.Tensor.float(X_test).cuda() if X_test is not None else None
    Y_test = torch.Tensor.float(Y_test).cuda() if Y_test is not None else None
    
    X_train = torch.Tensor.float(X_train).cuda() if X_train is not None else None
    Y_train = torch.Tensor.float(Y_train).cuda() if Y_train is not None else None
    
    X_val = torch.Tensor.float(X_val).cuda() if X_val is not None else None
    Y_val = torch.Tensor.float(Y_val).cuda() if Y_val is not None else None
    
    # 获取特征数量（从训练集获取，应该总是存在）
    num_features = X_train.shape[2] if X_train is not None else a.shape[1] - 1
    
    # 创建数据加载器
    train_loader = data_loader(X_train, Y_train, batch_size, shuffle=False, drop_last=False)
    val_loader = data_loader(X_val, Y_val, batch_size, shuffle=False, drop_last=False)
    test_loader = data_loader(X_test, Y_test, batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader, mmn, num_features
