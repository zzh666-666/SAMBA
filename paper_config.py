# -*- coding: utf-8 -*-
"""
论文配置参数

该文件包含与原始SAMBA论文完全一致的配置参数，确保实验结果的可重现性。
基于论文：《Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction》
会议：IEEE ICASSP 2025
"""

from config import ModelArgs, TrainingConfig


def get_paper_config(dataset_name="NYSE"):
    """
    获取与原始SAMBA论文匹配的配置参数
    
    该函数返回论文中使用的精确模型和训练配置，
    包括所有超参数设置，以确保结果的可重现性。
    
    参数:
        dataset_name: 数据集名称 ("NYSE", "NASDAQ", "DJIA", "NYSE_82", "NYSE_BASIC",
            "000016SH_BASIC", "000100SZ_BASIC", "000300SH_BASIC", "002129SZ_BASIC", "300015SZ_BASIC")
    
    返回:
        model_args: 模型架构配置
        training_config: 训练参数配置
    """
    
    # 模型配置（与论文完全一致）
    sequence_length = 5
    forecast_horizon = 1

    model_args = ModelArgs(
        d_model=64,           # 模型维度 (论文中E=64)
        n_layer=3,            # Mamba层数 (论文中R=3)
        vocab_size=82,        # 82个日度股票特征
        seq_in=sequence_length,  # 输入序列长度
        seq_out=forecast_horizon,  # 预测时间步长
        d_state=64,           # 状态维度 (论文中H=64)
        expand=2,             # 扩展因子
        dt_rank='auto',       # 自动计算时间步参数秩
        d_conv=3,             # 卷积核大小
        pad_vocab_size_multiple=8,  # 词汇表填充倍数
        conv_bias=True,       # 卷积偏置
        bias=False            # 线性层偏置
    )
    
    # 训练配置（与论文完全一致）
    training_config = TrainingConfig(
        dataset=dataset_name,     # 数据集名称
        lag=sequence_length,      # 输入序列长度
        horizon=forecast_horizon, # 预测时间步长
        num_nodes=82,             # 82个日度股票特征
        val_ratio=0.05,           # 5%验证集 (论文设置)
        test_ratio=0.15,          # 15%测试集 (论文设置)
        input_dim=1,              # 输入维度
        output_dim=1,             # 输出维度
        embed_dim=10,             # 嵌入维度 (论文中de=10)
        rnn_units=128,            # RNN单元数
        num_layers=3,             # 网络层数
        cheb_k=3,                 # 切比雪夫多项式阶数 (论文中K=3)
        d_in=64,                  # 输入特征维度 (与d_model一致)
        hid=32,                   # 隐藏层维度 (论文中U=32)
        batch_size=32,          # 批次大小 (最优配置)
        epochs=2500,              # 训练轮数 (最优配置)
        lr_init=0.0004,           # 初始学习率 (最优配置-关键!)
        lr_decay=False,           # 学习率衰减 (禁用-关键!)
        lr_decay_rate=0.8,        # 衰减率
        lr_decay_step=[1000, 1500, 2000],  # 衰减步骤
        early_stop=True,          # 早停
        early_stop_patience=500,  # 早停耐心值 (最优配置)
        grad_norm=False,          # 梯度裁剪
        max_grad_norm=5,          # 最大梯度范数
        loss_func='mae',          # 损失函数
        mae_thresh=None,          # MAE阈值
        mape_thresh=0,            # MAPE阈值
        device='cuda:0',          # 计算设备
        seed=1,                   # 随机种子
        debug=False,               # 调试模式(模型是否保存)
        log_step=20,              # 日志记录步长
        log_dir=f'./saved_models/{dataset_name}/'  # 数据集特定的日志目录
    )
    
    return model_args, training_config


def get_dataset_mapping():
    """
    获取数据集文件映射
    
    返回:
        dict: 数据集名称到文件路径的映射
    """
    return {
        "NYSE": "Dataset/combined_dataframe_NYSE.csv",
        "NYSE_82": "Dataset/combined_dataframe_NYSE.csv",
        "NYSE_BASIC": "Dataset/combined_dataframe_NYSE_basic_indicators.csv",
        "000016SH_BASIC": "Dataset/combined_dataframe_000016SH_basic_indicators.csv",
        "000100SZ_BASIC": "Dataset/combined_dataframe_000100SZ_basic_indicators.csv",
        "000300SH_BASIC": "Dataset/combined_dataframe_000300SH_basic_indicators.csv",
        "002129SZ_BASIC": "Dataset/combined_dataframe_002129SZ_basic_indicators.csv",
        "300015SZ_BASIC": "Dataset/combined_dataframe_300015SZ_basic_indicators.csv",
        "NASDAQ": "Dataset/combined_dataframe_IXIC.csv", 
        "DJIA": "Dataset/combined_dataframe_DJI.csv"
    }


def get_dataset_info():
    """
    获取论文中使用的三个数据集的信息
    
    返回包含数据集详细信息的字典，包括：
    - 数据集名称和文件名
    - 时间范围和特征数量
    - 论文和作者信息
    
    返回:
        dict: 包含数据集和论文信息的字典
    """
    return {
        'datasets': [
            {
                'name': 'NYSE_82',
                'file': 'combined_dataframe_NYSE.csv',
                'description': 'New York Stock Exchange with full 82-dimensional feature set',
                'period': 'January 2010 to November 2023',
                'features': 82
            },
            {
                'name': 'NYSE_BASIC',
                'file': 'combined_dataframe_NYSE_basic_indicators.csv',
                'description': 'New York Stock Exchange with basic indicator feature set',
                'period': 'January 2010 to November 2023',
                'features': 15
            },
            {
                'name': '000016SH_BASIC',
                'file': 'combined_dataframe_000016SH_basic_indicators.csv',
                'description': 'SSE 50 Index aligned to the NYSE basic indicator schema',
                'period': 'January 2010 to October 2023',
                'features': 15
            },
            {
                'name': '000100SZ_BASIC',
                'file': 'combined_dataframe_000100SZ_basic_indicators.csv',
                'description': 'TCL Technology aligned to the NYSE basic indicator schema',
                'period': 'January 2010 to October 2023',
                'features': 15
            },
            {
                'name': '000300SH_BASIC',
                'file': 'combined_dataframe_000300SH_basic_indicators.csv',
                'description': 'CSI 300 Index aligned to the NYSE basic indicator schema',
                'period': 'January 2010 to October 2023',
                'features': 15
            },
            {
                'name': '002129SZ_BASIC',
                'file': 'combined_dataframe_002129SZ_basic_indicators.csv',
                'description': 'TCL Zhonghuan aligned to the NYSE basic indicator schema',
                'period': 'January 2010 to October 2023',
                'features': 15
            },
            {
                'name': '300015SZ_BASIC',
                'file': 'combined_dataframe_300015SZ_basic_indicators.csv',
                'description': 'Aier Eye Hospital aligned to the NYSE basic indicator schema',
                'period': 'January 2010 to October 2023',
                'features': 15
            },
            {
                'name': 'NASDAQ',
                'file': 'combined_dataframe_IXIC.csv',
                'description': 'NASDAQ Composite Index',
                'period': 'January 2010 to November 2023',
                'features': 82
            },
            {
                'name': 'NYSE',
                'file': 'combined_dataframe_NYSE.csv',
                'description': 'New York Stock Exchange',
                'period': 'January 2010 to November 2023',
                'features': 82
            },
            {
                'name': 'DJIA',
                'file': 'combined_dataframe_DJI.csv',
                'description': 'Dow Jones Industrial Average',
                'period': 'January 2010 to November 2023',
                'features': 82
            }
        ],
        'total_features': 82,
        'time_period': 'January 2010 to November 2023',
        'paper_title': 'Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction',
        'conference': 'IEEE ICASSP 2025',
        'authors': ['Ali Mehrabian', 'Ehsan Hoseinzade', 'Mahdi Mazloum', 'Xiaohong Chen']
    }


def print_paper_info():
    """
    打印论文和数据集信息
    
    该函数以格式化的方式显示论文的详细信息，
    包括标题、作者、会议、数据集等。
    """
    info = get_dataset_info()
    
    print("=" * 70)
    print("SAMBA: A Graph-Mamba Approach for Stock Price Prediction")
    print("=" * 70)
    print(f"Paper: {info['paper_title']}")
    print(f"Conference: {info['conference']}")
    print(f"Authors: {', '.join(info['authors'])}")
    print(f"Time Period: {info['time_period']}")
    print(f"Total Features: {info['total_features']}")
    print()
    
    print("Datasets:")
    for dataset in info['datasets']:
        print(f"  • {dataset['name']} ({dataset['file']})")
        print(f"    - {dataset['description']}")
        print(f"    - {dataset['features']} features")
        print(f"    - Period: {dataset['period']}")
        print()
    
    print("=" * 70)


if __name__ == "__main__":
    print_paper_info()
