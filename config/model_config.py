# -*- coding: utf-8 -*-
"""
模型配置类
定义SAMBA模型架构和训练参数的配置类
"""

import math
from dataclasses import dataclass
from typing import Union, List, Dict, Any


@dataclass
class ModelArgs:
    """
    SAMBA模型架构配置类
    
    参数说明:
        d_model: 模型维度，控制模型的表示能力
        n_layer: Mamba层数，决定模型深度
        vocab_size: 词汇表大小，对应股票特征数量
        seq_in: 输入序列长度，历史时间步数
        seq_out: 输出序列长度，预测时间步数
        d_state: 状态维度，Mamba状态空间的维度
        expand: 扩展因子，控制内部维度扩展
        dt_rank: 时间步参数的秩，'auto'表示自动计算
        d_conv: 卷积核大小，1D卷积的核大小
        pad_vocab_size_multiple: 词汇表大小填充倍数
        conv_bias: 是否使用卷积偏置
        bias: 是否使用线性层偏置
    """
    d_model: int                    # 模型维度
    n_layer: int                    # Mamba层数
    vocab_size: int                 # 词汇表大小(特征数量)
    seq_in: int                     # 输入序列长度
    seq_out: int                    # 输出序列长度
    d_state: int = 128              # 状态维度
    expand: int = 2                 # 扩展因子
    dt_rank: Union[int, str] = 'auto'  # 时间步参数秩
    d_conv: int = 3                 # 卷积核大小
    pad_vocab_size_multiple: int = 8   # 词汇表填充倍数
    conv_bias: bool = True          # 卷积偏置
    bias: bool = False              # 线性层偏置

    def __post_init__(self):
        """
        初始化后处理，计算派生参数
        """
        # 计算内部维度：扩展因子 × 模型维度
        self.d_inner = int(self.expand * self.d_model)

        # 自动计算时间步参数的秩
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


@dataclass
class TrainingConfig:
    """
    训练参数配置类
    
    包含数据集参数、模型参数、训练参数、损失函数和系统参数等
    """
    # 数据集参数
    dataset: str = 'STOCK_DATA'     # 数据集名称
    lag: int = 5                    # 输入序列长度(滞后期)
    horizon: int = 1                # 预测时间步长
    num_nodes: int = 82             # 图节点数量(82个日度股票特征)
    val_ratio: float = 0.15         # 验证集比例
    test_ratio: float = 0.15        # 测试集比例
    
    # 模型参数
    input_dim: int = 1              # 输入维度
    output_dim: int = 1             # 输出维度
    embed_dim: int = 10             # 嵌入维度
    rnn_units: int = 128            # RNN单元数
    num_layers: int = 3             # 网络层数
    cheb_k: int = 3                 # 切比雪夫多项式阶数
    d_in: int = 32                  # 输入特征维度
    hid: int = 32                   # 隐藏层维度
    
    # 训练参数
    batch_size: int = 32            # 批次大小
    epochs: int = 1100              # 训练轮数
    lr_init: float = 0.001          # 初始学习率
    lr_decay: bool = True           # 是否使用学习率衰减
    lr_decay_rate: float = 0.5      # 学习率衰减率
    lr_decay_step: List[int] = None # 学习率衰减步骤
    early_stop: bool = True         # 是否启用早停
    early_stop_patience: int = 200  # 早停耐心值
    grad_norm: bool = False         # 是否使用梯度裁剪
    max_grad_norm: float = 5        # 最大梯度范数
    
    # 损失函数和评估指标
    loss_func: str = 'mae'          # 损失函数类型
    mae_thresh: float = None        # MAE阈值
    mape_thresh: float = 0          # MAPE阈值
    
    # 系统参数
    device: str = 'cuda:0'          # 计算设备
    seed: int = 1                   # 随机种子
    debug: bool = True              # 调试模式
    log_step: int = 20              # 日志记录步长
    log_dir: str = './'             # 日志目录
    
    def __post_init__(self):
        """
        初始化后处理，设置默认的学习率衰减步骤
        """
        if self.lr_decay_step is None:
            self.lr_decay_step = [40, 70, 100]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典格式
        
        返回:
            Dict[str, Any]: 包含所有配置参数的字典
        """
        return {
            'dataset': self.dataset,
            'mode': 'train',
            'device': self.device,
            'debug': self.debug,
            'model': 'SAMBA',
            'cuda': True,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'lag': self.lag,
            'horizon': self.horizon,
            'num_nodes': self.num_nodes,
            'tod': False,
            'normalizer': 'std',
            'column_wise': False,
            'default_graph': True,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embed_dim': self.embed_dim,
            'rnn_units': self.rnn_units,
            'num_layers': self.num_layers,
            'cheb_k': self.cheb_k,
            'loss_func': self.loss_func,
            'seed': self.seed,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr_init': self.lr_init,
            'lr_decay': self.lr_decay,
            'lr_decay_rate': self.lr_decay_rate,
            'lr_decay_step': self.lr_decay_step,
            'early_stop': self.early_stop,
            'early_stop_patience': self.early_stop_patience,
            'grad_norm': self.grad_norm,
            'max_grad_norm': self.max_grad_norm,
            'real_value': False,
            'mae_thresh': self.mae_thresh,
            'mape_thresh': self.mape_thresh,
            'log_dir': self.log_dir,
            'log_step': self.log_step,
            'plot': False,
            'teacher_forcing': False,
            'd_in': self.d_in,
            'hid': self.hid
        }
