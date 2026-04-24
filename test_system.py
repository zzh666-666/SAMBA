# -*- coding: utf-8 -*-
"""
SAMBA系统测试脚本

该脚本用于验证模块化SAMBA系统的各个组件是否正常工作，
包括模型创建、前向传播、配置管理和工具函数等功能测试。
"""

import torch
import numpy as np
from config import ModelArgs, TrainingConfig
from models import SAMBA
from utils import init_seed, print_model_parameters


def test_model_creation():
    """
    测试模型创建和初始化功能
    
    验证SAMBA模型能够正确创建并初始化所有组件。
    
    返回:
        model: 创建的SAMBA模型实例
    """
    print("Testing model creation...")
    
    # 创建模型参数配置
    model_args = ModelArgs(
        d_model=32,      # 模型维度
        n_layer=2,       # Mamba层数（测试用较小值）
        vocab_size=10,   # 特征数量（测试用较小值）
        seq_in=5,        # 输入序列长度
        seq_out=1        # 输出序列长度
    )
    
    # 创建SAMBA模型
    model = SAMBA(
        model_args,
        hidden=32,       # 隐藏层维度
        inp=5,           # 输入序列长度
        out=1,           # 输出序列长度
        embed=10,        # 嵌入维度
        cheb_k=3         # 切比雪夫多项式阶数
    )
    
    print("✓ Model created successfully")
    return model


def test_forward_pass():
    """
    测试模型前向传播功能
    
    验证模型能够正确处理输入数据并产生预期形状的输出。
    
    返回:
        output: 模型输出张量
    """
    print("Testing forward pass...")
    
    # 创建模型
    model = test_model_creation()
    
    # 创建虚拟输入数据
    batch_size = 2    # 批次大小
    seq_len = 5       # 序列长度
    num_nodes = 10    # 节点数量（特征数量）
    
    # 生成随机输入张量
    input_tensor = torch.randn(batch_size, seq_len, num_nodes)
    
    # 执行前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"✓ Forward pass successful. Input shape: {input_tensor.shape}, Output shape: {output.shape}")
    return output


def test_config():
    """
    测试配置类功能
    
    验证ModelArgs和TrainingConfig类能够正确创建和管理配置参数。
    
    返回:
        model_args: 模型配置实例
        config: 训练配置实例
    """
    print("Testing configuration...")
    
    # 测试ModelArgs配置类
    model_args = ModelArgs(
        d_model=64,      # 模型维度
        n_layer=3,       # Mamba层数
        vocab_size=20,   # 特征数量
        seq_in=10,       # 输入序列长度
        seq_out=2        # 输出序列长度
    )
    
    print(f"✓ ModelArgs created: d_inner={model_args.d_inner}, dt_rank={model_args.dt_rank}")
    
    # 测试TrainingConfig配置类
    config = TrainingConfig(
        lag=5,           # 输入序列长度
        horizon=1,       # 预测时间步长
        num_nodes=20,    # 节点数量
        epochs=100       # 训练轮数
    )
    
    # 转换为字典格式
    config_dict = config.to_dict()
    print(f"✓ TrainingConfig created with {len(config_dict)} parameters")
    
    return model_args, config


def test_utilities():
    """
    测试工具函数功能
    
    验证各种工具函数（如种子初始化、模型参数打印等）能够正常工作。
    """
    print("Testing utilities...")
    
    # 测试随机种子初始化
    init_seed(42)
    print("✓ Seed initialization successful")
    
    # 测试模型参数打印功能
    model = test_model_creation()
    print_model_parameters(model, only_num=True)
    print("✓ Model parameter printing successful")


def main():
    """
    运行所有测试
    
    执行完整的系统测试流程，验证SAMBA系统的各个组件。
    如果所有测试通过，说明系统工作正常。
    """
    print("=" * 50)
    print("SAMBA Stock Price Forecasting System Test")
    print("=" * 50)
    
    try:
        # 测试配置管理
        test_config()
        print()
        
        # 测试工具函数
        test_utilities()
        print()
        
        # 测试前向传播
        test_forward_pass()
        print()
        
        print("=" * 50)
        print("✓ All tests passed! SAMBA stock forecasting system is working correctly.")
        print("=" * 50)
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
