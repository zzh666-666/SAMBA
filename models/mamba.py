# -*- coding: utf-8 -*-
"""
Mamba模型实现

Mamba是一种基于状态空间模型的序列建模架构，能够高效处理长序列数据。
该实现包含完整的Mamba模型和残差块，支持双向处理和预训练模型加载。
"""

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from .normalization import RMSNorm
from .mamba_block import MambaBlock


class Mamba(nn.Module):
    """
    完整的Mamba模型，用于序列建模
    
    Mamba模型通过选择性状态空间机制处理序列数据，
    相比传统的Transformer具有更好的长序列建模能力和计算效率。
    """
    
    def __init__(self, args, hid):
        """
        初始化Mamba模型
        
        参数:
            args: 模型配置参数
            hid: 隐藏层维度
        """
        super().__init__()
        self.args = args
        self.nl = args.n_layer  # 层数
        
        # 嵌入层：将输入特征映射到模型维度
        self.embedding = nn.Linear(args.vocab_size, args.d_model)
        
        # Mamba层：双向处理的Mamba块
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.layers2 = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        
        # 残差连接的线性层
        self.lin = nn.ModuleList([
            # 第一层使用LayerNorm
            nn.Sequential(
                nn.LayerNorm(args.seq_in),
                nn.Linear(args.seq_in, hid),
                nn.ReLU(),
                nn.Linear(hid, args.seq_in)
            )
        ] + [
            # 中间层使用RMSNorm
            nn.Sequential(
                RMSNorm(args.seq_in),
                nn.Linear(args.seq_in, hid),
                nn.ReLU(),
                nn.Linear(hid, args.seq_in)
            ) for _ in range(args.n_layer - 2)
        ] + [
            # 最后一层使用RMSNorm
            nn.Sequential(
                RMSNorm(args.seq_in),
                nn.Linear(args.seq_in, hid),
                nn.ReLU(),
                nn.Linear(hid, args.seq_in)
            )
        ])
        
        # 最终归一化和输出投影
        self.norm_f = nn.LayerNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size)
        
        # 额外的投影层
        self.proj = nn.Sequential(
            nn.Linear(args.seq_in, hid),
            nn.ReLU(),
            nn.Linear(hid, args.seq_in)
        )
        self.nnl = nn.LayerNorm(args.vocab_size)
    
    def forward(self, input_ids):
        """
        Mamba模型的前向传播
        
        参数:
            input_ids: 输入张量 [B, T, N]
                      B: 批次大小
                      T: 序列长度  
                      N: 特征维度
                      
        返回:
            logits: 输出张量 [B, T, vocab_size]
        """
        # 嵌入层处理
        x = self.embedding(input_ids)  # [B, T, d_model]
        
        # 初始化双向处理的输入
        x1 = x  # 正向处理
        x2 = x  # 反向处理
        
        # 逐层处理
        for i in range(self.nl):
            # 正向Mamba处理
            x1 = self.layers[i](x1)
            # 反向Mamba处理（翻转序列）
            x2 = self.layers2[i](x2.flip([1]))
            
            # 双向信息融合
            x = x1 + x2.flip([1]) + x
            
            # 残差连接和线性变换
            x = self.lin[i](x.permute(0, 2, 1)).permute(0, 2, 1) + x
            
            # 更新双向输入
            x1 = x
            x2 = x
        
        # 最终归一化和输出投影
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """
        从HuggingFace加载预训练权重到模型中
        
        参数:
            pretrained_model_name: 预训练模型名称
            
        返回:
            model: 加载了预训练权重的Mamba模型
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            """加载HuggingFace配置文件"""
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            """加载HuggingFace权重文件"""
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        # 加载配置和权重
        config_data = load_config_hf(pretrained_model_name)
        from config import ModelArgs
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        
        # 加载并调整权重键名
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
        return model


class ResidualBlock(nn.Module):
    """
    简单的残差块，包装Mamba块并添加归一化和残差连接
    """
    
    def __init__(self, args):
        """
        初始化残差块
        
        参数:
            args: 模型配置参数
        """
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)           # Mamba块
        self.norm = nn.LayerNorm(args.d_model)  # 层归一化
    
    def forward(self, x):
        """
        残差块的前向传播
        
        参数:
            x: 输入张量 [B, T, d_model]
            
        返回:
            output: 输出张量 [B, T, d_model]
        """
        # 先归一化，再通过Mamba块处理
        output = self.mixer(self.norm(x))
        return output
