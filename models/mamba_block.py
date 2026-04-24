# -*- coding: utf-8 -*-
"""
Mamba块实现，包含选择性状态空间建模

Mamba块是Mamba模型的核心组件，实现了选择性状态空间机制。
该机制能够根据输入内容动态调整状态转移，实现高效的序列建模。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from .normalization import RMSNorm


class MambaBlock(nn.Module):
    """
    单个Mamba块实现
    
    如论文第3.4节图3所述，Mamba块包含以下核心组件：
    1. 输入投影和门控机制
    2. 1D卷积层
    3. 选择性状态空间建模(SSM)
    4. 输出投影
    """
    
    def __init__(self, args):
        """
        初始化Mamba块
        
        参数:
            args: 模型配置参数，包含维度、状态空间参数等
        """
        super().__init__()
        self.args = args
        
        # 输入投影层：将输入映射到内部维度的两倍（用于门控）
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.in_proj_r = nn.Linear(args.d_model, args.d_inner, bias=args.bias)
        
        # 1D卷积层：用于局部特征提取
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,      # 深度可分离卷积
            padding=args.d_conv - 1,  # 因果填充
        )
        
        # 状态空间参数投影层
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # 归一化和输出投影
        self.norm_f = RMSNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        
        # 时间步参数投影
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        
        # 状态空间参数初始化
        # A矩阵：状态转移矩阵，初始化为负数以保证稳定性
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        # D矩阵：跳跃连接参数
        self.D = nn.Parameter(torch.ones(args.d_inner))
        # 输出投影层
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
    
    def forward(self, x):
        """
        Mamba块的前向传播
        
        参数:
            x: 输入张量 [B, T, d_model]
            
        返回:
            output: 输出张量 [B, T, d_model]
        """
        (b, l, d) = x.shape
        
        # 输入投影，分为主分支和残差分支
        x_and_res = self.in_proj(x)  # [B, T, 2 * d_inner]
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        
        # 1D卷积处理（需要转换维度以适应Conv1d）
        x = rearrange(x, 'b l d_in -> b d_in l')  # [B, d_inner, T]
        x = self.conv1d(x)[:, :, :l]              # 因果卷积，保持序列长度
        x = rearrange(x, 'b d_in l -> b l d_in')  # [B, T, d_inner]
        
        # 激活函数
        x = F.silu(x)
        gate = x * (1 - F.sigmoid(res))  # 门控机制
        
        # 状态空间建模
        y = self.ssm(x)
        y = y * F.silu(res)  # 与残差分支结合
        
        # 输出投影
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x):
        """
        状态空间模型(SSM)的核心计算
        
        实现选择性状态空间机制，根据输入动态调整状态转移参数。
        
        参数:
            x: 输入张量 [B, T, d_inner]
            
        返回:
            y: SSM输出 [B, T, d_inner]
        """
        (d_in, n) = self.A_log.shape
        
        # 计算状态空间参数
        A = -torch.exp(self.A_log.float())  # 状态转移矩阵 [d_inner, d_state]
        D = self.D.float()                  # 跳跃连接参数 [d_inner]
        
        # 投影得到时间步和输入/输出矩阵
        x_dbl = self.x_proj(x)  # [B, T, dt_rank + 2*d_state]
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        
        # 时间步参数处理
        delta = F.softplus(self.dt_proj(delta))  # [B, T, d_inner]，确保为正数
        
        # 执行选择性扫描算法
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y
    
    def selective_scan(self, u, delta, A, B, C, D):
        """
        选择性扫描算法实现
        
        这是Mamba的核心算法，实现了高效的状态空间计算。
        算法通过选择性地更新状态，实现对长序列的高效建模。
        
        参数:
            u: 输入序列 [B, T, d_inner]
            delta: 时间步参数 [B, T, d_inner]
            A: 状态转移矩阵 [d_inner, d_state]
            B: 输入矩阵 [B, T, d_state]
            C: 输出矩阵 [B, T, d_state]
            D: 跳跃连接参数 [d_inner]
            
        返回:
            y: 输出序列 [B, T, d_inner]
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]  # 状态维度
        
        # 离散化连续参数 (A, B)
        # 使用零阶保持(ZOH)离散化
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # 执行选择性扫描
        x = torch.zeros((b, d_in, n), device=deltaA.device)  # 初始状态
        ys = []
        
        for i in range(l):
            # 状态更新：x_{t+1} = A_t * x_t + B_t * u_t
            x = deltaA[:, i] * x + deltaB_u[:, i]
            # 输出计算：y_t = C_t * x_t
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # [B, T, d_inner]
        
        # 添加跳跃连接
        y = y + u * D
        
        return y
