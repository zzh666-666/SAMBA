# -*- coding: utf-8 -*-
"""
SAMBA模型实现

SAMBA (State-space Mamba with Graph Neural Networks) 是结合状态空间Mamba模型
和图神经网络的股价预测架构。该模型通过高斯核生成自适应邻接矩阵，
使用切比雪夫多项式进行图卷积，实现对股票特征间空间关系的建模。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import Mamba


class SAMBA(nn.Module):
    """
    SAMBA: 结合状态空间Mamba和图神经网络的股价预测模型
    
    该模型包含以下核心组件:
    1. Mamba骨干网络: 处理时间序列的长期依赖
    2. 高斯核图生成: 自适应学习股票特征间的关系
    3. 切比雪夫图卷积: 捕获图结构中的空间依赖
    4. 输出投影层: 生成最终的预测结果
    """
    
    def __init__(self, model_args, hidden, inp, out, embed, cheb_k):
        """
        初始化SAMBA模型
        
        参数:
            model_args: 模型配置参数
            hidden: 隐藏层维度
            inp: 输入序列长度
            out: 输出序列长度
            embed: 嵌入维度
            cheb_k: 切比雪夫多项式阶数
        """
        super().__init__()
        self.args = model_args
        
        # Mamba骨干网络，用于处理时间序列
        self.mam1 = Mamba(model_args, hidden)
        
        # 图卷积参数
        self.cheb_k = cheb_k
        self.gamma = nn.Parameter(torch.tensor(1.))  # 高斯核的缩放参数
        
        # 可学习的邻接矩阵嵌入和权重
        self.adj = nn.Parameter(torch.randn(model_args.vocab_size, embed), requires_grad=True)
        self.embed_w = nn.Parameter(torch.randn(embed, embed), requires_grad=True)
        
        # 切比雪夫多项式的权重和偏置
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed, cheb_k, inp, out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed, out))
        
        # 输出投影层
        self.proj = nn.Linear(model_args.vocab_size, 1)      # 特征维度投影
        self.proj_seq = nn.Linear(model_args.seq_in, 1)      # 序列长度投影
    
    def gaussian_kernel_graph(self, E_A, x, gamma=1.0):
        """
        使用高斯核生成图邻接矩阵
        
        该方法通过计算节点嵌入之间的欧几里得距离，
        并应用高斯核函数来生成自适应的邻接矩阵。
        
        参数:
            E_A: 节点嵌入矩阵 [N, embed_dim]
            x: 输入特征 [B, T, N]
            gamma: 高斯核的缩放参数
            
        返回:
            A: 归一化的邻接矩阵 [N, N]
        """
        # 计算输入特征的时间平均值
        x_mean = torch.mean(x, dim=0)
        x_time = torch.mm(x_mean.permute(1, 0), x_mean)
        
        N = E_A.size(0)
        # 扩展维度以计算成对差异
        E_A_expanded = E_A.unsqueeze(0).expand(N, N, -1)      # [N, N, embed_dim]
        E_A_T_expanded = E_A.unsqueeze(1).expand(N, N, -1)    # [N, N, embed_dim]
        
        # 计算成对的欧几里得距离平方
        distance_matrix = torch.sum((E_A_expanded - E_A_T_expanded)**2, dim=2)
        
        # 应用高斯核函数
        A = torch.exp(-gamma * distance_matrix)
        
        # 应用dropout进行正则化
        dr = nn.Dropout(0.35)
        
        # 使用softmax进行行归一化
        A = F.softmax(A, dim=1)
        
        return dr(A)
    
    def forward(self, input_ids):
        """
        SAMBA模型的前向传播
        
        参数:
            input_ids: 输入张量 [B, T, N]
                      B: 批次大小
                      T: 时间步长
                      N: 特征数量(图节点数)
                      
        返回:
            output: 预测结果 [B, 1, 1]
        """
        # 通过Mamba网络处理时间序列
        xx = self.mam1(input_ids)  # [B, T, N]
        
        # 使用高斯核生成邻接矩阵
        ADJ = self.gaussian_kernel_graph(self.adj, xx, gamma=self.gamma)  # [N, N]
        
        # 单位矩阵，用于构建切比雪夫多项式
        I = torch.eye(input_ids.size(2)).cuda()
        
        # 构建切比雪夫多项式支撑集
        # T_0(L) = I, T_1(L) = L, T_k(L) = 2L*T_{k-1}(L) - T_{k-2}(L)
        support_set = [I, ADJ]
        
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * ADJ, support_set[-1]) - support_set[-2])
        
        supports = torch.stack(support_set, dim=0)  # [cheb_k, N, N]
        
        # 应用切比雪夫图卷积
        # 计算每个节点的权重: [N, cheb_k, inp, out]
        weights = torch.einsum('nd,dkio->nkio', self.adj, self.weights_pool)
        # 计算每个节点的偏置: [N, out]
        bias = torch.matmul(self.adj, self.bias_pool)
        
        # 图卷积操作: [B, cheb_k, N, inp]
        x_g = torch.einsum("knm,bmc->bknc", supports, xx.permute(0, 2, 1))
        x_g = x_g.permute(0, 2, 1, 3)  # [B, N, cheb_k, inp]
        
        # 最终的图卷积输出: [B, N, out]
        out = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        
        # 输出投影，得到最终预测结果
        return self.proj(out.permute(0, 2, 1))  # [B, 1, 1]
