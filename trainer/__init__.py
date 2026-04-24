# -*- coding: utf-8 -*-
"""
SAMBA模型的训练模块

该模块提供完整的模型训练管理功能，包括：
- 训练循环控制
- 验证和测试
- 早停机制
- 模型保存和加载
- 训练日志记录
"""

from .trainer import Trainer

__all__ = ['Trainer']
