# -*- coding: utf-8 -*-
"""
SAMBA多数据集训练脚本

该脚本实现了完整的SAMBA模型多数据集训练流程，包括：
1. 支持多数据集配置和训练
2. 历史记录管理
3. 自动化训练流程
4. 结果汇总和对比

基于论文：《Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction》
会议：IEEE ICASSP 2025
"""

import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from paper_config import get_paper_config, get_dataset_info, get_dataset_mapping
from models import SAMBA
from utils import (
    prepare_data, init_seed, print_model_parameters, 
    pearson_correlation, rank_information_coefficient, All_Metrics
)
from trainer import Trainer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")


def resolve_project_path(path):
    """Resolve paths relative to the SAMBA project directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def build_loss_function(loss_name, device):
    """Build the configured training loss."""
    loss_name = (loss_name or 'mae').lower()

    if loss_name == 'mae':
        return torch.nn.L1Loss().to(device)
    if loss_name == 'mse':
        return torch.nn.MSELoss().to(device)
    if loss_name in {'huber', 'smooth_l1', 'smoothl1'}:
        return torch.nn.SmoothL1Loss().to(device)

    raise ValueError(f"Unsupported loss_func: {loss_name}")


def dump_json(data, path, required=True):
    """Write JSON to disk and optionally tolerate file-lock issues."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except PermissionError:
        if required:
            raise
        print(f"Warning: skipped updating locked file: {path}")


def create_directories(dataset_names=None):
    """创建必要的目录结构"""
    datasets = dataset_names or ["NYSE", "NASDAQ", "DJIA"]
    for dataset in datasets:
        os.makedirs(os.path.join(SAVED_MODELS_DIR, dataset), exist_ok=True)
    os.makedirs(os.path.join(SAVED_MODELS_DIR, "summary"), exist_ok=True)
    print("✅ 目录结构创建完成")


def train_single_dataset(dataset_name, dataset_file):
    """训练单个数据集"""
    dataset_file = resolve_project_path(dataset_file)
    print(f"\n🎯 开始训练数据集: {dataset_name}")
    print(f"📊 数据集文件: {dataset_file}")
    print("-" * 50)
    
    # 检查数据集文件是否存在
    if not os.path.exists(dataset_file):
        print(f"❌ 数据集文件不存在: {dataset_file}")
        return None
    
    # 获取配置
    model_args, config = get_paper_config(dataset_name)
    config.dataset = dataset_name
    config.log_dir = os.path.join(SAVED_MODELS_DIR, dataset_name) + os.sep
    print(f"💾 模型保存路径: {config.log_dir}")
    
    # 执行训练
    results = run_training(model_args, config, dataset_file, dataset_name)
    
    if results:
        print(f"✅ {dataset_name} 训练完成!")
        print(f"📈 结果: IC={results['IC']:.4f}, RIC={results['RIC']:.4f}, RMSE={results['RMSE']:.4f}")
    
    return results


def run_training(model_args, config, dataset_file, dataset_name):
    """执行训练过程"""
    try:
        # 初始化随机种子
        init_seed(config.seed)
        
        # 准备数据
        train_loader, val_loader, test_loader, mmn, num_features = prepare_data(
            csv_file=dataset_file,
            window=config.lag,
            predict=config.horizon,
            test_ratio=config.test_ratio,
            val_ratio=config.val_ratio,
            batch_size=config.batch_size
        )
        
        # 更新配置
        config.num_nodes = num_features
        model_args.vocab_size = num_features
        model_args.seq_in = config.lag
        model_args.seq_out = config.horizon
        
        # 初始化模型
        model = SAMBA(
            model_args,
            config.hid,
            config.lag,
            config.horizon,
            config.embed_dim,
            config.cheb_k
        ).cuda()
        
        # 初始化模型参数
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        
        print_model_parameters(model, only_num=False)
        
        # 设置损失函数和优化器
        loss = build_loss_function(config.loss_func, config.device)
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=config.lr_init,
            eps=1.0e-8,
            weight_decay=0,
            amsgrad=False
        )
        
        # 设置学习率调度器
        lr_scheduler = None
        if config.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=[0.5 * config.epochs, 0.7 * config.epochs, 0.9 * config.epochs],
                gamma=0.1
            )
        
        # 训练
        trainer = Trainer(
            model, loss, optimizer, train_loader, val_loader, test_loader,
            args=config.to_dict(), lr_scheduler=lr_scheduler
        )
        
        start_time = time.time()
        y_pred, y_true = trainer.train()
        training_time = time.time() - start_time
        
        # 测试评估
        y1, y2 = trainer.test(trainer.model, trainer.args, test_loader, trainer.logger)
        
        # 计算最终指标
        y_p = np.array(y1[:, 0, :].cpu())
        y_t = np.array(y2[:, 0, :].cpu())
        
        y_p = mmn.inverse_transform(y_p)
        y_t = mmn.inverse_transform(y_t)
        
        y_p = torch.tensor(y_p)
        y_t = torch.tensor(y_t)
        
        # 计算收益率
        diff = y_p[1:] - y_p[:-1]
        return_p = diff / y_p[:-1]
        
        diff = y_t[1:] - y_t[:-1]
        return_t = diff / y_t[:-1]
        
        # 计算评估指标
        mae, rmse, _ = All_Metrics(return_p, return_t, None, None)
        IC = pearson_correlation(return_t, return_p)
        RIC = rank_information_coefficient(return_t[:, 0], return_p[:, 0])
        
        # 保存结果 - 支持历史记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "dataset": dataset_name,
            "dataset_file": dataset_file,
            "num_features": int(num_features),
            "IC": float(IC),
            "RIC": float(RIC),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "training_time_minutes": training_time / 60,
            "timestamp": datetime.now().isoformat(),
            "run_id": timestamp,
            "config": {
                "batch_size": config.batch_size,
                "lr_init": config.lr_init,
                "epochs": config.epochs,
                "early_stop_patience": config.early_stop_patience
            }
        }
        
        # 保存最新结果 (会覆盖)
        results_file = os.path.join(config.log_dir, "results.json")
        dump_json(results, results_file, required=False)
        
        # 保存历史记录 (追加模式)
        history_file = os.path.join(config.log_dir, "results_history.json")
        history_results = []
        
        # 读取已有历史记录
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history_results = json.load(f)
            except:
                history_results = []
        
        # 添加新结果
        history_results.append(results)
        
        # 保存更新后的历史记录
        dump_json(history_results, history_file, required=False)
        
        # 保存带时间戳的详细结果
        detailed_results_file = os.path.join(config.log_dir, f"results_{timestamp}.json")
        dump_json(results, detailed_results_file)
        
        # 保存配置备份 (带时间戳)
        config_file = os.path.join(config.log_dir, f"config_{timestamp}.json")
        dump_json(config.to_dict(), config_file)
        
        # 保存最新配置 (会覆盖)
        latest_config_file = os.path.join(config.log_dir, "config.json")
        dump_json(config.to_dict(), latest_config_file, required=False)
        
        print(f"\n{dataset_name} final results:")
        print(f"IC: {IC:.4f}")
        print(f"RIC: {RIC:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Training time: {training_time/60:.2f} minutes")
        print(f"Run ID: {timestamp}")
        
        return results
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        return None


def save_summary_results(all_results):
    """保存汇总结果"""
    summary_dir = os.path.join(SAVED_MODELS_DIR, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = os.path.join(summary_dir, "all_results_comparison.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 创建对比表格
    comparison_md = "# SAMBA 多数据集训练结果对比\n\n"
    comparison_md += "## 📊 结果汇总表\n\n"
    comparison_md += "| 数据集 | 特征数 | IC | RIC | RMSE | MAE | 训练时间(分钟) | 运行ID |\n"
    comparison_md += "|--------|--------|----|----|------|-----|----------------|--------|\n"
    
    for dataset, results in all_results.items():
        comparison_md += f"| {dataset} | {results.get('num_features', '')} | {results['IC']:.4f} | {results['RIC']:.4f} | {results['RMSE']:.4f} | {results['MAE']:.4f} | {results['training_time_minutes']:.1f} | {results['run_id']} |\n"
    
    with open(os.path.join(summary_dir, "training_summary.md"), 'w', encoding='utf-8') as f:
        f.write(comparison_md)
    
    print(f"📄 汇总结果已保存到: {summary_dir}")


def print_comparison_table(all_results):
    """打印对比表格"""
    print("\n" + "="*80)
    print("📊 多数据集训练结果汇总")
    print("="*80)
    
    print(f"{'数据集':<14} {'特征数':<8} {'IC':<8} {'RIC':<8} {'RMSE':<8} {'MAE':<8} {'训练时间':<10} {'运行ID':<15}")
    print("-" * 100)
    
    for dataset, results in all_results.items():
        print(f"{dataset:<14} {results.get('num_features', ''):<8} {results['IC']:<8.4f} {results['RIC']:<8.4f} {results['RMSE']:<8.4f} {results['MAE']:<8.4f} {results['training_time_minutes']:<10.1f} {results['run_id']:<15}")


def main():
    """主函数 - 在这里配置要训练的数据集"""
    
    # ===== 配置要训练的数据集 =====
    # 对比 NYSE 完整 82 维特征、NYSE 基础指标特征，以及新对齐的 A 股/指数基础指标数据
    datasets_to_train = [
        # ("NYSE_82", "Dataset/combined_dataframe_NYSE.csv"),
        # ("NYSE_BASIC", "Dataset/combined_dataframe_NYSE_basic_indicators.csv"),
        ("000016SH_BASIC", "Dataset/combined_dataframe_000016SH_basic_indicators.csv"),
        ("000100SZ_BASIC", "Dataset/combined_dataframe_000100SZ_basic_indicators.csv"),
        ("000300SH_BASIC", "Dataset/combined_dataframe_000300SH_basic_indicators.csv"),
        ("002129SZ_BASIC", "Dataset/combined_dataframe_002129SZ_basic_indicators.csv"),
        ("300015SZ_BASIC", "Dataset/combined_dataframe_300015SZ_basic_indicators.csv")
    ]
    
    # 如果只想训练单个数据集，可以这样配置：
    # datasets_to_train = [
    #     ("NYSE", "Dataset/combined_dataframe_NYSE.csv")
    # ]
    
    # 或者训练两个数据集：
    # datasets_to_train = [
    #     ("NYSE", "Dataset/combined_dataframe_NYSE.csv"),
    #     ("NASDAQ", "Dataset/combined_dataframe_IXIC.csv")
    # ]
    # ==========================================
    
    print("🚀 SAMBA 多数据集训练系统")
    print("="*50)
    print(f"📋 计划训练 {len(datasets_to_train)} 个数据集:")
    for dataset_name, dataset_file in datasets_to_train:
        print(f"  - {dataset_name}: {dataset_file}")
    print("="*50)
    
    # 创建目录结构
    create_directories([dataset_name for dataset_name, _ in datasets_to_train])
    
    # 依次训练每个数据集
    all_results = {}
    
    for i, (dataset_name, dataset_file) in enumerate(datasets_to_train, 1):
        print(f"\n📍 进度: {i}/{len(datasets_to_train)} - 当前数据集: {dataset_name}")
        results = train_single_dataset(dataset_name, dataset_file)
        
        if results:
            all_results[dataset_name] = results
        else:
            print(f"❌ {dataset_name} 训练失败，跳过")
            continue
    
    # 保存汇总结果
    if all_results:
        save_summary_results(all_results)
        print_comparison_table(all_results)
    
    print("\n✅ 所有数据集训练完成!")


if __name__ == "__main__":
    main()
