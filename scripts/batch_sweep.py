# -*- coding: utf-8 -*-
"""
SAMBA 批量实验脚本：自动搜索 batch size、learning rate、seed 组合。

主要功能：
1. 网格搜索 batch_size x lr x seed
2. 数据集选择风格与 main.py 一致（默认直接改脚本内列表）
3. 自动保存单次结果、聚合统计、每个数据集最佳配置
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

# 允许从项目根目录直接运行：python scripts/batch_sweep.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_config import get_paper_config, get_dataset_mapping


# ===== 数据集配置（与 main.py 同风格）=====
# 直接修改这个列表，控制默认跑哪些数据集。
# DEFAULT_DATASETS_TO_SWEEP = ["NYSE", "NASDAQ", "DJIA"]
DEFAULT_DATASETS_TO_SWEEP = ["NYSE"]
# ==========================================


def parse_csv_list(raw, caster):
    """将逗号分隔字符串转换为列表，并做类型转换。"""
    return [caster(x.strip()) for x in raw.split(",") if x.strip()]


def normalize_dataset_name(name):
    """统一数据集名称格式。"""
    return name.strip().upper()


def resolve_datasets(dataset_mapping, datasets_arg):
    """解析并校验数据集列表。"""
    selected_names = [normalize_dataset_name(x) for x in parse_csv_list(datasets_arg, str)]
    if not selected_names:
        raise ValueError("未选择数据集。请设置 DEFAULT_DATASETS_TO_SWEEP 或传入 --datasets。")

    unknown = [name for name in selected_names if name not in dataset_mapping]
    if unknown:
        raise ValueError(f"存在未知数据集: {unknown}。可用数据集: {list(dataset_mapping.keys())}")

    return [(name, dataset_mapping[name]) for name in selected_names]


def float_tag(value):
    """将浮点数转为适合目录名的字符串。"""
    text = f"{value:.8g}"
    return text.replace(".", "p").replace("-", "m")


def safe_mean_std(values):
    """计算均值和标准差（空列表安全）。"""
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def write_csv(path, rows, fieldnames):
    """写出 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_results(run_rows):
    """
    按 (dataset, batch_size, lr_init) 聚合统计。
    输出各指标 mean/std。
    """
    grouped = defaultdict(list)
    for row in run_rows:
        if row["status"] != "ok":
            continue
        key = (row["dataset"], row["batch_size"], row["lr_init"])
        grouped[key].append(row)

    summary_rows = []
    for (dataset, batch_size, lr_init), rows in grouped.items():
        ic_values = [r["IC"] for r in rows]
        ric_values = [r["RIC"] for r in rows]
        mae_values = [r["MAE"] for r in rows]
        rmse_values = [r["RMSE"] for r in rows]
        time_values = [r["training_time_minutes"] for r in rows]

        ic_mean, ic_std = safe_mean_std(ic_values)
        ric_mean, ric_std = safe_mean_std(ric_values)
        mae_mean, mae_std = safe_mean_std(mae_values)
        rmse_mean, rmse_std = safe_mean_std(rmse_values)
        time_mean, time_std = safe_mean_std(time_values)

        summary_rows.append(
            {
                "dataset": dataset,
                "batch_size": batch_size,
                "lr_init": lr_init,
                "runs": len(rows),
                "IC_mean": ic_mean,
                "IC_std": ic_std,
                "RIC_mean": ric_mean,
                "RIC_std": ric_std,
                "MAE_mean": mae_mean,
                "MAE_std": mae_std,
                "RMSE_mean": rmse_mean,
                "RMSE_std": rmse_std,
                "time_mean_min": time_mean,
                "time_std_min": time_std,
            }
        )

    summary_rows.sort(key=lambda x: (x["dataset"], -x["IC_mean"], -x["RIC_mean"]))
    return summary_rows


def select_best_by_dataset(summary_rows):
    """
    每个数据集选一组最佳配置。
    规则：优先 IC_mean，再看 RIC_mean，再看 IC_std（越小越好）。
    """
    best = {}
    for row in summary_rows:
        ds = row["dataset"]
        if ds not in best:
            best[ds] = row
            continue

        old = best[ds]
        old_key = (old["IC_mean"], old["RIC_mean"], -old["IC_std"])
        new_key = (row["IC_mean"], row["RIC_mean"], -row["IC_std"])
        if new_key > old_key:
            best[ds] = row

    best_rows = list(best.values())
    best_rows.sort(key=lambda x: x["dataset"])
    return best_rows


def main():
    parser = argparse.ArgumentParser(description="SAMBA 批量实验：batch/lr/seed 网格搜索")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS_TO_SWEEP),
        help="逗号分隔的数据集名称，如 NYSE,NASDAQ（默认来自 DEFAULT_DATASETS_TO_SWEEP）",
    )
    parser.add_argument("--batch-sizes", type=str, default="32,64,128", help="逗号分隔的 batch size")
    parser.add_argument("--learning-rates", type=str, default="3e-4,4e-4,6e-4", help="逗号分隔的学习率")
    parser.add_argument("--seeds", type=str, default="1,11,111", help="逗号分隔的随机种子")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖默认训练轮数")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="覆盖默认 early stop patience")
    parser.add_argument("--output-root", type=str, default="saved_models/sweeps", help="实验输出根目录")
    parser.add_argument("--sweep-name", type=str, default="", help="本次实验名称（用于输出子目录）")
    args = parser.parse_args()

    # 延迟导入：保证只看 --help 时不强依赖训练环境
    from main import run_training

    dataset_mapping = get_dataset_mapping()
    selected_datasets = resolve_datasets(dataset_mapping=dataset_mapping, datasets_arg=args.datasets)

    batch_sizes = parse_csv_list(args.batch_sizes, int)
    learning_rates = parse_csv_list(args.learning_rates, float)
    seeds = parse_csv_list(args.seeds, int)

    run_id = args.sweep_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = Path(args.output_root) / run_id
    sweep_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SAMBA 批量实验开始")
    print(f"运行ID: {run_id}")
    print(f"数据集: {[name for name, _ in selected_datasets]}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"学习率: {learning_rates}")
    print(f"随机种子: {seeds}")
    print("=" * 72)

    run_rows = []
    total_runs = len(selected_datasets) * len(batch_sizes) * len(learning_rates) * len(seeds)
    run_index = 0

    for dataset_name, dataset_file in selected_datasets:
        if not Path(dataset_file).exists():
            print(f"[跳过] {dataset_name}: 数据集文件不存在 -> {dataset_file}")
            continue

        for batch_size in batch_sizes:
            for lr_init in learning_rates:
                for seed in seeds:
                    run_index += 1
                    exp_name = f"bs{batch_size}_lr{float_tag(lr_init)}_seed{seed}"
                    exp_dir = sweep_root / dataset_name / exp_name
                    exp_dir.mkdir(parents=True, exist_ok=True)

                    print(
                        f"[{run_index}/{total_runs}] dataset={dataset_name} "
                        f"bs={batch_size} lr={lr_init} seed={seed}"
                    )

                    model_args, config = get_paper_config(dataset_name)
                    config.batch_size = batch_size
                    config.lr_init = lr_init
                    config.seed = seed
                    config.log_dir = str(exp_dir).replace("\\", "/")

                    if args.epochs is not None:
                        config.epochs = args.epochs
                    if args.early_stop_patience is not None:
                        config.early_stop_patience = args.early_stop_patience

                    result = run_training(
                        model_args=model_args,
                        config=config,
                        dataset_file=dataset_file,
                        dataset_name=dataset_name,
                    )

                    row = {
                        "dataset": dataset_name,
                        "dataset_file": dataset_file,
                        "batch_size": batch_size,
                        "lr_init": lr_init,
                        "seed": seed,
                        "epochs": config.epochs,
                        "early_stop_patience": config.early_stop_patience,
                        "log_dir": config.log_dir,
                        "status": "ok" if result else "failed",
                        "IC": result["IC"] if result else None,
                        "RIC": result["RIC"] if result else None,
                        "MAE": result["MAE"] if result else None,
                        "RMSE": result["RMSE"] if result else None,
                        "training_time_minutes": result["training_time_minutes"] if result else None,
                        "run_id": result["run_id"] if result else None,
                        "timestamp": result["timestamp"] if result else None,
                    }
                    run_rows.append(row)

    # 保存原始结果
    raw_json = sweep_root / "raw_results.json"
    raw_json.write_text(json.dumps(run_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    runs_csv_fields = [
        "dataset",
        "dataset_file",
        "batch_size",
        "lr_init",
        "seed",
        "epochs",
        "early_stop_patience",
        "status",
        "IC",
        "RIC",
        "MAE",
        "RMSE",
        "training_time_minutes",
        "run_id",
        "timestamp",
        "log_dir",
    ]
    write_csv(sweep_root / "runs.csv", run_rows, runs_csv_fields)

    # 聚合统计并挑选每个数据集最佳配置
    summary_rows = aggregate_results(run_rows)
    summary_fields = [
        "dataset",
        "batch_size",
        "lr_init",
        "runs",
        "IC_mean",
        "IC_std",
        "RIC_mean",
        "RIC_std",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "RMSE_std",
        "time_mean_min",
        "time_std_min",
    ]
    write_csv(sweep_root / "summary_by_setting.csv", summary_rows, summary_fields)

    best_rows = select_best_by_dataset(summary_rows)
    best_fields = [
        "dataset",
        "batch_size",
        "lr_init",
        "runs",
        "IC_mean",
        "IC_std",
        "RIC_mean",
        "RIC_std",
        "MAE_mean",
        "RMSE_mean",
        "time_mean_min",
    ]
    write_csv(sweep_root / "best_config_by_dataset.csv", best_rows, best_fields)

    print("\n批量实验完成。")
    print(f"结果已保存到: {sweep_root}")
    if best_rows:
        print("\n各数据集最佳配置:")
        for row in best_rows:
            print(
                f"- {row['dataset']}: bs={row['batch_size']}, lr={row['lr_init']}, "
                f"IC={row['IC_mean']:.4f}+/-{row['IC_std']:.4f}, "
                f"RIC={row['RIC_mean']:.4f}+/-{row['RIC_std']:.4f}"
            )
    else:
        print("没有成功运行的组合，请检查输出目录下日志。")


if __name__ == "__main__":
    main()
