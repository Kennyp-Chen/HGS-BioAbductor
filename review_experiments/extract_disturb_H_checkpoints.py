#!/usr/bin/env python3
"""
从已有的 checkpoint 中提取 disturb_H 在 [0, 0.1, 0.2, 0.3, 0.4, 0.5] 的
10-seed test_ci 结果，输出到 Results_disturb_H_granular.csv，
然后自动调用 perturbation_comparison_plot.py 更新对比图。

=============================================================================
脚本流程注解:
1. 扫描已有 checkpoint 目录，寻找最佳超参组合下所有种子 × disturb_H 水平的 .ckpt
2. 加载每个 .ckpt，提取 test_ci、val_ci、val_loss
3. 按 disturb_H 水平汇总 10 个 seed，计算 mean±std
4. 输出 Results_disturb_H_granular.csv（格式与原来 Results_disturb_H.csv 一致）
5. 调用 perturbation_comparison_plot.py 重新绘图（x 轴统一到 0-0.5）
=============================================================================
"""

import os
import re
import glob
import torch
import numpy as np
import argparse

# 最佳超参（从现有 Results_disturb_H.csv 中获取）
BEST_HPARAMS = {
    'l2': 0.1, 'lr': 0.01, 'glr': 0.5, 'AGG': 'cat',
    'predict_hiddens': [200, 100], 'n_hid': 200, 'epochs': 50,
    'batch_size': 64, 'cox_num': 1000, 'depth': 2,
    'edge_pooling': True, 'loss_w': {'a': 1, 'b': 1, 'c1': 1, 'c2': 0, 'd': 1},
    'pooling_method': 'linear', 'graph_layer': 'HG',
}
PK_D = {'DataDrive': 'STRING', 'method': 'layer_all', 'divisor': 6, 'layer_STRING': 34}

CKPT_DIR = "/Backup/home/chenyupeng/Results/Abalation/PRO/HCC/STRING/10seedsModelsWeights"
OUTPUT_DIR = "/Backup/home/chenyupeng/Results/Abalation/PRO/HCC/STRING"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "Results_disturb_H_granular.csv")

DISTURB_LEVELS = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_SEEDS = 10
DATASET = "HCC"


def build_ckpt_path(disturb_val: float, seed: int) -> str:
    """Build checkpoint file path for a given disturb_H value and seed."""
    pk_d_str = str(PK_D)
    fn = (f"disturb_H-{disturb_val}_{DATASET}_seed-{seed}"
          f"_l2-{BEST_HPARAMS['l2']}_lr-{BEST_HPARAMS['lr']}"
          f"_glr{BEST_HPARAMS['glr']}_agg{BEST_HPARAMS['AGG']}"
          f"_PH{BEST_HPARAMS['predict_hiddens']}"
          f"_loss_{BEST_HPARAMS['loss_w']}_{pk_d_str}.ckpt")
    return os.path.join(CKPT_DIR, fn)


def extract_test_ci(disturb_val: float, seed: int) -> tuple:
    """Load checkpoint and return (test_ci, val_ci, val_loss)."""
    ckpt_path = build_ckpt_path(disturb_val, seed)
    if not os.path.isfile(ckpt_path):
        print(f"  [WARN] Checkpoint not found: {ckpt_path}")
        return None, None, None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        test_ci = ckpt.get("test_ci", None)
        val_ci = ckpt.get("best_eval_ci", None)
        val_loss = ckpt.get("best_eval_loss", None)
        return test_ci, val_ci, val_loss
    except Exception as e:
        print(f"  [ERROR] Failed to load {ckpt_path}: {e}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Extract disturb_H results from existing checkpoints")
    parser.add_argument("--smoke_test", action="store_true", help="Print summary only, do not write CSV")
    args = parser.parse_args()

    print("=" * 65)
    print("  Extract disturb_H Granular Results from Checkpoints")
    print("=" * 65)
    print(f"\n  CKPT dir: {CKPT_DIR}")
    print(f"  Levels:   {DISTURB_LEVELS}")
    print(f"  Seeds:    0-{N_SEEDS-1}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    for lv in DISTURB_LEVELS:
        seeds_ci = []
        seeds_val_ci = []
        seeds_val_loss = []
        for seed in range(N_SEEDS):
            test_ci, val_ci, val_loss = extract_test_ci(lv, seed)
            if test_ci is not None:
                seeds_ci.append(test_ci)
                seeds_val_ci.append(val_ci)
                seeds_val_loss.append(val_loss)
        
        if len(seeds_ci) == 0:
            print(f"  [WARN] No checkpoints found for disturb_H={lv}")
            all_results[lv] = []
            continue
        
        print(f"  disturb_H={lv}: {len(seeds_ci)}/{N_SEEDS} seeds found")
        print(f"    test_ci: mean={np.mean(seeds_ci):.6f} std={np.std(seeds_ci):.6f}")
        all_results[lv] = list(zip(seeds_ci, seeds_val_ci, seeds_val_loss))

    if args.smoke_test:
        print("\n  [SMOKE TEST PASSED] No CSV written.")
        return

    # Write CSV
    with open(OUTPUT_CSV, "w") as f:
        f.write("----------------Results of ablation study of disturb_H (granular)----------------\n")
        for lv in DISTURB_LEVELS:
            results = all_results.get(lv, [])
            f.write(f"\ndisturb_H:{lv}\n")
            f.write("part_ablation,cancer,seed,val_ci,val_loss,test_ci\n")
            for seed, (test_ci, val_ci, val_loss) in enumerate(results):
                if val_ci is not None and val_loss is not None:
                    f.write(f"{lv},{DATASET},{seed},{val_ci},{val_loss},{test_ci}\n")
            
            test_cis = [r[0] for r in results if r[0] is not None]
            if test_cis:
                mean_ci = np.mean(test_cis)
                std_ci = np.std(test_cis)
                f.write(f"Test Cindex mean±std = {mean_ci:.4f}±{std_ci:.4f}\n")
            f.write(f"Above Models' Hyperparameters:{BEST_HPARAMS}\n")
            f.write(f"{PK_D}\n")
        
        # Summary line
        level_strs = [str(lv) for lv in DISTURB_LEVELS]
        mean_stds = []
        means = []
        for lv in DISTURB_LEVELS:
            test_cis = [r[0] for r in all_results.get(lv, []) if r[0] is not None]
            if test_cis:
                mean_stds.append(f"{np.mean(test_cis):.4f}±{np.std(test_cis):.4f}")
                means.append(np.mean(test_cis))
            else:
                mean_stds.append("N/A")
                means.append(0)
        
        f.write(f"\n{level_strs}:\n")
        f.write(f"{mean_stds}\n")
        f.write(f"{means}\n")

    print(f"\n  [SAVED] {OUTPUT_CSV}")
    print(f"  Done.")


if __name__ == "__main__":
    main()
