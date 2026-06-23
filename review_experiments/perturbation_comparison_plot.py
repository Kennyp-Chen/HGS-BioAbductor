#!/usr/bin/env python3
"""
Perturbation Comparison Plot
=============================
Compare random perturbation (disturb_H) vs Borda importance-based masking
effects on model performance (C-index) for HGS-BioAbductor.

Data sources:
  - Random perturbation: Results_disturb_H_granular.csv (extracted from existing checkpoints)
  - Borda masking:     Notebook cell outputs from get_borda_score_10seeds()

Generates:
  - Results/perturbation_comparison/random_vs_importance_hyperedges.png
  - Results/perturbation_comparison/random_vs_importance_nodes.png

Usage:
  python other_experiments/perturbation_comparison_plot.py
  python other_experiments/perturbation_comparison_plot.py --smoke_test

=============================================================================
脚本流程注解:
1. 读取随机扰动数据:
   - 从 Results_disturb_H_granular.csv 按 disturb_H 分段提取 seeds 0-9 的 test_ci
   - 扰动比例: [0, 0.1, 0.2, 0.3, 0.4, 0.5]（与 Borda 掩码对齐）
   - 对每个比例算 mean±std
2. 读取 Borda 重要性掩码数据（从 notebook stdout 提取的 10 seed 原始值）:
   - Top-down: 按 Borda 得分移除最重要的超边/节点
   - Bottom-up: 移除最不重要的超边/节点
   - 百分比: [0, 0.1, 0.2, 0.3, 0.4, 0.5]
   - 对每个百分比算 mean±std
3. 在同一张图上绘制对比曲线:
   - X 轴 = 扰动/移除比例, Y 轴 = C-index
   - 随机扰动曲线、Top-down 曲线、(可选) Bottom-up 曲线
4. 输出两张图到 Results/perturbation_comparison/
=============================================================================
"""

import os
import sys
import re
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Borda validation data (extracted from notebook stdout).
# get_borda_score_10seeds() for HCC PRO / hcluster ran 10 seeds with
# percent=0.5 → percent_space = [0, 0.1, 0.2, 0.3, 0.4, 0.5].
# Each entry: (top_remove_ci_h, bottom_remove_ci_h, top_remove_ci_n, bottom_remove_ci_n)
# ---------------------------------------------------------------------------
_BORDA_RAW = [
    # Seed 0
    ([0.6930881420418517, 0.6816740646797718, 0.6981610653138871,
      0.6867469879518072, 0.6785034876347495, 0.6544071020925808],
     [0.6930881420418517, 0.6873811033608117, 0.6886493341788206,
      0.6949904882688649, 0.6930881420418517, 0.6981610653138871],
     [0.6930881420418517, 0.6975269499048827, 0.689283449587825,
      0.6854787571337984, 0.6563094483195941, 0.6531388712745719],
     [0.6930881420418517, 0.6918199112238428, 0.6918199112238428,
      0.6886493341788206, 0.6873811033608117, 0.6861128725428028]),
    # Seed 1
    ([0.7029393370856786, 0.7154471544715447, 0.7041901188242652,
      0.7048155096935584, 0.6779237023139462, 0.6585365853658537],
     [0.7029393370856786, 0.7010631644777986, 0.7104440275171983,
      0.716072545340838, 0.7166979362101313, 0.7123202001250781],
     [0.7029393370856786, 0.7141963727329581, 0.6991869918699187,
      0.6879299562226392, 0.7023139462163852, 0.6772983114446529],
     [0.7029393370856786, 0.7035647279549718, 0.701688555347092,
      0.7041901188242652, 0.7023139462163852, 0.7041901188242652]),
    # Seed 2
    ([0.781750924784217, 0.7527743526510481, 0.7404438964241677,
      0.7422934648581998, 0.7219482120838471, 0.7299630086313194],
     [0.781750924784217, 0.7805178791615289, 0.7928483353884094,
      0.7990135635018496, 0.7774352651048089, 0.782983970406905],
     [0.781750924784217, 0.7601726263871763, 0.7848335388409371,
      0.7311960542540074, 0.6652281134401973, 0.49753390875462394],
     [0.781750924784217, 0.782367447595561, 0.7842170160295932,
      0.7891491985203453, 0.7934648581997534, 0.7916152897657214]),
    # Seed 3
    ([0.6721938775510204, 0.6619897959183674, 0.6473214285714286,
      0.6511479591836735, 0.6243622448979592, 0.6307397959183674],
     [0.6721938775510204, 0.6728316326530612, 0.6639030612244898,
      0.6632653061224489, 0.6600765306122449, 0.6536989795918368],
     [0.6721938775510204, 0.6556122448979592, 0.6186224489795918,
      0.5816326530612245, 0.6224489795918368, 0.5918367346938775],
     [0.6721938775510204, 0.6728316326530612, 0.6728316326530612,
      0.6619897959183674, 0.6588010204081632, 0.6607142857142857]),
    # Seed 4
    ([0.8007448789571695, 0.7883302296710117, 0.7839851024208566,
      0.770949720670391, 0.7603972687771571, 0.6846679081315953],
     [0.8007448789571695, 0.8013656114214773, 0.8013656114214773,
      0.8007448789571695, 0.8050900062073246, 0.7939168218497827],
     [0.8007448789571695, 0.8007448789571695, 0.7932960893854749,
      0.8081936685288641, 0.7957790192427064, 0.7951582867783985],
     [0.8007448789571695, 0.8007448789571695, 0.8013656114214773,
      0.803227808814401, 0.7970204841713222, 0.7939168218497827]),
    # Seed 5
    ([0.8612294583079733, 0.8587948874010956, 0.8277541083384053,
      0.8174071819841753, 0.81253804017042, 0.7857577601947656],
     [0.8612294583079733, 0.8587948874010956, 0.8545343883140597,
      0.8557516737674985, 0.8435788192331102, 0.837492391965916],
     [0.8612294583079733, 0.8472306755934267, 0.852099817407182,
      0.8381010346926354, 0.8070602556299452, 0.7748021911138162],
     [0.8612294583079733, 0.8630553864881315, 0.8636640292148509,
      0.8624467437614121, 0.8606208155812538, 0.8636640292148509]),
    # Seed 6
    ([0.7377347062386432, 0.709267110841914, 0.7104784978800727,
      0.6959418534221684, 0.6783767413688674, 0.663840096910963],
     [0.7377347062386432, 0.7431859479103574, 0.7425802543912781,
      0.7395517867958813, 0.7498485766202302, 0.7607510599636584],
     [0.7377347062386432, 0.6953361599030891, 0.67595396729255,
      0.6178073894609327, 0.5511811023622047, 0.5754088431253785],
     [0.7377347062386432, 0.7395517867958813, 0.74076317383404,
      0.7419745608721987, 0.7468201090248334, 0.7668079951544519]),
    # Seed 7
    ([0.7704180064308682, 0.760128617363344, 0.7337620578778135,
      0.7215434083601286, 0.7260450160771704, 0.7286173633440515],
     [0.7704180064308682, 0.7704180064308682, 0.7620578778135049,
      0.757556270096463, 0.757556270096463, 0.7401929260450161],
     [0.7704180064308682, 0.7678456591639872, 0.7665594855305466,
      0.782636655948553, 0.754983922829582, 0.7228295819935692],
     [0.7704180064308682, 0.7704180064308682, 0.7710610932475884,
      0.7710610932475884, 0.7697749196141479, 0.7627009646302251]),
    # Seed 8
    ([0.6786140979689367, 0.6816009557945042, 0.6887694145758662,
      0.6857825567502986, 0.6606929510155317, 0.6577060931899642],
     [0.6786140979689367, 0.6804062126642771, 0.6827956989247311,
      0.6792114695340502, 0.6845878136200717, 0.6762246117084827],
     [0.6786140979689367, 0.6589008363201911, 0.6445639187574671,
      0.6589008363201911, 0.6326164874551972, 0.6033452807646356],
     [0.6786140979689367, 0.6804062126642771, 0.6804062126642771,
      0.6780167264038232, 0.6786140979689367, 0.6768219832735962]),
    # Seed 9
    ([0.7245958429561201, 0.73094688221709, 0.7205542725173211,
      0.6997690531177829, 0.6968822170900693, 0.6893764434180139],
     [0.7245958429561201, 0.7176674364896074, 0.7107390300230947,
      0.7113163972286374, 0.7130484988452656, 0.6968822170900693],
     [0.7245958429561201, 0.7136258660508084, 0.680715935334873,
      0.6668591224018475, 0.6304849884526559, 0.5975750577367206],
     [0.7245958429561201, 0.7303695150115473, 0.7297921478060047,
      0.73094688221709, 0.7355658198614319, 0.7315242494226328]),
]


def read_disturb_h_csv(csv_path):
    """Parse the disturb_H CSV and return (levels, all_test_ci) where
    levels is list of disturb_H values and all_test_ci is a list of lists
    (one inner list per level, containing test_ci for 10 seeds)."""
    levels = []
    current_level = None
    current_cis = []
    all_test_ci = []

    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Check for section header: "disturb_H:0.3"
            m = re.match(r"^disturb_H:([\d.]+)$", line)
            if m:
                if current_level is not None:
                    levels.append(current_level)
                    all_test_ci.append(current_cis)
                current_level = float(m.group(1))
                current_cis = []
                continue
            # Skip header lines and summary lines
            if line.startswith("part_ablation"):
                continue
            if line.startswith("Test Cindex") or line.startswith("Above Model") or line.startswith("{"):
                continue
            if line.startswith("[") or line.startswith("-"):
                continue
            # Try to parse as data row: 0.3,HCC,0,...,...,0.6842
            parts = line.split(",")
            if len(parts) >= 6:
                try:
                    test_ci = float(parts[-1])
                    current_cis.append(test_ci)
                except ValueError:
                    pass

    # Don't forget the last section
    if current_level is not None:
        levels.append(current_level)
        all_test_ci.append(current_cis)

    return levels, all_test_ci


def mean_std_across_seeds(data_list):
    """Compute mean ± std across seeds for each masking level.
    data_list is a list of lists (one per seed) of equal length.
    Returns (means, stds)."""
    arr = np.array(data_list)  # shape (n_seeds, n_levels)
    return arr.mean(axis=0), arr.std(axis=0)


def _set_academic_style():
    """Apply matplotlib styling suitable for academic papers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


def plot_comparison(
    ax,
    x_rand,
    rand_mean,
    rand_std,
    x_borda,
    td_mean,
    td_std,
    bu_mean,
    bu_std,
    xlabel,
    title,
    show_bottom_up=True,
    delta_mode=False,
):
    """Plot one comparison panel onto *ax*.

    Parameters
    ----------
    delta_mode : bool
        When True, all curves are converted to ΔC-index (relative to their own
        baseline at level=0) so experiments with different absolute baselines
        can be compared fairly. When False, plots raw C-index values.
    """
    if delta_mode:
        rand_base = rand_mean[0]
        td_base   = td_mean[0]
        bu_base   = bu_mean[0] if show_bottom_up else 0
        r_plot = rand_mean - rand_base
        r_lo   = (rand_mean - rand_std) - rand_base
        r_hi   = (rand_mean + rand_std) - rand_base
        t_plot = td_mean - td_base
        t_lo   = (td_mean - td_std) - td_base
        t_hi   = (td_mean + td_std) - td_base
        if show_bottom_up:
            b_plot = bu_mean - bu_base
            b_lo   = (bu_mean - bu_std) - bu_base
            b_hi   = (bu_mean + bu_std) - bu_base
        ylabel = "Δ C-index (relative to baseline)"
    else:
        r_plot, r_lo, r_hi = rand_mean, rand_mean - rand_std, rand_mean + rand_std
        t_plot, t_lo, t_hi = td_mean, td_mean - td_std, td_mean + td_std
        if show_bottom_up:
            b_plot, b_lo, b_hi = bu_mean, bu_mean - bu_std, bu_mean + bu_std
        ylabel = "C-index"

    # Random perturbation
    ax.plot(x_rand, r_plot, "o-", color="tab:blue",
            linewidth=1.8, markersize=5,
            label="Random perturbation (disturb_H)", zorder=3)
    ax.fill_between(x_rand, r_lo, r_hi,
                    color="tab:blue", alpha=0.15, zorder=1)

    # Top-down (most important first)
    ax.plot(x_borda, t_plot, "s--", color="tab:red",
            linewidth=1.8, markersize=5,
            label="Top-down (most important first)", zorder=3)
    ax.fill_between(x_borda, t_lo, t_hi,
                    color="tab:red", alpha=0.12, zorder=1)

    # Bottom-up (least important first)
    if show_bottom_up:
        ax.plot(x_borda, b_plot, "d-.", color="tab:green",
                linewidth=1.8, markersize=5,
                label="Bottom-up (least important first)", zorder=3)
        ax.fill_between(x_borda, b_lo, b_hi,
                        color="tab:green", alpha=0.12, zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylabel(ylabel)
    if delta_mode:
        ax.axhline(y=0, color="grey", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_title(title)
    ax.legend(loc="best")

    if delta_mode:
        ax.set_ylim(-0.35, 0.10)
    else:
        ax.set_ylim(0.45, 0.95)


def main():
    parser = argparse.ArgumentParser(
        description="Generate perturbation comparison plots."
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Only read data and print summary statistics; skip plotting.",
    )
    args = parser.parse_args()

    # ---- paths ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    disturb_csv = os.path.join(
        os.path.dirname(project_root),
        "Results/Abalation/PRO/HCC/STRING/Results_disturb_H_granular.csv",
    )
    out_dir = os.path.join(project_root, "Results/perturbation_comparison")

    # ---- 1. Random perturbation data ----
    if not os.path.isfile(disturb_csv):
        print(f"[ERROR] disturb_H CSV not found: {disturb_csv}")
        sys.exit(1)

    rand_levels, rand_all_ci = read_disturb_h_csv(disturb_csv)

    # Transpose from (levels, seeds) to (seeds, levels)
    rand_by_seed = list(zip(*rand_all_ci))
    rand_means, rand_stds = mean_std_across_seeds(rand_by_seed)

    # ---- 2. Borda validation data ----
    percent_space = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # Unpack hardcoded data
    td_h = [s[0] for s in _BORDA_RAW]  # top-down hyperedge per seed
    bu_h = [s[1] for s in _BORDA_RAW]  # bottom-up hyperedge per seed
    td_n = [s[2] for s in _BORDA_RAW]  # top-down node per seed
    bu_n = [s[3] for s in _BORDA_RAW]  # bottom-up node per seed

    td_h_mean, td_h_std = mean_std_across_seeds(td_h)
    bu_h_mean, bu_h_std = mean_std_across_seeds(bu_h)
    td_n_mean, td_n_std = mean_std_across_seeds(td_n)
    bu_n_mean, bu_n_std = mean_std_across_seeds(bu_n)

    # ---- smoke test: print summary ----
    if args.smoke_test:
        print("=" * 68)
        print("  SMOKE TEST — Perturbation Comparison Data Summary")
        print("=" * 68)

        print("\n--- Random perturbation (disturb_H) ---")
        print(f"{'Level':>8}  {'Mean C-index':>14}  {'Std':>10}  {'n_seeds':>8}")
        for lv, mn, sd, ci_list in zip(rand_levels, rand_means, rand_stds, rand_all_ci):
            print(f"{lv:>8.1f}  {mn:>14.6f}  {sd:>10.6f}  {len(ci_list):>8}")

        print("\n--- Borda masking — Hyperedges ---")
        print(f"{'Percent':>8}  {'Top-down mean':>14}  {'Top-down std':>12}  "
              f"{'Bottom-up mean':>14}  {'Bottom-up std':>12}")
        for p, tm, ts, bm, bs in zip(percent_space,
                                      td_h_mean, td_h_std,
                                      bu_h_mean, bu_h_std):
            print(f"{p:>8.1f}  {tm:>14.6f}  {ts:>12.6f}  {bm:>14.6f}  {bs:>12.6f}")

        print("\n--- Borda masking — Nodes ---")
        print(f"{'Percent':>8}  {'Top-down mean':>14}  {'Top-down std':>12}  "
              f"{'Bottom-up mean':>14}  {'Bottom-up std':>12}")
        for p, tm, ts, bm, bs in zip(percent_space,
                                      td_n_mean, td_n_std,
                                      bu_n_mean, bu_n_std):
            print(f"{p:>8.1f}  {tm:>14.6f}  {ts:>12.6f}  {bm:>14.6f}  {bs:>12.6f}")

        print("\n[SMOKE TEST PASSED] No plots generated.")
        return

    # ---- 3. Generate plots ----
    os.makedirs(out_dir, exist_ok=True)
    plt = _set_academic_style()

    panels = [
        ("hyperedges", td_h_mean, td_h_std, bu_h_mean, bu_h_std),
        ("nodes",      td_n_mean, td_n_std, bu_n_mean, bu_n_std),
    ]

    for suffix, td_m, td_s, bu_m, bu_s in panels:
        # ── Absolute C-index ──
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        plot_comparison(
            ax,
            x_rand=rand_levels,
            rand_mean=rand_means, rand_std=rand_stds,
            x_borda=percent_space,
            td_mean=td_m, td_std=td_s,
            bu_mean=bu_m, bu_std=bu_s,
            xlabel="Perturbation / masking level",
            title=f"Random Perturbation vs Importance-based Masking ({suffix.capitalize()})",
            show_bottom_up=True,
            delta_mode=False,
        )
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"random_vs_importance_{suffix}.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[SAVED] {out_path}")

        # ── Δ C-index (relative) ──
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        plot_comparison(
            ax,
            x_rand=rand_levels,
            rand_mean=rand_means, rand_std=rand_stds,
            x_borda=percent_space,
            td_mean=td_m, td_std=td_s,
            bu_mean=bu_m, bu_std=bu_s,
            xlabel="Perturbation / masking level",
            title=f"Random Perturbation vs Importance-based Masking ({suffix.capitalize()})",
            show_bottom_up=True,
            delta_mode=True,
        )
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"random_vs_importance_{suffix}_delta.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[SAVED] {out_path}")

    print(f"\nDone. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
