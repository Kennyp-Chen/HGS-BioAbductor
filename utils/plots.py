
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
import pandas as pd
import numpy as np
import json
import logging

from models.Models_interpret import *
from utils.data_utils import *
from utils.hg_ops import *
import torch
from torch import optim
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind,pearsonr,mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.colors as mcolors
from sklearn.impute import KNNImputer
from utils.Interpret import *
from utils.plots import *
import os,torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from sklearn.metrics import confusion_matrix,roc_auc_score, roc_curve,RocCurveDisplay
from sklearn.preprocessing import MinMaxScaler
import time 
import json
from matplotlib.colors import LinearSegmentedColormap


def Difference_Analysis(group_h, group_l,df_corr,XAI):
    high_expr = df_corr.loc[group_h]
    low_expr = df_corr.loc[group_l]
    # 初始化存储结果
    ps_t = []
    log2_fc = []
    results = []
    # 对每个蛋白质进行差异分析
    for protein in XAI['proteins']:
        # 提取两组数据并去除缺失值
        h_data = high_expr[protein].dropna()
        l_data = low_expr[protein].dropna()
        
        # fold change 
        with np.errstate(divide='ignore', invalid='ignore'):
            fc = np.log2((h_data.mean()+1e-9)/(l_data.mean()+1e-9))
        log2_fc.append(fc)
        
        if len(h_data) > 1 and len(l_data) > 1:
            _, p_mwu= mannwhitneyu(h_data, l_data)
        ps_t.append(p_mwu)

        # 多重假设检验校正（Benjamini-Hochberg）
        results.append({'Protein':protein, 'log2FC':fc,'p_mwu':p_mwu})

    # 结果后处理 ----------------------------------------------------------
    df = pd.DataFrame(results)

    # 多重检验校正（仅校正实际使用的特征）
    df['adj_p'] = multipletests(df['p_mwu'], method='fdr_bh')[1]

    # 筛选显著差异蛋白（通常标准：adj_p<0.05 & |log2FC|>1）
    df['significant'] = (df['adj_p'] < 0.05) & \
                                (np.abs(df['log2FC']) > 1)
    # Compare with XAI res 
    df['XAI'] = XAI['XAI_scores']
    # Pearson Correlation
    df['abs_log2FC']=np.abs(df['log2FC'])
    df['corr_DA_XAI'] = df['abs_log2FC'].corr(df['XAI'],method='pearson')
    return df
def plot_scatter_regression(data,fs_base,title='Correlation between Absolute Protein Abundance Change and Model Interpretability'):
    '''
    data: pd.DataFrame({
    'abs_log2FC': [],
    'XAI_score': []
    'adj_p' [],
    })
    '''
    data=data.copy()

    # 计算相关性和回归线
    corr_coef, p_value = stats.pearsonr(data['XAI_score'], data['abs_log2FC'])
    slope, intercept = np.polyfit(data['XAI_score'], data['abs_log2FC'], 1)

    # 创建画布
    plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.gca()

    # 创建对数颜色的norm和颜色映射
    # 使用对数尺度更好地表示p值范围
    log_norm = mcolors.LogNorm(vmin=data['adj_p'].min(), 
                              vmax=data['adj_p'].max())

    # cmap = plt.cm.Reds_r  # 红色表示显著（低p值），白色表示不显著（高p值）
    
    # 为显著点添加边缘强调
    # 设置默认点边缘为无，显著点添加black边缘
    # edgecolors = np.where(data['adj_p'] < 0.05, 'black', 'none')
    # linewidths = np.where(data['adj_p'] < 0.05, 1.0, 0)
    
    scatter = ax.scatter(
        x='XAI_score', 
        y='abs_log2FC', 
        c=data['adj_p'],        # 颜色表示调整后的p值
        # cmap=cmap,               # 使用红色映射
        # norm=log_norm,           # 对数标准化
        cmap='coolwarm',   # 红蓝渐变色
        alpha=1,         # 透明度
        # edgecolor=edgecolors,     # 点边框
        # linewidth=linewidths,     # 边框粗细
        data=data
    )

    # 绘制回归线
    x_range = np.linspace(data['XAI_score'].min(), data['XAI_score'].max(), 100)
    ax.plot(x_range, slope*x_range + intercept, 
            color="#c50404", 
            linewidth=2.5, 
            linestyle='-',)
            # label=f'y = {np.round(slope, 5)}x + {np.round(intercept, 5)}')

    # 添加统计信息
    num_significant = np.sum(data['adj_p'] < 0.05)
    # text_box = f'Pearson r = {corr_coef:.2f}\np-value = {p_value:.2e}\nSignificant: {num_significant}/{len(data)}'
    text_box = f'Pearson r = {corr_coef:.2f}\np-value = {p_value:.2e}'
    
    ax.text(0.05, 0.95, text_box, 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),fontsize=fs_base-2)


    # 颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Adjusted p-value', rotation=270, labelpad=15,fontsize=fs_base,weight='bold')
    
    # 添加颜色条刻度标记（特别是0.05阈值）
    # cbar_tick_values = np.logspace(
    #     np.floor(np.log10(data['adj_p'].min())), 
    #     np.log10(data['adj_p'].max()), 
    #     num=2
    # )
    # cbar_tick_values = np.sort(np.append(cbar_tick_values, 0.05))
    # cbar.set_ticks(cbar_tick_values)
    # cbar.set_ticklabels([f"{v:.3f}" if v >= 0.001 else f"{v:.2e}" for v in cbar_tick_values])
    # cbar.mappable.set_clim(1e-7, 1)
    
    #############################################01
    # # 添加颜色条刻度标记
    cbar_tick_values = [0, 0.05, 1]  # 指定仅显示这三个值
    cbar.set_ticks(cbar_tick_values)  # 设置刻度位置
    cbar.set_ticklabels([f"{v:.2f}" for v in cbar_tick_values])  # 格式化标签

    # 确保颜色条的范围覆盖0到1（如果需要可添加这行）
    cbar.mappable.set_clim(0, 1)

    #############################################01

    # 坐标轴美化
    ax.set_xlabel('Normalized Borda Score', fontsize=fs_base+2, labelpad=10,weight='bold')
    ax.set_ylabel('Absolute Log2 Fold Change', fontsize=fs_base+2, labelpad=10,weight='bold')
    # ax.set_xticks(fontsize = fs_base)
    # ax.set_yticks(fontsize = fs_base)

    ax.set_title(title, 
                fontsize=fs_base+2, pad=15,weight = 'bold')

    # 网格线
    ax.grid(True, linestyle='--', alpha=0.4)
    sns.despine(trim=True)

    # 图例
    # ax.legend(loc='lower right', frameon=True)
    ax.legend( frameon=True,labels=['Protein', f'y = {np.round(slope, 3)}x + {np.round(intercept, 3)}'],fontsize=fs_base-2,loc="upper right")
    
    plt.tight_layout()
    plt.show()



def draw_venn(sets, labels=None, title="Venn Diagram", colors=('#1f77b4', '#ff7f0e', '#2ca02c'), 
              alpha=0.5, figsize=(8, 6), show_percentages=True, percentage_decimals=1,subset_label_fontsize=10):
    """
    绘制带百分比显示的韦恩图
    
    参数:
    sets -- 包含集合的列表 [set1, set2] 或 [set1, set2, set3]
    labels -- 可选，集合标签元组 ('A', 'B') 或 ('A', 'B', 'C')
    title -- 图表标题
    colors -- 圆圈颜色 (默认: 蓝色, 橙色, 绿色)
    alpha -- 透明度 (0-1)
    figsize -- 图表尺寸
    show_percentages -- 是否显示百分比 (默认True)
    percentage_decimals -- 百分比显示的小数位数 (默认1)
    """
    # 计算全集大小（所有元素的并集）
    total = len(set.union(*sets)) if sets else 1
    
    # 计算每个区域的数量和百分比
    def format_percentage(count):
        """格式化百分比显示"""
        if count == 0:
            return ""
        percentage = count / total * 100
        return f"{percentage:.{percentage_decimals}f}%" if show_percentages else ""
    
    def format_label(count):
        """格式化标签显示(数量 + 百分比)"""
        if count == 0:
            return ""
        percentage = count / total * 100
        return f"{count}\n({percentage:.{percentage_decimals}f}%)" if show_percentages else str(count)
    
    n_sets = len(sets)
    plt.figure(figsize=figsize)
    
    if n_sets == 2:
        # 计算各区域的数量
        A = sets[0]
        B = sets[1]
        only_A = len(A - B)
        only_B = len(B - A)
        intersect = len(A & B)
        subsets = {
            '10': len(A - B),  
            '11': len(A & B), 
            '01': len(B - A),  
        }
        # 创建图表
        venn = venn2(subsets=(only_A, only_B, intersect),
                     set_labels=labels,
                     set_colors=colors[:2],
                     alpha=alpha)
        
        # 添加百分比标签
        # if show_percentages:
        #     # 覆盖默认标签，同时显示数量和百分比
        #     # venn.get_label_by_id('10').set_text(format_label(only_A)).set_fontsize(subset_label_fontsize)
        #     # venn.get_label_by_id('11').set_text(format_label(intersect)).set_fontsize(subset_label_fontsize)
        #     # venn.get_label_by_id('01').set_text(format_label(only_B)).set_fontsize(subset_label_fontsize)
        #                 # 为所有可能区域设置百分比标签
        if show_percentages:
            # 覆盖默认标签，同时显示数量和百分比
            for region_id in ['10', '11', '01']:
                label = venn.get_label_by_id(region_id)
                if label:
                    count = intersect if region_id == '11' else only_A if region_id == '10' else only_B
                    label.set_text(format_label(count))
                    if region_id == '11':
                        label.set_fontsize(subset_label_fontsize-2)
                    else:
                        label.set_fontsize(subset_label_fontsize)

                
            # label.set_fontsize(subset_label_fontsize)
        
        # 添加圆圈轮廓
        venn2_circles(subsets=(only_A,only_B, intersect), 
                      linewidth=1.5, linestyle='dashed')
        
        # 在顶部添加总元素数
        plt.title(title , fontsize=16, pad=20,weight='bold')
        
    elif n_sets == 3:
        # 计算各区域的数量
        A, B, C = sets
        subsets = {
            '100': len(A - B - C),  # 仅A
            '010': len(B - A - C),  # 仅B
            '110': len(A & B - C),  # A与B的交集(不含C)
            '001': len(C - A - B),  # 仅C
            '101': len(A & C - B),  # A与C的交集(不含B)
            '011': len(B & C - A),  # B与C的交集(不含A)
            '111': len(A & B & C)   # A,B,C交集
        }
        
        # 创建图表
        venn = venn3(subsets=subsets, 
                     set_labels=labels, 
                     set_colors=colors,
                     alpha=alpha)
        
        # 添加百分比标签
        if show_percentages:
            # 为所有可能区域设置百分比标签
            for region_id in ['100', '010', '001', '110', '101', '011', '111']:
                count = subsets[region_id]
                label = venn.get_label_by_id(region_id)
                if label:
                    label.set_text(format_label(count))
                    label.set_fontsize(subset_label_fontsize)
            
            # 对于较小区域单独处理
            edge_positions = {
                '100': (-0.8, -0.55),  # 仅A
                '010': (0.5, -0.7),    # 仅B
                '001': (0.3, 0.75),    # 仅C
            }
            
            for region_id, position in edge_positions.items():
                count = subsets[region_id]
                if count > 0 and not venn.get_label_by_id(region_id):
                    plt.annotate(format_label(count), xy=position, 
                                fontsize=9, ha='center', va='center')
        
        # 添加圆圈轮廓
        venn3_circles(linewidth=1.5, linestyle='dashed')
        
        # 在顶部添加总元素数
        plt.annotate(f"Total Elements: {total}", xy=(0.5, 1.05), 
                     xycoords='axes fraction', ha='center', fontsize=11)
        plt.title(title , fontsize=16, pad=20,weight='bold')
        
    else:
        raise ValueError("只支持2个或3个集合的韦恩图")
    
    # 设置字体大小
    if labels:
        for label in venn.set_labels:
            if label:
                label.set_fontsize(12)
    
    # # 设置数字标签大小
    # for text in venn.subset_labels:
    #     if text:
    #         text.set_fontsize(9)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 为标题和总元素数留出空间
    plt.show()
    return venn

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = []
    n = len(y_label)
    
    # 避免概率为1导致除0错误
    safe_thresh_group = np.clip(thresh_group, 0.001, 0.999)
    
    for thresh in safe_thresh_group:
        y_pred_label = y_pred_score > thresh
        # 防止混淆矩阵计算错误
        try:
            tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        except ValueError:  # 当全预测为同一类别时
            if np.all(y_pred_label == 0):
                tn, fp, fn, tp = n, 0, np.sum(y_label), 0
            else:
                tn, fp, fn, tp = 0, n - np.sum(y_label), 0, np.sum(y_label)
        
        # 改进公式：更稳健的计算
        true_positive_benefit = tp / n
        false_positive_cost = (fp / n) * (thresh / (1 - thresh))
        net_benefit = true_positive_benefit - false_positive_cost
        
        net_benefit_model.append(net_benefit)
    
    return np.array(net_benefit_model)

def calculate_net_benefit_all(thresh_group, y_label):
    """
    正确计算Treat all和Treat none策略的净收益
    """
    n = len(y_label)
    p = np.sum(y_label)  # 正例数量
    treat_all_benefits = []
    treat_none_benefits = []
    
    for thresh in thresh_group:
        # Treat all策略: 预测所有人为阳性
        tp_all = p
        fp_all = n - p
        net_benefit_all = (tp_all / n) - (fp_all / n) * (thresh / (1 - thresh))
        treat_all_benefits.append(net_benefit_all)
        
        # Treat none策略: 预测所有人为阴性
        net_benefit_none = 0  # 没有真阳性，没有假阳性
        treat_none_benefits.append(net_benefit_none)
    
    return treat_all_benefits, treat_none_benefits

from sklearn.metrics import roc_curve, auc

def plot_DCA(ax, thresh_group, net_benefit_models, model_names, net_benefit_all, net_benefit_none,xmin = 0,xmax = 1,model_show = ['HGS','COX','RSF','TNM','BCLC'],title='DCA',fs=14):
    """
    多模型DCA绘图
    net_benefit_models:list of models net benefit
    """
    # 1. 绘制参考策略
    ax.plot(thresh_group, net_benefit_all, 
            color='black', linewidth=2, label='Treat all')
    ax.plot(thresh_group, net_benefit_none, 
            color='black', linestyle=':', linewidth=2, label='Treat none')
    
    # 2. 绘制模型曲线
    n_models = len(net_benefit_models)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_models))
#     linewidth=1
    # 动态调整Y轴范围
    min_val, max_val = 0, 0
    
    for i, (model_benefit, name) in enumerate(zip(net_benefit_models, model_names)):
        # 异常值
        clipped_benefit = np.clip(model_benefit, -0.1, None)
        model_min = np.nanmin(clipped_benefit) 
        model_max = np.nanmax(clipped_benefit)
        min_val = min(min_val, model_min)
        max_val = max(max_val, model_max)
        if name not in model_show:
            continue
        if name == 'HGS':
            ax.plot(thresh_group, clipped_benefit, 
                    color = '#C82423', linewidth=2, 
                    alpha=1, 
                    label=name,zorder=len(model_names)+1)
        else:
            ax.plot(thresh_group, clipped_benefit, 
                    # color=colors[i], linewidth=2.5, 
                    linewidth=2, alpha=1, 
                    label=name)
    
    padding = max(0.05, (max_val - min_val) * 0.1)
    ax.set_ylim(min_val - padding, max_val + padding)

    ax.set_xlim(xmax=xmax,xmin=xmin)
    ax.set_xlabel('Threshold Probability', fontsize=fs,weight='bold')
    ax.set_ylabel('Net Benefit', fontsize=fs,weight='bold')
    ax.grid(alpha=0.3)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    # 图例移到图形外避免遮挡
    ax.legend(loc='upper right',)
    plt.title(title, fontsize=fs+2)
    # plt.show()
    # plt.savefig(f'images/DCA/{title}.png', dpi=300, bbox_inches='tight')
    return ax

def plot_ROCs(y_scores_list, fs_base,model_names, y_true,model_ours="HGS",title='Receiver Operating Characteristic (ROC) Curves for model comparison'):
    '''
    绘制多个模型的ROC曲线在同一张图上进行比较
    
    参数：
        y_scores_list (list): 每个元素是一个模型对样本的预测概率（正类的概率）数组
        model_names (list): 每个模型的名称（字符串），与y_scores_list顺序对应
        y_true (array-like): 真实的标签（二进制，0或1）
    
    返回：
        None（显示并保存ROC比较图）
    '''
    # 设置图形和字体大小
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 14})
    
    # 绘制对角线（随机分类器的ROC）
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.5)')
    
    # 为每个模型计算ROC曲线和AUC
    for i, y_scores in enumerate(y_scores_list):

        # 计算FPR和TPR
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        # 计算AUC
        roc_auc = auc(fpr, tpr)
        
        # 格式化模型名称，保留两位小数的AUC值
        model_label = f"{model_names[i]} (AUC = {roc_auc:.4f})"
        
        # 绘制ROC曲线（使用不同的线条样式）
        line_style = '-'  # 默认实线
        if model_names[i] == model_ours:
            lw = 2  # 线宽
            alpha = 1  # 透明度
            color = '#C82423'  # 颜色
            # line_style = 'deepskyblue'  # 虚线
            plt.plot(fpr, tpr, 
                lw=lw,  # 线宽
                alpha=alpha,  # 透明度
                label=model_label,color=color,zorder=len(model_names)+1)
        else:
            lw = 2  # 线宽
            alpha = 1  # 透明度
            plt.plot(fpr, tpr, 
                    lw=lw,  # 线宽
                    alpha=alpha,  # 透明度
                    label=model_label,)
    
    # 设置图形属性
    plt.xlim([0.0, 1.0])  # 设置X轴范围
    plt.ylim([0.0, 1.05])  # 设置Y轴范围
    plt.xlabel('False Positive Rate', fontsize=fs_base,weight='bold')
    plt.ylabel('True Positive Rate', fontsize=fs_base,weight='bold')
    plt.title(title, fontsize=fs_base+2)
    plt.legend(loc='lower right', fontsize=fs_base)  # 图例在右下角
    
    # 设置网格线
    plt.grid(True, alpha=0.3)
    
    # # 添加文字标注
    # plt.figtext(0.5, 0.02, 'ROC curves for model comparison', 
    #             ha='center', fontsize=10, style='italic')
    # plt.savefig(f'images/ROC/{title}.png', dpi=300, bbox_inches='tight')
    
    # 保存图片
    plt.show()

    
    # 显示图形
def minmax_scale(x):   
    """
    Scale the input array to [0, 1]
    """
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)
def get_risk_dis_year(data,year,model):
    s=year*4
    return minmax_scale(get_cdf_dis(data,model)[:,:s].sum(axis=1))
def get_label_n_year(data,s):
    label = data[:,-2:].copy()
    # print(f"before process : {label[label[:,0]>=s,:]}")
    label[label[:,0]>=s,-1]=0# 对于发生在year之后的死亡或删失置为0，因为我们关注year之前的死亡。
    # print(f"after process : {label[label[:,0]>=s,:]}")
    return label
def get_CDF_con(data,model,t_obs):
    tensor = torch.tensor(data[:,:-2]).half().cuda()
    pdf = model.predict(tensor)
    cdf = 1-torch.cumprod(1 - pdf * torch.cat([torch.zeros(1), torch.ones(int(t_obs-1))]).cuda(), dim = 1)
    return cdf.cpu().detach().numpy()
def get_risk_CDF(cdf):
    risk_sum = cdf.sum(axis=1)
    return risk_sum
# RD['models'][5].predict(torch.tensor(data_test[:,:-2]).cuda().half())
def get_risk_con(data,model,t_obs):
    return minmax_scale(get_risk_CDF(get_CDF_con(data,model,t_obs)))
def get_risk_dis(data,model):
    return minmax_scale(get_cdf_dis(data,model).sum(axis=1))
def get_cdf_dis(data,model):
    try:
        tensor = torch.tensor(data[:,:-2]).half().cuda()
        pmf = model.predict(tensor)
    except:
        tensor = torch.tensor(data[:,:-2]).float().cuda()
        pmf = model.predict(tensor)
    cdf =  torch.cumsum(pmf, -1)
    return cdf.cpu().detach().numpy()
def plot_DCAs(xmin,xmax,data,risks,models_name,fs):
    bms=[]
    thresh_group = np.arange(0,1,0.01)
    y_label = data[:,-1]
    for risk in risks:
        bms.append(calculate_net_benefit_model(thresh_group, risk, y_label,))
    net_benefit_all, net_benefit_none = calculate_net_benefit_all(thresh_group, y_label)

    fig, ax = plt.subplots()

    ax = plot_DCA(ax, thresh_group, bms, models_name,net_benefit_all,net_benefit_none,xmin,xmax,fs=fs)
    # fig.savefig('fig1.png', dpi = 300)
    plt.show()

def minmax_df(df,by="row"):
    if by=="row":
        return df.apply(lambda x: minmax_scale(x),axis=1)
    elif by=="column":
        return df.apply(lambda x: minmax_scale(x),axis=0)
    else:
        raise ValueError("by should be either 'row' or 'column'")
def plot_heat_bar_rank(data,fs_base):
    # 处理数据
    df_text = pd.DataFrame(data).set_index('cancers')
    models = list(data.keys())[1:]

    # 提取均值和标准差
    df_mean = pd.DataFrame(index=df_text.index, columns=models)
    df_std = pd.DataFrame(index=df_text.index, columns=models)

    for model in models:
        for cancer in df_text.index:
            if df_text.loc[cancer, model]:
                mean_std = df_text.loc[cancer, model].split('±\n')
                df_mean.loc[cancer, model] = float(mean_std[0])
                if len(mean_std) > 1:
                    df_std.loc[cancer, model] = float(mean_std[1])

    # 计算排名 (不包括MEAN行)
    ranking_df = df_mean.drop('MEAN').copy()

    # 计算每个癌症类型中模型的排名（按降序排列）
    rankings = []
    for cancer in ranking_df.index:
        ranked_models = ranking_df.loc[cancer].sort_values(ascending=False).index
        rankings.append(list(zip(ranked_models, range(1, len(ranked_models)+1))))
        
    # 统计每个模型获得前三名的次数
    rank_counts = {model: {'1st': 0, '2nd': 0, '3rd': 0, 'other': 0} for model in models}

    for rank_list in rankings:
        for i, (model, rank) in enumerate(rank_list):
            if rank == 1:
                rank_counts[model]['1st'] += 1
            elif rank == 2:
                rank_counts[model]['2nd'] += 1
            elif rank == 3:
                rank_counts[model]['3rd'] += 1
            else:
                rank_counts[model]['other'] += 1

    # 创建绘图
    plt.figure(figsize=(20, 11))

    # 设置整体布局
    grid = plt.GridSpec(10, 1, hspace=0, height_ratios=[1, 0.01, 0.1, 0.01, 4, 0.01, 0.5, 0.01, 0.15, 0.01])
##########从下到上堆叠#######################
    # # 创建柱状图 - 展示排名
    # ax_bar = plt.subplot(grid[0])
    # ranks = ['1st', '2nd', '3rd', 'other']
    # colors = ["#0C4E9B", "#589FF3",  "#37AB78","#F3B169",]  
    # # colors = ["#"]# 金、银、铜、浅蓝
    # colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#589FF3'] 
    # colors = ['gold', 'silver', 'brown', '#589FF3'] 

    # # colors = [ "#76c68f", "#a9db7a", "#e5d354","#E6BB88"]  

    # # 为每个模型堆叠柱状图
    # bottom = np.zeros(len(models))
    # for i, rank in enumerate(ranks):
    #     counts = [rank_counts[model][rank] for model in models]
    #     x_pos = np.arange(len(models)) * 2
    #     ax_bar.bar(x_pos , counts, bottom=bottom, label=rank, color=colors[i],width=1,)
    #     ax_bar.set_xticks(x_pos)  
    #     bottom += counts


##########从下到上堆叠#######################

    # 修改后的代码 - 反转堆叠顺序
    ax_bar = plt.subplot(grid[0])
    ranks = ['other', '3rd', '2nd', '1st']  # 反转顺序
    colors = ['lightskyblue', 'brown', 'silver', 'gold']  # 对应的颜色也反转

    # 为每个模型堆叠柱状图（从底部开始堆叠，other在最底，1st在最顶）
    bottom = np.zeros(len(models))
    for i, rank in enumerate(ranks):
        counts = [rank_counts[model][rank] for model in models]
        x_pos = np.arange(len(models)) * 2
        ax_bar.bar(x_pos , counts, bottom=bottom, color=colors[i], width=0.8)
        bottom += counts

    # 添加图例时注意顺序要正确
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in colors[::-1]]  # 反转颜色顺序生成图例句柄
    labels = ['1st', '2nd', '3rd', 'other']  # 标签顺序保持不变
    ax_bar.legend(handles, labels, title='Rank',title_fontsize=fs_base, loc='upper right', bbox_to_anchor=(1.05, 1.0),fontsize=fs_base-6)

    ax_bar.set_ylabel('Counts',fontsize=fs_base,weight='bold')
    # ax_bar.set_title('Model Rankings Across Cancer Types', fontsize=14, pad=20)
    ax_bar.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)


    # 创建热力图 - 展示均值
    ax_heat = plt.subplot(grid[4])
    ax_cbar = plt.subplot(grid[8])

    # 自定义颜色映射
    colors = ["#6b1d1d", "#a33232", "#d86d3b", "#f1a34e", "#e5d354", "#a9db7a", "#76c68f", "#4cbf9d"]
    cmap = LinearSegmentedColormap.from_list("custom_diverging", colors)

    df_scale = minmax_df(df_mean,by="row")
    ax = sns.heatmap(df_scale.astype(float), 
                annot=df_text.values, 
                fmt='', 
                # cmap="cmap",
                cmap="Blues",
                # cmap="Reds",
                # cmap="RdBu_r",
                cbar=True,
                cbar_ax=ax_cbar,
                ax=ax_heat,
                annot_kws={'fontsize':fs_base},
                linewidths=0.5,
                cbar_kws={'label': 'Relative performance', 'orientation': 'horizontal','pad':0.1,}
                )
    ax_cbar.xaxis.label.set_size(20)
    ax_cbar.tick_params(labelsize=fs_base)
    # ax.figure.axes[-1].set_ylabel('Relative performance', size=20, weight='bold',loc='bottom',kws={ 'orientation': 'horizontal','pad':0.1})
    # ax_heat.set_title('Model Performance Across Cancer Types (Mean C-index ±\n SD)', fontsize=14)
    ax_heat.set_ylabel('Cancer Types',fontsize=fs_base,weight='bold')
    ax_heat.tick_params(axis='x', rotation=15)
    ax_heat.set_xticklabels(models, ha='center',fontsize=fs_base)
    ax_heat.set_yticklabels(df_text.index, ha='right',rotation=0,fontsize=fs_base)

    # # 添加分隔线区分MEAN行
    # ax_heat.axhline(y=len(df_text)-1, color='black', linewidth=1.5, linestyle='--')

    plt.tight_layout()