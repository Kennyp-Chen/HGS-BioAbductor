'''
画图
'''
import os,torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from sklearn.metrics import confusion_matrix,roc_auc_score, roc_curve,RocCurveDisplay
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler
import time 
import json

# js = "/Backup/home/chenyupeng/Results/Interpret/PRO-HCC-hcluster/AGG:cat-num_hgat:single-depth:2-l2:0.1-lr:0.01-glr:0-type_atten:additive-n_hid:200-gamma:0.99-patience:5-update_freq:1-type_know:hcluster-method:weighted-criterion:maxclust-num_clusters:59-divisor:10/bordas_dict.json"
# with open(js, 'r') as f:
#     bordas_dict = json.load(f)
# bordas_dict.keys()

# # 绘画横向直方图 for borada score
# import matplotlib.pyplot as plt
# import numpy as np
# alpha = 0.8
# edgecolor = "black"
# # 绘制横向直方图
# # 分别绘画，共三张图，并不合并三张图。字典中的值都是list
# # 1. node_borda 的 top10的分数，横坐标为node_borda的分数，纵坐标为node_name
# # 坐标轴range设置，横坐标范围为[0,1]
# node_borda = bordas_dict['node_borda']
# top10_node_borda = sorted(bordas_dict['node_borda'], reverse=True)[:10]
# top10_node_name = [bordas_dict['node_name'][i] for i in np.argsort(bordas_dict['node_borda'])[::-1][:10]]
# x = top10_node_borda
# y = top10_node_name
# plt.barh(y, x, color='yellow',alpha=alpha,edgecolor=edgecolor)
# plt.xlim(700, 800)
# plt.xlabel('Borda Score')
# plt.ylabel('Gene Name')
# plt.title('Top 10 Node Borda Score')
# plt.gca().invert_yaxis()  
# plt.show()
# # plt.close()
# # 2. hyperedge_borda 的 top10的分数，横坐标为hyperedge_borda的分数，纵坐标为hyperedge_name
# top10_hyperedge_borda = sorted(bordas_dict['hyperedge_borda'], reverse=True)[:10]
# top10_hyperedge_name = [bordas_dict['hyperedge_name'][i] for i in np.argsort(bordas_dict['hyperedge_borda'])[::-1][:10]]
# x = top10_hyperedge_borda
# y = top10_hyperedge_name
# plt.barh(y, x, color='g',alpha=alpha,edgecolor=edgecolor)
# plt.xlim(85, 110)
# plt.xlabel('Borda Score')
# plt.ylabel('Hyperedge Name')
# plt.title('Top 10 Hyperedge Borda Score')
# plt.gca().invert_yaxis()  
# plt.show()
# # plt.close()

# # 3. pair_borda 的 top10的分数，横坐标为pair_borda的分数，纵坐标为pair_name
# top10_pair_borda = bordas_dict['pair_borda'][:10]
# top10_pair_name = bordas_dict['pair_name'][:10]
# x = top10_pair_borda
# y = top10_pair_name
# plt.barh(y, x,color='b',alpha=alpha,edgecolor=edgecolor)
# plt.xlim(3, 4)
# plt.xlabel('Borda Score')
# plt.ylabel('Pair Name')
# plt.title('Top 10 Pair Borda Score')
# plt.gca().invert_yaxis()  
# plt.show()
# # plt.close()


def norm_test(df):
    '''
    df:gene*sample
    '''
    data = np.matrix(df).ravel().tolist()
    data = np.random.choice(np.squeeze(data), size=10000000)
    print(f"data standard deviation:",data[~np.isnan(data)].std())
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data, fill=True)
    plt.xlabel('Expression')
    plt.ylabel('Frequency')
    plt.title('Distribution of Expression Matrix')
    plt.show()
    
def get_risk_minmax(data,model):
    cdf = get_cdf(data,model)
    scaler = MinMaxScaler(feature_range=(0, 1))
    risk_sum = cdf.sum(axis=1)
    risk_minmax = scaler.fit_transform(risk_sum.reshape(-1, 1))
    return risk_minmax
def get_risk_mean(data,model):
    cdf = get_cdf(data,model)
    risk_mean = cdf.sum(axis=1)/cdf.shape[1]
    return risk_mean
def get_label_n_year(data,s):
    label = data[:,-2:].copy()
    # print(f"before process : {label[label[:,0]>=s,:]}")
    label[label[:,0]>=s,-1]=0# 对于发生在year之后的死亡或删失置为0，因为我们关注year之前的死亡。
    # print(f"after process : {label[label[:,0]>=s,:]}")
    return label
def get_death_CDF_year(pmf,s):
    cdf =  torch.cumsum(pmf, -1)
    return cdf[:,s].cpu().detach().numpy()
def get_cdf(data,model):
    '''
    Get the CDF on each time bin of the predicted risk for each patient.
    return np.array([
                    [cdf_11, cdf_12,..., cdf_1T],
                    [cdf_21, cdf_22,..., cdf_2T]
                    ...])
    where cdf_ij is the CDF of risk at time j for patient i.
    '''
    pmf = get_pmf(data,model)
    pmf = pmf.cpu().detach().numpy()
    cdf = np.cumsum(pmf, axis=1)
    # 限制最大最小值
    cdf[cdf>1]=1.
    cdf[cdf<0]=0.
    return cdf
def get_pmf(data,model):
    try:
        tensor = torch.tensor(data[:,:-2]).half().cuda()
        pmf = model.predict(tensor)
    except:
        tensor = torch.tensor(data[:,:-2]).float().cuda()
        pmf = model.predict(tensor)
    return pmf
def get_label_cdf_year(data,year,model):
    try:
        tensor = torch.tensor(data[:,:-2]).half().cuda()
        pmf = model.predict(tensor)
    except:
        tensor = torch.tensor(data[:,:-2]).float().cuda()
        pmf = model.predict(tensor)
    s = year*4 
    if s>pmf.shape[1]-2:
        s = int(pmf.shape[1]-2)
    print(f"years {year} = seasons {s}")
    label = get_label_n_year(data,s)
    cdf = get_death_CDF_year(pmf,s)
    return label,cdf

def get_CDF_con_year(data,year,model,t_obs):
    cdf = get_CDF_con(data,model,t_obs)
    s = year*4 
    if s>cdf.shape[1]-2:
        s = int(cdf.shape[1]-2)
    print(f"years {year} = seasons {s}")
    return cdf[:,s].cpu().detach().numpy()
def get_CDF_con(data,model,t_obs):
    tensor = torch.tensor(data[:,:-2]).half().cuda()
    pdf = model.predict(tensor)
    cdf = 1-torch.cumprod(1 - pdf * torch.cat([torch.zeros(1), torch.ones(int(t_obs-1))]).cuda(), dim = 1)
    return cdf.cpu().detach().numpy()
def get_risk_CDF(cdf):
    risk_sum = cdf.sum(axis=1)
    return risk_sum
def minmax_scale(x):   
    """
    Scale the input array to [0, 1]
    """
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)
def get_risk_con(data,model,t_obs):
    return minmax_scale(get_risk_CDF(get_CDF_con(data,model,t_obs)))

def get_risk_dis(data,model):
    return minmax_scale(get_cdf_dis(data,model).sum(axis=1))

def get_risk_dis_year(data,year,model):
    s=year*4
    return minmax_scale(get_cdf_dis(data,model)[:,:s].sum(axis=1))


def get_cdf_dis(data,model):
    try:
        tensor = torch.tensor(data[:,:-2]).half().cuda()
        pmf = model.predict(tensor)
    except:
        tensor = torch.tensor(data[:,:-2]).float().cuda()
        pmf = model.predict(tensor)
    cdf =  torch.cumsum(pmf, -1)
    return cdf.cpu().detach().numpy()


def get_HR(data,model):
    '''
    for deep surv
    '''
    tensor = torch.tensor(data[:,:-2]).half().cuda()
    h = model.predict(tensor).cpu().detach().numpy()
    return np.exp(h) 

def plot_ROC(y_true,y_scores,method=""):
    display = RocCurveDisplay.from_predictions(
        y_true, y_scores,
        name=f"Death vs. Survival",    
        plot_chance_level=True,
    )
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"{method} ROC curves",
    )


def plot_cancer_comparison(results,x_min=0.5,x_max=0.9):

    # 数据准备
    cancer_name = results['cancer']
    model_data = {k:v for k,v in results.items() if k != 'cancer'}
    
    # 数据解析
    entries = []
    max_ = 0.0
    min_ = 1.0
    for model, value in model_data.items():
        mean, std = map(float, value.split('±'))
        entries.append((model, mean, std))
        max_ = mean if mean > max_ else max_
        min_ = mean if mean < min_ else min_
    # entries.sort(key=lambda x: x[1], reverse=True)  # 降序排列
    if max_>0.8:
        x_max = 1.0
    if min_<0.5:
        x_min = 0.4
    # 可视化参数
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(entries))
    # colors = plt.cm.tab20(np.linspace(0, 1, len(entries)))
    #### Just For HGS 3 Know ####
    similar_colors = [
    [0.2, 0.4, 0.8, 1],  # 深蓝
    [0.4, 0.6, 1.0, 1],  # 中蓝
    [0.6, 0.8, 1.0, 1],  # 浅蓝
    ]

    # other_colors = plt.cm.tab20(np.linspace(0, 1, len(entries)-2 ))[1:1 + (len(entries) - 3)]
    other_colors = plt.cm.tab20(np.linspace(0, 1, len(entries) ))[1:1 + (len(entries) - 3)]

    colors = np.vstack([similar_colors, other_colors])
    ################

    # 绘制主图

    '''
    colors=
    array([[0.12156863, 0.46666667, 0.70588235, 1.        ],
       [1.        , 0.49803922, 0.05490196, 1.        ],
       [0.17254902, 0.62745098, 0.17254902, 1.        ],
       [0.83921569, 0.15294118, 0.15686275, 1.        ],
       [0.58039216, 0.40392157, 0.74117647, 1.        ],
       [0.76862745, 0.61176471, 0.58039216, 1.        ],
       [0.96862745, 0.71372549, 0.82352941, 1.        ],
       [0.78039216, 0.78039216, 0.78039216, 1.        ],
       [0.85882353, 0.85882353, 0.55294118, 1.        ],
       [0.61960784, 0.85490196, 0.89803922, 1.        ]])
    '''

    bars = plt.barh(y_pos, [e[1] for e in entries], 
                   xerr=[e[2] for e in entries], 
                   color=colors, alpha=0.8, 
                   capsize=5, ecolor='#555555')
    
    # 智能标签布局引擎
    for idx, (model, mean, std) in enumerate(entries):
        bar = bars[idx]
        total_width = mean + std
        
        # 可用空间检测（右侧）
        right_space = x_max - total_width  # 假设x轴上限0.9
        
        # 标签位置决策树
        if right_space > 0.05:  # 情况1：右侧有足够空间
            x_pos = mean + 0.01
            ha = 'left'
            va = 'center'
        elif mean - std > 0.55:  # 情况2：左侧有空间
            x_pos = mean - 0.01
            ha = 'right'
            va = 'center'
        else:  # 情况3：上下错位显示
            x_pos = mean + std + 0.005
            ha = 'left'
            va = 'bottom' if idx % 2 else 'top'
        
        # 绘制标签（带白色描边防遮挡）
        plt.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f"{mean:.3f} ± {std:.3f}",
                ha=ha, va=va, fontsize=9,
                color='black', fontweight='normal',
                path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])
    
    # 坐标轴美化
    plt.yticks(y_pos, [e[0] for e in entries], fontsize=11)
    plt.xticks(np.arange(0.5, 0.91, 0.1), fontsize=10)
    plt.xlim(x_min, x_max)
    
    # 辅助元素
    plt.title(f"Performance Comparison on {cancer_name}", fontsize=14, pad=20)
    plt.xlabel("C-index", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    
    # # 边框清理
    # for spine in ['top', 'right']:
    #     plt.gca().spines[spine].set_visible(False)
    plt.gca().invert_yaxis()  # 使最高性能显示在最上方
    plt.tight_layout()
    return plt
def batch_plot_cancer_comparisons(dict_results, output_dir='cancer_comparisons'):
    """
    批量生成每个癌症类型（包含均值）的模型对比图
    参数：
        dict_results : 包含所有数据的字典
        output_dir   : 输出图片的保存目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有癌症类型（包括均值）
    cancer_list = dict_results['cancers']
    
    # 获取模型列表（排除'cancers'）
    models = [k for k in dict_results.keys() if k != 'cancers']
    
    # 遍历每个癌症类型
    for cancer_idx, cancer_name in enumerate(cancer_list):
        # 准备数据容器
        results = {'cancer': cancer_name}
        
        # 收集每个模型在该癌症下的数据
        for model in models:
            # 检查数据有效性
            if len(dict_results[model]) != len(cancer_list):
                raise ValueError(f"Model {model}数据长度不匹配")
                
            # 提取数值并保留原始字符串格式
            value_str = dict_results[model][cancer_idx]
            results[model] = value_str
        
        # 生成单个癌症对比图（使用优化后的排序版本）
        plot = plot_cancer_comparison(results)

        
        # 构建保存路径
        filename = f"{cancer_name}_comparison.png".replace(' ', '_')
        save_path = os.path.join(output_dir, filename)
        
        # 保存图片
        plot.savefig(save_path, dpi=300, bbox_inches='tight')
        plot.close()
        
    print(f"成功生成 {len(cancer_list)} 张对比图，保存至：{output_dir}")


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

def plot_DCA(ax, thresh_group, net_benefit_models, model_names, net_benefit_all, net_benefit_none,fs=14):
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
    
    # 动态调整Y轴范围
    min_val, max_val = 0, 0
    
    for i, (model_benefit, name) in enumerate(zip(net_benefit_models, model_names)):
        # 异常值
        clipped_benefit = np.clip(model_benefit, -0.1, None)
        
        model_min = np.nanmin(clipped_benefit) 
        model_max = np.nanmax(clipped_benefit)
        min_val = min(min_val, model_min)
        max_val = max(max_val, model_max)
        
        # 绘制曲线
        ax.plot(thresh_group, clipped_benefit, 
                color=colors[i], linewidth=2.5, 
                label=name)
    
    padding = max(0.05, (max_val - min_val) * 0.1)
    ax.set_ylim(min_val - padding, max_val + padding)

    ax.set_xlim(0, 1)
    ax.set_xlabel('Threshold Probability', fontsize=fs)
    ax.set_ylabel('Net Benefit', fontsize=fs)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 图例移到图形外避免遮挡
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=3, frameon=False)
    
    return ax


# 批次效应pca可视化
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind,pearsonr,mannwhitneyu
from statsmodels.stats.multitest import multipletests

def plot_pca(data, batch_labels, names_dict, title=""):
    '''
    data: 2D array of features [n_samples, n_features]
    batch_labels: 1D array of labels for each sample
    names: 映射批次标签到名称的字典
                如: {0: 'Jiang', 1: 'Xing', 2: 'Gao'}
    '''
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(data)
    

    plt.figure(figsize=(10, 8))
    
    # 获取唯一批次标签
    unique_batches = np.unique(batch_labels)
    
    # 创建离散颜色映射
    color_map = plt.cm.get_cmap('viridis', len(unique_batches))
    batch_colors = {}
    
    # 为每个批次创建颜色映射
    for i, batch in enumerate(unique_batches):
        batch_colors[batch] = color_map(i / (len(unique_batches) - 1))
    
    # 为每个批次单独绘制散点图（这样才能设置不同标签）
    for batch in unique_batches:
        mask = batch_labels == batch
        color = batch_colors[batch]
        plt.scatter(
            pca_coords[mask, 0], 
            pca_coords[mask, 1], 
            c=[color],
            alpha=0.7,
            edgecolor='w',
            label=names_dict.get(batch, f'Batch {batch}')  # 使用批次名称作为标签
        )
    
    # 添加解释方差比
    var_ratio = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({var_ratio[0]:.1%} variance)", fontsize=12)
    plt.ylabel(f"PC2 ({var_ratio[1]:.1%} variance)", fontsize=12)
    
    # 添加批次中心点（黑色叉号）
    for batch in unique_batches:
        mask = batch_labels == batch
        center = pca_coords[mask].mean(axis=0)
        plt.scatter(
            center[0], center[1],
            marker='x',
            s=100,  # 符号大小
            c='black',
            edgecolor='w',
            zorder=5  # 确保在最上层
        )
    
    # 创建自定义图例（使用名称和颜色）
    legend = plt.legend(
        title="Batch",
        fontsize=10,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor='gray'
    )
    
    # 设置图例标题格式
    legend.get_title().set_fontsize(12)
    
    # 添加主标题
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格线
    plt.grid(alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    plt.show()
def plot_scatter_regression(data,title='Correlation between Absolute Protein Abundance Change and Model Interpretability'):
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
    # 设置默认点边缘为无，显著点添加红色边缘
    edgecolors = np.where(data['adj_p'] < 0.05, 'black', 'none')
    linewidths = np.where(data['adj_p'] < 0.05, 1.0, 0)
    
    scatter = ax.scatter(
        x='XAI_score', 
        y='abs_log2FC', 
        c=data['adj_p'],        # 颜色表示调整后的p值
        # cmap=cmap,               # 使用红色映射
        norm=log_norm,           # 对数标准化
        # cmap='coolwarm',   # 红蓝渐变色
        alpha=1,         # 透明度
        edgecolor=edgecolors,     # 点边框
        linewidth=linewidths,     # 边框粗细
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
    text_box = f'Pearson r = {corr_coef:.2f}\np-value = {p_value:.2e}\nSignificant: {num_significant}/{len(data)}'
    ax.text(0.05, 0.95, text_box, 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


    # 颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Adjusted p-value', rotation=270, labelpad=15)
    
    # 添加颜色条刻度标记（特别是0.05阈值）
    cbar_tick_values = np.logspace(
        np.floor(np.log10(data['adj_p'].min())), 
        np.log10(data['adj_p'].max()), 
        num=5
    )
    cbar_tick_values = np.sort(np.append(cbar_tick_values, 0.05))
    cbar.set_ticks(cbar_tick_values)
    cbar.set_ticklabels([f"{v:.3f}" if v >= 0.001 else f"{v:.2e}" for v in cbar_tick_values])

    # 坐标轴美化
    ax.set_xlabel('XAI Score (Feature Importance)', fontsize=12, labelpad=10)
    ax.set_ylabel('log2 Fold Change', fontsize=12, labelpad=10)
    ax.set_title(title, 
                fontsize=13, pad=15)

    # 网格线
    ax.grid(True, linestyle='--', alpha=0.4)
    sns.despine(trim=True)

    # 图例
    # ax.legend(loc='lower right', frameon=True)
    ax.legend( frameon=True,labels=['Protein', f'y = {np.round(slope, 5)}x + {np.round(intercept, 5)}'])

    plt.tight_layout()
    plt.show()

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
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
import numpy as np

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

def plot_ROC(y_true,y_scores,method=""):
    display = RocCurveDisplay.from_predictions(
        y_true, y_scores,
        name=f"Death vs. Survival",    
        plot_chance_level=True,
    )
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"{method} ROC curves",
    )
    
def get_cdf_risk_minmax(data,model):
    cdf = get_cdf(data,model)
    scaler = MinMaxScaler(feature_range=(0, 1))
    risk_sum = cdf.sum(axis=1)
    risk_minmax = scaler.fit_transform(risk_sum.reshape(-1, 1))
    return risk_minmax
def get_risk_mean(data,model):
    cdf = get_cdf(data,model)
    risk_mean = cdf.sum(axis=1)/cdf.shape[1]
    return risk_mean


def plot_ROCs(y_scores_list, model_names, y_true,model_ours="HGS"):
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
    plt.rcParams.update({'font.size': 12})
    
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
            color = 'dodgerblue'  # 颜色
            # line_style = 'deepskyblue'  # 虚线
            plt.plot(fpr, tpr, 
                lw=lw,  # 线宽
                alpha=alpha,  # 透明度
                label=model_label,color=color)
        else:
            lw = 1  # 线宽
            alpha = 1  # 透明度
            plt.plot(fpr, tpr, 
                    lw=lw,  # 线宽
                    alpha=alpha,  # 透明度
                    label=model_label,)
    
    # 设置图形属性
    plt.xlim([0.0, 1.0])  # 设置X轴范围
    plt.ylim([0.0, 1.05])  # 设置Y轴范围
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curves for model comparison', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)  # 图例在右下角
    
    # 设置网格线
    plt.grid(True, alpha=0.3)
    
    # # 添加文字标注
    # plt.figtext(0.5, 0.02, 'ROC curves for model comparison', 
    #             ha='center', fontsize=10, style='italic')
    
    # 保存图片
    plt.savefig('multi_model_roc_comparison.png', dpi=300, bbox_inches='tight')
    
    # 显示图形
    plt.show()







# if __name__ == "__main__":
#     # 数据

#     for DATA in ['PRO','RNA']:
#         if DATA == 'PRO':
#             dict_results = {
#             'cancers': ['CCRCC', 'HaNSCC', 'LA', 'LSCC', 'UCEC', 'HCC', 'GBM', 'PDA','MEAN'],
#             'HGS - Hcluster':         
#             ['0.8614±0.0958', '0.8447±0.0611', '0.8179±0.0858', '0.7131±0.1298', 
#             '0.8383±0.1306', '0.7393±0.0497', '0.74±0.063', '0.7621±0.0524','0.7896±0.0536'],        
            
#             # 'HGS - Reactome': ['0.8466±0.0932', '0.7552±0.1341', '0.8496±0.0945', '0.7703±0.0981', 
#             # '   ', '0.6897±0.0415', '0.6517±0.0888', '0.7064±0.0757','0.7388±0.0628'],
#             'HGS - Reactome': ['0.833±0.082', '0.7628±0.0751', '0.8644±0.0251', '0.6606±0.1328', '0.784±0.1592', 
#             '0.7064±0.0294', '0.671±0.0549', '0.7212±0.05','0.7504±0.0692'],
            
#             # 'HGS - STRING': ['0.8243±0.1285', '0.8132±0.0739', '0.8228±0.0699', '0.6961±0.154', 
#             # '0.8423±0.1282', '0.7258±0.0481', '0.6555±0.0736', '0.7124±0.0523','0.7616±0.0672'],
#             'HGS - STRING': ['0.8584±0.0649', '0.8301±0.088', '0.8216±0.089', '0.6875±0.1302', '0.8506±0.0984', 
#             '0.713±0.0498', '0.7054±0.0802', '0.7321±0.0424','0.7749±0.0671'],

#             'SHINE':['0.8174±0.1006', '0.6555±0.103', '0.8016±0.0693', '0.5829±0.1067', '0.5291±0.2462', 
#             '0.6994±0.0574', '0.6565±0.0657', '0.7047±0.0605','0.6809±0.0922'],
#             'Pnet':['0.6669±0.0749', '0.5856±0.1353', '0.6433±0.1245', '0.4383±0.1912',
#             '0.4791±0.2088', '0.7277±0.0606', '0.5234±0.0993', '0.5909±0.0698','0.5819±0.0915'],

#             'DH': ['0.8343±0.099', '0.7936±0.0812', '0.8068±0.1354', '0.6676±0.0702', '0.8098±0.1069', 
#                 '0.7165±0.0471', '0.6316±0.0706', '0.7161±0.033', '0.747±0.0697'],

#             'DS': ['0.8291±0.0886', '0.7902±0.082', '0.8258±0.0652', '0.5925±0.132', '0.8346±0.1325', 
#             '0.7142±0.0522', '0.6835±0.0528', '0.729±0.0433','0.7499±0.0804'],

#             'DR': ['0.7873±0.1114', '0.7793±0.0897', '0.8254±0.0803', '0.672±0.1554', 
#             '0.8389±0.0808', '0.6723±0.0708', '0.6719±0.0711', '0.6857±0.068','0.7416±0.0686'],

#             "RSF": [
#                 "0.7589±0.1167",
#                 "0.7452±0.0774",
#                 "0.8215±0.0803",
#                 "0.7051±0.1633",
#                 "0.8223±0.1133",
#                 "0.7363±0.0565",
#                 "0.6562±0.0895",
#                 "0.6904±0.0755",
#                 "0.742±0.0553"
#             ],
            
#             "COX": [
#                 "0.8068±0.0835",
#                 "0.7775±0.0855",
#                 "0.7753±0.0642",
#                 "0.6261±0.0913",
#                 "0.8091±0.1228",
#                 "0.731±0.053",
#                 "0.6902±0.0604",
#                 "0.6812±0.028",
#                 "0.7371±0.0621"
#                 ],
#             }

#         elif DATA == 'RNA':
#             dict_results = {
#             'cancers': ['LIHC', 'STAD', 'BLCA', 'OV', 'LUSC', 'LGG', 'LUAD', 'KIRC', 'HNSC', 'BRCA', 'MEAN'],
#             'HGS - Hcluster': ['0.778±0.0426', '0.6634±0.0473', '0.7123±0.0393', '0.6328±0.0458', '0.639±0.052', 
#             '0.8595±0.0165', '0.7108±0.0493', '0.7157±0.0356', '0.7318±0.0245', '0.7592±0.0313','0.7202±0.0648'],
#             'HGS - Reactome': ['0.761±0.0474', '0.6449±0.0518', '0.6566±0.0394', '0.5873±0.0174', '0.6245±0.0367', 
#             '0.846±0.0122', '0.6986±0.0291', '0.6933±0.0354', '0.7181±0.0373', '0.7285±0.0404','0.6959±0.0703'],
#             'HGS - STRING': ['0.7556±0.0461', '0.6232±0.0451', '0.6591±0.038', '0.6254±0.0261', '0.5591±0.0681', 
#             '0.8422±0.0107', '0.7081±0.0401', '0.6915±0.0402', '0.706±0.0341', '0.7129±0.0491','0.6883±0.0744'],
#             'SHINE':
#             ['0.7561±0.0593', '0.6241±0.0481', '0.6428±0.0271', '0.6145±0.0308', '0.6227±0.0336',
#             '0.8483±0.0131', '0.675±0.0376', '0.6799±0.0528', '0.6903±0.03', '0.7264±0.0885','0.688±0.0692'],
            
#             'Pnet': ['0.7159±0.0598', '0.5531±0.0617', '0.5885±0.0427', '0.5646±0.0546', '0.5025±0.0326',
#             '0.833±0.0211', '0.6209±0.0446', '0.6667±0.0288', '0.6102±0.0324', '0.6248±0.0446','0.628±0.0886'],
            
#             'DH': ['0.7699±0.0517', '0.6072±0.0561', '0.6782±0.037', '0.599±0.0615', '0.6099±0.0383', 
#             '0.8519±0.0175', '0.6977±0.0325', '0.6866±0.0376', '0.7057±0.0267', '0.7413±0.0296','0.6947±0.0755'],

#             'DS': ['0.7483±0.0417', '0.6042±0.0517', '0.67±0.0324', '0.5607±0.064', '0.619±0.0356', '0.8505±0.0169', 
#                 '0.6872±0.0499', '0.7031±0.0436', '0.6914±0.0395', '0.7372±0.0434','0.6872±0.0781'],
#             'DR': ['0.6946±0.0908', '0.622±0.054', '0.6875±0.0638', '0.5839±0.0483', '0.605±0.0507', 
#             '0.8375±0.0562', '0.6236±0.0352', '0.6486±0.0596', '0.655±0.0618', '0.7202±0.0461','0.6678±0.0694'],
#             'RSF': ['0.736±0.0653', '0.6012±0.0494', '0.6611±0.0335', '0.568±0.0384', '0.5566±0.0524',
#             '0.8292±0.0169', '0.6611±0.0318', '0.6751±0.0352', '0.668±0.021', '0.6696±0.0371','0.6626±0.0757'],

#             'COX': ['0.6928±0.0485', '0.5585±0.0631', '0.6551±0.0355', '0.6313±0.034', '0.6183±0.0422', 
#                 '0.8412±0.0205', '0.653±0.0306', '0.6809±0.0368', '0.6718±0.0338', '0.7481±0.0259','0.6751±0.0727'],

#             } 
#         # 生成所有对比图
#         batch_plot_cancer_comparisons(dict_results, output_dir=f'model_comparisons/{DATA}/')

