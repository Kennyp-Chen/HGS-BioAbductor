from captum.attr import (
    LayerConductance,
    LayerIntegratedGradients,
    LayerDeepLiftShap,
    LayerGradientShap,
    IntegratedGradients,
    DeepLiftShap,
    GradientShap
)
from gprofiler import GProfiler
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
# import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn as nn
from lifelines import CoxPHFitter
from scipy.stats import ttest_ind
from functools import partial
import multiprocessing as mul
from tqdm import tqdm
import gseapy as gp
import logging
from Bio import Entrez
import time
from scipy.stats import pearsonr
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import json
class Interpreter:
    def __init__(self, model: nn.Module, input: pd.DataFrame, target: int = None):
        """
        model: Pytorch neural network
        input: Dataframe with samples and features, 2D
        """
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        self.model = model
        if isinstance(input, pd.DataFrame):
            self.input = torch.Tensor(input.values).cuda()
        if isinstance(input, np.ndarray):
            self.input = torch.Tensor(input).cuda()
        self.target = target
        # hook = Hook(self.model.cuda(), backward=False, hook_to_module_input=False)
        # self.CDF = hook.CDFhook()
    def layer_conductance(self, layer, l2_norm=True, baseline=None):
        """
        LayerConductance method
        """
        hook = Hook(self.model.cuda(), backward=False, hook_to_module_input=False)
        CDFhook = hook.CDFhook()
        cond = LayerConductance(self.model.cuda(), layer)
        cond_vals = cond.attribute(self.input, target=self.target, baselines=baseline).detach().cpu().numpy()
        hyperedge_cond = cond_vals.sum(-1)
        if l2_norm:
            hyperedge_cond /= np.linalg.norm(hyperedge_cond.astype(np.float32), ord=2)
        hook.close()
        return hyperedge_cond

    def layer_integrated_gradients(self, layer, l2_norm=True, baseline=None, n_steps=50):
        """
        LayerIntegratedGradients method
        """
        hook = Hook(self.model.cuda(), backward=False, hook_to_module_input=False)
        CDFhook = hook.CDFhook()

        lig = LayerIntegratedGradients(self.model.cuda(), layer)
        attributions, delta = lig.attribute(
            self.input,
            baselines=baseline,
            target=self.target,
            return_convergence_delta=True,
            n_steps=n_steps,
        )

        attr_vals = attributions.detach().cpu().numpy()

        hyperedge_attr = attr_vals.sum(-1)
        if l2_norm:
            try:
                norm = np.linalg.norm(hyperedge_attr.astype(np.float32), ord=2)
            except np.linalg.LinAlgError:
                norm = np.sqrt(np.sum(hyperedge_attr.astype(np.float32) ** 2))
            if norm != 0:
                hyperedge_attr /= norm
        
        hook.close()
        return hyperedge_attr

    def layer_deeplift_shap(self, layer, l2_norm=True, baseline=None):
        """
        LayerDeepLiftShap method
        """
        hook = Hook(self.model.cuda(), backward=False, hook_to_module_input=False)
        CDFhook = hook.CDFhook()

        if baseline is None:
            raise ValueError("Baselines must be provided for DeepLiftShap.")
        lds = LayerDeepLiftShap(self.model.cuda(), layer)
        attributions = lds.attribute(
            self.input, baselines=baseline, target=self.target
        )
        attr_vals = attributions.detach().cpu().numpy()
        hyperedge_attr = attr_vals.sum(-1)
        if l2_norm:
            try:
                norm = np.linalg.norm(hyperedge_attr.astype(np.float32), ord=2)
            except np.linalg.LinAlgError:
                norm = np.sqrt(np.sum(hyperedge_attr.astype(np.float32) ** 2))
            if norm != 0:
                hyperedge_attr /= norm
        
        hook.close()
        return hyperedge_attr

    def layer_gradient_shap(self, layer, l2_norm=True, baseline=None, stdevs=0.0, n_samples=20):
        """
        LayerGradientShap method
        """
        hook = Hook(self.model.cuda(), backward=False, hook_to_module_input=False)
        CDFhook = hook.CDFhook()
        gs = LayerGradientShap(self.model.cuda(), layer)
        attributions = gs.attribute(
            self.input,
            baselines=baseline,
            target=self.target,
            n_samples=n_samples,
            stdevs=stdevs,
        )
        attr_vals = attributions.detach().cpu().numpy()
        hyperedge_attr = attr_vals.sum(-1)
        if l2_norm:
            try:
                norm = np.linalg.norm(hyperedge_attr.astype(np.float32), ord=2)
            except np.linalg.LinAlgError:
                norm = np.sqrt(np.sum(hyperedge_attr.astype(np.float32) ** 2))
            if norm != 0:
                hyperedge_attr /= norm
        
        hook.close()
        return hyperedge_attr
    
    def integrated_gradients(self, l2_norm=True, baseline=None, n_steps=50):
        """
        IntegratedGradients method
        """
        hook = Hook(self.model.cuda(), backward=False, hook_to_module_input=False)
        CDFhook = hook.CDFhook()

        ig = IntegratedGradients(self.model.cuda())
        attributions, delta = ig.attribute(
            self.input,
            baselines=baseline,
            target=self.target,
            return_convergence_delta=True,
            n_steps=n_steps,
        )
        feature_attr = attributions.detach().cpu().numpy()
        if l2_norm:
            try:
                norm = np.linalg.norm(feature_attr.astype(np.float32), ord=2)
            except np.linalg.LinAlgError:
                norm = np.sqrt(np.sum(feature_attr.astype(np.float32) ** 2))
            if norm != 0:
                feature_attr /= norm
        
        hook.close()
        return feature_attr
    
    def gradient_shap(self, l2_norm=True, baseline=None, stdevs=0.0, n_samples=20):
        """
        GradientShap method
        """
        hook = Hook(self.model.cuda(), backward=False, hook_to_module_input=False)
        CDFhook = hook.CDFhook()
        gs = GradientShap(self.model.cuda())
        attributions = gs.attribute(
            self.input,
            baselines=baseline,
            target=self.target,
            n_samples=n_samples,
            stdevs=stdevs,
        )
        feature_attr = attributions.detach().cpu().numpy()
        if l2_norm:
            try:
                norm = np.linalg.norm(feature_attr.astype(np.float32), ord=2)
            except np.linalg.LinAlgError:
                norm = np.sqrt(np.sum(feature_attr.astype(np.float32) ** 2))
            if norm != 0:
                feature_attr /= norm
        
        hook.close()
        return feature_attr
    
    def deeplift_shap(self, l2_norm=True, baseline=None):
        """
        DeepLiftShap method
        """
        hook = Hook(self.model.cuda(), backward=False, hook_to_module_input=False)
        CDFhook = hook.CDFhook()

        if baseline is None:
            raise ValueError("Baselines must be provided for DeepLiftShap.")
        ds = DeepLiftShap(self.model.cuda())
        attributions = ds.attribute(
            self.input, baselines=baseline, target=self.target
        )
        feature_attr = attributions.detach().cpu().numpy()
        if l2_norm:
            try:
                norm = np.linalg.norm(feature_attr.astype(np.float32), ord=2)
            except np.linalg.LinAlgError:
                norm = np.sqrt(np.sum(feature_attr.astype(np.float32) ** 2))
            if norm != 0:
                feature_attr /= norm
        
        hook.close()
        return feature_attr
    
    # def group_risk_divide(self,):
    #     hook = Hook(self.model.cuda(), backward=False, hook_to_module_input=False)
    #     CDFhook = hook.five_years_hook()
    #     risk_5years = self.model(self.input)
    #     median_risk = risk_5years.median()
    #     high_risk_group = (risk_5years >= median_risk).detach().cpu().numpy() 
    #     low_risk_group = (risk_5years < median_risk).detach().cpu().numpy()
    #     return high_risk_group,low_risk_group,risk_5years
    def survival(self, prefix, survival_time, survival_status, interval,risk_method="time_median"):
        """
        Survival analysis for model
        """
        hook = Hook(self.model.cuda(), backward=False, hook_to_module_input=False)
        if risk_method == 'time_median':
            CDFhook = hook.time_median_hook()
        elif risk_method == 'time_sum':
            CDFhook = hook.CDFhook()
        # CDFhook = hook.five_years_hook()
        risk = self.model(self.input)
        median_risk = risk.median()
        print(f"{risk_method} median risk:{median_risk}")
        high_risk_group = (risk >= median_risk).detach().cpu().numpy() 
        low_risk_group = (risk < median_risk).detach().cpu().numpy()
        T_high = survival_time[high_risk_group]
        T_low = survival_time[low_risk_group]
        E_high = survival_status[high_risk_group]
        E_low = survival_status[low_risk_group]
        results = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)
        
        # 输出p值
        print(f" Log-rank p-value: {results.p_value}")
        print(f"Plotting survival curve ")
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        
        plt.figure(figsize=(10, 6), dpi=300)
        T_high *=(interval/30)
        T_low *=(interval/30)
        T_high[T_high > 60] = 60
        kmf_high.fit(T_high, event_observed=E_high, label=f'High risk')
        kmf_low.fit(T_low, event_observed=E_low, label=f'Low risk')
        
        kmf_high.plot_survival_function()
        kmf_low.plot_survival_function()
        
        plt.title(f"Survival Curve  (p-value = {results.p_value})")
        plt.xlabel("Time (months)")
        plt.xlim(0, 60)
        plt.ylabel("Survival Probability")
        # output_dir = 'survival'
        # if not os.path.exists(output_dir):
        #     print(f"Creating output directory: {output_dir}")
        #     os.makedirs(output_dir)
        plt.savefig( f'{prefix}-survival.svg')
        plt.close()
        hook.close()
        
    def average_sort_attr(self, attr):
        average_attr = np.mean(attr, axis=0)
        ascending_indices = np.argsort(np.abs(average_attr))
        return ascending_indices[::-1]
    
    def attr_score(self, *args):
        """
        **args: tuples of (descending indices, ascending indices) for each method
        """

        num_methods = len(args)
        N = args[0].shape[0]  # Number of features from the first set of indices (assuming all are the same length)


        borda_scores = np.zeros(N)

        # Loop through each method's descending and ascending indices
        for method in range(num_methods):
            desc_indices = args[method]

            # Update positive Borda scores using descending indices
            for position, feature_index in enumerate(desc_indices):
                borda_scores[feature_index] += N - position

            # Update negative Borda scores using ascending indices

        # Final Borda scores: Choose positive score if it's greater, otherwise take negative with a minus sign
        borda_scores
        return borda_scores
    
    # def all_borda_rank(self, layer:nn.Module, baselineTensor:torch.tensor, lr_norm=True, IGsteps=50, GSsamples=50):
    #     """
    #     ene to end method to calculate borda ranks.
    #     baselinetensor: torch.zeros(feature_num)/ means of data train features
    #     """
    #     Gradientbaseline = baselineTensor.unsqueeze(0).expand(self.input.shape).cuda()
    #     DLbaseline = baselineTensor.unsqueeze(0).expand(32, self.input.shape[1]).cuda()
    #     ## train data mean
    
    #     layerIG = self.average_sort_attr(self.layer_integrated_gradients(layer, baseline=Gradientbaseline, l2_norm=lr_norm, n_steps=IGsteps))
    #     nodeIG = self.average_sort_attr(self.integrated_gradients(l2_norm=True, baseline=Gradientbaseline, n_steps=IGsteps))
    #     # layerGS =  self.average_sort_attr(self.layer_gradient_shap(layer, baseline = Gradientbaseline, l2_norm=lr_norm, n_samples=GSsamples))
    #     # nodeGS =  self.average_sort_attr(self.gradient_shap(baseline = Gradientbaseline, l2_norm=lr_norm, n_samples=GSsamples)) 
    #     layerDL = self.average_sort_attr(self.layer_deeplift_shap(layer, baseline=DLbaseline, l2_norm=lr_norm))
    #     nodeDL = self.average_sort_attr(self.deeplift_shap(l2_norm=lr_norm, baseline=DLbaseline))
    #     hyperedge_attr_score = self.attr_score(layerIG, layerDL)
    #     node_attr_score = self.attr_score(nodeIG, nodeDL)
    #     return node_attr_score, hyperedge_attr_score
    # 低显存
    def hyperedge_attr_score_(self,layer,Gradientbaseline,DLbaseline,lr_norm=True,IGsteps=50):
        return self.attr_score(
            self.average_sort_attr(self.layer_integrated_gradients(layer, baseline=Gradientbaseline, l2_norm=lr_norm, n_steps=IGsteps)),
            self.average_sort_attr(self.layer_deeplift_shap(layer, baseline=DLbaseline, l2_norm=lr_norm))
        )
    def node_attr_score_(self,Gradientbaseline,DLbaseline,lr_norm=True,IGsteps=50):
        return self.attr_score(
            self.average_sort_attr(self.integrated_gradients(l2_norm=True, baseline=Gradientbaseline, n_steps=IGsteps)),
            self.average_sort_attr(self.deeplift_shap(l2_norm=lr_norm, baseline=DLbaseline))
        )
    def all_borda_rank(self, layer:nn.Module, baselineTensor:torch.tensor, lr_norm=True, IGsteps=50, GSsamples=50):
        """
        ene to end method to calculate borda ranks.
        baselinetensor: torch.zeros(feature_num)/ means of data train features
        """
        Gradientbaseline = baselineTensor.unsqueeze(0).expand(self.input.shape).cuda()
        DLbaseline = baselineTensor.unsqueeze(0).expand(32, self.input.shape[1]).cuda()
        
        return self.node_attr_score_(Gradientbaseline,DLbaseline,lr_norm,IGsteps), self.hyperedge_attr_score_(layer,Gradientbaseline,DLbaseline,lr_norm,IGsteps)

    def hypergraph_borda_rank(self, nodelayer:nn.Module, hyperedgelayer:nn.Module, baselineTensor:torch.tensor, lr_norm=True, IGsteps=50, GSsamples=50):
        """
        ene to end method to calculate borda ranks.
        baselinetensor: torch.zeros(feature_num)/ means of data train features
        """
        Gradientbaseline = baselineTensor.unsqueeze(0).expand(self.input.shape).cuda()
        DLbaseline = baselineTensor.unsqueeze(0).expand(32, self.input.shape[1]).cuda()
        ## train data mean

        layerIG = self.average_sort_attr(self.layer_integrated_gradients(hyperedgelayer, baseline=Gradientbaseline, l2_norm=lr_norm, n_steps=IGsteps))
        nodeIG = self.average_sort_attr(self.layer_integrated_gradients(nodelayer,l2_norm=True, baseline=Gradientbaseline, n_steps=IGsteps))
        layerGS =  self.average_sort_attr(self.layer_gradient_shap(hyperedgelayer, baseline = Gradientbaseline, l2_norm=lr_norm, n_samples=GSsamples))
        nodeGS =  self.average_sort_attr(self.layer_gradient_shap(nodelayer, baseline = Gradientbaseline, l2_norm=lr_norm, n_samples=GSsamples)) 
        layerDL = self.average_sort_attr(self.layer_deeplift_shap(hyperedgelayer, l2_norm=lr_norm, baseline = DLbaseline))
        nodeDL = self.average_sort_attr(self.layer_deeplift_shap(nodelayer,l2_norm=lr_norm, baseline = DLbaseline))
        hyperedge_attr_score = self.attr_score(layerDL, layerIG, layerGS)
        node_attr_score = self.attr_score(nodeIG, nodeGS, nodeDL)
        return node_attr_score, hyperedge_attr_score


def attr_visualization(feature_names, attr, plot=True, top=10, title="Average Feature Importances", axis_title="Features", descend=True,outpath=None):
    outpath+='/borda_scores_bar'
    if descend:
        sorted_index = np.argsort(np.abs(attr))[::-1]
    else:
        sorted_index = np.argsort(np.abs(attr))
    top_attr = attr[sorted_index][:top]
    top_feature_names = np.array(feature_names)[sorted_index][:top]

    # 打印每个特征及其归因值
    for i in range(len(top_feature_names)):
        print(top_feature_names[i], ": ", '%.3f' % (top_attr[i]))

    x_pos = np.arange(len(top_feature_names))

    if plot:
        # 设置颜色，正向贡献用蓝色，负向贡献用红色
        colors = ['#1f77b4' if val >= 0 else '#ff7f0e' for val in top_attr]

        # 创建画布并调整大小
        plt.figure(figsize=(10, 6), dpi=300)

        # 绘制条形图，设置 bar 宽度
        bars = plt.bar(x_pos, top_attr, align='center', color=colors, edgecolor='black', width=0.6)

        # 设置x轴的特征名，并调整旋转角度
        plt.xticks(x_pos, top_feature_names, wrap=True, rotation=45, ha='right', fontsize=12)

        # 设置标题和标签
        plt.xlabel(axis_title, fontsize=14)
        plt.ylabel('Borda Score', fontsize=14)
        plt.title(title, fontsize=16, weight='bold')

        # 设置y轴范围，使差异更加明显
        plt.ylim(min(top_attr) * 0.9, max(top_attr) * 1.1)

        # 添加条形图上方的数值注释
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, '%.2f' % yval, ha='center', va='bottom', fontsize=10)

        # 添加图例
        plt.axhline(0, color='black',linewidth=0.8)


        plt.tight_layout()

        # 保存为svg
        os.makedirs(outpath,exist_ok=True)
        plt.savefig(f'{outpath}/{title}_attr.svg', format='svg', bbox_inches='tight', dpi=300)
        plt.show()


def compute_y_positions(values):
    y_positions = [0.01]
    for i in range(1, len(values)):
        previous_y = y_positions[-1]
        # Space based on the relative size of the values and add a buffer to avoid overlap
        y_spacing = 0.01 + 0.9 * ((values[i - 1] + values[i]) / 2 / sum(values))
        y_positions.append(previous_y + y_spacing)
    return y_positions

# def plot_sankey_hypergraph(node_borda_scores, hyperedge_borda_scores, incidence_matrix,
                        #   node_labels=None, hyperedge_labels=None, top_n_nodes=10, top_n_hyperedges=10,
                        #   title="Sankey Diagram of Gene and Pathway Borda Scores",
                        #   ):
    # """
    # Generate a Sankey diagram using Borda scores for nodes and hyperedges.
    
    # Parameters:
    # - node_borda_scores: ndarray of shape (num_nodes,), importance scores for each node.
    # - hyperedge_borda_scores: ndarray of shape (num_hyperedges,), importance scores for each hyperedge.
    # - incidence_matrix: ndarray of shape (num_nodes, num_hyperedges), incidence matrix showing connections.
    # - node_labels: List of node names.
    # - hyperedge_labels: List of hyperedge names.
    # - top_n_nodes: Number of top nodes to display.
    # - top_n_hyperedges: Number of top hyperedges to display.
    # - title: Title of the Sankey diagram.
    # - other_scaling_factor: Scaling factor for "Other" values to make them smaller.

    # Returns:
    # - Plotly Figure object.
    # """
    # num_nodes = len(node_borda_scores)
    # num_hyperedges = len(hyperedge_borda_scores)
    # if incidence_matrix.shape != (num_nodes, num_hyperedges):
    #     raise ValueError(f"Incidence matrix shape {incidence_matrix.shape} does not match node ({num_nodes}) and hyperedge ({num_hyperedges}) dimensions.")

    # # 如果未提供标签，则生成默认标签
    # if node_labels is None:
    #     node_labels = [f'Gene{i+1}' for i in range(num_nodes)]
    # if hyperedge_labels is None:
    #     hyperedge_labels = [f'Pathway{i+1}' for i in range(num_hyperedges)]

    # # 排序并选择顶级节点和超边
    # top_node_indices = np.argsort(node_borda_scores)[::-1][:top_n_nodes]
    # other_node_indices = np.setdiff1d(np.arange(num_nodes), top_node_indices)

    # top_hyperedge_indices = np.argsort(hyperedge_borda_scores)[::-1][:top_n_hyperedges]
    # other_hyperedge_indices = np.setdiff1d(np.arange(num_hyperedges), top_hyperedge_indices)
    
    # # 计算“Others”中的节点和超边数量
    # num_other_nodes = len(other_node_indices)
    # num_other_hyperedges = len(other_hyperedge_indices)

    # # 创建包含“Others”的标签
    # new_node_labels = [node_labels[i] for i in top_node_indices] + ['Other Nodes']
    # new_hyperedge_labels = [hyperedge_labels[i] for i in top_hyperedge_indices] + ['Other Hyperedges', 'Output']

    # # 更新节点和超边的Borda分数，包含“Others”的总和
    # new_node_scores = node_borda_scores[top_node_indices].tolist() + [node_borda_scores[other_node_indices].sum()]
    # new_hyperedge_scores = hyperedge_borda_scores[top_hyperedge_indices].tolist() + [hyperedge_borda_scores[other_hyperedge_indices].sum()]

    # # 构建新的关联矩阵，包含“Others”连接
    # # 节点部分
    # new_incidence_matrix_nodes = incidence_matrix[top_node_indices, :]
    # others_nodes_connections = incidence_matrix[other_node_indices, :].sum(axis=0, keepdims=True)
    # new_incidence_matrix_nodes = np.vstack([new_incidence_matrix_nodes, others_nodes_connections])

    # # 超边部分
    # new_incidence_matrix = new_incidence_matrix_nodes[:, top_hyperedge_indices]
    # others_hyperedges_connections = new_incidence_matrix_nodes[:, other_hyperedge_indices].sum(axis=1, keepdims=True)
    # new_incidence_matrix = np.hstack([new_incidence_matrix, others_hyperedges_connections])
    # new_H_df = pd.DataFrame(new_incidence_matrix, index=new_node_labels, columns = new_hyperedge_labels[:-1])
    # # 更新节点和超边的数量
    # new_num_nodes = len(new_node_labels)
    # new_num_hyperedges = len(new_hyperedge_labels) - 1  # 不包括“Output”

    # # 合并所有标签
    # all_labels = new_node_labels + new_hyperedge_labels

    # # 计算每个超边的总连接强度

    # # 构建Sankey图的source、target和values
    # source_indices = []
    # target_indices = []
    # values = []
    # x = []
    # y = []
    # node_values = np.zeros(new_num_nodes)
    # hyperedge_values = np.zeros(new_num_nodes)
    # # 添加节点到超边的连接
    # for node_idx in range(new_num_nodes):
    #     x.append(0.01)
    #     node_values[node_idx] = 0
    #     for hyperedge_idx in range(new_num_hyperedges):
    #         connection_strength = new_incidence_matrix[node_idx, hyperedge_idx]
    #         if connection_strength > 0:
    #             source_indices.append(node_idx)
    #             target_indices.append(new_num_nodes + hyperedge_idx)
                
    #             if node_idx == new_num_nodes - 1:                    
    #                 if hyperedge_idx == new_num_hyperedges - 1:
    #                 # 对“Other Nodes”的连接强度进行调整
    #                     flow_value = connection_strength / num_other_nodes

                    
    #             else:
    #                 flow_value = connection_strength
    #             node_values[node_idx] += flow_value
    #             hyperedge_values[hyperedge_idx] += flow_value
    #             values.append(flow_value)



    # # 添加超边到输出的连接
    # output_index = len(all_labels) - 1  # Output是最后一个节点
    # node_hyperedge_tuple = tuple(zip(source_indices, target_indices, values))
    # for hyperedge_idx in range(new_num_hyperedges):
    #     x.append(0.45)
    #     source_indices.append(new_num_nodes + hyperedge_idx)
    #     target_indices.append(output_index)
        

    #     flow_value = sum([i[-1] for i in node_hyperedge_tuple if i[1] == new_num_nodes + hyperedge_idx])
    #     hyperedge_values[hyperedge_idx] += flow_value
    #     values.append(flow_value)


    # node_y_positions = compute_y_positions(node_values)
    # hyperedge_y_positions = compute_y_positions(hyperedge_values)
    # y = node_y_positions + hyperedge_y_positions
    # x.append(0.9)
    # y.append(0.5)
    # # 设置节点颜色
    # node_colors = [f'rgba(255, 0, 0, {(len(new_node_scores)-node_rank)/len(new_node_scores)})' for node_rank, node_score in enumerate(new_node_scores)]
    # hyperedge_colors = [f'rgba(0, 255, 0, {(len(new_hyperedge_scores)-hyperedge_rank)/len(new_hyperedge_scores)})' for hyperedge_rank, hyperedge_score in enumerate(new_hyperedge_scores)]
    # output_color = ['rgba(0, 0, 255, 1)']
    # node_colors += hyperedge_colors + output_color

    # # 设置链接颜色
    # link_colors = []
    # for node, hyperedge in tuple(zip(source_indices, target_indices)):
    #     node_indices = node
    #     hyperedge_indices = hyperedge
    #     if hyperedge_indices != output_index:
    #         link_colors.append(node_colors[node_indices])
    #     else:
    #         link_colors.append(hyperedge_colors[hyperedge_indices-1-len(new_node_scores)])

    # # 创建Sankey图
    # fig = go.Figure(
    #     data=[go.Sankey(
    #         node=dict(
    #             pad=15,
    #             thickness=20,
    #             line=dict(color="black", width=0.5),
    #             label=all_labels,
    #             color=node_colors,
    #             x=x, 
    #             y=y
    #         ),
    #         link=dict(
    #             source=source_indices,
    #             target=target_indices,
    #             value=values,
    #             color=link_colors,
    #             hovertemplate='Source: %{source.label}<br>Target: %{target.label}<br>Value: %{value}<extra></extra>'
    #         )
    #     )]
    # )

    # # 更新布局和标题
    # fig.update_layout(
    #     title_text=title,
    #     font_size=12,
    #     plot_bgcolor='white',
    #     paper_bgcolor='white',
    #     autosize=False,
    #     width=1000,
    #     height=600
    # )
    # return fig, new_H_df





def node_enrich_pathway(query:dict, background_genes:list):
    gp = GProfiler(return_dataframe=True)
    results = gp.profile(
        organism='hsapiens', 
        query=query, 
        user_threshold=1, 
        sources=['GO:BP','GO:MF', 'GO:CC', 'REAC', 'KEGG', 'WP', 'CORUM'],   
        background=background_genes
    )
    return results

def enrich_bubble_plot(df:pd.DataFrame, prefix:str):
    plt.figure(figsize=(12, 8))
    df = df.sort_values('p_value').head(50)
    sns.scatterplot(data=df, x='p_value', y='name', size='precision', hue='query', sizes=(20, 200), alpha=0.6, palette='viridis')

    # 添加标题和标签
    plt.title('Relationship between Clusters and Enriched Pathways')
    plt.xlabel('P-value')
    plt.ylabel('Pathway Name')
    plt.xscale('log')  # 使用对数坐标以更好地展示p值
    plt.legend(title='Cluster')
    plt.subplots_adjust(left=0.5, right=0.9, top=0.95, bottom=0.05)

    if not os.path.exists(dn:='bubbule'):
        os.makedirs(dn)

    plt.savefig(f'{dn}/{prefix}_bubble.svg')
    

    

def survival_analysis(borda_scores, expression_matrix:pd.DataFrame, survival_time, survival_status,interval, K, prefix, top=True):
    # Step 1: 选取Borda Scores绝对值最高的K个基因
    if top:
        top_genes_indices = np.argsort(np.abs(borda_scores))[-K:][::-1]  # 取绝对值最高的前K个基因的索引
    else:
        top_genes_indices = np.argsort(np.abs(borda_scores))[:K]  # 取绝对值最高的前K个基因的索引
    top_genes = expression_matrix.columns[top_genes_indices]  # 根据索引获取基因名
    
    # Step 2: 对每个基因进行病人划分
    for gene in top_genes:
        gene_expression = expression_matrix[gene]
        median_expression = gene_expression.median()  # 基于该基因的中位数划分
        # 将病人分为高表达组和低表达组

        high_expression_group = gene_expression >= median_expression
        low_expression_group = gene_expression < median_expression
        if high_expression_group.sum() == 0 or low_expression_group.sum() == 0:
            print(f"No sufficient patiencts for Gene {gene} to divide into two groups")
            continue
        # Step 3: Log-rank生存分析
        T_high = survival_time[high_expression_group]
        T_low = survival_time[low_expression_group]
        E_high = survival_status[high_expression_group]
        E_low = survival_status[low_expression_group]

        results = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)
        
        # 输出p值
        print(f"Gene {gene}, Log-rank p-value: {results.p_value}")
        
        # Step 4: 绘制生存曲线
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        
        plt.figure(figsize=(10, 6), dpi=300)
        T_high *=(interval/30)
        T_low *=(interval/30)
        T_high[T_high > 60] = 60
        kmf_high.fit(T_high, event_observed=E_high, label=f'High Expression - {gene}')
        kmf_low.fit(T_low, event_observed=E_low, label=f'Low Expression - {gene}')
        
        kmf_high.plot_survival_function()
        kmf_low.plot_survival_function()
        
        plt.title(f"Survival Curve for Gene {gene} (p-value = {results.p_value:.4f})")
        plt.ylabel("Survival Probability")
        plt.xlabel("Time (months)")
        plt.xlim(0, 60)
        if not os.path.exists(dn:='survival'):
            os.makedirs(dn)
        plt.savefig(f'{dn}/{prefix}_{gene}-survival.svg')


def survival_analysis_with_cox(borda_scores, expression_matrix: pd.DataFrame, survival_time, survival_status, interval, K, prefix):
    # Step 1: Select top 10 and bottom 10 genes
    top_indices = np.argsort(borda_scores)[-K:]  # Top 10
    bottom_indices = np.argsort(borda_scores)[:K]  # Bottom 10
    indices = [top_indices, bottom_indices]
    annot = ['top', 'bottom']
    # Prepare the data for Cox regression
    for selected_indices, direction in list(zip(indices, annot)):
        # selected_genes = expression_matrix.columns[selected_indices]
        # cox_data = expression_matrix[selected_genes].copy()
        cox_data = pd.DataFrame(expression_matrix[:, selected_indices])
        cox_data['survival_time'] = survival_time
        cox_data['survival_status'] = survival_status

        # Step 2: Fit the Cox model
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col='survival_time', event_col='survival_status')

        # Step 3: Predict risk values
        cox_data['risk'] = cph.predict_partial_hazard(cox_data)

        # Step 4: Divide patients based on median risk value
        median_risk = cox_data['risk'].median()
        high_risk_group = cox_data['risk'] >= median_risk
        low_risk_group = cox_data['risk'] < median_risk

        # Step 5: Log-rank survival analysis
        T_high = survival_time[high_risk_group]
        T_low = survival_time[low_risk_group]
        E_high = survival_status[high_risk_group]
        E_low = survival_status[low_risk_group]

        results = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)

        # Output p-value
        print(f"Log-rank p-value: {results.p_value}")

        # Step 6: Plot survival curves
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        plt.figure(figsize=(10, 6), dpi=300)
        T_high *= (interval / 30)
        T_low *= (interval / 30)
        T_high[T_high > 60] = 60

        kmf_high.fit(T_high, event_observed=E_high, label='High Risk Group')
        kmf_low.fit(T_low, event_observed=E_low, label='Low Risk Group')

        kmf_high.plot_survival_function()
        kmf_low.plot_survival_function()

        plt.title(f"Survival Curve (p-value = {results.p_value:.4f})")
        plt.ylabel("Survival Probability")
        plt.xlabel("Time (months)")
        plt.xlim(0, 60)
        
        # Save the plot
        if not os.path.exists(dn := 'survival'):
            os.makedirs(dn)
        plt.savefig(f'{dn}/{prefix}_{direction}_survival_curve.svg')




def attention_survival_analysis(hyperedge_borda_scores, hyperedge_attention_matrix: np.array, expression_matrix: pd.DataFrame, survival_time, survival_status, interval,K, N, prefix, top=True):
    # 检查输入数据的有效性
    print("Checking validity of input data...")
    if len(hyperedge_borda_scores) != hyperedge_attention_matrix.shape[0]:
        raise ValueError("Length of hyperedge Borda scores must match the number of hyperedges.")
    
    if len(survival_time) != expression_matrix.shape[0] or len(survival_status) != expression_matrix.shape[0]:
        raise ValueError("Length of survival data must match the number of patients in the expression matrix.")
    
    # Step 1: 选取Hyperedge Borda Scores绝对值最高的K个超边
    print(f"Selecting top {K} hyperedges based on absolute Borda scores...")
    if top:
        top_hyperedges_indices = np.argsort(np.abs(hyperedge_borda_scores))[-K:][::-1]  # 取绝对值最高的前K个超边的索引
    else:
        top_hyperedges_indices = np.argsort(np.abs(hyperedge_borda_scores))[:K]  # 取绝对值最高的前K个超边的索引
        
    print(f"Top hyperedges indices: {top_hyperedges_indices}")
    
    # Step 2: 对每个超边选取Attention值最高的N个基因
    top_genes = []
    for hyperedge_idx in top_hyperedges_indices:
        print(f"Processing hyperedge index: {hyperedge_idx}")
        attention_scores = hyperedge_attention_matrix[hyperedge_idx, :]
        top_genes_indices = np.argsort(attention_scores)[-N:][::-1]  # 取Attention值最高的前N个基因的索引
            
        print(f"Top {N} genes indices for hyperedge {hyperedge_idx}: {top_genes_indices}")
        top_genes.extend(expression_matrix.columns[top_genes_indices])
    top_genes = list(set(top_genes))  # 去重
    print(f"Total unique genes selected for survival analysis: {len(top_genes)}")
    
    # 输出文件夹
    output_dir = 'survival'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 3: 对每个基因进行生存分析
    for gene in top_genes:
        print(f"Performing survival analysis for gene: {gene}")
        gene_expression = expression_matrix[gene]
        median_expression = gene_expression.median()  # 基于该基因的中位数划分
        print(f"Median expression for gene {gene}: {median_expression}")
        
        # 将病人分为高表达组和低表达组
        high_expression_group = gene_expression >= median_expression
        low_expression_group = gene_expression < median_expression
        
        if high_expression_group.sum() == 0 or low_expression_group.sum() == 0:
            print(f"No sufficient patients for Gene {gene} to divide into two groups")
            continue
        
        # Step 4: Log-rank生存分析
        print(f"Performing log-rank test for gene: {gene}")
        T_high = survival_time[high_expression_group]
        T_low = survival_time[low_expression_group]
        E_high = survival_status[high_expression_group]
        E_low = survival_status[low_expression_group]

        results = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)
        
        # 输出p值
        print(f"Gene {gene}, Log-rank p-value: {results.p_value}")
        
        
        # Step 5: 绘制生存曲线
        print(f"Plotting survival curve for gene: {gene}")
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        
        plt.figure(figsize=(10, 6), dpi=300)
        T_high *=(interval/30)
        T_low *=(interval/30)
        T_high[T_high > 60] = 60
        kmf_high.fit(T_high, event_observed=E_high, label=f'High Expression - {gene}')
        kmf_low.fit(T_low, event_observed=E_low, label=f'Low Expression - {gene}')
        
        kmf_high.plot_survival_function()
        kmf_low.plot_survival_function()
        
        plt.title(f"Survival Curve for Gene {gene} (p-value = {results.p_value:.4f})")
        plt.xlabel("Time (months)")
        plt.xlim(0, 60)
        plt.ylabel("Survival Probability")
        
        plt.savefig(os.path.join(output_dir, f'{prefix}_{gene}-survival.svg'))
        plt.close()
        print(f"Survival curve saved for gene: {gene}")

def hyperedge_survival_analysis(H, borda_scores, expression_matrix, survival_time, survival_status, K, top=True):
    # Step 1: 选取Borda Scores绝对值最高的K个通路
    if top:
        top_hyperedge_indices = np.argsort(np.abs(borda_scores))[-K:][::-1]  # 取绝对值最高的前K个通路的索引
    else:
        top_hyperedge_indices = np.argsort(np.abs(borda_scores))[:K]
    expression_matrix = expression_matrix @ H
    expression_matrix.columns = [i[0] for i in expression_matrix.columns.str.split('_')]
    top_hyperedges = expression_matrix.columns[top_hyperedge_indices]  # 根据索引获取通路名
    
    # Step 2: 对每个基因进行病人划分
    for hyperedge in top_hyperedges:
        hyperedge_expression = expression_matrix[hyperedge]
        median_expression = hyperedge_expression.median()  # 基于该基因的中位数划分
        
        # 将病人分为高表达组和低表达组
        high_expression_group = hyperedge_expression >= median_expression
        low_expression_group = hyperedge_expression < median_expression
        
        # Step 3: Log-rank生存分析
        T_high = survival_time[high_expression_group]
        T_low = survival_time[low_expression_group]
        E_high = survival_status[high_expression_group]
        E_low = survival_status[low_expression_group]
        
        results = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)
        
        # 输出p值
        print(f"Hyperedge {hyperedge}, Log-rank p-value: {results.p_value}")
        
        # Step 4: 绘制生存曲线
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        
        plt.figure(figsize=(10, 6), dpi=300)
        
        kmf_high.fit(T_high, event_observed=E_high, label=f'High Expression - {hyperedge}')
        kmf_low.fit(T_low, event_observed=E_low, label=f'Low Expression - {hyperedge}')
        
        kmf_high.plot_survival_function()
        kmf_low.plot_survival_function()
        
        plt.title(f"Survival Curve for Gene {hyperedge} (p-value = {results.p_value:.4f})")
        plt.xlabel("Time (bins)")
        plt.ylabel("Survival Probability")
        plt.savefig(f'{hyperedge}-survival.svg')
        

def attention_heatmap_top_borda(attention_scores:np.ndarray, node_borda:np.ndarray, hyperedge_borda:np.ndarray, prefix:str, node_name:list, hyperedge_name:list, node_num=20, hyperedge_num=10, top=True):
    top_node_indices = np.argsort(np.abs(node_borda))[-node_num:][::-1]
    top_hyperedge_indices = np.argsort(np.abs(hyperedge_borda))[-hyperedge_num:][::-1]
    heatmap_data = attention_scores[np.ix_(top_hyperedge_indices, top_node_indices)]
    heatmap_xticklabels = [node_name[i] for i    in top_node_indices]
    heatmap_yticklabels = [hyperedge_name[i] for i in top_hyperedge_indices]
    # 绘制热图
    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, cmap='viridis', linewidths=.5,
                xticklabels=heatmap_xticklabels,
                yticklabels=heatmap_yticklabels)
    plt.text(1.15, 0.5, 'Attention', fontsize=16, ha='left', va='center', transform=plt.gca().transAxes)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.title('Top Borda Pathways to Top Borda Genes with attention')
    plt.xlabel('Gene')
    plt.ylabel('Hyperedge')
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.2)
    if not os.path.exists(dn:='attention_heatmap'):
        os.makedirs(dn)
    plt.savefig(f'{dn}/{prefix}_Borda_Hyperedge_gene.attention.svg')
    
def get_top_atten_item(attention, borda_scores, K, item1_name, item2_name,outpath):
    """
    Summarize attention values for the top K item2(genes or hyperedges) in the top 10 hyperedges or genes by Borda score.

    Parameters:
    - attention: ndarray of shape (num_items1, num_items2), attention scores for each node-hyperedge pair.
    - borda_scores: ndarray of shape (num_items2,), Borda scores for each items2.
    - K: int, number of top genes to consider per hyperedge.
    - item1_name: list of str, names of the items1 (genes or hyperedges).
    - item2_name: list of str, names of the items2 (hyperedges or genes).

    Returns:
    - summary: dict, contains the top K genes and their attention values for each of the top 10 hyperedges.
    """
    # Find the indices of the top 10 hyperedges by Borda score
    top_item2_indices = np.argsort(borda_scores)[::-1][:10]

    summary = {}
    attn_str = ''
    # Iterate over the top 10 hyperedges
    for item2_idx in top_item2_indices:
        # Extract attention values for the current hyperedge from all nodes
        hyperedge_attention = attention[item2_idx, :]
        # Find the indices of the top K genes by attention value
        top_gene_indices = np.argsort(hyperedge_attention)[::-1][:K]
        # Get the gene names and attention values
        top_genes = [(item1_name[i], hyperedge_attention[i]) for i in top_gene_indices]
        # Store in summary dictionary
        summary[item2_name[item2_idx]] = top_genes

    item1_attention_over_top_10_items2 = []
    for i in summary:
        cluster_dict = {}
        cluster_dict['hyperedge'] = i
        cluster_dict['genes'] = []
        attn_str += f'    - {i}('
        for j in summary[i]:
            if j[1] == 0:
                continue
            attn_str += f'{j[0]},'
            cluster_dict['genes'].append(j[0])
        attn_str = attn_str[:-1]
        attn_str += f')\n'
        item1_attention_over_top_10_items2.append(cluster_dict)
    os.makedirs(outpath,exist_ok = True)
    visualize_attention_summary(summary, outpath)
    return item1_attention_over_top_10_items2


def summarize_attention_top_genes(attention, hyperedge_borda, K, node_name, hyperedge_name):
    """
    Summarize attention values for the top K genes in the top 10 hyperedges by Borda score.

    Parameters:
    - attention: ndarray of shape (num_nodes, num_hyperedges), attention scores for each node-hyperedge pair.
    - hyperedge_borda: ndarray of shape (num_hyperedges,), Borda scores for each hyperedge.
    - K: int, number of top genes to consider per hyperedge.
    - node_name: list of str, names of the nodes (genes).
    - hyperedge_name: list of str, names of the hyperedges.

    Returns:
    - summary: dict, contains the top K genes and their attention values for each of the top 10 hyperedges.
    """
    # Find the indices of the top 10 hyperedges by Borda score
    top_hyperedge_indices = np.argsort(hyperedge_borda)[::-1][:10]

    summary = {}

    # Iterate over the top 10 hyperedges
    for hyperedge_idx in top_hyperedge_indices:
        # Extract attention values for the current hyperedge from all nodes
        hyperedge_attention = attention[hyperedge_idx, :]
        # Find the indices of the top K genes by attention value
        top_gene_indices = np.argsort(hyperedge_attention)[::-1][:K]
        # Get the gene names and attention values
        top_genes = [(node_name[i], hyperedge_attention[i]) for i in top_gene_indices]
        # Store in summary dictionary
        summary[hyperedge_name[hyperedge_idx]] = top_genes

    return summary

# New method to visualize the summarized attention values
def visualize_attention_summary(summary, prefix):
    """
    Visualize the attention values for the top K genes in the top 10 hyperedges.

    Parameters:
    - summary: dict, contains the top K genes and their attention values for each of the top 10 hyperedges.
    """
    min_values = {j: min(i[1] for i in summary[j]) for j in summary}
    summary = {
        j: [i for i in summary[j] if i[1] != min_values[j]] for j in summary
    }
    for hyperedge, genes in summary.items():
        gene_names = [gene[0] for gene in genes]
        attention_values = [gene[1] for gene in genes]

        plt.figure(figsize=(10, 6))
        plt.barh(gene_names, attention_values, color='skyblue')
        plt.xlabel('Attention Value')
        plt.title(f'Top Genes for {hyperedge}')
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest attention at the top
        plt.tight_layout()
        hyperedge = hyperedge.replace('/', '_')
        plt.savefig(f'{prefix}_{hyperedge}_attention_summary.svg')
def get_top_nodes_for_hyperedges(node_borda_scores, hyperedge_borda_scores, incidence_matrix, 
                                 node_labels, hyperedge_labels, top_n_nodes=10, top_n_hyperedges=10):
    # 获取前 top_n_hyperedges 个超边的索引
    top_hyperedge_indices = np.argsort(hyperedge_borda_scores)[::-1][:top_n_hyperedges]
    
    # 创建一个字典存储结果
    hyperedge_node_dict = {}

    # 对于每个超边，找到其关联的节点并选择前 top_n_nodes 个
    for hyperedge_index in top_hyperedge_indices:
        # 获取与此超边连接的节点的索引
        connected_node_indices = np.where(incidence_matrix[:, hyperedge_index] > 0)[0]

        # 根据节点的Borda分数对这些节点进行排序
        sorted_node_indices = connected_node_indices[np.argsort(node_borda_scores[connected_node_indices])[::-1]]

        # 获取前 top_n_nodes 个节点的名字
        top_node_names = [node_labels[i] for i in sorted_node_indices[:top_n_nodes]]

        # 获取超边的名字
        hyperedge_name = hyperedge_labels[hyperedge_index]

        # 将超边名字和节点名字列表添加到字典中
        hyperedge_node_dict[hyperedge_name] = top_node_names

    return hyperedge_node_dict

class Hook:
    """
    Create hooks on modules, e.g.,
    back_hook = Hook(layer[1], backward=True)
    fwd_hook = Hook(layer[1])
    """

    def __init__(self, module, backward=False, hook_to_module_input=False):
        self.lin = []
        self.out = []
        self.module = module
        self.hook_to_module_input = hook_to_module_input

    def backward(self):
        if self.hook_to_module_input:
            self.hook = self.module.register_full_backward_hook(self.hook_fn_module_input)
        else:
            self.hook = self.module.register_full_backward_hook(self.hook_fn_module_backward_output)

    def forward(self):
        if self.hook_to_module_input:
            self.hook = self.module.register_forward_hook(self.hook_fn_module_input)
        else:
            self.hook = self.module.register_forward_hook(self.hook_fn_module_output)

    def CDFhook(self):
        self.hook = self.module.register_forward_hook(self.PMF2CDF)
        return self.hook

    def Survivalhook(self):
        self.hook = self.module.register_forward_hook(self.PMF2Survival)
        return self.hook
    
    def five_years_hook(self):
        self.hook = self.module.register_forward_hook(self.PMFto5years)
    def time_median_hook(self):
        self.hook = self.module.register_forward_hook(self.PMFtoMedian)
    def Riskhook(self):
        self.hook = self.module.register_forward_hook(self.PMF2risk)

    def hook_fn_module_input(self, module, input, output):
        f = input[0].detach().cpu()
        self.lin.append(f)

    def hook_fn_module_output(self, module, input, output):
        f = output[0].detach().cpu()
        self.out.append(f)
        print("Forward hook executed. Output:", output)

    def hook_fn_module_backward_output(self, module, input, output):
        f = output[0].detach().cpu()
        self.out.append(f)
        print("Backward hook executed. Output:", output)

    def PMF2CDF(self, module, input, output):
        # if type of foutput is tuple
        if type(output) == tuple:
            output = output[0]
        cdf = torch.cumsum(output, -1)
        cdf_sum = cdf.sum(-1)
        return cdf_sum
 

    def PMF2Survival(self, module, input, output):
        S = 1 - output
        survival = torch.sum(S, dim=-1)
        return survival
    
    # def PMFto5years(self, module, input, output):
    #     survival = 1 - torch.cumsum(output, -1)
        
    #     hazard_function = output/survival
    #     cumulative_hazard_function = torch.cumsum(hazard_function, -1)

    #     return cumulative_hazard_function[:, -2]
    
    def PMFto5years(self, module, input, output):
        cdf =  torch.cumsum(output, -1)
        return cdf[:, 20]
    def PMFtoMedian(self, module, input, output):   
        cdf =  torch.cumsum(output, -1)
        median = output.shape[1]//2
        print(f"median of time bins is {median}")
        return cdf[:, median]
    def PMF2risk(self, module, input, output):
        
        survival_probability = torch.zeros_like(output)
        num_patients, num_time_bins = output.shape
        for i in range(num_patients):
            for j in range(num_time_bins):
                survival_probability[i, j] = output[i, j:].sum()

        # 计算风险值 h(t) = P(X = t) / P(X >= t)
        risk_matrix = output / survival_probability

        return risk_matrix[:,20]
    
    def close(self):
        self.hook.remove()
        del self.out[:], self.lin[:]


def univariate_regression(
    data: pd.DataFrame, duration_col: str, event_col: str,penalizer:float, feature: str,
):
    """
    Use CoxPHFitter to do univariate regression.

    Params:
        data (DataFrame): Input data with shape (n_samlpes, n_features+2).
        duration_col (str): Name of the duration column.
        event_col (str): Name of the event column.
        feature (str): Name of the feature to do regression.
    Returns:
        p (float): P value of the feature.
    """
    cph = CoxPHFitter(penalizer=penalizer)# slove converge problems set penalizer=0.0001

    cph.fit(
    df=data, duration_col=duration_col, event_col=event_col, formula=feature,#show_progress=True
    )
    hazard_ratios = cph.hazard_ratios_
    # 获取pvalue
    p = cph.summary.loc[feature, 'p']
    return hazard_ratios,p
def get_HRs(
    data: pd.DataFrame,
    duration_col: str='Time',
    event_col: str='Event',
    process_num: int = 10,
    penalizer: float = 0.,
):
    """
        Use CoxPHFitter to select feature.

        Params:
            data(DataFrame): Input data with shape (n_samlpes, n_features+2).
            duration_col(str): Name of the duration column.
            event_col(str): Name of the event column.
            process_num(int): Number of processes to use.
            penalizer(float): Penalizer for CoxPHFitter.
        Returns:
            HRs (DataFrame): Hazard Ratios of all features.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("feature_matrix should be of type pd.DataFrame!")
    features=data.columns[:-2].tolist()
    print(features)
    hrs = []
    p_values = []
    # Use multiprocessing to accelerate the process
    print('Cox univariate regression for computing hr...')
    try:
        regression = partial(univariate_regression, data, duration_col, event_col,penalizer)
        pool = mul.Pool(processes=process_num)
        print("using multiprocessing...")
        with tqdm(total=len(features)) as t:
            for hr in pool.imap(regression, features):# 单个基因进行回归分析
                # print()
                hr,p = univariate_regression(data, duration_col, event_col,penalizer,feature)
                p_values.append(p)
                hrs.append(hr)
                t.update()
    except:
        print("multiprocessing error,using for loop...")
        for feature in features:
            hr,p = univariate_regression(data, duration_col, event_col,penalizer,feature)
            hrs.append(hr)
            p_values.append(p)
    HRs = pd.DataFrame(pd.concat(hrs))
    HRs['p_values']=p_values
    return HRs


def get_death_CDF_sum(model,data):
    if isinstance(data,pd.DataFrame):
        data = data.values
    data = torch.tensor(data).float().cuda()
    pmf = model.predict(data)
    cdf =  torch.cumsum(pmf, -1)
    cdf_sum = cdf.sum(-1)
    return cdf_sum


class TripletsAnalysis:
    def __init__(self,model,H_df,data,out_path,dn_data,hyperedge_borda,node_borda,n_triplets,PK_D,fn_map,dn_esdb):
        # ensemble2gene = pd.read_table('/Backup/home/chenyupeng/DATA/mart_export (1).txt', index_col=0)
        # self.dn_esdb = f"/Backup/home/chenyupeng/DATA/GSVA/"
        ensemble2gene = pd.read_table(fn_map, index_col=0)
        self.n_triplets = n_triplets
        self.outpath = out_path
        self.PK_D = PK_D
        self.H_df = H_df
        self.model = model
        self.ENSG2GN=ensemble2gene['Gene name'].to_dict()
        self.data = data
        self.expr = data.iloc[:,:-2]
        self.genes = self.expr.columns
        self.time_event = data.iloc[:,-2:]
        self.dn_data = dn_data
        self.dn_esdb = dn_esdb

        self.node_name = self.get_node_name()
        self.hyperedge_name = self.get_hyperedge_name()
        self.pair_borda_scores = self.calculate_weighted_borda_scores(node_borda,hyperedge_borda)
        self.HRs_genes,self.data_es = self.get_HRsgene_GSVA(dn_data)
    
    def GeneClusterEnrichment(self,clusters):
        '''
        cluster: dict, key: cluster name, value: list of gene names;
        self.ENSG2GN: dict or pd.DataFrame, key: ENSG, value: gene name;
        self.outpath: str, output path;
        self.dn_esdb: str, enrichment database .gmt files path;
        Returns: dict, key: cluster name, value: cluster enriched description.
        '''
        enrich_results = {}
        for culster,gs_clu in clusters.items():
            gs_clu = [self.ENSG2GN[ensg] for ensg in gs_clu]
            eres = [] 
            # TODO判断文件是否存在
            dn_gsea = f'{self.outpath}/ClusterGSEA/'
            os.makedirs(dn_gsea,exist_ok=True)
            if os.path.isfile(f'{dn_gsea}{culster}.csv'):
                logger.info(f"using existing file: {dn_gsea}{culster}.csv")
                df_gsea = pd.read_csv(f'{dn_gsea}{culster}.csv',index_col = 0)
            else:
                fs_gsea = os.listdir(self.dn_esdb)
                # fs_gsea.remove('CORUM.gmt')
                for file in fs_gsea:
                    if file.endswith('.gmt'):
                        logger.info(f"GSEA with database : {file.split('.')[0]}")
                        try: 
                            enr = gp.enrich(gene_list=gs_clu,gene_sets=self.dn_esdb+file, outdir=None,verbose=True)
                            eres.append(enr.results)					
                        except ValueError:
                            logger.info(f"Did not enrich any... Continue")
                if len(eres)>0:
                    eres = [res for res in eres if not isinstance(res, list)]
                    if eres == []:
                        df_gsea = None
                    else:
                        df_gsea= pd.concat(eres).set_index("Term").sort_values(by='Adjusted P-value',ascending=True)
                        df_gsea.to_csv(f'{self.outpath}/ClusterGSEA/{culster}.csv')
                else:
                    df_gsea = None
            # filter by adj p-value 0.05, and sort by adj p-value and combined score
            # if did not enrich any, then use the gene list itself as description
            if df_gsea is not None:
                # p_top = df_gsea['Adjusted P-value'].min()
                description = df_gsea[df_gsea['Adjusted P-value'] < 0.05].sort_values('Combined Score',ascending=False).index[0]

                enrich_results[culster] = description 
            else:
                logger.info(f"{culster} did not enrich any description...")
                enrich_results[culster] = gs_clu
        return enrich_results
    
    def get_regulation_diff(self,ES_matrix):
        risk = get_death_CDF_sum(self.model,self.expr)
        group_h = risk>risk.median()
        group_l = risk<risk.median()
        ES_matrix_h = ES_matrix.loc[group_h.cpu().numpy(),:].iloc[0,:]
        ES_matrix_l = ES_matrix.loc[group_l.cpu().numpy(),:].iloc[0,:]
        tt_info = ttest_ind(ES_matrix_h, ES_matrix_l)# (t_statistic, p_value)
        ES_sum_h = ES_matrix_h.sum(0)
        ES_sum_l = ES_matrix_l.sum(0)
        diff_hl_es = ES_sum_h - ES_sum_l
        # if p_value < 0.05:
        #     ES_sum_h = ES_matrix_h.sum(0)
        #     ES_sum_l = ES_matrix_l.sum(0)
        #     diff_hl_es = ES_sum_h - ES_sum_l
        #     if diff_hl_es > 0:
        #         regulation_pathway_diff = 'activation'
        #     elif diff_hl_es < 0:		
        #         regulation_pathway_diff = 'inhibition'
        # else:
        #     print(f"p_value:{p_value} is not significant, skip!")
        #     regulation_pathway_diff = "no-regulation"
        return diff_hl_es,tt_info


    def get_paper_count(self,name_hyperedge,triplet:dict):
        '''
        triplet: {'gene':gene_symbol, 'cancer':cancer_name, 'pathway':pathway_name}
        '''
        # Entrez.email = "qymyc04@live.com"
        gene_symbol = triplet['gene']
        cancer_name = triplet['cancer']
        pathway_name = triplet[name_hyperedge]
        # Construct gene search term
        max_retries = 3

        for attempt in range(max_retries):
            try:
                gene_term = f'{gene_symbol}[Gene Name] AND Homo sapiens[Organism]'
                handle = Entrez.esearch(db='gene', term=gene_term, usehistory=True)
                gene_record = Entrez.read(handle)
                break
            except Exception as e:
                logger.warning(f"Entrez.read failed on attempt {attempt + 1}/{max_retries} with error: {e}. Retrying...")
                time.sleep(2)
                if attempt == max_retries - 1:
                    raise

        handle.close()
        # Check if the IdList is empty (no results)
        if not gene_record.get('IdList'):
            print(f"No gene found for: {gene_symbol}")
        time.sleep(1)
        disease_term = f'({cancer_name}[mh])'
        print(disease_term)
        
        
        for attempt in range(max_retries):
            try:
                handle = Entrez.elink(dbfrom='gene', db='pubmed', id=gene_record['IdList'], term=disease_term)
                record = Entrez.read(handle)
                break
            except Exception as e:
                logger.warning(f"Entrez.read failed on attempt {attempt + 1}/{max_retries} with error: {e}. Retrying...")
                time.sleep(2)
                if attempt == max_retries - 1:
                    raise
        handle.close()
        time.sleep(1)
        num_gene_disease = len(record[0]['LinkSetDb'][0]['Link'])
        for attempt in range(max_retries):
            try:
                handle = Entrez.elink(dbfrom='gene', db='pubmed', id=gene_record['IdList'], term=pathway_name)
                record = Entrez.read(handle)
                break
            except Exception as e:
                logger.warning(f"Entrez.read failed on attempt {attempt + 1}/{max_retries} with error: {e}. Retrying...")
                time.sleep(2)
                if attempt == max_retries - 1:
                    raise
        
        handle.close()
        time.sleep(1)
        num_gene_path = len(record[0]['LinkSetDb'][0]['Link'])
        pathway_disease_query = f'{pathway_name} AND {disease_term}'
        print(pathway_disease_query)
        for attempt in range(max_retries):
            try:
                handle = Entrez.esearch(db='pubmed', term=pathway_disease_query, usehistory=True)
                record = Entrez.read(handle)
                break
            except Exception as e:
                logger.warning(f"Entrez.read failed on attempt {attempt + 1}/{max_retries} with error: {e}. Retrying...")
                time.sleep(2)
                if attempt == max_retries - 1:
                    raise
            

        num_pathway_disease = len(record['Count'])
        # print(f'pubmed all: {num_pathway_disease+num_gene_path+num_gene_disease}')
        handle.close()
        time.sleep(1)

        dis_pat_term = f'({cancer_name}[mh]) AND {pathway_name}'
        for attempt in range(max_retries):
            try:
                handle = Entrez.elink(dbfrom='gene', db='pubmed', id=gene_record['IdList'], term=dis_pat_term)
                record = Entrez.read(handle)
                break
            except Exception as e:
                logger.warning(f"Entrez.read failed on attempt {attempt + 1}/{max_retries} with error: {e}. Retrying...")
                time.sleep(2)
                if attempt == max_retries - 1:
                    raise

        count_triplet = len(record[0]['LinkSetDb'][0]['Link'])

        count_diction = {
                    f'papers count of cancer-gene-{name_hyperedge}':count_triplet,
                    f'papers count of gene-{name_hyperedge}':num_gene_path, 
                    f'papers count of cancer-gene':num_gene_disease, 
                    f'papers count of cancer-{name_hyperedge}':num_pathway_disease}
        
        return count_diction

    def get_node_name(self):
        return [self.ENSG2GN[ensg] for ensg in self.genes]
    
    def get_hyperedge_name(self):
        '''
        将超图邻接矩阵中的超边和节点id转换为描述：
        超边：聚类知识、STRING：富集分析；Reactome：description
        节点：ensg->gene_name
        '''
        if self.PK_D['type_know'] =='hcluster' or (self.PK_D['type_know']=='STRING' and self.PK_D['Description']=="Enrichment"):
            # GSEA for description
            ## map for cluster-genes
            clus_genes={}
            for i in range(len(self.H_df.columns)):
                clus_genes[f"{i}"] = np.array(self.genes)[self.H_df.values[:,i].nonzero()[0]].tolist()
            enrich_results = self.GeneClusterEnrichment(clus_genes)
            hyperedge_name = [enrich_results[f"{i}"] for i in range(len(self.H_df.columns))]
        elif self.PK_D['type_know'] == 'Reactome':
            # H = pd.read_csv("/data/HGCluster/reactome_P12/AcrossCohorts/RNA/test-['ICGC_LIRI']-cox200/H1.csv", index_col =0)
            reactome_description = pd.read_table('/Backup/home/chenyupeng/Projects240423/HGS-Explainer/HSA_ReactomePathways.txt', index_col = 0, header=None)# TODO
            reactome_description = reactome_description[1].to_dict()
            hyperedge_name = [reactome_description[hsa[0]] for hsa in self.H_df.columns.str.split('_')]
        elif self.PK_D['type_know']=='STRING' and self.PK_D['Description']=="STRING":
            STRING_description = pd.read_csv('/Backup/home/chenyupeng/DATA/Graph/STRING/clusters.protein.ensg.csv',index_col='cluster_id')
            STRING_description = STRING_description['best_described_by'].to_dict()
            # TODO：key error 如果不存在则找其父，如果仍然不存在，则不进行解释;暂时直接保留 ;判断是否存在没有解释的cluster，若存在在字典中添加其值为 unexplained
            if sum([1-(ppi[0] in (STRING_description.keys())) for ppi in self.H_df.columns.str.split('_')])>0:
                out_idxs = [bool(1-(ppi[0] in (STRING_description.keys()))) for ppi in self.H_df.columns.str.split('_')]
                for clu_out in self.H_df.columns[out_idxs].to_list():
                    STRING_description[clu_out]='unexplained' 
            hyperedge_name = [STRING_description[ppi[0]] for ppi in self.H_df.columns.str.split('_')]
        return hyperedge_name

    def calculate_weighted_borda_scores(self,node_borda,hyperedge_borda):
        '''
        Compute the weighted borda scores of all pairs of nodes and hyperedges.
        Details:
        -------
            The weighted borda score of a pair of a node and a hyperedge is defined as:
            B_ij(k,l) = (1/|V|) * (b_node_pair) + (1/|E|) * b_hyperedge_pair
            where b_node_pair and b_hyperedge_pair are the borda scores of the nodes and hyperedge in one pair,
            |V| and |E| are the number of nodes and hyperedges.

        Parameters:
        -----------
        node_borda : np.array, shape=(num_nodes,), dtype=float
        hyperedge_borda : np.array, shape=(num_hyperedges,), dtype=float
        H : np.array, shape=(num_nodes, num_hyperedges), dtype=int # Hypergraph adjacency matrix

        Returns:
        --------
        pair_borda_scores : np.array, shape=(3,num_pairs), dtype=float
            pair_borda_scores[0]: borda score of pairs；
            pair_borda_scores[1]: node index of pairs；
            pair_borda_scores[2]: hyperedge index of pairs；
        '''
        H = self.H_df.values
        node_w = 1/len(node_borda)
        edge_w = 1/len(hyperedge_borda)
        pair = H.nonzero()
        pair_borda_scores = np.array([])
        for i in range(len(pair[0])):
            node_score = node_borda[pair[0][i]]*node_w
            edge_score = hyperedge_borda[pair[1][i]]*edge_w
            pair_borda_scores = np.append(pair_borda_scores,node_score+edge_score)
        # 构建三维矩阵，分别存储:0: pair的borda score；1:pair的节点索引；2: pair的超边索引
        pair_borda_scores = np.concatenate((pair_borda_scores.reshape(1,-1),pair[0].reshape(1,-1),pair[1].reshape(1,-1)),axis=0)
        # 根据borda score由大到小排序
        pair_borda_scores = pair_borda_scores[:,np.argsort(pair_borda_scores[0])[::-1]]
        return pair_borda_scores

    def get_HRsgene_GSVA(self,dn_data):
        ## GSVA for {name_hyperedge} expression
        expr_t = self.expr.T
        expr_t.index = self.node_name
        es_list=[]
        fs_esdb = os.listdir(self.dn_esdb)
        # fs_esdb.remove('CORUM.gmt')
        if self.PK_D['type_know']=='Reactome':
            fs_esdb = [f for f in fs_esdb if (f.startswith('Reactome') and f.endswith('.gmt'))]
        for file in fs_esdb:
            if file.endswith('.gmt'):
                dn_gsva = dn_data+f"/GSVA/"
                ds = dn_gsva+f"{file.split('.')[0]}/"
                logger.info(f"GSVA with database : {file.split('.')[0]}")
                os.makedirs(ds,exist_ok=True)
                fn_es = ds+"gseapy.gene_set.gsva.report.csv"
                if os.path.isfile(fn_es): 
                    logger.info(f"using existing file {fn_es}")
                    df_es = pd.read_csv(fn_es).pivot(index='Term', columns='Name', values='ES')
                else:
                    es = gp.gsva(data=expr_t,# as a matrix of expression values where rows correspond to genes and columns correspond to samples.
                                gene_sets=self.dn_esdb+file,
                                min_size=1,
                                outdir=ds)
                    df_es = es.res2d.pivot(index='Term', columns='Name', values='ES')
                es_list.append(df_es)
        df_es = pd.concat(es_list)
        df_es.to_csv(dn_gsva+f"PathwayExpressionMatrix.csv")
        # es_index = df_es.index.to_list()
        logger.info(f"num GSVA pathway: {df_es.shape[0]}")
        # data_es = df_es.reset_index(inplace=False,drop=True).T
        # data_es.columns = [(col[0]+'_'+str(i)) for i,col in enumerate(df_es.index)]
        data_es = df_es.T
        data_es['Time']=self.time_event.loc[data_es.index,:].iloc[:,-2]
        data_es['Event']=self.time_event.loc[data_es.index,:].iloc[:,-1]

        # Gene Expression Cox Univariate Regression HRs 
        fn_HRs = dn_data+"Genes_HazardRatios_Pvalue.csv"
        if os.path.isfile(fn_HRs):
            logger.info(f"using existing HRs file {fn_HRs}")
            HRs = pd.read_csv(fn_HRs,index_col=0)
        else:
            logger.info(f"calculating HRs for {len(self.genes)} genes by cox univariate regression")
            # HRs = get_HRs(data,H,genes,FEAT)
            HRs = get_HRs(self.data,self.data.columns[-2],self.data.columns[-1],50)
            HRs.to_csv(fn_HRs)
        
        return HRs,data_es

    def get_triplets_statistics(self,cancer_name):
        '''
        获得所有三元组的统计学分数
        '''
        triplets_statistics = {}
        data_es= self.data_es.copy()
        # time_event = self.data_es.iloc[:,-2:]
        data_es.columns = data_es.columns.str.lower()
        # 遍历所有高分对
        num_unr_e = sum([isinstance(i,list) for i in self.hyperedge_name])
        logger.info(f"There are {num_unr_e} edges in total {len(self.hyperedge_name)} edges, which are not enriched in any module !")
        n=0
        for i,(s_p,i_n,i_h) in enumerate(self.pair_borda_scores.T):
            i_n,i_h = int(i_n),int(i_h)
            gene_name = self.node_name[i_n]
            pathway_name = self.hyperedge_name[i_h]
            if pathway_name is not None and not isinstance(pathway_name,list):
                n+=1
                # 判断hyperedhe是通路或是功能
                MF = pd.read_csv(f"{self.dn_data}/GSVA/GO_Molecular_Function_2023/gseapy.gene_set.gsva.report.csv")['Term']
                BP = pd.read_csv(f"{self.dn_data}/GSVA/GO_Biological_Process_2023/gseapy.gene_set.gsva.report.csv")['Term']
                CC = pd.read_csv(f"{self.dn_data}/GSVA/GO_Cellular_Component_2023/gseapy.gene_set.gsva.report.csv")['Term']
                if pathway_name in MF.values:
                    name_hyperedge = "molecular_function"
                elif pathway_name in BP.values:
                    name_hyperedge = "biological_process"
                elif pathway_name in CC.values:
                    name_hyperedge = "cellular_component"
                else:
                    name_hyperedge = "pathway"

                triplets_statistics[f'triplet{n}']={}
                ts = triplets_statistics[f'triplet{n}']
                logger.info(f"writing triplet{n} ... ...")
                ts['cancer name'] = cancer_name
                ts['gene name'] = gene_name
                ts[f'{name_hyperedge} name'] = pathway_name

                # ### Optional
                ts[f"XAI scores of gene '{gene_name}' and {name_hyperedge} '{pathway_name}' in '{cancer_name}' cancer"]=s_p
                ts[f"Papers count of gene '{gene_name}' and {name_hyperedge} '{pathway_name}' and '{cancer_name}' cancer in Pubmed"] = self.get_paper_count(name_hyperedge,{'gene':gene_name, 'cancer':cancer_name, f'{name_hyperedge}':pathway_name})
                # ### Optional

                ensgid = self.genes[i_n]
                hr_gene,p_gene = self.HRs_genes.loc[ensgid,:]
                # regu inform write
                ts[f"gene '{gene_name}' bioinformatics statistics indicators"] = f"cox hazard ratio for gene expression: {hr_gene}; cox p value for gene expression: {p_gene};"

                # 小写化以方便进行检索
                pathway_name = pathway_name.lower()

                # 计算pearson相关性系数between gene and {name_hyperedge} 以得到基因通路调控关系
                cor_pearson,p_pearson=pearsonr(self.expr.loc[:,ensgid].T.values,data_es.loc[:,pathway_name].values)
                ts[f"gene '{gene_name}' and {name_hyperedge} '{pathway_name}' bioinformatics statistics indicators"] = f"Pearson correlation coefficient between genes and pathways: {cor_pearson}; Pearson correlation coefficient p value: {p_pearson};"

                if pathway_name not in data_es.columns:
                    logger.info(f"pathway {pathway_name} not in gene set enrichment results")
                    hsa = self.H_df.columns[[pathway_name in he for he in [hyperedge.lower() for hyperedge in self.hyperedge_name]]][0]
                    if self.H_df.loc[:,hsa].sum()==1:
                        logger.info(f"pathway {pathway_name} only has one gene!")
                else:
                    expr_path = data_es.loc[:,[pathway_name,'time','event']]#此时已经小写化
                    expr_path.columns = ['ES','time','event']
                    ##  cox uni for pathwat expression
                    logger.info(f"calculating HRs for {pathway_name} pathways by cox univariate regression")
                    HR_pathway = get_HRs(expr_path,'time','event',1)
                    p_path_cox,hr_path = HR_pathway.values[0,:]
                    ts[f"pathway '{pathway_name}' bioinformatics statistics indicators"]=\
                    f"Cox hazard ratio for enrichment score: {hr_path}; Cox p value for enrichment score: {p_path_cox}; "
                    diff_es,(t_statistic,p_value) = self.get_regulation_diff(expr_path)

                    ts[f"pathway '{pathway_name}' bioinformatics statistics indicators"]+=\
                    f"The difference between the sum of {name_hyperedge} enrichment scores of high-risk and low-risk populations of {cancer_name} cancer: {diff_es}; The ttest information (t_statistic, p_value) of between the {name_hyperedge} enrichment scores of high-risk and low-risk populations of {cancer_name} cancer: {((t_statistic,p_value))};"
                    logger.info(f"{json.dumps(ts,indent=4)}")
            else:
                self.n_triplets+=1
            if i==self.n_triplets-1:
                break
        return triplets_statistics