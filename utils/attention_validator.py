import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
import numpy.typing as npt
from typing_extensions import override
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.survfunc_utils import get_ci_td

class AttentionValidator:
    def __init__(self, 
                 attention: torch.Tensor, 
                 rows: str = 'gene') -> None:
        """
        attention: Attention matrix of genes over hyperedges or hyperedges over genes. 
                  Shape [batch_size, gene_num, hyperedge_num] or [batch_size, hyperedge_num, gene_num].
        gene_id: Array of gene symbols.
        hyperedge_id: Array of hyperedge identifiers.
        rows: "gene" if rows represent genes, else "hyperedge".
        """
        self.Atn = attention
        self.original_atn = attention.detach().clone()
        self.rows = rows
        

    
    def get_top_indices(self, percent: float):
        # 统计每个 hyperedge 的非零值数量（沿 gene 维度）
        non_zero_mask = self.original_atn > 0  # (batch, hyperedge, gene)
        non_zero_counts = non_zero_mask.sum(dim=-1)  # (batch, hyperedge)

        # 计算每个 hyperedge 需要移除的 gene 数量
        remove_k_per_row = (non_zero_counts * percent).long()  # (batch, hyperedge)
        remove_k_per_row = torch.clamp(remove_k_per_row, min=0)

        # 对每个 hyperedge 的 gene 注意力值降序排序
        sorted_values, sorted_indices = torch.sort(
            self.original_atn, 
            dim=-1, 
            descending=True
        )  # sorted_values: (batch, hyperedge, gene)

        # 生成掩码：标记需要移除的 top-k gene 索引
        batch_size, num_hyperedges, num_genes = sorted_values.shape
        device = sorted_values.device
        remove_mask = torch.zeros_like(sorted_values, dtype=torch.bool)

        for b in range(batch_size):
            for h in range(num_hyperedges):
                k = remove_k_per_row[b, h].item()
                if k > 0:
                    remove_mask[b, h, :k] = True  # 移除前 k 个（最高值）

        # 获取需要移除的 (batch, hyperedge, gene) 索引
        batch_indices, hyperedge_indices, sorted_gene_indices = torch.where(remove_mask)
        gene_indices = sorted_indices[batch_indices, hyperedge_indices, sorted_gene_indices]
        return batch_indices, hyperedge_indices, gene_indices
    
    def update_model_node_attention(self, model, batch_indices, row_indices, col_indices):
        """
        Updates the model's attention layers with the modified attention values.
        model: The model containing the attention layers.
        """
        with torch.no_grad():
            model.hgc_encoder[0].hgat_layers[0].evaluate = 'node'
            model.hgc_encoder[0].hgat_layers[0].mask = (batch_indices, row_indices, col_indices)

    def update_model_hyperedge_attention(self, model, batch_indices, row_indices, col_indices):
        """
        Updates the model's attention layers with the modified attention values.
        model: The model containing the attention layers.
        """
        with torch.no_grad():
            model.hgc_encoder[0].hgat_layers[0].evaluate = 'edge'
            model.hgc_encoder[0].hgat_layers[0].mask = (batch_indices, row_indices, col_indices)

    def validate(self, model, data, step_percent=0.1, max_percent=1):
        """
        Validate the impact of attention masking.
        model: Trained model to evaluate.
        step_percent: Step size for percentage removal.
        max_percent: Maximum percentage of removal.
        """
        c_index_changes = []
        attentions = []
        percentages = np.arange(0, max_percent + step_percent, step_percent)

        for percent in percentages:
            batch_indices, row_indices, col_indices = self.get_top_indices(percent)
            # Re-run the model and calculate c-index
            self.update_model_hyperedge_attention(model, batch_indices, row_indices, col_indices)

            PMF = model.predict(torch.Tensor(data[:, :-2]).cuda().float())
            surv = torch.clamp(1. - torch.cumsum(PMF, dim=-1), 0, 1)
            ci = get_ci_td(
                s_time=torch.from_numpy(data[:, -2]).cuda(),
                cens=1 - torch.from_numpy(data[:, -1]).cuda(),
                f=surv
            )
            c_index_changes.append(ci)
            attentions.append(model.hgc_encoder[0].hgat_layers[0].attention_edge.detach().clone())
            # Restore original attention matrix for the next iteration
            self.Atn = self.original_atn.detach().clone()


        return c_index_changes, attentions
        


    def compare_attention(self):
        """
        Compare the attention scores between patients.
        Returns pairwise comparison of attention for each patient.
        """
        # 获取原始 attention 张量
        attention_tensor = self.original_atn  # Shape: (batch_size, num_hyperedges, num_genes)

        # 计算每个患者的平均 attention 向量
        mean_attention = attention_tensor.mean(dim=1)  # Shape: (batch_size, num_genes)

        # 初始化差异矩阵
        batch_size = mean_attention.shape[0]
        attention_diff = torch.zeros(batch_size, batch_size)

        # 计算患者间的余弦相似度
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # 计算余弦相似性
                distance = F.cosine_similarity(mean_attention[i].unsqueeze(0), mean_attention[j].unsqueeze(0))
                attention_diff[i, j] = distance
                attention_diff[j, i] = distance  # 对称矩阵

        return attention_diff
    

def jaccard_distance_tensor(tensor_a, tensor_b):
    """
    Compute the Jaccard distance between two tensors of indices.
    tensor_a, tensor_b: Two tensors (PyTorch tensors of indices).
    """
    # 获取唯一值

    # 计算交集和并集
    intersection = (tensor_a == tensor_b).sum()
    print(f'intersection:{intersection}')
    union = len(tensor_a)
    print(f'union:{union}')
    # 计算 Jaccard 距离
    return 1 - (intersection / union if union != 0 else 0.0)