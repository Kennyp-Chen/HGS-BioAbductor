import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import bootstrap
from utils.survfunc_utils import get_ci_td
from copy import deepcopy
class BordaScoreValidator:
    def __init__(self, model_interprete, num_resamples=50):
        self.model_interprete = model_interprete
        self.num_resamples = num_resamples
        self.node_ranks_df = None
        self.hyperedge_ranks_df = None


    def bootstrap_borda_ranks(self, baseline, model_layer, GSsamples=50, IGsteps=50):
        node_bordas = []
        hyperedge_bordas = []
        for _ in range(self.num_resamples):
            node_borda, hyperedge_borda = self.model_interprete.all_borda_rank(
                model_layer, baselineTensor=baseline, GSsamples=GSsamples, IGsteps=IGsteps
            )
            node_bordas.append(node_borda)
            hyperedge_bordas.append(hyperedge_borda)

        self.node_ranks_df = pd.DataFrame(node_bordas).T
        self.hyperedge_ranks_df = pd.DataFrame(hyperedge_bordas).T
        return self.node_ranks_df, self.hyperedge_ranks_df

    def calculate_confidence_intervals(self):
        node_confidence_intervals = self.node_ranks_df.apply(lambda x: (np.percentile(x, 2.5), np.percentile(x, 97.5)), axis=1)
        hyperedge_confidence_intervals = self.hyperedge_ranks_df.apply(lambda x: (np.percentile(x, 2.5), np.percentile(x, 97.5)), axis=1)
        return node_confidence_intervals, hyperedge_confidence_intervals

    def select_top_bottom_k(self, scores_df, k=10):
        sorted_df = scores_df.sort_values("Borda_Rank")
        top_k = sorted_df.head(k)
        bottom_k = sorted_df.tail(k)
        return top_k, bottom_k

    def calculate_stability(self):
        stability_scores = self.node_ranks_df.std(axis=1)
        return stability_scores

    def jaccard_index(self, set1, set2):
        return len(set1 & set2) / len(set1 | set2)

    def top_k_jaccard(self, k=10, instance='node'):
        if instance == "node":
            df = self.node_ranks_df
        else:
            df = self.hyperedge_ranks_df
        top_k_sets = [set(df.iloc[:, i].nlargest(k).index) for i in range(df.shape[1])]
        jaccard_scores = [
            self.jaccard_index(top_k_sets[i], top_k_sets[j])
            for i in range(len(top_k_sets)) for j in range(i + 1, len(top_k_sets))
        ]
        return np.mean(jaccard_scores)
    
    def bottom_k_jaccard(self, k=10, instance='node'):
        if instance == "node":
            df = self.node_ranks_df
        else:
            df = self.hyperedge_ranks_df
        bottom_k_sets = [set(df.iloc[:, i].nsmallest(k).index) for i in range(df.shape[1])]
        jaccard_scores = [
            self.jaccard_index(bottom_k_sets[i], bottom_k_sets[j])
            for i in range(len(bottom_k_sets)) for j in range(i + 1, len(bottom_k_sets))
        ]
        return np.mean(jaccard_scores)

    def validate_node_borda_scores(self, data_test, node_borda, k, model, outdir):
        top_node = np.argsort(node_borda)[::-1]
        bottom_node = np.argsort(node_borda)
        
        top_remove_ci = []
        bottom_remove_ci = []

        def compute_ci(data, nodes_to_remove):
            """Helper function to compute CI after removing specified nodes."""
            data_less_important = torch.from_numpy(data[:, :-2]).cuda()
            data_less_important[:, list(nodes_to_remove)] = 0
            PMF = model.predict(data_less_important.float())
            surv = torch.clamp(1. - torch.cumsum(PMF, dim=-1), 0, 1)
            ci = get_ci_td(
                s_time=torch.from_numpy(data[:, -2]).cuda(),
                cens=1 - torch.from_numpy(data[:, -1]).cuda(),
                f=surv
            )
            return ci

        for i in range(1, k):
            ci = compute_ci(data_test, top_node[:i])
            top_remove_ci.append(ci)
            print(f'Top nodes removed: {i}, test CI: {ci}')

        for i in range(1, k):
            ci = compute_ci(data_test, bottom_node[:i])
            bottom_remove_ci.append(ci)
            print(f'Bottom nodes removed: {i}, test CI: {ci}')

        # Plotting the c-index changes for both top and bottom node removals
        x_values = range(1, k)
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, top_remove_ci, label='Top Nodes Removed', linestyle='-', marker='.')
        plt.plot(x_values, bottom_remove_ci, label='Bottom Nodes Removed', linestyle='--', marker='o')
        
        # Adding labels, legend, and title
        plt.xlabel('Number of Nodes Removed')
        plt.ylabel('C-Index')
        plt.title('C-Index Changes with Node Removals')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(f'{outdir}/remove_feature_c_index.svg')
        plt.close()
        
        return top_remove_ci, bottom_remove_ci



    
    

    

def validate_hyperedge_borda_scores(data_test, hyperedge_borda, adj_matrix, percent, model, method='baseline'):
    """
    Validate hyperedge Borda scores by zeroing out all node features connected to the top-K and bottom-K hyperedges.
    
    Parameters:
    - data_test: Test dataset.
    - hyperedge_borda: Scores for each hyperedge.
    - adj_matrix: Adjacency matrix indicating node-hyperedge connections.
    - percent: percent of top and bottom hyperedges to iteratively remove.
    - models: The prediction.
    - outdir: Directory to save the plots.
    - method baseline: remove hyperedges based on the feature values. connection removed hyperedges based on incidence connection.
    Returns:
    - top_remove_ci: C-Index scores after removing nodes for top-K hyperedges.
    - bottom_remove_ci: C-Index scores after removing nodes for bottom-K hyperedges.
    """
    top_hyperedges = np.argsort(hyperedge_borda)[::-1]
    bottom_hyperedges = np.argsort(hyperedge_borda)
    top_remove_ci = []
    bottom_remove_ci = []
    original_adj = model.H0[0].clone().detach()
    def compute_ci(data, hyperedges_to_remove):
        data_tensor = torch.from_numpy(data[:, :-2]).cuda()       

        if len(hyperedges_to_remove) != 0:
            connected_nodes = np.unique(np.where(adj_matrix[:, hyperedges_to_remove] > 0)[0])
            if method == 'baseline':
                data_tensor[:, connected_nodes] = 0
            if method == 'connection':
                model.H0[0][list(hyperedges_to_remove), :] = 0
        
        PMF = model.predict(data_tensor.float())
        surv = torch.clamp(1. - torch.cumsum(PMF, dim=-1), 0, 1)
        ci = get_ci_td( 
            s_time=torch.from_numpy(data[:, -2]).cuda(),
            cens=1 - torch.from_numpy(data[:, -1]).cuda(),
            f=surv
        )
        model.H0[0] = original_adj.clone().detach()

        return ci
    percent_space = np.linspace(0, percent, num=round(percent*10+1))
    for percentage in percent_space:
        remove_hyperedges_num = int(percentage*len(hyperedge_borda))
        ci_top = compute_ci(data_test, top_hyperedges[:remove_hyperedges_num])
        top_remove_ci.append(ci_top)
        print(f'Top {remove_hyperedges_num} hyperedges removed, test CI: {ci_top}')
        ci_bottom = compute_ci(data_test, bottom_hyperedges[:remove_hyperedges_num])
        bottom_remove_ci.append(ci_bottom)
        print(f'Bottom {remove_hyperedges_num} hyperedges removed, test CI: {ci_bottom}')
    
    return percent_space, top_remove_ci, bottom_remove_ci

def validate_node_borda_scores(data_test, node_borda, adj_matrix, percent, model, method='baseline'):
    """
    Validate node Borda scores by zeroing out all node features connected to the top-K percent and bottom-K node hyperedges.
    
    Parameters:
    - data_test: Test dataset.
    - node_borda: Scores for each node.
    - adj_matrix: Adjacency matrix indicating node-hyperedge connections.
    - percent: Percent of top and bottom nodes to iteratively remove.
    - model: The prediction model.
    - method: 'baseline' removes based on feature values; 'connection' removes hyperedges based on incidence connection.
    
    Returns:
    - percent_space: Percentage of nodes removed.
    - top_remove_ci: C-Index scores after removing nodes for top-K hyperedges.
    - bottom_remove_ci: C-Index scores after removing nodes for bottom-K hyperedges.
    """
    top_nodes = np.argsort(node_borda)[::-1]
    bottom_nodes = np.argsort(node_borda)
    top_remove_ci = []
    bottom_remove_ci = []

    # Use clone() to create an independent copy of the adjacency matrix
    original_adj = model.H0[0].clone().detach()

    def compute_ci(data, nodes_to_remove):
        data_tensor = torch.from_numpy(data[:, :-2]).cuda()
        if method == 'baseline':
            data_tensor[:, list(nodes_to_remove)] = 0
        elif method == 'connection':
            model.H0[0][:, list(nodes_to_remove)] = 0
        PMF = model.predict(data_tensor.float())
        surv = torch.clamp(1. - torch.cumsum(PMF, dim=-1), 0, 1)
        ci = get_ci_td(
            s_time=torch.from_numpy(data[:, -2]).cuda(),
            cens=1 - torch.from_numpy(data[:, -1]).cuda(),
            f=surv
        )
        # Restore the original adjacency matrix
        model.H0[0] = original_adj.clone().detach()
        return ci

    # Generate percentages for node removal
    percent_space = np.linspace(0, percent, num=round(percent * 10 + 1))
    for percentage in percent_space:
        remove_nodes_num = int(percentage * len(node_borda))
        
        # Compute CI for top nodes
        ci_top = compute_ci(data_test, top_nodes[:remove_nodes_num])
        top_remove_ci.append(ci_top)
        print(f'Top {remove_nodes_num} nodes removed, test CI: {ci_top}')
        
        # Compute CI for bottom nodes
        ci_bottom = compute_ci(data_test, bottom_nodes[:remove_nodes_num])
        bottom_remove_ci.append(ci_bottom)
        print(f'Bottom {remove_nodes_num} nodes removed, test CI: {ci_bottom}')

    return percent_space, top_remove_ci, bottom_remove_ci
