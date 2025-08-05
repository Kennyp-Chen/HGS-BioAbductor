import numpy as np
import pandas as pd
import pickle as pkl
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfTransformer
# using scipy hier cluster
from scipy.cluster.hierarchy import linkage,fcluster,maxRstat,maxinconsts,inconsistent
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt
from utils.ReactomeNet import *
import os
import torch
# define pearson correlation matrix calculation function
def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)

def hcluster(criterion,geneTree,t):
    if criterion=="monocrit":
        R = inconsistent(geneTree)
        MR = maxRstat(Z = geneTree,R = R, i=3)
        labels=fcluster(geneTree, t=t, criterion='monocrit', monocrit=MR)
    elif criterion=="maxclust_monocrit":
        R = inconsistent(geneTree)
        MI = maxinconsts(geneTree, R)
        if t<1:
            raise
        labels=fcluster(geneTree, t=t, criterion='maxclust_monocrit', monocrit=MI)# t>1 int
    else:
        if (criterion=="maxclust" and t<1)or(criterion!="maxclust" and t>1):
            raise
        labels = fcluster(geneTree,t=t,criterion=criterion)
    return labels
def Construct_Hierachy_Clust_H(df_exp,method="average",t=0.3,criterion='distance'):
    '''
    Using Pearson correlation compute the gene coexpression out, distance matrix=1-out, then hierarchy clusterize to get computed Hypergraph Incidence Matrix H.

    df_exp: gene expression data frame, index:patients id , columns: genes id
    t:      The threshold to apply when forming flat clusters.  If list_type gen hier H ,elif num_type then flat H.
    methods: linkage methods are used to compute the distance between two clusters. Including: single,complete,average,weighted,centroid,median,ward 
    criterion_list  = ["inconsistent","distance","maxclust","monocrit","maxclust_monocrit"]
    method_list     = ["single","complete","average","weighted","centroid","median","ward"]
    
    '''
    X = df_exp.values
    out = np_pearson_cor(X,X)
    out[np.isnan(out)] = 0

    # Pearson Cor Mat[Sim Mat]
    out = np.abs(out)

    # Distance Mat
    dm = 1 - out

    a = squareform(dm.round(decimals=8), checks=False)
    geneTree = linkage(a, method=method)# method 聚类计算距离的方式，可变化
    # Clusterize the data
    # criterion=="monocrit" or "maxclust" t should>1 and int type

    if str(type(t))=="<class 'list'>" and len(t)>0:
        H=[]
        for i in range(len(t)):
            a=t[i]
            s1 = X.shape[1] if i==0 else eval(f"labels{i-1}.max()") # last labels max 
            exec(f"labels{i} = hcluster(criterion=criterion,geneTree=geneTree,t=a)")
            exec(f"s2 = labels{i}.max()")# this labels max 
            H_array = eval("np.zeros([s1,s2])")
            # df_Hcom = pd.DataFrame(data=0,index=df_exp.columns,columns=np.unique(labels))
            for j in range(X.shape[1]):
                col = eval(f"labels{i}[j]") if i==0 else eval(f"labels{i}[j]")
                idx = j if i==0 else eval(f"labels{i-1}[j]")
                H_array[idx-1,col-1] = 1
            H.append(H_array)
        return H# list
    elif str(type(t))=="<class 'float'>" or str(type(t))=="<class 'int'>":
        labels = hcluster(criterion=criterion,geneTree=geneTree,t=t)
        # Keep the indices to sort labels
        print("Hyperedges computed by Pearson-Hcluster=",labels.max())# hyperedge num 计算出来的超边数量
        # construct H by compute Prior knowledge
        df_Hcom = pd.DataFrame(data=0,index=df_exp.columns,columns=np.unique(labels))
        for i in range(X.shape[1]):
            col = labels[i]
            df_Hcom.iloc[i,col-1] = 1
        return df_Hcom
    else:
        print("wrong type of t or empty list")
        raise




def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable, this is likely useful to penalize very large pathways
    :return: G
    """
    H = np.array(H, dtype=float)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G



def construct_Hexp(fn_H, fn_m, dataset='MC3'):
    """
    init multi-scale hypergraph Vertex-Edge matrix from sparse tensor file
the dataset_gene_pathway.hg file is generated by the gene_pathway_group.R file fro the respective dataset
    :param spt: sparse tensor file
    :return: N_genes x M_hyperedge (pathways)
    """
    # m is patient x gene matrix, y is label
    f = open(fn_m, 'rb')
    if dataset == 'MC3':
        [m, cf, y] = pkl.load(f)
        # get unique label list
        y, yuniques = pd.factorize(y, sort=True)
        idx1 = m.index.tolist()
        # idx1[0] = '0003' # done: need to pre-process and re-save tcga data
        idx1 = ['p|' + x for x in idx1] # p| distinguishes pt node labels
        m.index = idx1
        cf.index = idx1
    elif dataset == 'disgenet':
        [m, y] = pkl.load(f)
        yuniques = y.columns.values
        y = y.values
    else:
        sys.exit(f'unrecognized dataset {dataset}')        
    f.close()

    # H is the gene hypergraph, pathway as hyperedge
    H = pd.read_csv(fn_H, index_col=0)
    
    # Hexp is the patient gene hypergraph, pathway and patient-mutation profile as hyperedge
    Hexp = pd.DataFrame(np.zeros((H.shape[0]+m.shape[0],
                                  H.shape[1]+m.shape[0])),
                        index = H.index.tolist()+m.index.tolist(),
                        columns = H.columns.tolist()+m.index.tolist(),
                        dtype=float)
    Hexp.loc[H.index, H.columns] = H
    Hexp.loc[m.index, m.index] = np.identity(len(m.index))
    Hexp.loc[H.index, m.index] = m.T # [H.index, m.index]

    gene_idx = slice(0, len(H.index), 1)
    subj_idx = slice(len(H.index), len(Hexp.index), 1) # can be pt or dx
    if dataset == 'MC3':
        return Hexp, gene_idx, subj_idx, cf, y, yuniques
    elif dataset == 'disgenet':
        return Hexp, gene_idx, subj_idx, y, yuniques

def construct_Hexp_inductive(H, m, tfidf = False):
    """
    init multi-scale hypergraph Vertex-Edge matrix from sparse tensor file
the dataset_gene_pathway.hg file is generated by the gene_pathway_group.R file fro the respective dataset
    :param 
    fn_H: pathway hyperedge file
    m: training matrix
    :return: N_genes x M_hyperedge (pathways)
    """
    # H is the gene hypergraph, pathway as hyperedge
    # H = pd.read_csv(fn_H, index_col=0).astype('float')
    
    # Hexp adds the training subject-mutation profile as hyperedge
    Hexp = pd.DataFrame(np.zeros((H.shape[0],
                                  H.shape[1]+m.shape[0])),
                        index = H.index.tolist(),
                        columns = H.columns.tolist()+m.index.tolist(),
                        dtype=float)
    Hexp.loc[H.index, H.columns] = H
    Hexp.loc[H.index, m.index] = m.T # [H.index, m.index]

    pathway_idx = slice(0, len(H.index), 1)
    subj_idx = slice(len(H.index), len(Hexp.index), 1)
    if tfidf:
        tx = TfidfTransformer()
        Hexp = tx.fit_transform(Hexp).todense()
    return Hexp, pathway_idx, subj_idx

def construct_Hexp_KNN_inductive(H, m, k_neig=3, is_probH=False, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param corr_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    # H is the gene hypergraph, pathway as hyperedge
    # H = pd.read_csv(fn_H, index_col=0).astype('float')
    
    # Hexp adds the gene KNNs from training subject-mutation profile as hyperedge
    genes = m.columns.tolist()
    Hexp = pd.DataFrame(np.zeros((H.shape[0],
                                  H.shape[1]+m.shape[1])),
                        index = H.index.tolist(),
                        columns = H.columns.tolist() + genes,
                        dtype=float)
    Hexp.loc[H.index, H.columns] = H
    
    corr_mat = np.corrcoef(m, rowvar=False)
    corr_mat = np.nan_to_num(corr_mat)

    n_obj = corr_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj

    pathway_idx = slice(0, len(H.index), 1)
    knn_idx = slice(len(H.index), len(Hexp.index), 1)
    
    for center_idx in range(n_obj):
        corr_vec = corr_mat[center_idx]
        nearest_idx = np.array(np.argsort(corr_vec)).squeeze()
        avg_dis = np.average(corr_vec)
        if not np.any(nearest_idx[-k_neig:] == center_idx):
            nearest_idx[-k_neig] = center_idx

        for node_idx in nearest_idx[-k_neig:]:
            if is_probH:
                Hexp.loc[genes[node_idx], genes[center_idx]] = np.exp(-corr_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                Hexp.loc[genes[node_idx], genes[center_idx]] = 1.0

    return Hexp, pathway_idx, knn_idx


def construct_H(fn_H, tfidf = False):
    """
    init multi-scale hypergraph Vertex-Edge matrix from sparse tensor file
the dataset_gene_pathway.hg file is generated by the gene_pathway_group.R file fro the respective dataset
    :param spt: sparse tensor file
    :return: N_genes x M_hyperedge (pathways)
    """
    # H is the gene hypergraph, pathway as hyperedge
    H = pd.read_csv(fn_H, index_col=0).astype('float')
    if tfidf:
        tx = TfidfTransformer()
        H_tfidf = tx.fit_transform(H).todense()
        H = pd.DataFrame(H_tfidf,index = H.index.tolist(),
                        columns = H.columns.tolist(),
                        dtype=float)
    # H.values=H1
    return H



class construct_H_knn:
    def __init__(self, expression_matrix):
        self.expression_matrix = expression_matrix
        self.num_patients, self.num_genes = expression_matrix.shape
    
    def gene_coexpression(self, methods:str = 'pearson'):
        """
        return: gene coexpression matrix with spearman or pearson coefficient
        """
        return self.expression_matrix.corr(methods).abs()

    
    def calculate_distance_distribution(self, cancerType):
        """
        计算距离分布
        :return: np.array, 距离数组
        """
        # 计算基因共表达矩阵
        coexpression = self.gene_coexpression()
        

        # 计算距离矩阵
        distance_matrix = euclidean_distances(coexpression)
        # 距离矩阵的上三角部分（不包括对角线）
        triu_indices = np.triu_indices_from(distance_matrix, k=1)
        distances = distance_matrix[triu_indices]
        plt.figure(figsize=(10, 6))
        # sns.histplot(distances, bins=30, kde=True)
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title('Distribution of Distances in Gene Co-expression Matrix')
        plt.show()
        plt.savefig(f'{cancerType}_distance_distribution.pdf')
        return distances
    
    def build_knn_incidence_matrix(self, K):
        """
        构建 KNN 超图的关联矩阵
        :param K: int, K 值
        :return: np.array, KNN 关联矩阵, coexpression 基因名称
        """
        coexpression = self.gene_coexpression()
        knn_graph = kneighbors_graph(coexpression, K, mode='connectivity', include_self=True)
        knn_matrix = knn_graph.toarray()
        
        return knn_matrix.T, coexpression.index
    
def dataFrame_gen(df:pd.DataFrame, columns:list):
    for i, row in df.iterrows():
        yield tuple(row[col] for col in columns)

def construct_H_STRING(gene_list:list, layer_STRING:int, config=None):
    '''
    layer_STRING:1-104
    '''
    if config is None:
        raise ValueError("config parameter is required for construct_H_STRING")
    
    # Get paths from config
    paths = config.get_paths()
    
    # Read STRING data files
    clusters_relation = pd.read_table(paths['cluster_relation_file'], sep='\t')    
    gene_cluster_relation = pd.read_csv(paths['string_gene_list'])
    
    Gclusters_relation = dataFrame_gen(clusters_relation, ['parent_cluster_id', 'child_cluster_id'])
    Ggene_cluster_relation = dataFrame_gen(gene_cluster_relation, ['protein_id', 'cluster_id', 'best_described_by'])
    
    STRING = ReactomeNet(Gclusters_relation, Ggene_cluster_relation)
    STRING.prune_to_genes(set(gene_list))
    STRING.prune_to_level(layer_STRING)
    H = STRING.incidence_mat(layer_STRING, layer_STRING-1)
    H = H.loc[H.sum(axis=1)>0, H.sum(axis=0)>0]
    print("H_String shape:", H.shape)
    return H.values

def construct_H_list_STRING(gene_list,layer_STRING_list:list=[104,74,59,24],H_path=None):
    H_masks = []
    H_path+=f"/{layer_STRING_list}/"
    if os.path.isfile(H_path+"/H1.csv"):
        for i in range(len(layer_STRING_list)):
            H = pd.read_csv(H_path+f"H{4-i}.csv",index_col=0).loc[gene_list,:]
            H= H.loc[H.sum(axis=1)>0,H.sum(axis=0)>0]
            H_masks.append(H)
            gene_list=H.columns.tolist()
    else:
        os.makedirs(H_path, exist_ok=True)
        clusters_relation = pd.read_table('/Backup/home/chenyupeng/DATA/Graph/STRING/9606.clusters.tree.v12.0.txt', sep='\t')
        gene_cluster_relation=pd.read_csv("/Backup/home/chenyupeng/DATA/Graph/STRING/clusters.protein.ensg.csv")
        Gclusters_relation = dataFrame_gen(clusters_relation, ['parent_cluster_id', 'child_cluster_id'])
        Ggene_cluster_relation = dataFrame_gen(gene_cluster_relation, ['protein_id', 'cluster_id', 'best_described_by'])
        STRING = ReactomeNet(Gclusters_relation, Ggene_cluster_relation)
        STRING.prune_to_genes(set(gene_list))
        STRING.prune_to_level(max(layer_STRING_list))
        for i in range(len(layer_STRING_list)):
            if i==0:
                H = STRING.incidence_mat(layer_STRING_list[i],layer_STRING_list[i]-1)
            else:
                # 通过连续矩阵相乘
                for j in range(layer_STRING_list[i-1]-layer_STRING_list[i]-1):
                    j+=1
                    if j==1:
                        #first (104-1,104-2) last (104-29,104-30)
                        H = STRING.incidence_mat(layer_STRING_list[i-1]-j,layer_STRING_list[i-1]-j-1)
                                    # first (104-2,104-3) last (104-30,104-31)
                    H = H.dot(STRING.incidence_mat(layer_STRING_list[i-1]-j-1,layer_STRING_list[i-1]-j-2))
            H.to_csv(H_path+f"/H{4-i}.csv")
            H_masks.append(H)

    return H_masks
def convert_hypergraph_to_graph(H):
    """
        # # 示例用法
        # H = torch.tensor([[1, 0, 1],
        #                   [1, 1, 0],
        #                   [0, 1, 1]], dtype=torch.float32)

        # edge_index, adjacency_matrix = convert_hypergraph_to_graph(H)
        # print("Edge Index:\n", edge_index)
        # print("Adjacency Matrix:\n", adjacency_matrix)

    Convert a hypergraph adjacency matrix to a graph's adjacency matrix and edge index.
    
    Parameters:
    H (torch.Tensor): The hypergraph adjacency matrix of shape [num_nodes, num_hyperedges].
    
    Returns:
    torch.Tensor: The graph's edge index with shape [2, num_edges].
    torch.Tensor: The adjacency matrix of the graph with shape [num_nodes, num_nodes].
    """
    # 获取节点数量和超边数量
    num_nodes, num_edges = H.shape
    
    # 创建一个空的列表用于存储边的索引
    edge_index = []
    
    # 遍历每个超边
    for hyperedge_index in range(num_edges):
        # 找到参与该超边的节点
        nodes_in_hyperedge = (H[:, hyperedge_index] > 0).nonzero(as_tuple=True)[0]
        
        # 获取所有节点对 (combinations) 形成边
        if len(nodes_in_hyperedge) > 1:
            # 使用组合得到边
            for i in range(len(nodes_in_hyperedge)):
                for j in range(i + 1, len(nodes_in_hyperedge)):
                    edge_index.append((nodes_in_hyperedge[i].item(), nodes_in_hyperedge[j].item()))
    
    # 转换为 tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).T  # 形状: [2, num_edges]

    # 构造图的邻接矩阵
    num_graph_nodes = H.shape[0]
    adjacency_matrix = torch.zeros((num_graph_nodes, num_graph_nodes), dtype=torch.float32)

    # 填充邻接矩阵
    for edge in edge_index.T:  # 转置为 [num_edges, 2]
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1  # 无向图

    return edge_index, adjacency_matrix

def disturb_H(H,percent=0.1,seed=0):
    '''
    扰动超图,根据percent的比例随机选择num_disturb个超边进行扰动，并随机打乱其节点
    :param H: 超图邻接矩阵[num_nodes,num_hyperedges]
    :param percent: 扰动比例
    :param seed: 随机种子
    :return: 扰动后的超图邻接矩阵
    ''' 
    np.random.seed(seed)
    H_disturb = H.copy()
    num_edges = H.shape[1]
    num_disturb = int(num_edges*percent)
    disturb_idx = np.random.choice(num_edges,num_disturb,replace=False)
    # 随机打乱超边节点,但是超边的节点不能和原来一样，并且尽量保持度相同，如无法做到就取能取值的最大
    for i in disturb_idx:
        nodes_in_hyperedge = (H[:, i] == 1).nonzero()[0]
        nodes_out_hyperedge = (H[:, i] == 0).nonzero()[0]
        degree_i = len(nodes_in_hyperedge)
        degree_o = len(nodes_out_hyperedge)
        if degree_o>=degree_i:
            # 从中随机选择nodes_out_hyperedge随机选择degree_i个索引将超边连接到这些节点
            # 将noeds_in_hyperedge的索引的连接从超边中删除
            nodes_out_hyperedge_idx = np.random.choice(nodes_out_hyperedge,degree_i,replace=False)
            H_disturb[nodes_out_hyperedge_idx,i] = 1
            H_disturb[nodes_in_hyperedge,i] = 0
        else:
            H_disturb[nodes_out_hyperedge,i] = 1
            H_disturb[nodes_in_hyperedge,i] = 0
            
    return H_disturb
    