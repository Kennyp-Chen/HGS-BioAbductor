import os
import pandas as pd
import numpy as np
import torch
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from utils.hg_ops import *
import time
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.base import SurvivalAnalysisMixin
from lifelines import statistics, CoxPHFitter
import configparser
from utils.ReactomeNet import *
from typing import Tuple, Any, List
from torch.utils.data import Dataset
from gprofiler import GProfiler
from sklearn.preprocessing import StandardScaler 
from utils.config_loader import ConfigLoader
from utils.hg_ops import construct_H_STRING
# 全局加载config
config = ConfigLoader("config.yaml")

def read_config(config_file):
    ''' 
    Performs read config file and return a 2-layer dict.

    :param config_file: (str) the path of a .ini file.
    :return config_infor: (dict) the dictionary of information in config_file.
    '''
    def _build_dict(items):
        return {item[0]: eval(item[1]) for item in items}
    # create configparser object
    cf = configparser.ConfigParser()
    # read .config file
    cf.read(config_file)
    config = {sec: _build_dict(cf.items(sec)) for sec in cf.sections()}

    return config

# def load_TCGA_data(data_path, endpoint, Filt_num):
#     # get RNAseq matrix
#     data_df = pd.read_csv(
#         os.path.join(data_path +  '/features.csv'), 
#         index_col= 0,
#     )

#     # get survival info
#     survival_df = pd.read_csv(
#         "/Backup/home/chenyupeng/DATA/COX_Selection/ClinicalDataFrame_DiscreteTime-Cut15Years.csv",
#         index_col= 0,
#     )

#     # get gene info
#     gene_info_df = pd.read_csv(
#         os.path.join(data_path + '/genes_list.csv'),
#         index_col= 0,
#     )
#     # Pats Filt
#     sdf=survival_df.loc[survival_df["PatientID"].isin(data_df.columns,),:]
#     survival_df = sdf.loc[~sdf.duplicated(),:]
#     # FS Filt
#     data_df=data_df.iloc[:Filt_num,:]
#     gene_info_df=gene_info_df.iloc[:Filt_num,:]
   
#     print(endpoint)
#     # transfer to numpy format
#     data = data_df.to_numpy().transpose()
#     patients = survival_df["PatientID"].to_numpy()
#     # gene_id = gene_info_df["Gene_ID"].to_numpy()
#     # gene_symbol = gene_info_df["Gene_symbol"].to_numpy()
#     # event_status = survival_df[endpoint + ' Status'].to_numpy()
#     event_status = survival_df['OS Status'].to_numpy()
#     event_time = survival_df[endpoint].to_numpy()

#     # wrap to a dataset dictionary
#     dataset = {
#         'data': data,
#         'patients': patients,
#         # 'gene_id': gene_id,
#         # 'gene_symbol': gene_symbol,
#         'event_status': event_status,
#         'event_time': event_time,
#     }
#     data = np.c_[data,event_time,event_status]

#     return dataset, data


def load_TCGA_data(fn_data, endpoint, Filt_num):
    '''
    load survival data and features data
    return dataframe, index: patients ; cols: genes+time+event
    '''
    data_df = pd.read_csv(fn_data, index_col=0)
    survival_path = config.get_paths()['survival_data_template']
    survival_df = pd.read_csv(survival_path, index_col=0)
    sdf = survival_df.loc[survival_df["PatientID"].isin(data_df.columns,), :]
    survival_df = sdf.loc[~sdf.duplicated(), :]
    if Filt_num == -1:
        Filt_num = data_df.shape[0]
    data_df = data_df.iloc[:Filt_num, :]
    data = data_df.to_numpy().transpose()
    event_status = survival_df['OS Status'].to_numpy()
    event_time = survival_df[endpoint].to_numpy()
    data = np.c_[data, event_time, event_status]
    col = data_df.index.to_list() + ["time", "event"]
    data_df = pd.DataFrame(data=data, index=data_df.columns, columns=col)
    return data_df

def load_TCGA_H_data(H_path, fn_data, endpoint='OS_60', Filt_num=1000, multilevel=False):
    data = load_TCGA_data(fn_data, endpoint, Filt_num=Filt_num)
    H = pd.read_csv(os.path.join(H_path, 'H1.csv'), index_col=0)
    genes = data.columns[:-2]
    H = H.loc[H.index.isin(genes), :]
    # 填补H： data中有的而H中没有的基因连接于统一邻接于一个新的超边，权重为1
    genes_out = genes[~genes.isin(H.index)]
    print(f"{len(genes_out)} genes are not in H, and are set to 0 in H")
    if len(genes_out) > 0:
        H = pd.concat([H, pd.DataFrame(np.zeros((Filt_num - H.shape[0], H.shape[1])), index=genes_out, columns=H.columns)], axis=0)
        H['unconnected'] = 0
        H.loc[genes_out, 'unconnected'] = 1
    # H 和 data 基因顺序对齐
    H = H.loc[genes, :]
    # path purne
    H = H.loc[:, H.sum(axis=0) != 0]

    return data, H

def load_cohort_data(fn_H, fn_data, num_filt, DF=False):
    data_df = pd.read_csv(fn_data, index_col=0)
    label_df = data_df.iloc[:, -2:]
    if fn_H is not None and os.path.isfile(fn_H):
        H = pd.read_csv(fn_H, index_col=0)
        H = H.loc[H.index.isin(data_df.columns), :]
        print("Num of feature matrix's genes in H: ", data_df.columns.isin(H.index).sum())
        feature_matrix = data_df.loc[:, data_df.columns.isin(H.index)]
        if num_filt > data_df.shape[1] - 2:
            raise ValueError("num of filting features are more than the total number of features, please check the num_filt")
        feature_matrix = feature_matrix.iloc[:, :num_filt]
        H = H.loc[feature_matrix.columns, :]
        H = H.loc[:, H.sum(axis=0) != 0]
        print(f"Incidence matrix shape: {H.shape}")
    else:
        if num_filt > data_df.shape[1] - 2:
            raise ValueError("num of filting features are more than the total number of features, please check the num_filt")
        feature_matrix = data_df.iloc[:, :num_filt]
        H = None
    data_df = pd.concat([feature_matrix, label_df], axis=1)
    if not DF:
        return data_df.values, H.values if H is not None else None
    return data_df, H


def load_proteomic_data(data_path, Filt_num, paper="Nature", n=0, e=0, log2=False):

    if log2 == True and paper=="Cell":
        data_df = pd.read_csv(os.path.join(data_path, paper + "-DATA-zscore-log2.csv"), index_col=0)
    else:
        data_df = pd.read_csv(os.path.join(data_path, paper + "-DATA-zscore.csv"), index_col=0)

        
    label_df = data_df[["TimeBin_month","OS"]]# Time event
    fn_H1 = os.path.join(data_path, paper + "-H.csv")
    feature_df = data_df.loc[:, ~data_df.columns.isin(["OS month", "TimeBin_month", "OS"])]
    H = construct_H(fn_H1, tfidf=False)
    if paper == "Nature":
        clinical_df = feature_df.iloc[:, :4]
        proteomic_df = feature_df.iloc[:, 4:]
    else:
        clinical_df = None
        proteomic_df = feature_df

    # 取交集
    proteomic_df = proteomic_df.loc[:, proteomic_df.columns.isin(H.index)]
    H = H.loc[H.index.isin(proteomic_df.columns), :]
    # features fil by cox
    proteomic_df = proteomic_df.iloc[:, :Filt_num]
    H = H.loc[proteomic_df.columns, :]
    H = H.loc[:, H.sum(axis=0) != 0]

    data = pd.concat([proteomic_df, label_df], axis=1).values

    # LapMat Compute
    if e == 0 and n == 0:
        G = None
    else:
        if e > 0 and n <= 0:
            G = generate_G_from_H(H.T)
        else:
            G = generate_G_from_H(H)
    print(f"After Preprocess: DATA shape - {data.shape} H : {H.shape}")

    return data, np.array(H), clinical_df, G


def load_data_hier(data_path, H_path, endpoint, Filt_num, e=0, n=0, tfidf=False):
    '''
    Add P4-6 H
    e standards for use edge_loss or no, 0 for no
    '''
    data_df = pd.read_csv(os.path.join(data_path, 'feature_matrix.csv'), index_col=0)
    survival_path = config.get_paths()['survival_data_template']
    survival_df = pd.read_csv(survival_path, index_col=0)
    gene_info_df = pd.read_csv(os.path.join(data_path, 'genes_list.csv'), index_col=0)
    sdf = survival_df.loc[survival_df["PatientID"].isin(data_df.columns,), :]
    survival_df = sdf.loc[~sdf.duplicated(), :]
    n_H = "H1.csv"
    fn_H1 = os.path.join(H_path, n_H)
    H = construct_H(fn_H1, tfidf=tfidf)
    data_df = data_df.loc[data_df.index.isin(H.index), :]
    gene_info_df = gene_info_df.loc[data_df.index, :]
    H1 = H.loc[data_df.index, :]

    # Reactome cox fil
    H1 = H1.iloc[:Filt_num, :]
    H1 = H1.loc[:, H1.sum(axis=0) != 0]
    # if coe:
    #     H1 = H1.loc[:,H1.sum(axis=0)>50]#path purne coexp 

    # FS Filt
    data_df = data_df.iloc[:Filt_num, :]
    gene_info_df = gene_info_df.iloc[:Filt_num, :]
    if (data_df.index != H1.index).sum() + (H1.index != gene_info_df.index).sum() > 0:
        raise
    data = data_df.to_numpy().transpose()
    patients = survival_df["PatientID"].to_numpy()
    event_status = survival_df['OS Status'].to_numpy()
    event_time = survival_df[endpoint].to_numpy()

    # datadf stand
    data_mmstd = ((data_df - data_df.min()) / (data_df.max() - data_df.min())).T
    # wrap to a dataset dictionary
    dataset = {
        'data_mmstd': data_mmstd,
        'data_z': data,
        'patients': patients,
        'event_status': event_status,
        'event_time': event_time,
        'Incidence Matrix': H1,
    }
    data = np.c_[data, event_time, event_status]
    col = data_df.index.to_list() + [endpoint, 'OS Status']
    data_df = pd.DataFrame(data=data, index=data_df.columns, columns=col)
    if e == 0 and n == 0:
        G = None
    else:
        if e > 0 and n <= 0:
            G = generate_G_from_H(H1.T)
        else:
            G = generate_G_from_H(H1)
    print(f"After Preprocess: DATA shape - {data.shape} H1 : {H1.shape}")

    return dataset, data, np.array(H1), G, data_df


def data_split(seed, data, dataset: str = None, HD={"knowledge_base": "hcluster", "layer_STRING": 100}, construct_H=False, H_df=False):
    (label, time) = (data[:, -1], data[:, -2]) if isinstance(data, np.ndarray) else (data.iloc[:, -1], data.iloc[:, -2])
    t_obs = time.max() + 2 
    data_train_val, data_test, y_train_val, y_test = train_test_split(data, label, test_size=0.2, random_state=seed, shuffle=True, stratify=label)
    data_train, data_valid, _, _ = train_test_split(data_train_val, y_train_val, test_size=0.25, random_state=seed, shuffle=True, stratify=y_train_val)
    if construct_H:
        if HD["type_know"] == "hcluster":
            H = Construct_Hierachy_Clust_H(data_train.iloc[:, :-2], method=HD['method'], criterion=HD['criterion'], t=HD['num_clusters'])
            H = H if H_df else H.values
            return data_train.values, data_valid.values, data_test.values, t_obs, H
        elif HD["type_know"] == "STRING":

            return data_train.values, data_valid.values, data_test.values, t_obs


    else:
        if isinstance(data, np.ndarray):
            return data_train, data_valid, data_test, t_obs
        elif isinstance(data, pd.DataFrame):
            return data_train.values, data_valid.values, data_test.values, t_obs


def construct_Pnet_H_Reactome(H_path, data, layers_list: list = [1, 2, 3, 4]):
    genes = data.columns.tolist()
    H_list = []
    layer_max = max(layers_list)
    if os.path.isfile(os.path.join(H_path, f"H{layer_max}.csv")):
        for i in layers_list:
            H_list.append(pd.read_csv(os.path.join(H_path, f"H{i}.csv"), index_col=0))
    else:
        os.makedirs(H_path, exist_ok=True)
        p2p = pd.read_table(config.get_paths()['reactome_p2p'], sep='\t')
        g2p = pd.read_table(config.get_paths()['reactome_g2p'], sep='\t')
        pathways_relation = dataFrame_gen(p2p, ['Parent', 'Child'])
        gene_pathway = dataFrame_gen(g2p, ['Gene', 'Identifier', 'Orthologous Events'])
        reactome_net = ReactomeNet(pathways_relation, gene_pathway)
        reactome_net.prune_to_genes(set(genes))
        reactome_net.prune_to_level(layer_max)
        for i in layers_list:
            H = reactome_net.incidence_mat(i, i - 1)
            H.to_csv(os.path.join(H_path, f"H{i}.csv"))
            H_list.append(H)
    H_gene = H_list[-1]
    fm = data.loc[:, data.columns.isin(H_gene.index)].iloc[:, :1000]
    data = pd.concat([fm, data.iloc[:, -2:]], axis=1)
    H = H_gene.loc[fm.columns, :]
    H = H.loc[H.sum(axis=1) != 0, H.sum(axis=0) != 0]
    H_list[-1] = H
    pathway_masks = []
    pathway_masks.append(H)
    for i in range(layer_max - 1):
        i = -(i + 2)
        H = H_list[i].loc[H_list[i + 1].columns, :]
        H = H.loc[H.sum(axis=1) != 0, H.sum(axis=0) != 0]
        H_list[i] = H
        pathway_masks.append(H)
    return pathway_masks, data

# def load_TCGA_data(data_path,H_path,Filt_num,endpoint,DF=False,pathway_mask=False):
    # '''
    # For loading P3-P12 layers Reactome H ; select root in 29 differents root path ;  
    # Add Computed Prior Knowledges; Pearson+Hcluster
    # e standards for use edge_loss or no, 0 for no
    # '''
    # # get RNAseq expression matrix
    # data_df = pd.read_csv(os.path.join(data_path + '/feature_matrix.csv'), index_col= 0,)

    # # get patients' survival data
    # survival_df = pd.read_csv("/Backup/home/chenyupeng/DATA/TCGA-RNAseq/COX_Selection/ClinicalDataFrame_DiscreteTime-Cut15Years.csv",index_col= 0,)

    # # get gene info; which selected by cox uni reg
    # gene_info_df = pd.read_csv(os.path.join(data_path +  '/genes_list.csv'),index_col= 0,)
    # # gene_info_df = pd.read_csv(os.path.join(data_path +  '/genes_information.csv'),index_col= 1,)
    
    # H = pd.read_csv(H_path+"/H1.csv",index_col=0)

    # # SurData Pats Filt
    # sdf=survival_df.loc[survival_df["PatientID"].isin(data_df.columns,),:]
    # survival_df = sdf.loc[~sdf.duplicated(),:]

    # # 取reactome 与feature matrix 交集
    # data_df = data_df.loc[data_df.index.isin(H.index),:]
    # # 排序
    # gene_info_df=gene_info_df.loc[data_df.index,:]
    # H = H.loc[data_df.index,:]

    # # Reactome cox fil
    # H = H.iloc[:Filt_num,:]#gene purne
    # H = H.loc[:,H.sum(axis=0)!=0]#path purne

    # # FS Filt
    # data_df = data_df.iloc[:Filt_num,:]
    # gene_info_df=gene_info_df.iloc[:Filt_num,:]
    # if (data_df.index != H.index).sum()+(H.index != gene_info_df.index).sum()>0:
    #     raise
    # H_df=H
    # H=H.values
    # data = data_df.to_numpy().transpose()
    # event_status = survival_df['OS Status'].to_numpy()
    # event_time = survival_df[endpoint].to_numpy()
    # data = np.c_[data,event_time,event_status]
    # if pathway_mask:
    #     pathway_masks=[]
    #     pathway_masks.append(H_df)
    #     for i in range(3):
    #         # 2-4
    #         Hn=pd.read_csv(H_path+f"/H{i+2}.csv",index_col=0)
    #         H_df=Hn.loc[H_df.columns,:]
    #         H_df = H_df.loc[:,H_df.sum(axis=0)!=0]#path purne
    #         pathway_masks.append(H_df)
    #     return data,pathway_masks
    # print(f"After Preprocess: DATA shape - {data.shape} H shape : {H.shape}")
    # if DF:
    #     data_df =  pd.DataFrame(data=data,index=data_df.columns,columns=data_df.index.to_list()+["time","event"],)
    #     return data_df,H_df
    # else:
    #     return data,H
                
def load_opt_data(data_path, H_path, Filt_num, endpoint, DF=False, pathway_mask=False):
    data_df = pd.read_csv(os.path.join(data_path, 'feature_matrix.csv'), index_col=0)
    survival_path = config.get_paths()['survival_data_template']
    survival_df = pd.read_csv(survival_path, index_col=0)
    gene_info_df = pd.read_csv(os.path.join(data_path, 'genes_list.csv'), index_col=0)
    H = pd.read_csv(os.path.join(H_path, 'H.csv'), index_col=0)
    sdf = survival_df.loc[survival_df["Patient"].isin(data_df.columns,), :]
    survival_df = sdf.loc[~sdf.duplicated(), :]
    data_df = data_df.loc[data_df.index.isin(H.index), :]
    gene_info_df = gene_info_df.loc[data_df.index, :]
    H = H.loc[data_df.index, :]
    H = H.iloc[:Filt_num, :]
    H = H.loc[:, H.sum(axis=0) != 0]
    data_df = data_df.iloc[:Filt_num, :]
    gene_info_df = gene_info_df.iloc[:Filt_num, :]
    if (data_df.index != H.index).sum() + (H.index != gene_info_df.index).sum() > 0:
        raise
    H_df = H
    H = H.values
    data = data_df.to_numpy().transpose()
    event_status = survival_df['OS Status'].to_numpy()
    event_time = survival_df[endpoint].to_numpy()
    data = np.c_[data, event_time, event_status]
    if pathway_mask:
        pathway_masks = []
        pathway_masks.append(H_df)
        for i in range(3):
            Hn = pd.read_csv(os.path.join(H_path, f"H{i+2}.csv"), index_col=0)
            H_df = Hn.loc[H_df.columns, :]
            H_df = H_df.loc[:, H_df.sum(axis=0) != 0]
            pathway_masks.append(H_df)
        return data, pathway_masks
    print(f"After Preprocess: DATA shape - {data.shape} H shape : {H.shape}")
    if DF:
        data_df = pd.DataFrame(data=data, index=data_df.columns, columns=data_df.index.to_list() + ["time", "event"])
        return data_df, H_df
    else:
        return data, H


def load_opt_hier_data(data_path, H_path, Filt_num, endpoint, e=0, n=0):
    data_df = pd.read_csv(os.path.join(data_path, 'feature_matrix.csv'), index_col=0)
    survival_path = config.get_paths()['survival_data_template']
    survival_df = pd.read_csv(survival_path, index_col=0)
    gene_info_df = pd.read_csv(os.path.join(data_path, 'genes_list.csv'), index_col=0)
    H1 = pd.read_csv(os.path.join(H_path, 'H1.csv'), index_col=0)
    H2_path = os.path.join(H_path, 'H2.csv')
    H3_path = os.path.join(H_path, 'H3.csv')
    if os.path.exists(H2_path) and os.path.exists(H3_path):
        H2 = pd.read_csv(H2_path, index_col=0)
        H3 = pd.read_csv(H3_path, index_col=0)
    sdf = survival_df.loc[survival_df["PatientID"].isin(data_df.columns,), :]
    survival_df = sdf.loc[~sdf.duplicated(), :]
    data_df = data_df.loc[data_df.index.isin(H1.index), :]
    gene_info_df = gene_info_df.loc[data_df.index, :]
    H1 = H1.loc[data_df.index, :]
    H1 = H1.iloc[:Filt_num, :]
    H1 = H1.loc[:, H1.sum(axis=0) != 0]
    data_df = data_df.iloc[:Filt_num, :]
    gene_info_df = gene_info_df.iloc[:Filt_num, :]
    if (data_df.index != H1.index).sum() + (H1.index != gene_info_df.index).sum() > 0:
        raise
    if os.path.exists(H2_path) and os.path.exists(H3_path):
        H2 = H2.loc[H1.columns, :]
        H2 = H2.loc[:, H2.sum(axis=0) != 0]
        H3 = H3.loc[H2.columns, :]
        H3 = H3.loc[:, H3.sum(axis=0) != 0]
        H_list = [H1.values, H2.values, H3.values]
        H1 = H_list[0]
    else:
        H1 = H1.values
    data = data_df.to_numpy().transpose()
    event_status = survival_df['OS Status'].to_numpy()
    event_time = survival_df[endpoint].to_numpy()
    data = np.c_[data, event_time, event_status]
    if e == 0 and n == 0:
        G = None
    else:
        if e > 0 and n <= 0:
            G = generate_G_from_H(H1.T)
        else:
            G = generate_G_from_H(H1)
    print(f"After Preprocess: DATA shape - {data.shape} H shape : {H1.shape}")
    if os.path.exists(H2_path) and os.path.exists(H3_path):
        H = H_list
    else:
        H = H1
    return data, H, G

def load_data_general(DATA, PK_D, HD, fn_data, config=None):
    if DATA == 'PRO':
        if PK_D["DataDrive"] == 'STRING':
            # Load PRO data for STRING
            data, H = load_cohort_data(None, fn_data, HD['cox_num'], DF=True)
            
            # Get gene list from data
            gene_list = data.columns[:-2].tolist()  # Exclude time and event columns
            
            # Import construct_H_STRING function

            
            # Construct H matrix for STRING
            H = construct_H_STRING(gene_list, PK_D['layer_STRING'], config)
        else:
            data, H = load_cohort_data(None, fn_data, HD['cox_num'], DF=True)
        
    elif DATA == 'RNA':
        if PK_D["DataDrive"] == 'hcluster':
            data = load_TCGA_data(fn_data, HD["end_point"], Filt_num=HD["cox_num"])
            H = None
        else:  # Reactome
            H_path = config.get_paths()['reactome_graph_template'].format(
                layer=config.get_knowledge_params()['layer_reactome']
            )
            data, H = load_opt_data(
                data_path=os.path.dirname(fn_data),
                H_path=H_path,
                endpoint=HD["end_point"],
                Filt_num=HD['cox_num'],
                DF=False
            )
    return data, H

def create_random_param_sequence(param_option_dict, seed):
    keys = []
    params_opts = []
    for key in param_option_dict.keys():
        keys.append(key)
        params_opts.append(param_option_dict[key])
    all_combs = product(params_opts)
    param_dicts = []
    for comb in all_combs:
        param_dict = {}
        for key, param in zip(keys, comb):
            param_dict[key] = param
        param_dicts.append(param_dict)

    # shuffle the sequence
    np.random.seed(seed)
    rand_ids = np.random.choice(np.arange(len(param_dicts)), len(param_dicts), replace=False)
    param_dicts = [param_dicts[idx] for idx in rand_ids]
    return param_dicts


def train_test_data_process(dataset, test_rate, seed):
    data = np.c_[
        dataset['data'], 
        dataset['event_time'], 
        dataset['event_status'],
    ]

    data_train, data_test, _, _ = train_test_split(data, data[:,-1],
                                                test_size=test_rate,
                                                random_state=seed,
                                                shuffle=True,
                                                stratify=data[:,-1])
    # N, _ = data.shape
    # np.random.seed(seed)
    # ids_test = np.random.choice(np.arange(N), int(N * test_rate), replace=False)
    # ids_test = np.sort(ids_test)
    # ids_train = np.array([i for i in range(N) if i not in ids_test])
    # data_train = data[ids_train, :].astype(np.float32)
    # data_test = data[ids_test, :].astype(np.float32)
    return data_train, data_test

def get_ci(t, e, r):
    # t: event time, e: event status, r: ln(bx)
    t, e, r = t.detach().cpu().numpy(), e.detach().cpu().numpy(), np.exp(r.detach().cpu().numpy())
    r = np.squeeze(r)
    if (~np.isfinite(r)).sum() > 0:
        try:
            r_max = np.abs(r[(np.isfinite(r))]).max()
            r[[~np.isfinite(r < 0)] and r < 0] = -(r_max * 2)
            r[[~np.isfinite(r > 0)] and r > 0] = r_max * 2
            ci = concordance_index_censored(e.astype(bool), t, r)[0]
        except:
            ci = 0.0
    else:
        ci = concordance_index_censored(e.astype(bool), t, r)[0]
    return ci

def get_ci_np(t, e, r):
    r = np.squeeze(r)
    ci = concordance_index_censored(e.astype(bool), t, r)[0]
    return ci


def get_ibs(t, e, r):
    r = np.squeeze(r)
    # scale the predicted risk or there will be inf or nan values in BreslowEstimator.fit()
    if np.amax(r) > 5.:
        r = r / np.amax(r) * 5.

    baseline_model = BreslowEstimator()
    baseline_model.fit(r, e, t)
    cbh = baseline_model.cum_baseline_hazard_
    bs = baseline_model.baseline_survival_ 
    sam = SurvivalAnalysisMixin()
    survs = sam._predict_survival_function(baseline_model, r, False)
    times = np.arange(np.amin(t), np.amax(t) - 1.)
    preds = np.asarray([[fn(t) for t in times] for fn in survs])
    preds[np.isinf(preds)] = np.amax(preds[~np.isinf(preds)])

    y = np.array(
        [(bool(a), b) for a, b in zip(e, t)],
        dtype=[('status', 'bool'), ('time', 'float')]
    )

    score = integrated_brier_score(y, y, preds, times)
    return score


def product(lists):
    # create the cartisian product of multiple lists
    pools = [tuple(pool) for pool in lists]
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


# TODO： select useful module    
def save_results(log_dict, save_path):
    df = pd.DataFrame(data=log_dict)
    df.to_csv(save_path)

# def select_cancer_types(data_path, endpoint, min_num):
#     df = pd.read_csv(
#         join(data_path, 'ClinicalDataFrame-Cut15Year.csv'),
#         header=[0]
#     )
#     df = df.dropna(subset=[endpoint, endpoint+' Status'])
#     cancer_types = df['CancerType'].to_list()
#     type_num_dict = {}
#     for cancer_type in cancer_types:
#         if cancer_type not in type_num_dict.keys():
#             type_num_dict[cancer_type] = 1
#         else:
#             type_num_dict[cancer_type] += 1
#     return [key for key in type_num_dict.keys() if type_num_dict[key] >= min_num]


def get_km_curve(times, events, clip_time=60):
    unique_times = np.asarray(list(set(times)))
    sorted_unique_times = np.sort(unique_times)
    S_list = [1.]
    time_list = [0.]
    censor_list = [False]
    at_risk_list = [len(times)]
    live_at_the_start = len(times)
    S_t = 1.
    start_time = 0
    RMST = 0.
    for i in range(len(sorted_unique_times)):
        end_time = sorted_unique_times[i]
        event_num = np.sum(events[times==end_time])
        at_risk_list.append(live_at_the_start)
        live_at_the_start = np.sum(times >= end_time)
        if end_time <= clip_time:
            RMST += (S_t * (end_time - start_time))
        S_list.append(S_t)
        S_list.append(S_t)
        time_list.append(end_time)
        time_list.append(end_time)
        censor_list.append(0 in events[times==end_time])
        censor_list.append(0 in events[times==end_time])
        at_risk_list.append(live_at_the_start)
        start_time = end_time
    if np.amax(times) < clip_time:
        RMST += (S_t * (60 - end_time))
    return S_list, time_list, censor_list, at_risk_list, RMST

def get_logrank_p(times, events, assignments):
    logrank_results = statistics.pairwise_logrank_test(times, assignments, events)
    return logrank_results.p_value

# if __name__ == '__main__':
#     cancer_types = select_cancer_types(r'D:\Data\standardFileFoldTCGA', 'OS', 200)
#     print(cancer_types ,len(cancer_types))

def get_H_from_reactome(num_layer, root_rm_list, genes):
    '''
    num_layer : int ,the number of reactome hierachy net layer for H;depth option
    root_rm_list : root ids list ;width option
    genes: feature matrix genes for purne useless genes.
    '''
    start = time.time()
    # 通路-通路
    p2p = pd.read_table(config.get_paths()['reactome_p2p'], sep='\t')
    # 节点-通路
    g2p = pd.read_table(config.get_paths()['reactome_g2p'], sep='\t')

    ## 根节点
    root = pd.read_csv(config.get_paths()['reactome_roots_children'], index_col=0)
    root_ids = root.index.unique()

    # 删除的根节点列表
    root_remove = root_ids[root_rm_list]
    root_remain = list(set(root_ids) - set(root_remove))
    
    # 选择留下的根节点数量和原始根结点数了
    opt_num = len(root_remain)
    root_num = p2p.loc[~p2p['Parent'].isin(p2p['Child']),]['Parent'].unique().shape[0]

    print(f"{root_num} roots remain {opt_num} roots:{root_remain}")

    # 初始根结点数29
    # 判断是否剩余的根节点是否等于选择的根节点数量，相等代表修剪完成
    while(root_num != opt_num):
        # 不相等就删除选择根节点外的其他根节点
        p2p = p2p.loc[~p2p["Parent"].isin(root_remove),]
        # 更新后的根节点，未进行unique操作
        root_now = p2p.loc[~p2p['Parent'].isin(p2p['Child']),]['Parent']
        # 更新下一个循环应该去除的节点
        root_remove = root_now[~root_now.isin(root_remain)].unique()
        # 更新根节点数目
        root_num = root_now.unique().shape[0]
    
    # 更新基因通路映射
    g2p = g2p.loc[g2p["Identifier"].isin(np.unique(p2p.values.reshape(-1))),]

    # 构建reactomeH
    pathways_relation = dataFrame_gen(p2p, ['Parent', 'Child'])
    gene_pathway = dataFrame_gen(g2p, ['Gene', 'Identifier', 'Orthologous Events'])
    reactome_net = ReactomeNet(pathways_relation, gene_pathway)
    print(f"Construct layer {num_layer} reactome H")
    reactome_net.prune_to_genes(set(genes))
    reactome_net.prune_to_level(num_layer)# max=12 这步操作相当于将n作为最后一层基因

    root_ids = reactome_net.get_all_root_ids()
    print(f"{len(root_ids)} roots")
    H_rea = reactome_net.incidence_mat(num_layer, num_layer - 1)
    # 通过表达矩阵中的基因列表进行筛选
    H = H_rea.iloc[H_rea.index.isin(genes), :]
    H = H.loc[H.sum(axis=1) != 0, H.sum(axis=0) != 0]
    H = H.loc[~H.index.duplicated(), ~H.T.index.duplicated()]
    print(f"{time.time()-start} sconds are used for Hbio building")
    return H

def preprocess_H(feature_matrix, H):
    '''
    feature_matrix: cols:genes; index:pats;
    H: cols:pathways; index:genes;
    preprocess H for feature_matrix
    '''
    H = H.loc[feature_matrix.columns, :]
    H = H.loc[:, H.sum() != 0]
    return H






def construct_gene_enrich_pathway_hypergraph(gene_set, H_df, filt_method, threshold=None, remain_handle="remove", gsea_path=None):
    '''
    gene_set: pd.Index , data_df's genes
    data_df: Pats * Genes+2(event,time); expression matrix +labels
    filt_method: "p_value" | "top_num" | "significant"
    remain_handle: method to handle remain genes 
    '''
    gp = GProfiler(return_dataframe=True)
    fn_gsea = os.path.join(gsea_path, "1000genes_GSEA.csv")
    if os.path.exists(fn_gsea):
        df_enrich = pd.read_csv(fn_gsea, index_col=0)
    else:
        df_enrich = gp.profile(organism='hsapiens', sources=["REAC"], # ["GO:MF","GO:CC","GO:BP","KEGG","REAC","WP","TF","MIRNA","HPA","CORUM","HP"],
            query=gene_set.to_list(), user_threshold=1) # output rank
        if not os.path.exists(gsea_path):
            os.makedirs(gsea_path)
        df_enrich.to_csv(fn_gsea)
    print(f"There are {df_enrich.shape[0]} pathways can be used")
    print(f"num of pathway that p<0.05 and significant are :", df_enrich[df_enrich["p_value"] <= 0.05].shape[0], df_enrich[df_enrich["significant"] == True].shape[0])
    # filt enrich path
    if filt_method == "p_value":
        df_enrich = df_enrich[df_enrich["p_value"] <= threshold]
    elif filt_method == "significant":
        df_enrich = df_enrich[df_enrich["significant"] == True]
    elif filt_method == "top_num" and isinstance(threshold, int):
        df_enrich = df_enrich.iloc[:threshold, :]
    elif filt_method == "all" :
        pass
    else:
        raise Exception(f"wrong input filt_num or threshold, please check the arguments...")
    # # prepare enri_hsa name 
    enri_hsa = []
    for hsa_str in df_enrich["native"]:
        if hsa_str[5] == "R":
            enri_hsa.append(hsa_str[5:])
    # purne the pathway
    # 去除重复通路和交集外的通路，并将H转换为原名矩阵
    # 获取重复通路原名
    ori_col = H_df.columns
    paths_duplicte, clear_col = get_duplicate_path(ori_col)
    # 获取富集分析通路和邻接矩阵交集外的通路,并且整合重复通路
    # paths_enri = pd.Index(enri_hsa)
    clear_col = pd.Index(clear_col)
    print(f"There are {len(clear_col) - len(clear_col.unique())} pathways duplicated")
    paths_pruned = list(clear_col[~clear_col.isin(enri_hsa)]) + paths_duplicte
    # paths_pruned = list(paths_enri[~paths_enri.isin(clear_col)])+paths_duplicte
    paths_pruned = np.unique(paths_pruned)
    H_df_pruned = H_df
    # 通路修剪
    for pp in paths_pruned:
        idx, = np.where(np.array(clear_col) == pp)
        if len(idx) > 1:
            idx_del = idx[1:] # 保留第一个,删除其他
        else:
            idx_del = idx
        H_df_pruned = H_df_pruned.drop(columns=ori_col[idx_del])
    print(f"{len(clear_col.unique()) - H_df_pruned.shape[1]} pathways has been removed! ")
    node_idx = []
    # 虚拟或者删除
    if (H_df_pruned.sum(axis=1) == 0).sum() > 0: # check if exist genes that unconnect to edges  
        if remain_handle == "hyperedge": # create a new hyperedge
            H_df_pruned['edge4remain'] = (H_df_pruned.sum(axis=1) == 0).replace({True: 1, False: 0})
        elif remain_handle == "remove": # remove in hypergraph, but linear cat
            H_df_pruned = H_df_pruned.loc[H_df_pruned.sum(axis=1) != 0, :]
            # TODO: add threshold
            # data_df = pd.concat([fm.loc[:,H_df_pruned.index],data_df.iloc[:,-2:]],axis=1)
            node_idx, = np.where(gene_set.isin(H_df_pruned.index))
            # nodeidx

    return H_df_pruned, node_idx # test the adj

def find_duplicates(lst):
    return list(set([x for x in lst if lst.count(x) > 1]))

def get_duplicate_path(col_H,):
    col = []
    for path_str in col_H:
        if "_" in path_str:
            col.append(path_str[:path_str.index('_')])
        else:
            col.append(path_str)
    paths_duplicte = find_duplicates(col)
    return paths_duplicte, col


def build_hiddens(num_ini, divisor, max_depth=5):
    pooling_hiddens = []
    num_node = num_ini
    while(num_node > 1 and len(pooling_hiddens) < max_depth ):
        pooling_hiddens.append(num_node)
        num_node //= divisor
    return pooling_hiddens

def build_deep_hiddens(num_ini, divisor=2, num_min=100, depth=3):
    pooling_hiddens = []
    num_node = num_ini
    while(num_node >= num_min and len(pooling_hiddens) < depth ):
        pooling_hiddens.append(num_node)
        num_node //= divisor
        if num_node <= num_min:
            num_node = num_min
    return pooling_hiddens


# def load_data_H_G(data_path,H_path,HD):
#     '''
#     HD.keys() must contain ["end_point","edge_pooling","fil_num"]
#     '''
#     data = load_data(data_path,HD["end_point"],Filt_num=-1)# all data
#     H = pd.read_csv(H_path,index_col=0)
#     H=H.loc[H.sum(axis=1)!=	0,H.sum(axis=0)!=0]# purne nodes and edges without any connection
#     data=data.loc[:,data.columns[:-2].isin(H.index).tolist()+[True,True]]
#     H=H.loc[data.columns[:-2],:]
#     H=H.loc[H.sum(axis=1)!=	0,H.sum(axis=0)!=0]# purne nodes and edges without any connection

#     # feature filter
#     if HD["fil_num"]!=-1:
#         data=pd.concat([data.iloc[:,:HD["fil_num"]],data.iloc[:,-2:]],axis=1)
#         H=H.loc[H.index.isin(data.columns),:]
#         H=H.loc[H.sum(axis=1)!=	0,H.sum(axis=0)!=0]# purne nodes and edges without any connection
#     # check H.index == data.columns[:-2]
#     assert H.index.equals(data.columns[:-2])
#     G=generate_G_from_H(H.T) if HD["edge_pooling"] else generate_G_from_H(H)
#     return data,H,G
