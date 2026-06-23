import argparse
from functools import partial
from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
import multiprocessing as mul
from tqdm import tqdm
import warnings
import time
import os
# from ReactomeNet.ReactomeNet import ReactomeNet
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
import matplotlib.pyplot as plt

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

def load_feature_matrix(fn_feature_matrix:str,cancer_type):
    print(f'Load {cancer_type} feature matrix ...')
    fm = pd.read_csv(fn_feature_matrix,index_col=None,header=None) 
    SampleType = fm.columns[fm.iloc[2] == '1'].append(fm.columns[fm.iloc[2] == '01'])
    print('Only select SampleType 1')
    if len(SampleType) == 0:
        raise Exception("Invalid SampleType for TCGA expression data!!! No SampleType1 sample, please check the dataset!",SampleType)
    SampleType = [0,1] + list(SampleType)# 0 has been the index col,here is 1
    fm = fm[SampleType].drop([0,2]) # 'EnsembleID'])
    col = pd.concat([fm.iloc[1,0:2],fm.iloc[0,2:]])
    fm.columns = col
    fm.drop(fm.index[0:2], inplace=True)
    fm = fm.reset_index().drop('index',axis=1)
    fm = fm.apply(pd.to_numeric, errors = 'ignore')
    return fm

# function cut years and output clinical csv
def cut_prognosis_matrix(cut_years=0,to_csv=False):
    '''
    cut_years should lower than 30 years
    '''
    prognosis_matrix = pd.read_csv("TCGA/standardFileFoldTCGA/ClinicalDataFrame.csv")
    OS_max = prognosis_matrix["OS"].max()/365
    PFI_max = prognosis_matrix["PFI"].max()/365
    print(f"--Before Cut--\nOS_max:{int(OS_max)} years,PFI_max:{int(PFI_max)} years")
    # cut prognosis to n years
    # because the infuence of the features should not be too long
    if cut_years > 0:
        # because TCGA use days, transfer year to day
        cut_days = cut_years * 365 + int(cut_years / 4.0) * 1
        # change OS, patients with survival > cut_days were regarded as NO event
        prognosis_matrix.loc[prognosis_matrix["OS"] > cut_days, "OS Status"] = 0
        prognosis_matrix.loc[prognosis_matrix["OS"] > cut_days, "OS"] = cut_days
        # change PFI
        prognosis_matrix.loc[prognosis_matrix["PFI"] > cut_days, "PFI Status"] = 0
        prognosis_matrix.loc[prognosis_matrix["PFI"] > cut_days, "PFI"] = cut_days
        
    OS_max = prognosis_matrix["OS"].max()/365
    PFI_max = prognosis_matrix["PFI"].max()/365
    print(f"--After Cut--\nOS_max:{int(OS_max)} years,PFI_max:{int(PFI_max)} years")
    print('Use day as the unit of survival time')
    if to_csv:
        prognosis_matrix.to_csv(f"TCGA/standardFileFoldTCGA/ClinicalDataFrame-Cut{cut_years}Year.csv",index=False)
    return prognosis_matrix

def preprocess_prognosis_matrix(prognosis_matrix, feature_matrix, cancer_type="ALL"):
    print(f'Load {cancer_type} prognosis matrix ...')

    # choose selected cancer type
    if cancer_type != "ALL":
        prognosis_matrix = prognosis_matrix.loc[prognosis_matrix["CancerType"] == cancer_type,:]
    
    # Delete duplicated patients
    prognosis_matrix = prognosis_matrix.loc[-prognosis_matrix["PatientID"].duplicated(),]

    # Keep the ones in expression matrix
    prognosis_matrix = prognosis_matrix.loc[prognosis_matrix["PatientID"].isin(feature_matrix.columns),]

    # Remove samples without prognosis information
    prognosis_matrix = prognosis_matrix.loc[-prognosis_matrix["OS"].isna(),]
    prognosis_matrix = prognosis_matrix.loc[-prognosis_matrix["PFI"].isna(),]
    
    min_PFI = prognosis_matrix["PFI"].min()
    max_PFI = prognosis_matrix["PFI"].max()
    min_OS = prognosis_matrix["OS"].min()
    max_OS = prognosis_matrix["OS"].max()
    if min_OS==0 or min_PFI==0:
        raise Exception("Prognosis Matrix contain 0 PFI or OS",prognosis_matrix)
    print(f"new Prognosis Matrix PFI:{min_PFI}days-{max_PFI}days; OS:{min_OS}days-{max_OS}days.")
    return prognosis_matrix

def preprocess_feature(feature_matrix, prognosis_matrix,nan_percent = 0.7, top_nan =0,top_var = 2000):
    print('Preprocess Feature matrixs ... ')
    # take out genelist
    gene_infor = feature_matrix.iloc[:, 0:2]
    gene_infor.columns = np.array(["Gene_ID", "Gene_symbol"])
    
    # select patients with prognosis information
    feature_matrix = feature_matrix.loc[:, prognosis_matrix["PatientID"]]

    # remove features with too much NaN (has been transformed to a low level)
    if nan_percent > 0.0:
        print("filt by nan_percent")
        feature_matrix_min_value = feature_matrix.min().min()
        min_value_percent = (feature_matrix == feature_matrix_min_value).sum(axis=1) / feature_matrix.shape[1]
        feature_matrix = feature_matrix.loc[min_value_percent < nan_percent, ]
        gene_infor = gene_infor.loc[min_value_percent < nan_percent]
        print(30*'-') 
        print(f'after remove features with too much NaN ,features:{feature_matrix.shape[0]}')
    
    if top_nan > 0.0:
        print("filt by top_nan")
        feature_matrix_min_value = feature_matrix.min().min()
        min_value_percent = (feature_matrix == feature_matrix_min_value).sum(axis=1) / feature_matrix.shape[1]
        min_value_percent = min_value_percent.sort_values(ascending=True)[:top_nan]# 选取nan百分比最低的前n个基因
        print(min_value_percent,f"\nThere are {(min_value_percent > 0.5).sum()} features' nan value percent higher than 0.5 ")#[-int(top_nan/10):]
        feature_matrix = feature_matrix.loc[min_value_percent.index, ]
        gene_infor = gene_infor.loc[min_value_percent.index]
        print(30*'-')
        print(f'after remove features with too much NaN ,features:{feature_matrix.shape[0]}')
    # remove gene without expression and features with low variance
    if top_var>0: 
        feature_matrix_trans = feature_matrix.T
        feature_matrix_trans = (feature_matrix_trans - feature_matrix_trans.min(axis=0)) / (
            feature_matrix_trans.max(axis=0) - feature_matrix_trans.min(axis=0))# 0-1scaler标准化
        gene = feature_matrix_trans.var().dropna().sort_values(ascending=False)[:top_var]
        print(gene.head())
        feature_matrix = feature_matrix.loc[gene.index, ]
        gene_infor = gene_infor.loc[gene.index, ]
        print(30*'-')
        print(f'after filt low variance,retain {top_var}, features:{feature_matrix.shape[0]}')
    return feature_matrix, gene_infor

def univariate_regression(
    data: pd.DataFrame, duration_col: str, event_col: str, feature: str
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
    cph = CoxPHFitter(penalizer=0.0001)# slove converge problems set penalizer=0.0001
    # try:
    # cph.new
    try:
        cph.fit(
        df=data, duration_col=duration_col, event_col=event_col, formula=feature,#show_progress=True
        )
        p = cph.summary.loc[feature, "p"]
    except BaseException as e:
        print(e)
        p = 1

    return p, feature

def cox_feature_selection(
    time_col: str,event_col: str,
    feature_matrix: pd.DataFrame,
    label_matrix: pd.DataFrame,
    process_num: int = 10,
):
    """
        Use CoxPHFitter to select feature.

        Params:
            feature_matrix (DataFrame): Input features with shape (n_samlpes, n_features).
            label_matrix (DataFrame): Input label with shape (n_samlpes, 2).
                Time,Event
        Returns:
            data (DataFrame): Concatenation of feature_matrix and label_matrix with shape (n_samples, n_features+2).
            Cox_Feat: Selected features lower than the threshold.
    """
    if not isinstance(feature_matrix, pd.DataFrame):
        raise TypeError("feature_matrix should be of type pd.DataFrame!")
    if not isinstance(label_matrix, pd.DataFrame):
        raise TypeError("label_matrix should be of type pd.DataFrame!")
    data = pd.concat([feature_matrix, label_matrix], axis=1).dropna(axis=1)
    pval = []

    # Use multiprocessing to accelerate the process
    print('Cox univariate regression ...')
    regression = partial(univariate_regression, data, time_col,event_col)
    pool = mul.Pool(processes=process_num)
    with tqdm(total=len(feature_matrix.columns.tolist())) as t:
        for p, feat_name in pool.imap(regression, feature_matrix.columns.tolist()):
            pval.append(p)
            t.update()
    return data,pval

def cox_filter(feature_matrix, prognosis_matrix, gene_infor, col_use='PFI', use_threshold=False, cox_top=0, process_num=16):
    print('Cox filter ...')
    print(f'top {cox_top}')
    y = prognosis_matrix[[(col_use + ' Status'), col_use]]
    y.index = prognosis_matrix['PatientID']
    y.columns = ['event', 'time']
    y.loc[:,'event'] = np.array(y['event'], dtype='int')
    
    feature_matrix_T = feature_matrix.T
    feature_matrix_T.columns = np.array(feature_matrix_T.columns, dtype = "str")
    feature_matrix_T.columns = "f" + feature_matrix_T.columns# each Geneid add "f"
    since = time.time()
    cox_p_value = np.array(cox_feature_selection(time_col="time",event_col="event",feature_matrix=feature_matrix_T,label_matrix= y, process_num=process_num))
    cox_p_value_df = pd.DataFrame([feature_matrix.index, cox_p_value]).T
    gene_infor["cox_p"] = cox_p_value

    if cox_top == 0:
        cox_top = len(cox_p_value)
    
    top_ids = np.argsort(cox_p_value)[:cox_top]
    
    # if len(top_ids) <= len(feature_matrix.index):
    cox_p_value = cox_p_value[top_ids]
    gene_infor = gene_infor.iloc[top_ids]
    feature_matrix = feature_matrix.iloc[gene_infor.index, :]
    time_elapsed = time.time() - since
    
    print(f'Cox feature filter complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('feature_matrix shape:',feature_matrix.shape)
    return feature_matrix, gene_infor

def discrete_time_bin(df:pd.DataFrame,endpoint:str, interval:int):
    '''
    df: contain time
    the unit of interval time shold be the same as the unit of time in df
    '''
    df_bin=df.copy()
    maxos = df[endpoint].max()
    minos = df[endpoint].min()
    t = round((maxos - minos)/interval) + 1
    for i in range(t):
        top = ((i+1)*interval+df[endpoint].min())
        bottom  = (i * interval + df[endpoint].min())
        df_bin.loc[( df[endpoint] >= bottom )==( df[endpoint] < top ),endpoint] = i+1
    return df_bin

def fil_min_nan_threshold(feature_matrix,nan_thre=0.5,):
    '''
    remove genes that nan percent > thre
    feature_matrix: genes * pats
    '''
    print(f"filt by nan_threshold: {nan_thre}")
    feature_matrix_min_value = feature_matrix.min().min()
    min_value_percent = (feature_matrix == feature_matrix_min_value).sum(axis=1) / feature_matrix.shape[1]
    feature_matrix = feature_matrix.loc[min_value_percent < nan_thre, ]
    print(30*'-') 
    print(f'after remove features with {nan_thre} NaN ,features:{feature_matrix.shape[0]}')
    return feature_matrix

# fn_cell = "Proteomic/Proteomics_Biomarker_Pepline/Fudan_Cell_HCC159_6478-nolog2-intensity-preprocessed.csv"
# fn_nature = "Proteomic/Proteomics_Biomarker_Pepline/Jiang-Nature-2019-intensity-preprocessed.csv"
# df_cell = pd.read_csv(fn_cell,index_col=0)
# df_nature = pd.read_csv(fn_nature,index_col=0).iloc[:,4:]
# # fm_ce = df_cell.iloc[:,2:]#159病人 × 6405 基因
# # la_ce = df_cell.iloc[:,:2]

# # fm_na = df_nature.iloc[:,6:] # nature 1176个基因，101个样本
# # la_na = df_nature[["OS month","OS"]]
# # 方差阈值筛选
# for df in [df_nature]:#df_nature
#     feature_matrix=df.iloc[:,2:]
#     label = df[["OS month","OS"]]
#     print(f"Original genes num:{feature_matrix.shape[1]};")
#     # feature_matrix_mm = (feature_matrix - feature_matrix.min(axis=0)) / (
#     # feature_matrix.max(axis=0) - feature_matrix.min(axis=0))# 0-1scaler标准化
#     # gene = feature_matrix_mm.var().dropna().sort_values(ascending=False)
#     # gene=gene[gene>0.03]# 阈值筛选>0.03 Nature:1176->994 Cell:6405->2719
#     # feature_matrix = feature_matrix.loc[:,gene.index]

#     # gene_infor = gene_infor.loc[gene.index, ]
#     print(30*'-')
#     print(f'Original genes num:{df.shape[1]-2};after filt low variance,retain features:{feature_matrix.shape[1]}')
#     # feature_matrix, gene_list = preprocess_feature(feature_matrix, prognosis_matrix,nan_percent=0,top_nan=top_var,top_var=top_var)

#     cox_p_value = np.array(cox_feature_selection(time_col="OS month",event_col="OS", 
#                         feature_matrix=feature_matrix, label_matrix=label, process_num=80))
#     cox_p_value_df = pd.DataFrame([feature_matrix.columns, cox_p_value]).T
#     cox_p_value_df= cox_p_value_df.sort_values(by=1)
#     cox_p_value_df.columns=["Ensemble_ID","Cox_p_value"]
#     cox_p_value_df= cox_p_value_df.set_index("Ensemble_ID",drop=True)
#     feature_matrix= feature_matrix[cox_p_value_df.index]
#     cox_p_value_df.to_csv("Proteomic/PreprocessedData/Nature-Cox-genes.csv")
#     df = pd.concat([label,feature_matrix],axis=1)
#     df.to_csv("Proteomic/PreprocessedData/Cell-DATA-Cox-zscore.csv")

    # print(feature_matrix,cox_p_value_df)
    # TimeBins process
    # df_sorted = df.sort_values("OS month")
    # Tmax=125
    # Tmin=7
    # Tmax=50
    # Tmin=0
    # Tbin = []
    # for t in label_cell["OS month"]:
    #     for i in range(int(Tmax-Tmin)):
    #         if t>=Tmin+i and t<Tmin+i+1:
    #             Tbin.append(i+1)
    #             print("append")
        # if t==Tmax:#t>Tmax-1 and
        #     Tbin.append(50)   
        #     print("max") 
# label_cell = df_cell[["OS month","OS"]]
# label_cell = label_cell.sorted("OS month")

# def _main():
    
#     '''
#     RNA-seq
#     数据预处理：特征筛选(60498-10893-10000)，预后信息处理（2month-15years）；
#     最终生成cox10000的特征矩阵、多层邻接矩阵、特征coxp列表和预后信息矩阵。
#     '''

#     # cancer_list = pd.read_csv("TCGA/TCGA400CancerList.csv")
#     # cancer_list = cancer_list['cancer']
#     # cox_fil = True
#     # cox_list = [1000,2000,3000]
#     # cox_num_list = [250,500,1500,2000]
#     # for cox_top_num in cox_num_list:
#     #     top_var = cox_top_num * 2
#     # for cox_top_num in cox_list:
    
#     use_pThre = False 
#     top_var = 10000 # 10893 -> 10000
#     cox_top_num = 10000
#     to_csv = True
#     cancer_type="BRCA"
#     endpoint = 'OS'
#     cox_proc_num = 32

#     df_prog = pd.read_csv(f'TCGA/standardFileFoldTCGA/ClinicalDataFrame-Cut15Years.csv')
#     OS_min = df_prog["OS"].min()
#     PFI_min = df_prog["PFI"].min()
#     print(f"min time check: {OS_min},{PFI_min}")
#     # output file 
#     out_path = f'TCGA/dataset_{cancer_type}/'
#     if not os.path.exists(out_path):
#         os.makedirs(out_path)
#     # nf = f'{out_path}{cancer_type}_cox{cox_top_num}'
#     pf = f'{out_path}prognosis.csv'# 不受基因筛选影响
#     nf = f'{out_path}cox{cox_top_num}_'
#     ff = f'{nf}features.csv' 
#     gf = f'{nf}genes.csv'
#     f_H1 = f'{nf}H1.csv'
#     f_H2 = f'{nf}H2.csv'
#     f_H3 = f'{nf}H3.csv'
#     H1 = pd.read_csv("ReactomeNet/H1(1).csv",header=0,index_col=0)
#     H2 = pd.read_csv("ReactomeNet/H2(1).csv",header=0,index_col=0)
#     H3 = pd.read_csv("ReactomeNet/H3(1).csv",header=0,index_col=0)
#     # load feature
#     fn_feat = f"TCGA/standardFileFoldTCGA/{cancer_type}-Intensity-Matrix.csv"
#     # file_exist = os.path.exists(ff) & os.path.exists(gf) & os.path.exists(pf) & os.path.exists(f_H1) & os.path.exists(f_H2) & os.path.exists(f_H3)
    
#     # if not file_exist:
#         # load feature and preprocess prog mat
#     feature_matrix = load_feature_matrix(fn_feat,cancer_type=cancer_type)
#     print(f'feature matrix shape: {feature_matrix.shape}')

#     prognosis_matrix = preprocess_prognosis_matrix(
#         prognosis_matrix=df_prog,feature_matrix=feature_matrix,cancer_type=cancer_type)
#     print(f'feature matrix shape: {feature_matrix.shape}; prognosis matrix shape: {prognosis_matrix.shape}')
#     print(30*'-')

#     # Remove the genes out of Reactome from TCGA Data
#     print('remove the DATA not in Ractome and TCGA ...')
#     feature_matrix= feature_matrix.loc[feature_matrix['EnsembleID'].isin(H1.index),:]
#     print(f'feature matrix shape: {feature_matrix.shape}; prognosis matrix shape: {prognosis_matrix.shape}')
#     print(30*'-')

#     # here features : 60498->Bottom up 11113
#     # 新 Top down 10893
#     feature_matrix, gene_list = preprocess_feature(feature_matrix, prognosis_matrix,nan_percent=0,top_nan=top_var,top_var=top_var)
#     print(f'feature matrix shape: {feature_matrix.shape}; prognosis matrix shape: {prognosis_matrix.shape}')
#     print(30*'-')
#     # here features : 11113->9862->top var

#     #cox filter
#     feature_matrix_cox, gene_list_cox = cox_filter(feature_matrix, prognosis_matrix, gene_list,col_use=endpoint,
#                                                     use_threshold = use_pThre, cox_top= cox_top_num, process_num=cox_proc_num)# without threshold select
#     print(f'features after cox: {feature_matrix.shape[0]}->{feature_matrix_cox.shape[0]}')
#     gene_list = gene_list_cox
#     feature_matrix = feature_matrix_cox
#     print(f'feature matrix shape: {feature_matrix.shape}; prognosis matrix shape: {prognosis_matrix.shape}')
#     print(30*'-')
#     # here features : top var->top cox

#     # Purne the Incidence_Matrix H1 H2 H3
#     H1 = H1.loc[gene_list['Gene_ID'],:]
#     H1 = H1.loc[:,H1.sum(axis=0)!=0]
#     H2 = H2.loc[H1.columns,:]
#     H2 = H2.loc[:,H2.sum(axis=0)!=0]
#     H3 = H3.loc[H2.columns,:]
#     H3 = H3.loc[:,H3.sum(axis=0)!=0]
#     print(f"H1,H2,H3:{H1.shape},{H2.shape},{H3.shape}")
    
#     if to_csv:
#         H1.to_csv(f_H1)
#         H2.to_csv(f_H2)
#         H3.to_csv(f_H3)
#         feature_matrix.to_csv(ff)
#         prognosis_matrix.to_csv(pf,index = False)
#         gene_list.to_csv(gf)

#         # else:
        
#         #     print(f'Skip the {cancer_type} data ,cause files has been existed already')


# if __name__ == '__main__':
#     _main()

