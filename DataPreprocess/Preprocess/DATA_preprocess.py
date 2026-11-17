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
