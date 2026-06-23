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
from ReactomeNet.ReactomeNet import ReactomeNet
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# preprocess prognsis matrix
def preprocess_prognosis_matrix(prognosis_matrix_fn, feature_matrix, cancer_type="ALL",cut_years = 5):
    print(f'Load {cancer_type} prognosis matrix ...')
    prognosis_matrix = pd.read_csv(prognosis_matrix_fn)

    # choose selected cancer type
    if cancer_type != "ALL":
        prognosis_matrix = prognosis_matrix.loc[prognosis_matrix["CancerType"] == cancer_type]
    
    # 截断时间外的样本仍然保留,状态设置为0
    # cut prognosis to 5 years
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

    print('Use day as the unit of survival time')
    
    # Delete duplicated patients
    prognosis_matrix = prognosis_matrix.loc[-prognosis_matrix["PatientID"].duplicated(),]

    # Keep the ones in expression matrix
    prognosis_matrix = prognosis_matrix.loc[prognosis_matrix["PatientID"].isin(feature_matrix.columns),]

    # Remove samples without prognosis information 
    prognosis_matrix = prognosis_matrix.loc[-prognosis_matrix["OS"].isna(),]
    prognosis_matrix = prognosis_matrix.loc[-prognosis_matrix["PFI"].isna(),]


    return prognosis_matrix

# feature matrix preprocess
def preprocess_feature(feature_matrix, prognosis_matrix,nan_percent = 0.5, top_var = 2000):
    print('Preprocess matrixs ... ')
    # take out genelist
    gene_infor = feature_matrix.iloc[:, 0:2]
    gene_infor.columns = np.array(["Gene_ID", "Gene_symbol"])
    
    # select patients with prognosis information
    feature_matrix = feature_matrix.loc[:, prognosis_matrix["PatientID"]]

    # remove features with too much NaN (has been transformed to a low level)
    if nan_percent > 0.0:
        feature_matrix_min_value = feature_matrix.min().min()
        min_value_percent = (feature_matrix == feature_matrix_min_value).sum(axis=1) / feature_matrix.shape[1]
        feature_matrix = feature_matrix.loc[min_value_percent < nan_percent, ]
        gene_infor = gene_infor.loc[min_value_percent < nan_percent]

    # remove gene without expression and features with low variance 
    feature_matrix_trans = feature_matrix.T
    feature_matrix_trans = (feature_matrix_trans - feature_matrix_trans.min(axis=0)) / (
        feature_matrix_trans.max(axis=0) - feature_matrix_trans.min(axis=0))# 0-1scaler标准化
    
    # modifed the sort
    gene = feature_matrix_trans.var().dropna().sort_values(ascending=False)[:top_var]
    feature_matrix = feature_matrix.loc[gene.index, ]
    gene_infor = gene_infor.loc[gene.index, ]
    print(30*'-')
    print(f'after filt low variance,retain 2000, features:{feature_matrix.shape[0]}')
    return feature_matrix, gene_infor

# feature selection

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
    cph = CoxPHFitter()
    try:
        cph.fit(
            df=data, duration_col=duration_col, event_col=event_col, formula=feature
        )
        p = cph.summary.loc[feature, "p"]
    except BaseException as e:
        print(e)
        p = 1

    return p, feature

def cox_feature_selection(
    feature_matrix: pd.DataFrame,
    label_matrix: pd.DataFrame,
    process_num: int = 10,
):
    """
        Use CoxPHFitter to select feature.

        Params:
            feature_matrix (DataFrame): Input features with shape (n_samlpes, n_features).
            label_matrix (DataFrame): Input label with shape (n_samlpes, 2).

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
    regression = partial(univariate_regression, data, "time", "event")
    pool = mul.Pool(processes=process_num)
    # with tqdm(total=len(feature_matrix.columns.tolist())) as t:
    for p, feat_name in pool.imap(regression, feature_matrix.columns.tolist()):
        pval.append(p)
    #         t.update()
    return pval

def cox_filter(feature_matrix, prognosis_matrix, gene_infor, col_use='PFI', use_threshold=False, cox_top=0, process_num=16):
    print('Cox filter ...')
    print(f'top {cox_top}')
    y = prognosis_matrix[[(col_use + ' Status'), col_use]]
    y.index = prognosis_matrix['PatientID']
    y.columns = ['event', 'time']
    y.loc[:,'event'] = np.array(y['event'], dtype='int')

    feature_matrix_T = feature_matrix.T
    feature_matrix_T.columns = np.array(feature_matrix_T.columns, dtype = "str")
    feature_matrix_T.columns = "f" + feature_matrix_T.columns
    since = time.time()
    cox_p_value = np.array(cox_feature_selection(feature_matrix_T, y, process_num=process_num))
    cox_p_value_df = pd.DataFrame([feature_matrix.index, cox_p_value]).T
    gene_infor["cox_p"] = cox_p_value

    # 阈值筛选，理论0.05,暂时不使用
    if use_threshold:
        feature_matrix = feature_matrix.loc[cox_p_value < 0.05, ]
        gene_infor = gene_infor[cox_p_value < 0.05]
        cox_p_value = cox_p_value[cox_p_value < 0.05]

    if cox_top == 0:
        cox_top = len(cox_p_value)
    
    top_ids = np.argsort(cox_p_value)[:cox_top]
    
    if len(top_ids)<len(feature_matrix.index):
        cox_p_value = cox_p_value[top_ids]
        feature_matrix = feature_matrix.iloc[top_ids, :]
        gene_infor = gene_infor.iloc[top_ids]
    
    time_elapsed = time.time() - since
    
    print(f'Cox feature filter complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('feature_matrix shape:',feature_matrix.shape)
    return feature_matrix, gene_infor

# create H inscidence matrix 
def relation_generator(df:pd.DataFrame, columns:list):
    for i, row in df.iterrows():
        yield tuple(row[col] for col in columns)

def create_HyperIncidence_matrix(feature_matrix = pd.DataFrame,gene_list = pd.DataFrame) ->pd.DataFrame :
    '''
    # create Hyper incidence matrix by cancer genes list
    # modify the feature matrix and gene list
    '''
    
    print("Construct Hyper Incidece Matrix...")
    since = time.time()
    feature_matrix = pd.concat([gene_list,feature_matrix],axis = 1)
    gene_list = list(feature_matrix['Gene_ID'])
    # 通路-通路
    reactome_pathways_relation = pd.read_table('ReactomeNet/ReactomePathwaysRelation_hsa.txt', sep='\t')
    df12 = relation_generator(reactome_pathways_relation, ['Parent', 'Child'])
    # 节点-通路
    ensemble2reactome = pd.read_table('ReactomeNet/Ensembl2Reactome_hsa.txt', sep = '\t')
    df01 = relation_generator(ensemble2reactome, ['Gene', 'Identifier', 'Orthologous Events'])
    # instance 
    reactome_net = ReactomeNet(df12, df01)
    #prune by genelist
    reactome_net.prune_to_genes(set(gene_list))
    # incidence matrix
    H01 = reactome_net.incidence_mat(0,1)
    H12 = reactome_net.incidence_mat(1,2)
    feature_matrix = feature_matrix.loc[feature_matrix['Gene_ID'].isin(H01.index),:]
    gene_list = feature_matrix.iloc[:, 0:3]
    feature_matrix = feature_matrix.iloc[:,3:]
    
    time_elapsed = time.time() - since
    print(f'Constructing H01 H02 complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('H01.shape:',H01.shape,'H12.shape:',H12.shape)
    print(f'After reactome filter, genes:{feature_matrix.shape[0]}')

    return H01,H12,feature_matrix,gene_list


def select_TCGA_dataset(prognosis_matrix_fn:str,sample_threshold_num:int, print_detail = False)->list:
    prognosis_matrix = pd.read_csv(prognosis_matrix_fn)
    cancer_uni = prognosis_matrix['CancerType'].unique()
    print(f'len_uni:{len(cancer_uni)}; TCGA Cancer list: {cancer_uni}; if SampleID duplicated? {sum(prognosis_matrix["SampleID"].duplicated())} ')
    cancer_avai_list = []
    print(f'Samples > 400 Cancer types :') 
    for ct in cancer_uni:
        sample_len = sum(prognosis_matrix['CancerType']==ct)
        if sample_len >=  sample_threshold_num:
            if print_detail:# print the choosen cancer types
                print(ct,':',sample_len)
            cancer_avai_list.append(ct)
    print(f'total:{len(cancer_avai_list)} types')
    return cancer_avai_list,prognosis_matrix


def parse_input() -> argparse.Namespace:
    """
    Parse input argument
    """
    # argument parser
    parser = argparse.ArgumentParser(
        description="Call HRDscan model to predict HRD risk score from CNV segmentation file"
    )
    parser.add_argument(
        "--endpoint", default="PFI", type=int, help="the ClinicalDataFrame Endpoint for feature selection(cox)",
    )
    parser.add_argument(
        "--cox_top_num", default=1000 , type=int, help="top feature selection num",# 是否使用
    )
    return parser.parse_args()


def _main():
    to_csv = True
    fn_prog = 'TCGA/standardFileFoldTCGA/ClinicalDataFrame-Cut15Year.csv'
    endpoint = 'OS'
    cancer_list = pd.read_csv("TCGA/TCGA400CancerList.csv")
    cancer_list = cancer_list['cancer']
    cut_years = 5
    
    # cox param 
    cox_proc_num = 16
    cox_fil = True
    cox_top_num = 1000
    use_pThre = False # if True you 
    # low var fil
    top_var = cox_top_num * 2
    
    # cox_num_list = [250,500,1500,2000]
    # for cox_top_num in cox_num_list:
    #     top_var = cox_top_num * 2
    for cancer_type in cancer_list:
        # output file 
        print(f"process {len(cancer_list)} cancer datasets")
        out_path = f'TCGA/dataset_{endpoint}/{cancer_type}/'
        H1 = pd.read_csv("ReactomeNet/H1(1).csv",header=0,index_col=0)
        H2 = pd.read_csv("ReactomeNet/H2(1).csv",header=0,index_col=0)
        H3 = pd.read_csv("ReactomeNet/H3(1).csv",header=0,index_col=0)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        nf = f'{out_path}{cancer_type}'
        if cox_fil:
            nf = f'{out_path}{cancer_type}_cox{cox_top_num}'
        ff = f'{nf}_features.csv' 
        pf = f'{nf}_prognosis_cut{cut_years}.csv'
        gf = f'{nf}_genes.csv'
        f_H1 = f'{nf}_H1.csv'
        f_H2 = f'{nf}_H2.csv'
        f_H3 = f'{nf}_H3(1).csv'#test
        # load feature
        fn_feat = f"TCGA/standardFileFoldTCGA/{cancer_type}-Intensity-Matrix.csv"
        file_exist = os.path.exists(ff) & os.path.exists(gf) & os.path.exists(pf) & os.path.exists(f_H1) & os.path.exists(f_H2) & os.path.exists(f_H3)
        
        if not file_exist:
            # load feature and preprocess prog mat
            feature_matrix = load_feature_matrix(fn_feat,cancer_type=cancer_type)
            prognosis_matrix = preprocess_prognosis_matrix(
                prognosis_matrix_fn=fn_prog,feature_matrix=feature_matrix,cancer_type=cancer_type,cut_years=cut_years)
            print(f'feature matrix shape: {feature_matrix.shape}; prognosis matrix shape: {prognosis_matrix.shape}')
            print(30*'-')

            # Remove the genes out of Reactome from TCGA Data
            print('remove the DATA not in Ractome and TCGA ...')
            feature_matrix= feature_matrix.loc[feature_matrix['EnsembleID'].isin(H1.index),:]
            print(f'feature matrix shape: {feature_matrix.shape}; prognosis matrix shape: {prognosis_matrix.shape}')
            print(30*'-')

            # here features : 60498->11113
            # 10893
            feature_matrix, gene_list = preprocess_feature(feature_matrix, prognosis_matrix,top_var=top_var)
            print(f'feature matrix shape: {feature_matrix.shape}; prognosis matrix shape: {prognosis_matrix.shape}')
            print(30*'-')
            # here features : 11113->9862->2000
            # 

            #cox filter
            if cox_fil:
                feature_matrix_cox, gene_list_cox = cox_filter(feature_matrix, prognosis_matrix, gene_list,col_use=endpoint,
                                                                use_threshold = use_pThre, cox_top= cox_top_num, process_num=cox_proc_num)# without threshold select
                print(f'features after cox: {feature_matrix.shape[0]}->{feature_matrix_cox.shape[0]}')
                gene_list = gene_list_cox
                feature_matrix = feature_matrix_cox
                print(f'feature matrix shape: {feature_matrix.shape}; prognosis matrix shape: {prognosis_matrix.shape}')
                print(30*'-')
            # here features : 2000->1000

            # Purne the Incidence_Matrix H1 H2 H3
            H1 = H1.loc[gene_list['Gene_ID'],:]
            H1 = H1.loc[:,H1.sum(axis=0)!=0]
            H2 = H2.loc[H1.columns,:]
            H2 = H2.loc[:,H2.sum(axis=0)!=0]
            H3 = H3.loc[H2.columns,:]
            H3 = H3.loc[:,H3.sum(axis=0)!=0]
            print(f"H1,H2,H3:{H1.shape},{H2.shape},{H3.shape}")
            
            if to_csv:
                H1.to_csv(f_H1)
                H2.to_csv(f_H2)
                H3.to_csv(f_H3)
                feature_matrix.to_csv(ff)
                prognosis_matrix.to_csv(pf)
                gene_list.to_csv(gf)
    

    else:
        print(f'Skip the {cancer_type} data ,cause files has been existed already')


if __name__ == '__main__':
    _main()

