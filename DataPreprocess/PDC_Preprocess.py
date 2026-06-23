import pandas as pd
import numpy as np
import os
from Preprocess.DATA_preprocess import cox_feature_selection,discrete_time_bin,fil_min_nan_threshold
from scipy.stats import zscore 
def _main():
    KNOW='Reactome'# Reactome,STRING,ALL
    na_threshold=0.5
    process_num=90
    # for KNOW in ['STRING']:
    fn_label="/Backup/home/chenyupeng/DATA/CPTAC/PDC221025/PRO_Pan-cancer_Time-Event.csv"
    df_TE=pd.read_csv(fn_label,index_col=0)        
    # 读取ensg-pdc文件
    data_path="/Backup/home/chenyupeng/DATA/CPTAC/PDC221025/ENSG/"
    for KNOW in ['Reactome','ALL']:
        for file in os.listdir(data_path):
            if 'DDA' in file:
                # [] [cancer in file[i] for i in range(len(file))]:
                print(file)

                try:
                    cancer=file.split("-")[0]
                    method=file.split("-")[1]
                    feature_matrix = pd.read_csv(f"{data_path}/{file}",index_col=0)
                except:
                    # 下一个循环
                    print('pass')
                    continue
                # 交集
                feature_matrix = feature_matrix.loc[:,feature_matrix.columns.isin(df_TE.index)]
                time_event = df_TE.loc[feature_matrix.columns,:]
                if (time_event['Event'].sum()+time_event['Time'].sum())==0:
                    print('time wrong pass')
                    continue
                print("Oringinal feature matrix shape:",feature_matrix.shape)

                if KNOW=='STRING':
                    genes_STRING=pd.read_csv("/Backup/home/chenyupeng/DATA/Graph/STRING/clusters.protein.ensg.csv")['protein_id'].to_list()
                    genes_STRING = sorted(list(set(genes_STRING)))# 11171
                    gene_set=feature_matrix.index
                    gene_set=gene_set[gene_set.isin(genes_STRING)]#LSCC(11344->6570)
                    feature_matrix = feature_matrix.loc[gene_set.to_list(),:]
                elif KNOW=="Reactome":
                    gene_set = pd.read_csv("/Backup/home/chenyupeng/DATA/Graph/Reactome/ENSG2HSA.csv",index_col=0)['Gene']
                    feature_matrix = feature_matrix.loc[feature_matrix.index.isin(gene_set),:]# 7305

                elif KNOW=="ALL":
                    pass
                print(f"After KNOW {KNOW} filter, feature matrix shape:",feature_matrix.shape)

                # 癌症名称提取，再文件名的 -D 字符串之前
                out_path=f'{data_path}/{na_threshold}nafilter_CoxSort/{KNOW}/{cancer}-{method}/'
                file_dataset=out_path+f"/dataset.csv"
                file_genes=out_path+f"/genes_list.csv"
                # 判断dataset文件是否存在
                os.makedirs(out_path,exist_ok=True)
                # 判断是否存在dataset文件和
                if not os.path.isfile(file_dataset):
                    # log2
                    # feature_matrix = np.log2(feature_matrix+1)
                    
                    # min value fill na
                    print(f"Before fil_min_nan_threshold, feature matrix shape: {feature_matrix.shape}")
                    feature_matrix = feature_matrix.fillna(feature_matrix.min().min())
                    
                    # filt nan
                    feature_matrix = fil_min_nan_threshold(feature_matrix,na_threshold)# gene x sample
                
                    # 数据标准化(MEAN=0 NORM FOR DEEP LEARNING) z-score
                    # feature_matrix = zscore(feature_matrix,axis=1)# 每一个基因在所有样本上的均值接近零，方差接近一 
                    
                    feature_matrix = feature_matrix.T
                    print(f"After fillna and filt na, feature matrix shape: {feature_matrix.shape}")

                    if not os.path.isfile(file_genes):
                        # cox rank 
                        # feature_matrix (DataFrame): Input features with shape (n_samlpes, n_features).
                        # label_matrix (DataFrame): Input label with shape (n_samlpes, 2).
                        dataset,cox_p_value = cox_feature_selection(time_col="Time",event_col="Event",feature_matrix=feature_matrix,label_matrix= time_event, process_num=process_num)
                        gene_list = pd.DataFrame(index=dataset.columns[:-2],columns=["cox_p_value"],data = cox_p_value ).sort_values("cox_p_value")
                        gene_list.to_csv(file_genes)
                        print(f"Save gene_list in {file_genes} successfully !")
                    else:
                        print(f"Using existed Genes list file: {file_genes}")
                        gene_list = pd.read_csv(file_genes,index_col=0)

                    # Time Bins Preprocess
                    time_event_bin = discrete_time_bin(time_event,"Time",90)
                    print(f"{cancer} time max:{time_event_bin['Time'].max()} seasons")
                    dataset_sorted = pd.concat([feature_matrix.loc[:,gene_list.index.to_list()],time_event_bin['Time'],time_event_bin['Event']] ,axis=1)

                    dataset_sorted.to_csv(file_dataset)
                    print(f"Save dataset in {file_dataset} successfully !")

                    # H1.to_csv(out_path+f"/H1.csv")
                    # # 多队列数据集构建
                    # # Jiang\Gao\Xing
                    # Jiang = pd.read_csv(save_path+f"/Jiang/dataset.csv",index_col=0)
                    # Gao = pd.read_csv(save_path+f"/Gao/dataset.csv",index_col=0)
                    # xing = pd.read_csv(save_path+f"/xing/dataset.csv",index_col=0)
                    
                    # data_list=[Jiang,Gao,xing]
                    # [Jiang,Gao,xing]=cohorts_fetures_aggregation(data_list,500)
                    # Jiang.to_csv(save_path+f"/Jiang|Gao|Xing/Jiang-dataset.csv")
                    # Gao.to_csv(save_path+f"/Jiang|Gao|Xing/Gao-dataset.csv")
                    # xing.to_csv(save_path+f"/Jiang|Gao|Xing/Xing-dataset.csv")
if __name__ == '__main__':
    _main()