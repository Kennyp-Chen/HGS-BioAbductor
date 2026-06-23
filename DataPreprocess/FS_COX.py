from Preprocess.DATA_preprocess import cox_feature_selection,discrete_time_bin,fil_min_nan_threshold
# from DATA_preprocess import cox_feature_selection,discrete_time_bin,fil_min_nan_threshold
import pandas as pd
import os
# from utils.AcrossCohort_utils import cohorts_fetures_aggregation
def _main():
    KNOW='STRING'# Reactome,STRING,ALL
    for KNOW in ['STRING','ALL']:
        omics = "Protein"

        na_threshold=0.5
        process_num=90
        save_path=f"/Backup/home/chenyupeng/DATA/HCC_MultiCohorts/{omics}Cohorts_{na_threshold}nafilter_CoxSort/{KNOW}/"

        # data_df = pd.read_csv('/Backup/home/chenyupeng/DATA/AcrossCohort/HCC_transcript/clinic.csv',index_col=0)
        if omics=='Protein':
            data_df = pd.read_csv("/Backup/home/chenyupeng/DATA/HCC_MultiCohorts/ProteomicsCohorts_clinic.csv",index_col=0)
        else:
            data_df = pd.read_csv('/Backup/home/chenyupeng/DATA/HCC_MultiCohorts/clinic.csv',index_col=0)

        ungene_cols = data_df.columns[:8].to_list()
        # 取Reactome交集
        # 加入取Reactome交集、取STRING交集、不取交集三个可选项
        if KNOW=='Reactome':
            H1 = pd.read_csv("/Backup/home/chenyupeng/DATA/AcrossCohort/HCC_transcript/TCGA_LIHC/H1.csv",header=0,index_col=0)
            data_df = data_df.loc[:,ungene_cols+H1.index.to_list()]
        elif KNOW=='STRING':
            genes_STRING=pd.read_csv("/Backup/home/chenyupeng/DATA/Graph/STRING/clusters.protein.ensg.csv")['protein_id'].to_list()
            genes_STRING = list(set(genes_STRING))# 11171
            gene_set=data_df.columns[8:]# 3509
            gene_set=gene_set[gene_set.isin(genes_STRING)]
            data_df = data_df.loc[:,ungene_cols+gene_set.to_list()]

        elif KNOW=="ALL":
            pass
        # 单一队列数据集构建
        cohorts=data_df.loc[data_df['omics']==omics,]["cohort"].unique()
        print(f"{omics} omis cohorts:{cohorts}")
        for cohort in cohorts:
            print(cohort)
            out_path = save_path+f"/{cohort}/"
            file_dataset=out_path+f"/dataset.csv"
            file_genes=out_path+f"/genes_list.csv"
            # 判断dataset文件是否存在
            os.makedirs(out_path,exist_ok=True)
            # 取出队列
            data_cohort = data_df.loc[data_df["cohort"]==cohort,]
            if not os.path.isfile(file_dataset):
                # 取出特征矩阵
                feature_matrix = data_cohort.iloc[:,8:]# pats,genes

                # min value fill na
                feature_matrix = feature_matrix.fillna(feature_matrix.min().min())
                # fil nan
                feature_matrix_trans = fil_min_nan_threshold(feature_matrix.T,na_threshold)
                feature_matrix = feature_matrix_trans.T
                
                if not os.path.isfile(file_genes):
                    time_event = data_cohort.loc[:,["OS time","death"]]# pats,2
                    # cox rank 
                    dataset,cox_p_value = cox_feature_selection(time_col="OS time",event_col="death",feature_matrix=feature_matrix,label_matrix= time_event, process_num=process_num)
                    gene_list = pd.DataFrame(index=dataset.columns[:-2],columns=["cox_p_value"],data = cox_p_value ).sort_values("cox_p_value")
                    gene_list.to_csv(file_genes)
                    print(f"Save gene_list in {file_genes} successfully !")
                else:
                    print(f"Using existed Genes list file: {file_genes}")
                    gene_list = pd.read_csv(file_genes,index_col=0)

                # Time Bins Preprocess
                data_cohort_dt = discrete_time_bin(data_cohort,"OS time",3)# 一季度
                print(f"{cohort} time max:{data_cohort_dt['OS time'].max()} seasons")

                time_event = data_cohort_dt.loc[:,["OS time","death"]]# pats,2
                dataset_sorted = pd.concat([feature_matrix.loc[:,gene_list.index.to_list()],time_event],axis=1)

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