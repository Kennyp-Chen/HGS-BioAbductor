import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd
import numpy as np
import os
from DataPreprocess.Preprocess.DATA_preprocess import cox_feature_selection,discrete_time_bin,fil_min_nan_threshold
from scipy.stats import zscore
def _main():
    parser = argparse.ArgumentParser(description="Prepare legacy Cox-ranked PDC cohort files")
    parser.add_argument("--labels", default="data/CPTAC/PRO_Pan-cancer_Time-Event.csv")
    parser.add_argument("--data-dir", default="data/CPTAC/ENSG")
    parser.add_argument("--output-root", default="data/CPTAC/preprocessed")
    parser.add_argument("--prior-knowledge", nargs="+", choices=["STRING", "Reactome", "ALL"], default=["Reactome", "ALL"])
    parser.add_argument("--na-threshold", type=float, default=0.5)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--string-genes", default="data/PriorKnow/STRING/clusters.protein.ensg.csv")
    parser.add_argument("--reactome-mapping", default="data/PriorKnow/Reactome/ENSG2HSA.csv")
    args = parser.parse_args()
    na_threshold=args.na_threshold
    process_num=args.processes
    # for KNOW in ['STRING']:
    fn_label=args.labels
    df_TE=pd.read_csv(fn_label,index_col=0)
    # 读取ensg-pdc文件
    data_path=args.data_dir
    for KNOW in args.prior_knowledge:
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
                    genes_STRING=pd.read_csv(args.string_genes)['protein_id'].to_list()
                    genes_STRING = sorted(list(set(genes_STRING)))# 11171
                    gene_set=feature_matrix.index
                    gene_set=gene_set[gene_set.isin(genes_STRING)]#LSCC(11344->6570)
                    feature_matrix = feature_matrix.loc[gene_set.to_list(),:]
                elif KNOW=="Reactome":
                    gene_set = pd.read_csv(args.reactome_mapping,index_col=0)['Gene']
                    feature_matrix = feature_matrix.loc[feature_matrix.index.isin(gene_set),:]# 7305

                elif KNOW=="ALL":
                    pass
                print(f"After KNOW {KNOW} filter, feature matrix shape:",feature_matrix.shape)

                # 癌症名称提取，再文件名的 -D 字符串之前
                out_path=f'{args.output_root}/{na_threshold}nafilter_CoxSort/{KNOW}/{cancer}-{method}/'
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


if __name__ == "__main__":
    _main()
