from Preprocess.DATA_preprocess import cox_filter, load_feature_matrix
import pandas as pd
def _main():
    H1 = pd.read_csv("H1.Edge.Ten2Hundred.csv",header=0,index_col=0)
    H2 = pd.read_csv("H2.Edge.Ten2Hundred.csv",header=0,index_col=0)
    H3 = pd.read_csv("H3.Edge.Ten2Hundred.csv",header=0,index_col=0)
    prognosis_matrix=pd.read_csv("TCGA/dataset_BRCA/prognosis_days.csv")
    gene_list = pd.read_csv("gene.csv")
    fn_feat_BRCA = f"TCGA/standardFileFoldTCGA/BRCA-Intensity-Matrix.csv"
    feature_matrix = load_feature_matrix(fn_feat_BRCA ,cancer_type="BRCA")

    # select patients with prognosis information 
    genes = feature_matrix.loc[feature_matrix["EnsembleID"].isin(gene_list["Gene_ID"]), ].iloc[:,:2]
    feature_matrix = feature_matrix.loc[feature_matrix["EnsembleID"].isin(gene_list["Gene_ID"]), prognosis_matrix["PatientID"]]

    # feature_matrix_cox, gene_list_cox = cox_filter(feature_matrix, prognosis_matrix, gene_list,col_use="OS",
    #                                                     use_threshold = False, cox_top=0, process_num=10)# without threshold select
    # gene_list = gene_list_cox
    # feature_matrix = feature_matrix_cox
    feature_matrix_trans = feature_matrix.T
    lowvar_sort = feature_matrix_trans.var().dropna().sort_values(ascending=False)
    feature_matrix = feature_matrix.loc[lowvar_sort.index, ]
    genes=genes.loc[lowvar_sort.index, ]
    # Purne the Incidence_Matrix H1 H2 H3
    H1 = H1.loc[gene_list['Gene_ID'],:]
    H1 = H1.loc[:,H1.sum(axis=0)!=0]
    H2 = H2.loc[H1.columns,:]
    H2 = H2.loc[:,H2.sum(axis=0)!=0]
    H3 = H3.loc[H2.columns,:]
    H3 = H3.loc[:,H3.sum(axis=0)!=0]
    print(f"H1,H2,H3:{H1.shape},{H2.shape},{H3.shape}")

    # save csv
    out_path = f'TCGA/dataset_BRCA/remove_extreme_path_dataset_LowVarSelect_BRCA/'
    ff = f'{out_path}features.csv' 
    gf = f'{out_path}genes_list.csv'
    f_H1 = f'{out_path}H1.csv'
    f_H2 = f'{out_path}H2.csv'
    f_H3 = f'{out_path}H3.csv'
    H1.to_csv(f_H1)
    H2.to_csv(f_H2)
    H3.to_csv(f_H3)
    feature_matrix.to_csv(ff)
    gene_list.to_csv(gf)

if __name__ == '__main__':
    _main()