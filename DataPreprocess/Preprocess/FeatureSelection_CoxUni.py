from Preprocess.DATA_preprocess import cox_filter, load_feature_matrix
import pandas as pd
import os
def _main():
    '''
    此版本未将H的index进行排序，需要再load时进行对齐。
    '''
    os.chdir("/home/chenyupeng/Projects/BaseLine/TCGA/")
    # save csv
    fn_out="Reactome-P4"
    out_path = f'{fn_out}/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    H1 = pd.read_csv("reactome_P4/H1.csv",header=0,index_col=0)
    H2 = pd.read_csv("reactome_P4/H2.csv",header=0,index_col=0)
    H3 = pd.read_csv("reactome_P4/H3.csv",header=0,index_col=0)
    prognosis_matrix=pd.read_csv("dataset_BRCA/prognosis_days.csv")
    gene_list = pd.read_csv("reactome_P4/genes_list.csv").iloc[:,-2:]
    # reset p_cox
    gene_list["cox_p"]=1
    fn_feat_BRCA = f"standardFileFoldTCGA/BRCA-Intensity-Matrix.csv"
    feature_matrix = load_feature_matrix(fn_feat_BRCA ,cancer_type="BRCA")

    gene_index = gene_list.loc[:,"Gene_ID"]
    fm = feature_matrix.set_index("EnsembleID")
    feature_matrix = fm.loc[gene_index,prognosis_matrix["PatientID"]]

    # # Cut 
    # gene_list = gene_list.iloc[:300,:]
    # feature_matrix = feature_matrix.iloc[:300,:]
    
    feature_matrix_cox, gene_list_cox = cox_filter(feature_matrix, prognosis_matrix, gene_list,col_use="OS",use_threshold = False, cox_top=0, process_num=60)# without threshold select
    gene_list = gene_list_cox
    feature_matrix = feature_matrix_cox

    # Purne the Incidence_Matrix H1 H2 H3
    H1 = H1.loc[feature_matrix.index,:]
    H1 = H1.loc[:,H1.sum(axis=0)!=0]
    H2 = H2.loc[H1.columns,:]
    H2 = H2.loc[:,H2.sum(axis=0)!=0]
    H3 = H3.loc[H2.columns,:]
    H3 = H3.loc[:,H3.sum(axis=0)!=0]
    print(f"H1,H2,H3:{H1.shape},{H2.shape},{H3.shape}")


    ff = f'{out_path}features.csv' 
    gf = f'{out_path}genes_list.csv'
    f_H1 = f'{out_path}H1.csv'
    f_H2 = f'{out_path}H2.csv'
    f_H3 = f'{out_path}H3.csv'
    H1.to_csv(f_H1)
    H2.to_csv(f_H2)
    H3.to_csv(f_H3)
    feature_matrix.to_csv(ff)
    gene_list.to_csv(gf,index=None)

if __name__ == '__main__':
    _main()