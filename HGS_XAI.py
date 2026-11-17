import torch
import numpy as np
from models.Models import HGS_Interpret
import os
import torch.optim as optim
from utils.data_utils import *
import matplotlib.pyplot as plt
from utils.optuna_utils import *
from utils.hg_ops import *
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from utils.Interpret import *
from utils.interpret_prompt import *
import json
from utils.Borda_Score_validator import *
from utils.config_loader import ConfigLoader
from pathlib import Path
def get_borda_score_10seeds(HD, PK_D, dataset, data, Model, DATA):
    """Calculate Borda scores across 10 seeds and return averaged results"""
    td_h_10s, td_n_10s, bu_h_10s, bu_n_10s = [], [], [], []
    
    for seed in range(10):
        if PK_D['type_know'] == 'hcluster':
            data_train, data_valid, data_test, t_obs, H_df = data_split(
                seed=seed, dataset=dataset, data=data, HD=PK_D, construct_H=True, H_df=True
            )
        else:
            data_train, data_valid, data_test, t_obs = data_split(
                seed=seed, dataset=dataset, data=data, HD=PK_D, construct_H=False
            )
        
        H = H_df.values
        G = generate_G_from_H(H.T) if HD["edge_pooling"] else generate_G_from_H(H)
        logger.info(f"H:{H.shape}; data:{data.shape}; G:{G.shape}; t_obs:{t_obs}")
        logger.info(f"{dataset} Best Paras-{seed}:L2{HD['l2']};lr{HD['lr']};glr{HD['glr']};agg{HD['AGG']};ph{HD['predict_hiddens']}")
        
        HD['pooling_hiddens'] = build_hiddens(H.shape[1], PK_D['divisor'])
        logger.info(f"Pooling_hiddens:{HD['pooling_hiddens']}\n")
        
        fn_ckpt = f"borda_scores/ckpt/seed{seed}"
        os.makedirs(f"borda_scores/ckpt/", exist_ok=True)
        
        # Initialize model for Borda score calculation
        model = HGS(HD, data_train=data_train, data_eval=data_valid, data_test=data_test, 
                   H=H, fn_ckpt=fn_ckpt, t_obs=t_obs, G=G, seed=seed)
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=HD['lr'], weight_decay=HD['l2'])
        model = model.fit(optimizer=optimizer, logger=logger, num_epochs=HD["epochs"], 
                         batch_size=HD["batch_size"], loss_dict=HD['loss_w'])
        
        (top_remove_ci_h, bottom_remove_ci_h), (top_remove_ci_n, bottom_remove_ci_n), percent_space = \
            borda_score_validation(model, data_base=data_train, data_test=data_test, H=H, seed=seed)
        
        td_h_10s.append(top_remove_ci_h)
        td_n_10s.append(top_remove_ci_n)
        bu_h_10s.append(bottom_remove_ci_h)
        bu_n_10s.append(bottom_remove_ci_n)
    
    td_h, td_n, bu_h, bu_n = np.array(td_h_10s).mean(axis=0), np.array(td_n_10s).mean(axis=0), \
                              np.array(bu_h_10s).mean(axis=0), np.array(bu_n_10s).mean(axis=0)
    
    plot_bs_validation("10mean", percent_space, td_h, bu_h, percent_space, td_n, bu_n)
    return td_h_10s, td_n_10s, bu_h_10s, bu_n_10s

def borda_score_validation(model, data_base, data_test, H, seed):
    """Calculate Borda scores for a single model"""
    # Calculate Borda score
    model_interpreter = Interpreter(model, data_test[:, :-2])
    ## Use training set mean as baseline
    baseline = torch.mean(torch.Tensor(data_base[:, :-2]).cuda(), dim=0)
    node_borda, hyperedge_borda = model_interpreter.all_borda_rank(
        model.hgc_encoder[0], baselineTensor=baseline, IGsteps=50
    )
    
    # Borda Score Validate
    percent_space_h, top_remove_ci_h, bottom_remove_ci_h = validate_hyperedge_borda_scores(
        data_test, hyperedge_borda, H, 0.5, model, method='baseline'
    )
    percent_space_n, top_remove_ci_n, bottom_remove_ci_n = validate_node_borda_scores(
        data_test, node_borda, H, 0.5, model, method='baseline'
    )
    
    plot_bs_validation(seed, percent_space_h, top_remove_ci_h, bottom_remove_ci_h, 
                      percent_space_n, top_remove_ci_n, bottom_remove_ci_n)
    
    return (top_remove_ci_h, bottom_remove_ci_h), (top_remove_ci_n, bottom_remove_ci_n), percent_space_n

def plot_bs_validation(seed, percent_space_h, top_remove_ci_h, bottom_remove_ci_h, 
                      percent_space_n, top_remove_ci_n, bottom_remove_ci_n):
    """Plot Borda score validation results"""
    # Plot grid and line points
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].plot(percent_space_h, top_remove_ci_h, label='top', marker='o')
    ax[0].plot(percent_space_h, bottom_remove_ci_h, label='bottom', marker='o')
    ax[0].set_xlabel('Percentage of removed edges')
    ax[0].set_ylabel('Borda Score')
    ax[0].set_title('Hyperedge Borda Score')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(percent_space_n, top_remove_ci_n, label='top', marker='o')
    ax[1].plot(percent_space_n, bottom_remove_ci_n, label='bottom', marker='o')
    ax[1].set_xlabel('Percentage of removed nodes')
    ax[1].set_ylabel('Borda Score')
    ax[1].set_title('Node Borda Score')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'borda_scores/borda_score_validate_seed{seed}.svg')

def split_xai_data(data, dataset, PK_D, seed, H_df=None):
    """Split data while retaining a fixed prior or building one on training data."""
    if PK_D['type_know'] != 'Reactome':
        data_train, data_valid, data_test, t_obs, H_df = data_split(
            seed=seed, dataset=dataset, data=data, HD=PK_D, construct_H=True, H_df=True
        )
    else:
        data_train, data_valid, data_test, t_obs = data_split(
            seed=seed, dataset=dataset, data=data, HD=PK_D, construct_H=False
        )
    
    if H_df is None:
        raise ValueError("A prior-knowledge incidence matrix is required for this experiment")
    return data_train, data_valid, data_test, t_obs, H_df


def train_and_explain_model(HD, PK_D, dataset, data, Model, DATA, out_path, fn_ckpt, seed, logger, H_df=None):
    """Train model and perform explanation analysis"""
    data_train, data_valid, data_test, t_obs, H_df = split_xai_data(data, dataset, PK_D, seed, H_df)
    H = H_df.values
    G = generate_G_from_H(H.T) if HD["edge_pooling"] else generate_G_from_H(H)
    logger.info(f"H:{H.shape}; data:{data.shape}; G:{G.shape}; t_obs:{t_obs}")
    
    # Build pooling layers
    HD['pooling_hiddens'] = build_hiddens(H.shape[1], PK_D['divisor'])
    logger.info(f"Pooling_hiddens:{HD['pooling_hiddens']}\n")
    
    # Initialize and train model
    model = HGS_Interpret(HD, data_train=data_train, data_eval=data_valid, data_test=data_test,
                H=H, fn_ckpt=fn_ckpt, t_obs=t_obs, G=G, seed=seed)
    model = model.cuda()
    
    if os.path.isfile(f'{fn_ckpt}.ckpt'):
        logger.info(f'Using existing {fn_ckpt}.ckpt')
        model.load_state_dict(torch.load(f'{fn_ckpt}.ckpt')['state_dict'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=HD['lr'], weight_decay=HD['l2'])
        model = model.fit(optimizer=optimizer, logger=logger, num_epochs=HD["epochs"],
                         batch_size=HD["batch_size"], loss_dict=HD['loss_w'])
    
    # Calculate Borda scores
    model_interpreter = Interpreter(model, data_test[:, :-2])
    baseline = torch.mean(torch.Tensor(data_train[:, :-2]).cuda(), dim=0)
    node_borda, hyperedge_borda = model_interpreter.all_borda_rank(
        model.hgc_encoder[0], baselineTensor=baseline, IGsteps=50
    )
    
    # Borda Score Validation
    percent_space_h, top_remove_ci_h, bottom_remove_ci_h = validate_hyperedge_borda_scores(
        data_test, hyperedge_borda, H, 0.5, model, method='baseline'
    )
    percent_space_n, top_remove_ci_n, bottom_remove_ci_n = validate_node_borda_scores(
        data_test, node_borda, H, 0.5, model, method='baseline'
    )
    
    # Plot results
    plot_borda_validation_results(percent_space_h, top_remove_ci_h, bottom_remove_ci_h,
                                percent_space_n, top_remove_ci_n, bottom_remove_ci_n, out_path)
    
    return model, node_borda, hyperedge_borda, data_train, data_test, H_df

def plot_borda_validation_results(percent_space_h, top_remove_ci_h, bottom_remove_ci_h,
                                percent_space_n, top_remove_ci_n, bottom_remove_ci_n, out_path):
    """Plot Borda validation results"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].plot(percent_space_h, top_remove_ci_h, label='top')
    ax[0].plot(percent_space_h, bottom_remove_ci_h, label='bottom')
    ax[0].set_xlabel('Percentage of masked edges')
    ax[0].set_ylabel('C-index')
    ax[0].set_title('Hyperedge Borda Score')
    ax[0].legend()
    
    ax[1].plot(percent_space_n, top_remove_ci_n, label='top')
    ax[1].plot(percent_space_n, bottom_remove_ci_n, label='bottom')
    ax[1].set_xlabel('Percentage of masked nodes')
    ax[1].set_ylabel('C-index')
    ax[1].set_title('Node Borda Score')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{out_path}/borda_score_validate.png')

def perform_triplets_analysis(model, H_df, data, out_path, dn_data, hyperedge_borda, node_borda, 
                            n_triplets, PK_D, fn_map, dn_esdb):
    """Perform triplets analysis and save results"""
    TA = TripletsAnalysis(model, H_df, data, out_path, dn_data, hyperedge_borda, node_borda, 
                         n_triplets, PK_D, fn_map, dn_esdb)
    
    # Create pair names
    pair_name = [f"{[TA.node_name[int(i)] for i in TA.pair_borda_scores[1]][j]}-{[TA.hyperedge_name[int(m)] for m in TA.pair_borda_scores[2]][j]}" 
                 for j in range(TA.pair_borda_scores.shape[1])]
    
    # Save Borda scores
    bordas_dict = {
        "node_borda": list(node_borda),
        "hyperedge_borda": list(hyperedge_borda),
        "node_name": TA.node_name,
        "hyperedge_name": TA.hyperedge_name,
        "pair_name": pair_name,
        "pair_borda": list(TA.pair_borda_scores[0])
    }
    
    with open(f"{out_path}/bordas_dict.json", "w") as f:
        json.dump(bordas_dict, f)
    
    # Get triplets statistics
    triplets_statistics = TA.get_triplets_statistics('liver neoplasm')
    with open(out_path + f'/{n_triplets}triplets_statistics.json', 'w') as f:
        json.dump(triplets_statistics, f, indent=4)
    
    return TA, bordas_dict, triplets_statistics

def load_xai_data(omics, data_file, parameters, knowledge, paths):
    """Load a cohort and its fixed prior; clustering builds its prior after splitting."""
    kind = knowledge['type_know']
    if kind not in ('hcluster', 'STRING', 'Reactome'):
        raise ValueError(f"Unsupported knowledge type: {kind}")
    h_path = paths['reactome_graph_template'].format(layer=knowledge.get('layer_Reactome', 11))
    if omics == 'PRO':
        matrix = str(Path(h_path) / 'H1.csv') if kind == 'Reactome' else None
        if matrix and not Path(matrix).is_file():
            raise FileNotFoundError(matrix)
        return load_cohort_data(matrix, data_file, parameters['cox_num'], DF=True)
    if omics != 'RNA':
        raise ValueError(f"Unsupported omics type: {omics}")
    if kind == 'Reactome':
        return load_opt_data(data_path=str(Path(data_file).parent), H_path=h_path,
                             endpoint=parameters['end_point'], Filt_num=parameters['cox_num'], DF=True)
    return load_TCGA_data(data_file, parameters['end_point'], Filt_num=parameters['cox_num']), None


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="Train and explain an HGS experiment")
    parser.add_argument("--experiment", default="configs/xai_hcc.yaml", help="XAI experiment YAML")
    args = parser.parse_args()
    with open(args.experiment, encoding="utf-8") as f:
        experiment = yaml.safe_load(f)
    DATA, dataset = experiment['omics_data'], experiment['dataset']
    HD = experiment['model_parameters']
    HD['dataset'] = dataset
    PK_D = experiment['knowledge_parameters']
    KNOW, Model = PK_D['type_know'], "HGS"
    torch.cuda.set_device(f"cuda:{experiment.get('device', 0)}")
    logger = logging.getLogger(__name__)

    print(f"Selected parameters for {DATA}-{dataset}-{KNOW}:")
    print(f"HD: {HD}")
    print(f"PK_D: {PK_D}")

    config = ConfigLoader()
    path_config = config.get_paths()
    data_root = path_config['data_root']
    # Select paths from the experiment, independently of the shared default omics type.
    template = 'proteomic_data_template' if DATA == 'PRO' else 'rna_data_template'
    dn_data = path_config[template].format(dataset=dataset)
    data, H_df = load_xai_data(DATA, dn_data, HD, PK_D, path_config)
    out_path = f"Results/Interpret/{DATA}-{dataset}-{KNOW}/"
    out_path += str(list(HD.items())[:HD['n_hp_ckpt']]).replace("), (", "-").replace(",", ":").replace("'", "").replace(" ", "")[2:-2]
    out_path += "-" + str(list(PK_D.items())).replace("), (", "-").replace(",", ":").replace("'", "").replace(" ", "")[2:-2]
    os.makedirs(out_path, exist_ok=True)

    fn_ckpt = out_path + f"/{Model}-{dataset}"

    print(f"Output directory: {out_path}")
    print(f"Checkpoint file: {fn_ckpt}")
    model, node_borda, hyperedge_borda, data_train, data_test, H_df = train_and_explain_model(
        HD, PK_D, dataset, data, Model, DATA, out_path, fn_ckpt, HD['seed'], logger, H_df=H_df
    )
    n_triplets = 30
    fn_map = config.get_paths()['mart_export'].format(data_root=data_root)
    dn_esdb = config.get_paths()['gsva_dir'].format(data_root=data_root)

    TA, bordas_dict, triplets_statistics = perform_triplets_analysis(
        model, H_df, data, out_path, str(Path(dn_data).parent), hyperedge_borda, node_borda,
        n_triplets, PK_D, fn_map, dn_esdb
    )

if __name__ == "__main__":
    main()
