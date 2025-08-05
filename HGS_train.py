#!/usr/bin/env python3

import torch
import numpy as np
import logging
from utils.config_loader import ConfigLoader
from models.Models_interpret import HGS
from utils.data_utils import *
from utils.optuna_utils import *
from utils.hg_ops import *
import torch.optim as optim

def main():
    """Main training function using configuration"""
    
    # Load configuration
    config = ConfigLoader("config.yaml")
    
    # Get configurations
    hardware_config = config.get_hardware_config()
    data_config = config.get_data_config()
    model_config = config.get_model_config()
    
    # Setup hardware
    if hardware_config['use_cuda']:
        torch.cuda.set_device(hardware_config['cuda_device'])
    
    # Initialize results dictionary
    RD = {
        "Model": model_config['name'],
        "datasets": data_config['datasets'],
        "TestCims": [],
        "BestValCims": [],
        "TestMean": [],
        "ValMean": [],
        "H_shape": [],
        "EventRate": [],
        "OptDicList": []
    }
    
    # Process each dataset
    for dataset in RD["datasets"]:
        logger = logging.getLogger(f"{model_config['name']} - {dataset}")
        
        # Get knowledge parameters
        PK_D = config.build_PK_D_dict()
        
        # Create output directories
        config.create_directories(dataset)
        
        # Get file paths
        fn_data = config.get_data_path(dataset)
        fn_results = config.get_results_file_path(dataset)
        
        # Load data
        data, H = load_data_with_config(config, fn_data, dataset)
        
        # Bayesian optimization if enabled
        if model_config['bayesian_optimization']:
            HD = config.build_HD_dict(dataset=dataset)
            study, PK_D = KNOW_Opt_SC(
                DATA=data_config['type'],
                PK_D=PK_D,
                HD=HD,
                out_path=config.get_output_path(dataset),
                fn_data=fn_data,
                logger=logger,
                config=config
            )
        
        # Grid search
        best_metric = -np.inf
        OptDic = {"JM": model_config['judge_metric']}
        
        # Get all parameter combinations
        param_combinations = config.get_grid_search_params()
        
        for params in param_combinations:
            # Build HD with current parameters
            HD = config.build_HD_dict(**params)
            
            # Build prediction layers based on aggregation method
            n = 1 if params['AGG'] == 'noAGG' else 2
            HD['predict_hiddens'] = build_deep_hiddens(
                num_ini=n * HD['n_hid'],
                divisor=2,
                num_min=100,
                depth=params['depth']
            )
            
            logger.info(f"Grid Search Params: {params}")
            
            # Initialize results tracking
            ci_list, loss_list, test_list = [], [], []
            
            # Multiple runs with different seeds
            for seed in range(HD["repeat_time"]):
                # Get checkpoint path
                fn_ckpt = config.get_checkpoint_path(dataset, seed, params)
                
                # Check if checkpoint exists
                if os.path.isfile(fn_ckpt) and config.get_validation_config()['use_existing_checkpoints']:
                    logger.info(f'Using existing checkpoint: {fn_ckpt}')
                else:
                    # Train model
                    model = train_single_model(HD, PK_D, data, dataset, seed, fn_ckpt, logger, H)
                
                # Load results
                ckpt = torch.load(fn_ckpt)
                val_ci = ckpt[f'{model_config["judge_metric"]}_eval_ci']
                val_loss = ckpt[f'{model_config["judge_metric"]}_eval_loss']
                test_ci = ckpt['test_ci']
                
                # Record results
                ci_list.append(val_ci)
                loss_list.append(val_loss)
                test_list.append(test_ci)
                
                logger.info(f"{dataset} seed,val_ci,val_loss: {seed},{val_ci},{val_loss}")
            
            # Calculate statistics
            ci_mean = np.mean(ci_list)
            ci_std = np.std(ci_list)
            test_mean = np.mean(test_list)
            test_std = np.std(test_list)
            loss_mean = np.mean(loss_list)
            
            val_ms = f'{round(ci_mean, 4)}±{round(ci_std, 4)}'
            test_ms = f'{round(test_mean, 4)}±{round(test_std, 4)}'
            
            # Update best parameters
            if ci_mean > best_metric:
                best_metric = ci_mean
                OptDic.update({
                    "opt_lr": params['lr'],
                    "opt_agg": params['AGG'],
                    "best_ci": ci_mean,
                    "best_loss": loss_mean,
                    "test_ci": test_mean,
                    "test_std": test_std,
                    "BestCims": val_ms,
                    "TestCims": test_ms
                })
        
        # Record final results
        RD["BestValCims"].append(OptDic["BestCims"])
        RD["ValMean"].append(OptDic['best_ci'])
        RD["TestCims"].append(OptDic['TestCims'])
        RD["TestMean"].append(OptDic['test_ci'])
        RD["OptDicList"].append(OptDic)
        
        try:
            RD["H_shape"].append(H.shape)
        except:
            pass
        
        event_rate = data[:, -1].sum() / data.shape[0] if isinstance(data, np.ndarray) else data.iloc[:, -1].sum() / data.shape[0]
        RD["EventRate"].append(event_rate)
    
    # Calculate total statistics
    RD["TotalCiMean"] = f"{round(np.mean(RD['TestMean']), 4)}±{round(np.std(RD['TestMean']), 4)}"
    
    # Save final results
    with open(fn_results, 'a') as f:
        f.write(f'RD_{model_config["name"]} = {RD}\n')
        f.write(f'optimaDict_{model_config["name"]} = {OptDic}\n')
    
    logger.info(f"Training completed. Best metric: {best_metric}")
    logger.info(f"Results saved to: {fn_results}")

def load_data_with_config(config, fn_data, dataset):
    data_config = config.get_data_config()
    PK_D = config.build_PK_D_dict()
    HD = config.build_HD_dict()
    # 直接调用
    return load_data_general(data_config['type'], PK_D, HD, fn_data, config)

def train_single_model(HD, PK_D, data, dataset, seed, fn_ckpt, logger, H=None):
    """Train a single model with given parameters"""
    
    # Split data
    if PK_D['DataDrive'] == 'hcluster':
        data_train, data_valid, data_test, t_obs, H_Pro = data_split(
            seed=seed, dataset=dataset, data=data, HD=PK_D, construct_H=True
        )
        # Use H from data_split for non-Reactome
        H = H_Pro
    else:
        # For Reactome, use H from load_data_general
        data_train, data_valid, data_test, t_obs = data_split(
            seed=seed, dataset=dataset, data=data, HD=PK_D, construct_H=False
        )
        # H should be passed from load_data_general for Reactome
        if H is None:
            raise ValueError("H must be provided for Reactome data from load_data_general")
    
    # Build pooling layers
    HD['pooling_hiddens'] = build_hiddens(H.shape[1], PK_D['divisor'])
    logger.info(f"Pooling_hiddens: {HD['pooling_hiddens']}")
    
    # Generate graph
    G = generate_G_from_H(H.T) if HD["edge_pooling"] else generate_G_from_H(H)
    logger.info(f"H: {H.shape}; data: {data.shape}; G: {G.shape}")
    
    # Initialize model
    model = HGS(
        HD, data_train=data_train, data_eval=data_valid, data_test=data_test,
        H=H, fn_ckpt=fn_ckpt, t_obs=t_obs, G=G, seed=seed
    )
    model = model.cuda()
    
    # Train model
    optimizer = optim.Adam(model.parameters(), lr=HD['lr'], weight_decay=HD['l2'])
    model = model.fit(
        optimizer=optimizer,
        logger=logger,
        num_epochs=HD["epochs"],
        batch_size=HD["batch_size"],
        loss_dict=HD['loss_w']
    )
    
    return model

if __name__ == "__main__":
    main() 