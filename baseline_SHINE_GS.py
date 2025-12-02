"""
Grid Search for SHINE Model on all cancers.
Optimized version with improved readability, efficiency, and automatic logging.
"""
import pandas as pd
import os
import sys
import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim

from models.Models import SHINE
from utils.data_utils import load_cohort_data, load_TCGA_data, data_split, load_H_G

warnings.filterwarnings("ignore")


class ExperimentLogger:
    """Handles logging setup for experiments."""
    
    def __init__(self, script_name: str, args=None):
        self.script_name = script_name
        self.args = args
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self._create_log_filename()
        self._setup_logging()
        
    def _create_log_filename(self) -> Path:
        """Create log filename with timestamp, script name, and arguments."""
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        
        # Add argument information to filename if available
        if self.args:
            arg_str = f"-{self.args.omics_data}-re{self.args.repeat_time}"
            return self.log_dir / f"{timestamp}-{self.script_name}{arg_str}.log"
        else:
            return self.log_dir / f"{timestamp}-{self.script_name}.log"
    
    def _setup_logging(self):
        """Configure logging to both file and console."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter(log_format)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        logging.info(f"Log file created: {self.log_file}")


class GridSearchConfig:
    """Configuration for grid search hyperparameters."""
    
    SEARCH_SPACE = {
        'n_hid': [100, 200],
        'l2': [0.005, 0.0005],
        'batch_size': [32, 1024],
        'jump_know': [True, False],
        'glr': [0.1, 0],
    }
    
    BASE_HYPERPARAMS = {
        "lr": 0.001,
        "act": "Tanh",
        "risk_type": "discrete_time",
        'cox_num': 400,
        "batch_norm": True,
        "epochs": 50,
        "dropout": 0.5,
        "loss_w": {'a': 1, 'b': 1, 'c1': 1, 'c2': 0, 'd': 1},
        "tyep_know": "SHINE",
    }


class SHINEGridSearch:
    """Main class for running grid search on SHINE model."""
    
    DATASETS = {
        'PRO': ['CCRCC', 'HaNSCC', 'LA', 'LSCC', 'UCEC', 'HCC', 'GBM', 'PDA'],
        'RNA': ['LIHC', 'STAD', 'BLCA', 'OV', 'LUSC', 'LGG', 'LUAD', 'KIRC', 'HNSC', 'BRCA']
    }
    
    def __init__(self, omics_data: str, device: str, repeat_time: int):
        self.omics_data = omics_data
        self.device = device
        self.repeat_time = repeat_time
        self.res_path = Path("Results")
        self.data_path = Path("data")
        self.h_path = self.data_path / "PriorKnow/SHINE/gene_pathway_msig_ENSG_protein_combine.csv"
        self._setup_device()
        self.out_base = self.res_path / "GridSearch" / self.omics_data / "SHINE" 
        
    def _setup_device(self):
        """Configure CUDA device."""
        if torch.cuda.is_available():
            torch.cuda.set_device(f"cuda:{self.device}")
            logging.info(f"Using CUDA device: {self.device}")
        else:
            logging.warning("No CUDA device found, using CPU")
    
    def _load_data(self, cohort: str, cox_num: int) -> Tuple:
        """Load data based on omics type."""
        if self.omics_data == 'PRO':
            fn_data = self.data_path / self.omics_data / cohort / "dataset.csv"
            data, _ = load_cohort_data(None, str(fn_data), cox_num, DF=True)
        elif self.omics_data == 'RNA':
            fn_data = self.data_path / self.omics_data / cohort / "feature_matrix.csv"
            data = load_TCGA_data(str(fn_data), "OS_60", Filt_num=cox_num)
        else:
            raise ValueError(f"Unknown omics data type: {self.omics_data}")
        
        # Load hierarchical structure
        data, H, G = load_H_G(data, str(self.h_path))
        return data, H, G
    
    def _create_checkpoint_name(self, hyper_params: Dict) -> str:
        """Create checkpoint filename from grid search hyperparameters."""
        # Only use SEARCH_SPACE parameters for checkpoint naming
        search_params = GridSearchConfig.SEARCH_SPACE.keys()
        name_parts = [f"{k}:{hyper_params[k]}" for k in search_params if k in hyper_params]
        return "-".join(name_parts).replace(" ", "")
    
    def _train_or_load_model(self, cohort: str, data, hyper_params: Dict,
                            fn_ckpt: str, logger,H) -> Dict:
        """Train model or load existing checkpoint."""
        results = {'ci_list': [], 'loss_list': [], 'test_list': []}
        
        for seed in range(self.repeat_time):
            fn_ckpt_seed = f"{fn_ckpt}-seed:{seed}"
            ckpt_path = Path(f"{fn_ckpt_seed}.ckpt")
            
            if ckpt_path.exists():
                logger.info(f'Using existing checkpoint: {ckpt_path.name}')
            else:
                # Split data
                data_train, data_valid, data_test, t_obs = data_split(
                    seed=seed, dataset=cohort, data=data, HD=None, construct_H=False
                )
                
                # Update hyperparameters with time bins
                hyper_params['time_bins'] = t_obs
                
                # Create and train model
                model = SHINE(
                    hyper_params,
                    data_train=data_train,
                    data_eval=data_valid,
                    data_test=data_test,
                    H=H,
                    fn_ckpt=fn_ckpt_seed,
                    seed=seed
                )
                model = model.cuda()
                
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=hyper_params['lr'],
                    weight_decay=hyper_params['l2']
                )
                
                logger.info(f"Training seed: {seed}")
                model.fit(
                    optimizer=optimizer,
                    logger=logger,
                    num_epochs=hyper_params["epochs"],
                    batch_size=hyper_params["batch_size"],
                    loss_dict=hyper_params['loss_w']
                )
            
            # Load checkpoint and extract results
            ckpt = torch.load(
                f'{fn_ckpt_seed}.ckpt',
                map_location={"cuda:0": f"cuda:{self.device}"}
            )
            
            results['ci_list'].append(ckpt['final_eval_ci'])
            results['loss_list'].append(ckpt['final_eval_loss'])
            results['test_list'].append(ckpt['final_test_ci'])
        
        return results
    
    def _compute_statistics(self, results: Dict) -> Dict:
        """Compute mean and std statistics from results."""
        ci_mean = np.mean(results['ci_list'])
        ci_std = np.std(results['ci_list'])
        test_mean = np.mean(results['test_list'])
        test_std = np.std(results['test_list'])
        loss_mean = np.mean(results['loss_list'])
        
        return {
            'val_ci_mean': ci_mean,
            'val_ci_std': ci_std,
            'test_ci_mean': test_mean,
            'test_ci_std': test_std,
            'loss_mean': loss_mean,
            'val_cims': f'{ci_mean:.4f}±{ci_std:.4f}',
            'test_cims': f'{test_mean:.4f}±{test_std:.4f}'
        }
    
    def _run_cohort_grid_search(self, cohort: str) -> Dict:
        """Run grid search for a single cohort."""
        logger = logging.getLogger(f"SHINE-{cohort}")
        
        # Initialize hyperparameters
        hyper_params = GridSearchConfig.BASE_HYPERPARAMS.copy()
        hyper_params['cancer'] = cohort
        hyper_params['repeat_times'] = self.repeat_time
        
        # Log loss configuration
        if hyper_params["risk_type"] == "discrete_time":
            if hyper_params["loss_w"] == {'a': 1, 'b': 1, 'c1': 0, 'c2': 0, 'd': 0}:
                logger.info("Using DeepHit Head")
            elif hyper_params["loss_w"] == {'a': 1, 'b': 1, 'c1': 1, 'c2': 0, 'd': 1}:
                logger.info("Using Triplet loss {'a': 1, 'b': 1, 'c1': 1, 'c2': 0, 'd': 1}")
        else:
            logger.info("Using Deep Surv Head")
        
        logger.info(f"Hyperparameters for {cohort}: {hyper_params}")
        
        # Setup output paths
        self.out_path =   self.out_base / cohort / "Results_Files" / "GridSearch"
        self.out_path.mkdir(parents=True, exist_ok=True)
        
        fn_results = self.out_path / "Results.csv"
        
        # Load data
        data, H, G = self._load_data(cohort, hyper_params['cox_num'])
        
        
        # Initialize results file
        with open(fn_results, 'w') as f:
            f.write(f'GridSearchSpace:\n{GridSearchConfig.SEARCH_SPACE}\n')
            f.write(f'Hyperparameters for {cohort} Knowledge Optimization: {hyper_params}\n')
            f.write(f'dataset,seed,val_ci,test_ci,val_loss\n')
        
        # Grid search
        opt_dict = {
            "cancer": cohort,
            "HyperParas": {},
            "BestHyperParas": {},
            "best_metric": -np.inf
        }
        
        for n_hid in GridSearchConfig.SEARCH_SPACE['n_hid']:
            for l2 in GridSearchConfig.SEARCH_SPACE['l2']:
                for batch_size in GridSearchConfig.SEARCH_SPACE['batch_size']:
                    for jump_know in GridSearchConfig.SEARCH_SPACE['jump_know']:
                        for glr in GridSearchConfig.SEARCH_SPACE['glr']:
                            # Update hyperparameters
                            hyper_params.update({
                                'n_hid': n_hid,
                                'l2': l2,
                                'batch_size': batch_size,
                                'jump_know': jump_know,
                                'glr': glr
                            })
                            
                            # Create checkpoint name
                            fn_ckpt = self.out_path / self._create_checkpoint_name(hyper_params)
                            logger.info(f"Checkpoint: {fn_ckpt.name}")
                            
                            # Train or load model
                            results = self._train_or_load_model(
                                cohort, data, hyper_params, str(fn_ckpt), logger,H
                            )
                            
                            # Write individual results
                            with open(fn_results, 'a') as f:
                                f.write(30 * "-" + '\n')
                                f.write(f'{hyper_params}\n')
                                f.write(f'dataset,seed,val_ci,test_ci,val_loss\n')
                                for seed, (ci, test, loss) in enumerate(zip(
                                    results['ci_list'],
                                    results['test_list'],
                                    results['loss_list']
                                )):
                                    f.write(f"{cohort},{seed},{ci},{test},{loss}\n")
                            
                            # Compute statistics
                            stats = self._compute_statistics(results)
                            
                            # Write statistics
                            with open(fn_results, 'a') as f:
                                f.write(f"final valid Cindex mean±std = {stats['val_cims']}\n")
                                f.write(f"final test Cindex mean±std = {stats['test_cims']}\n")
                            
                            logger.info(f"{cohort} valid Cindex mean±std = {stats['val_cims']}")
                            logger.info(f"{cohort} test Cindex mean±std = {stats['test_cims']}")
                            
                            # Update best parameters
                            if stats['val_ci_mean'] > opt_dict['best_metric']:
                                with open(fn_results, 'a') as f:
                                    f.write(f"--UPDATE BEST PARAM--\n")
                                
                                opt_dict.update({
                                    'best_metric': stats['val_ci_mean'],
                                    'BestHyperParas': hyper_params.copy(),
                                    'best_loss': stats['loss_mean'],
                                    'best_val_mean': stats['val_ci_mean'],
                                    'best_test_mean': stats['test_ci_mean'],
                                    'best_val_cims': stats['val_cims'],
                                    'best_test_cims': stats['test_cims']
                                })
        
        return opt_dict
    
    def run(self):
        """Run grid search for all datasets."""
        logging.info(f"\n{'='*60}\nStarting Grid Search for SHINE\n{'='*60}")
        
        datasets = self.DATASETS[self.omics_data]
        
        results_dict = {
            "Model": "SHINE",
            "datasets": datasets,
            "TestCims": [],
            "BestValCims": [],
            "TestMean": [],
            "ValMean": [],
            "OptDicList": []
        }
        
        # Run grid search for each cohort
        for cohort in datasets:
            logging.info(f"\nProcessing cohort: {cohort}")
            opt_dict = self._run_cohort_grid_search(cohort)
            
            results_dict["BestValCims"].append(opt_dict["best_val_cims"])
            results_dict["ValMean"].append(opt_dict["best_val_mean"])
            results_dict["TestCims"].append(opt_dict['best_test_cims'])
            results_dict["TestMean"].append(opt_dict['best_test_mean'])
            results_dict["OptDicList"].append(opt_dict)
        
        # Compute overall statistics
        total_mean = np.mean(results_dict['TestMean'])
        total_std = np.std(results_dict['TestMean'])
        results_dict["TotalCiMean"] = f"{total_mean:.4f}±{total_std:.4f}"
        
        logging.info(f"\nSHINE Overall Results:")
        logging.info(f"Total C-index Mean±Std: {results_dict['TotalCiMean']}")
        logging.info(f"Results Dictionary: {results_dict}")
        # Final Save
        
        pd.DataFrame(results_dict).to_csv(self.out_base / f"Final_Results.csv")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Grid Search for SHINE Survival Model'
    )
    parser.add_argument(
        "--omics_data",
        type=str,
        default="RNA",
        choices=["RNA", "PRO"],
        help="Omics data type: RNA or PRO"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device to use (e.g., 0, 1, ...)"
    )
    parser.add_argument(
        "--repeat_time",
        type=int,
        default=10,
        help="Number of repetitions for data split"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging with arguments
    script_name = Path(__file__).stem
    exp_logger = ExperimentLogger(script_name, args)
    
    logging.info(f"Starting experiment with parameters:")
    logging.info(f"  Omics Data: {args.omics_data}")
    logging.info(f"  Device: {args.device}")
    logging.info(f"  Repeat Time: {args.repeat_time}")
    
    # Run grid search
    grid_search = SHINEGridSearch(
        omics_data=args.omics_data,
        device=args.device,
        repeat_time=args.repeat_time
    )
    grid_search.run()
    
    logging.info(f"\nExperiment completed. Log saved to: {exp_logger.log_file}")


if __name__ == "__main__":
    main()
