"""
Grid Search for Deep Learning Models on all cancers.
Supports: DeepSurv, DeepHit, DRSA, Pnet

Optimized version with improved readability, efficiency, and automatic logging.
"""

import os
import sys
import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from models.Models import DeepSurv, DeepHit, DRSA, Pnet
from utils.data_utils import load_opt_data, load_cohort_data, data_split
from utils.ReactomeNet import get_BINN_Pathways

warnings.filterwarnings("ignore")


class GridSearchConfig:
    """Configuration for grid search hyperparameters."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.l2_values = [0.0005, 0.005, 0.1]
        self.lr_values = [0.001, 0.01]
        self.num_layers = self._get_num_layers()
        
    def _get_num_layers(self) -> List[int]:
        """Get number of layers based on model type."""
        layer_config = {
            "DeepHit": [0, 1, 2],
            "Pnet": [2, 3, 4, 5],
        }
        return layer_config.get(self.model_name, [1, 2, 3])


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
            
            # Add models information if specified
            if hasattr(self.args, 'models') and self.args.models:
                models_str = "_".join(self.args.models)
                arg_str += f"-{models_str}"
            
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


class DeepModelGridSearch:
    """Main class for running grid search on deep learning models."""
    
    ALL_MODELS = ['DeepHit', 'Pnet', 'DeepSurv', 'DRSA']
    
    DATASETS = {
        'PRO': ['CCRCC', 'HaNSCC', 'LA', 'LSCC', 'UCEC', 'HCC', 'GBM', 'PDA'],
        'RNA': ['LIHC', 'STAD', 'BLCA', 'OV', 'LUSC', 'LGG', 'LUAD', 'KIRC', 'HNSC', 'BRCA']
    }
    
    def __init__(self, omics_data: str, device: str, repeat_time: int, models: List[str] = None):
        self.omics_data = omics_data
        self.device = device
        self.repeat_time = repeat_time
        self.batch_size = 64
        self.cox_num = 400
        self.res_path = Path("Results")
        self.data_path = Path("data")
        self.pnet_pathways = Path("data") / "PriorKnow/PNET/PNET_pathways.tsv"
        self.pnet_map =  Path("data") / "PriorKnow/PNET/ENSG2HSA.csv"
        
        # Set models to train
        if models is None:
            self.models = self.ALL_MODELS
        else:
            # Validate model names
            invalid_models = [m for m in models if m not in self.ALL_MODELS]
            if invalid_models:
                raise ValueError(f"Invalid model names: {invalid_models}. Valid models: {self.ALL_MODELS}")
            self.models = models
        
        logging.info(f"Models to train: {self.models}")
        self._setup_device()
        
    def _setup_device(self):
        """Configure CUDA device."""
        if torch.cuda.is_available():
            torch.cuda.set_device(f"cuda:{self.device}")
            logging.info(f"Using CUDA device: {self.device}")
        else:
            logging.warning("No CUDA device found, using CPU")
    
    def _load_data(self, cohort: str, model_name: str) -> Tuple:
        """Load data based on omics type and model."""
        if self.omics_data == 'RNA':
            return self._load_rna_data(cohort, model_name)
        elif self.omics_data == 'PRO':
            return self._load_pro_data(cohort, model_name)
        else:
            raise ValueError(f"Unknown omics data type: {self.omics_data}")
    
    def _load_rna_data(self, cohort: str, model_name: str,) -> Tuple:
        """Load RNA data."""
        fn_data = self.data_path / self.omics_data / cohort / "dataset_Reactome12_1000genes.csv"
        if model_name != 'Pnet':
            data,H = load_cohort_data(None,str(fn_data),num_filt=self.cox_num,DF=True)
            pathway = None
        else:
            # Reactome 4 Pnet
            fn_data = self.data_path / self.omics_data / cohort 
            H_path = f"data/PriorKnow/PNET/Reactome/{self.omics_data}/{cohort}"
            data,H = load_opt_data(data_path=str(fn_data),H_path=H_path,endpoint='OS_60',Filt_num=-1,DF=True) 
            pathway_mask,data = get_BINN_Pathways(data,4)

        return data, H, pathway_mask
    
    def _load_pro_data(self, cohort: str, model_name: str) -> Tuple:
        """Load PRO data."""
        fn_data = self.data_path / self.omics_data / cohort / "dataset.csv"
        if model_name != 'Pnet':
            data, H = load_cohort_data(None, str(fn_data), self.cox_num, DF=True)
            pathway = None 
        else:
            data = pd.read_csv(fn_data,index_col=0)
            pathway_mask, data = get_BINN_Pathways(data, 4, cox_num=self.cox_num,fn_pathways =self.pnet_pathways,fn_mapping = self.pnet_map)
            H = None
        return data, H, pathway_mask

    
    def _create_model(self, model_name: str, data_train, data_valid, data_test, 
                     seed: int, num_layers: int, fn_ckpt: str, 
                     pathway_mask=None, t_obs=None):
        """Create model instance based on model type."""
        model_params = {
            'data_train': data_train,
            'data_eval': data_valid,
            'data_test': data_test,
            'nn_seed': seed,
            'dropout': 0.5,
            'fn': fn_ckpt
        }
        
        if model_name == "DeepSurv":
            return DeepSurv(**model_params, ACT="LeakyReLU", NumLayers=num_layers)
        
        elif model_name == "DeepHit":
            return DeepHit(**model_params, ACT="LeakyReLU", NumLayers=num_layers, t_obs=t_obs)
        
        elif model_name == "DRSA":
            return DRSA(**model_params, lstm_layers=num_layers, t_obs=t_obs)
        
        elif model_name == 'Pnet':
            mask = pathway_mask[:num_layers]
            return Pnet(**model_params, pathway_mask=mask, ACT="tanh", t_obs=t_obs)

        else:
            raise ValueError(f"Unknown model: {model_name}")
        

    
    def _train_or_load_model(self, model_name: str, cohort: str, seed: int,
                            l2: float, lr: float, num_layers: int,
                            data, pathway_mask, out_path: Path, logger) -> Dict:
        """Train model or load existing checkpoint."""
        # Create checkpoint filename
        fn_ckpt = out_path / f"{cohort}_50eps_seed-{seed}_l2-{l2}_lr-{lr}_nl{num_layers}_bs{self.batch_size}"
        ckpt_path = Path(f"{fn_ckpt}.ckpt")
        
        if ckpt_path.exists():
            logger.info(f'Using existing checkpoint: {ckpt_path.name}')
        else:
            # Split data
            data_train, data_valid, data_test, t_obs = data_split(
                seed=seed, data=data, dataset=cohort, construct_H=False
            )
            
            # Create and train model
            model = self._create_model(
                model_name, data_train, data_valid, data_test,
                seed, num_layers, str(fn_ckpt), pathway_mask, t_obs
            )
            model = model.cuda()
            
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
            model.fit(optimizer=optimizer, logger=logger, num_epochs=50, batch_size=self.batch_size)
        
        # Load checkpoint
        ckpt = torch.load(f'{fn_ckpt}.ckpt', map_location={"cuda:0": f"cuda:{self.device}"})
        
        return {
            'val_ci': ckpt['final_eval_ci'],
            'val_loss': ckpt['final_eval_loss'],
            'test_ci': ckpt['final_test_ci']
        }
    
    def _run_cohort_grid_search(self, model_name: str, cohort: str, 
                               config: GridSearchConfig) -> Dict:
        """Run grid search for a single cohort."""
        logger = logging.getLogger(f"{model_name}-{cohort}")
        # if model_name == "Pnet":
        #     self.batch_size = 32
        # else:
        #     self.batch_size = 64
        # Setup output paths
        self.out_base = self.res_path / "Benchmark" / self.omics_data / model_name 
        out_path = self.out_base / cohort / "GridSearch"
        out_path.mkdir(parents=True, exist_ok=True)
        
        fn_results = self.out_base / cohort /"Results.csv"
        
        # Load data
        data, H, pathway_mask = self._load_data(cohort, model_name)
        
        # Initialize results file
        with open(fn_results, 'w') as f:
            f.write(f'GridSearchSpace:\n')
            f.write(f'LR = {config.lr_values}, L2 = {config.l2_values}, NL = {config.num_layers}\n')
            f.write(f'cohort,seed,L2,lr,nl,val_ci,val_loss,test_ci\n')
        
        best_metric = -np.inf
        opt_params = {}
        
        # Grid search
        for l2 in config.l2_values:
            for lr in config.lr_values:
                for nl in config.num_layers:
                    logger.info(f"Grid Search Params: L2={l2}, lr={lr}, nl={nl}")
                    
                    ci_list, loss_list, test_list = [], [], []
                    
                    # Run multiple seeds
                    for seed in range(self.repeat_time):
                        results = self._train_or_load_model(
                            model_name, cohort, seed, l2, lr, nl,
                            data, pathway_mask, out_path, logger
                        )
                        
                        # Record results
                        with open(fn_results, 'a') as f:
                            f.write(f"{cohort},{seed},{l2},{lr},{nl},"
                                  f"{results['val_ci']},{results['val_loss']},{results['test_ci']}\n")
                        
                        ci_list.append(results['val_ci'])
                        loss_list.append(results['val_loss'])
                        test_list.append(results['test_ci'])
                        
                        logger.info(f"{cohort} seed={seed}, val_ci={results['val_ci']:.4f}, "
                                  f"val_loss={results['val_loss']:.4f}")
                    
                    # Compute statistics
                    ci_mean, ci_std = np.mean(ci_list), np.std(ci_list)
                    test_mean, test_std = np.mean(test_list), np.std(test_list)
                    loss_mean = np.mean(loss_list)
                    
                    val_ms = f'{ci_mean:.4f}±{ci_std:.4f}'
                    test_ms = f'{test_mean:.4f}±{test_std:.4f}'
                    
                    with open(fn_results, 'a') as f:
                        f.write(f"valid Cindex mean±std = {val_ms}\n")
                        f.write(f"test Cindex mean±std = {test_ms}\n")
                    
                    # Update best parameters
                    if ci_mean > best_metric:
                        with open(fn_results, 'a') as f:
                            f.write(f"--UPDATE BEST PARAM--\n")
                        
                        best_metric = ci_mean
                        opt_params = {
                            "opt_l2": l2,
                            "opt_lr": lr,
                            "opt_nl": nl,
                            "best_ci": ci_mean,
                            "best_loss": loss_mean,
                            "test_ci": test_mean,
                            "test_std": test_std,
                            "BestCims": val_ms,
                            "TestCims": test_ms
                        }
        
        # Write final results
        with open(fn_results, 'a') as f:
            f.write(f'\nOptimal Parameters: {opt_params}\n')
        
        return opt_params
    
    def run(self):
        """Run grid search for all models and datasets."""
        for model_name in self.models:
            logging.info(f"\n{'='*60}\nStarting Grid Search for {model_name}\n{'='*60}")
            
            config = GridSearchConfig(model_name)
            datasets = self.DATASETS[self.omics_data]
            
            results_dict = {
                "Model": model_name,
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
                opt_params = self._run_cohort_grid_search(model_name, cohort, config)
                
                results_dict["BestValCims"].append(opt_params["BestCims"])
                results_dict["ValMean"].append(opt_params['best_ci'])
                results_dict["TestCims"].append(opt_params['TestCims'])
                results_dict["TestMean"].append(opt_params['test_ci'])
                results_dict["OptDicList"].append(opt_params)
            
            # Compute overall statistics
            total_mean = np.mean(results_dict['TestMean'])
            total_std = np.std(results_dict['TestMean'])
            results_dict["TotalCiMean"] = f"{total_mean:.4f}±{total_std:.4f}"
            
            logging.info(f"\n{model_name} Overall Results:")
            logging.info(f"Total C-index Mean±Std: {results_dict['TotalCiMean']}")
            logging.info(f"Results Dictionary: {results_dict}")
            
            # Save final results
            final_results_file = self.out_base / f"Final_Results.csv"
            pd.DataFrame(results_dict).to_csv(final_results_file, index=False)
            logging.info(f"Final results saved to: {final_results_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Grid Search for Deep Learning Survival Models'
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
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=None,# None
        choices=["DeepHit", "Pnet", "DeepSurv", "DRSA"],
        help="List of models to train (e.g., --models DeepHit DeepSurv). If not specified, all models will be trained."
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
    logging.info(f"  Models: {args.models if args.models else 'All models'}")
    
    # Run grid search
    grid_search = DeepModelGridSearch(
        omics_data=args.omics_data,
        device=args.device,
        repeat_time=args.repeat_time,
        models=args.models
    )
    grid_search.run()
    
    logging.info(f"\nExperiment completed. Log saved to: {exp_logger.log_file}")


if __name__ == "__main__":
    main()
