"""
Grid Search for Base Survival Analysis Models on all cancers.
Supports: Cox (CSA), Random Survival Forest (RSF)

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
from itertools import product

import numpy as np
import pandas as pd
import torch

from models.Models import COX_train_eval, RSF_train_eval
from utils.data_utils import load_opt_data, load_cohort_data, data_split

warnings.filterwarnings("ignore")


class GridSearchConfig:
    """Configuration for grid search hyperparameters."""
    
    SEARCH_SPACES = {
        "RSF": {
            "n_estimators": [500, 1000],
            "min_samples_split": [10, 50],
            "max_features": ['log2', 'sqrt'],
            "min_samples_leaf": [5, 20]
        },
        "CSA": {
            "alpha": 10.0 ** np.linspace(-2, 2, 5),
            "ties": ["breslow", "efron"]
        }
    }
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.search_space = self.SEARCH_SPACES[model_name]
    
    def get_param_combinations(self) -> List[Dict]:
        """Generate all parameter combinations for grid search."""
        keys = self.search_space.keys()
        values = self.search_space.values()
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations


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


class BaseModelGridSearch:
    """Main class for running grid search on base models."""
    
    MODELS = ['CSA', 'RSF']
    
    DATASETS = {
        'PRO': ['CCRCC', 'HaNSCC', 'LA', 'LSCC', 'UCEC', 'HCC', 'GBM', 'PDA'],
        'RNA': ['LIHC', 'STAD', 'BLCA', 'OV', 'LUSC', 'LGG', 'LUAD', 'KIRC', 'HNSC', 'BRCA']
    }
    
    def __init__(self, omics_data: str, device: str, repeat_time: int):
        self.omics_data = omics_data
        self.device = device
        self.repeat_time = repeat_time
        self.cox_num = 400
        self.res_path = Path("Results")
        self.data_path = Path("data")
        
        self._setup_device()
        
    def _setup_device(self):
        """Configure CUDA device."""
        if torch.cuda.is_available():
            torch.cuda.set_device(f"cuda:{self.device}")
            logging.info(f"Using CUDA device: {self.device}")
        else:
            logging.warning("No CUDA device found, using CPU")
    
    def _load_data(self, cohort: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data based on omics type."""
        if self.omics_data == 'PRO':
            fn_data = self.data_path / self.omics_data / cohort / "dataset.csv"
            data, H = load_cohort_data(None, str(fn_data), self.cox_num, DF=True)
        
        elif self.omics_data == 'RNA':
            fn_data = self.data_path / self.omics_data / cohort
            data, H = load_opt_data(
                data_path=str(fn_data),
                H_path=str(fn_data),
                endpoint='OS_60',
                Filt_num=self.cox_num,
                DF=False
            )
        else:
            raise ValueError(f"Unknown omics data type: {self.omics_data}")
        
        return data, H
    
    def _train_or_load_model(self, model_name: str, cohort: str, seed: int,
                            hyper_params: Dict, data: np.ndarray, 
                            out_path: Path, logger) -> Dict:
        """Train model or load existing checkpoint."""
        # Create checkpoint filename
        param_str = "_".join([f"{k}-{v}" for k, v in hyper_params.items()])
        fn_ckpt = out_path / f"{cohort}_seed-{seed}_{param_str}"
        ckpt_path = Path(f"{fn_ckpt}.ckpt")
        if ckpt_path.exists():
            logger.info(f'Using existing checkpoint: {fn_ckpt.name}.ckpt')
            ckpt = torch.load(
                f'{fn_ckpt}.ckpt',
                map_location={"cuda:0": f"cuda:{self.device}"}
            )
        else:
            # Split data
            data_train, data_valid, data_test, t_obs = data_split(seed=seed, data=data)
            logger.info(f"Data shape: {data.shape}, Time bins observed: {t_obs}")
            
            # Train model
            if model_name == "CSA":
                ckpt = COX_train_eval(
                    data_train=data_train,
                    data_eval=data_valid,
                    data_test=data_test,
                    HD=hyper_params,
                    fn=str(fn_ckpt)
                )
            elif model_name == "RSF":
                ckpt = RSF_train_eval(
                    data_train=data_train,
                    data_eval=data_valid,
                    data_test=data_test,
                    seed=seed,
                    HD=hyper_params,
                    fn=str(fn_ckpt)
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")
        
        ckpt["HyperParas"] = hyper_params
        return ckpt
    
    def _evaluate_hyperparameters(self, model_name: str, cohort: str,
                                  hyper_params: Dict, data: np.ndarray,
                                  out_path: Path, fn_results: Path) -> Dict:
        """Evaluate a set of hyperparameters across multiple seeds."""
        logger = logging.getLogger(f"{model_name}-{cohort}")
        logger.info(f"Hyperparameters: {hyper_params}")
        
        val_list, test_list = [], []
        
        for seed in range(self.repeat_time):
            ckpt = self._train_or_load_model(
                model_name, cohort, seed, hyper_params, data, out_path, logger
            )
            
            val_ci = ckpt['eval_ci']
            test_ci = ckpt['test_ci']
            
            # Record results
            with open(fn_results, 'a') as f:
                f.write(f"{cohort},{seed},{val_ci},{test_ci},{hyper_params}\n")
            
            val_list.append(val_ci)
            test_list.append(test_ci)
            
            logger.info(f"{cohort} seed={seed}, val_ci={val_ci:.4f}, test_ci={test_ci:.4f}")
        
        # Compute statistics
        val_mean, val_std = np.mean(val_list), np.std(val_list)
        test_mean, test_std = np.mean(test_list), np.std(test_list)
        
        val_ms = f'{val_mean:.4f}±{val_std:.4f}'
        test_ms = f'{test_mean:.4f}±{test_std:.4f}'
        
        with open(fn_results, 'a') as f:
            f.write(f'Valid Cindex Mean±Std = {val_ms}\n')
        
        return {
            'val_mean': val_mean,
            'val_std': val_std,
            'test_mean': test_mean,
            'test_std': test_std,
            'val_ms': val_ms,
            'test_ms': test_ms,
            'hyper_params': hyper_params
        }

    
    def _run_cohort_grid_search(self, model_name: str, cohort: str,
                               config: GridSearchConfig) -> Dict:
        """Run grid search for a single cohort."""
        logger = logging.getLogger(f"{model_name}-{cohort}")
        
        # Setup output paths
        search_keys = list(config.search_space.keys())
        out_base = self.res_path / "GridSearch" / self.omics_data / model_name / cohort / str(search_keys)
        out_path = out_base / "Results_Files"
        out_path.mkdir(parents=True, exist_ok=True)
        
        fn_results = out_base / "Results.csv"
        
        # Load data
        data, H = self._load_data(cohort)
        
        # Initialize results file
        with open(fn_results, 'w') as f:
            f.write(f'GridSearchSpace:\n{config.search_space}\n')
            f.write(f'cohort,seed,val_ci,test_ci,hyper_params\n')
        
        # Track best parameters
        best_metric = -np.inf
        best_params = None
        best_val_ci = None
        best_test_ci = None
        best_val_ms = None
        best_test_ms = None
        
        # Grid search over all parameter combinations
        param_combinations = config.get_param_combinations()
        logger.info(f"Total parameter combinations: {len(param_combinations)}")
        
        for hyper_params in param_combinations:
            results = self._evaluate_hyperparameters(
                model_name, cohort, hyper_params, data, out_path, fn_results
            )
            
            # Update best parameters
            if results['val_mean'] > best_metric:
                best_metric = results['val_mean']
                best_params = hyper_params
                best_val_ci = results['val_mean']
                best_test_ci = results['test_mean']
                best_val_ms = results['val_ms']
                best_test_ms = results['test_ms']
                
                logger.info(f"Updating best parameters. Best valid C-index: {best_metric:.4f}")
                
                with open(fn_results, 'a') as f:
                    f.write(f'Updating The Best Option\n')
                    f.write(f'Test Cindex Mean±Std = {best_test_ms}\n')
        
        # Write final best results
        opt_dict = {
            "cohort": cohort,
            "BestHyperParas": best_params,
            "best_val_ci": best_val_ci,
            "test_ci": best_test_ci,
            "BestValCims": best_val_ms,
            "BestTestCims": best_test_ms
        }
        
        with open(fn_results, 'a') as f:
            f.write(f'\nOptimal Parameters: {opt_dict}\n')
        
        return opt_dict
    
    def run(self):
        """Run grid search for all models and datasets."""
        for model_name in self.MODELS:
            logging.info(f"\n{'='*60}\nStarting Grid Search for {model_name}\n{'='*60}")
            
            config = GridSearchConfig(model_name)
            datasets = self.DATASETS[self.omics_data]
            
            results_dict = {
                "Model": model_name,
                "cohorts": datasets,
                "TestCims": [],
                "BestValCims": [],
                "TestMean": [],
                "ValMean": [],
                "OptDicList": []
            }
            
            # Run grid search for each cohort
            for cohort in datasets:
                logging.info(f"\nProcessing cohort: {cohort}")
                opt_dict = self._run_cohort_grid_search(model_name, cohort, config)
                
                results_dict["BestValCims"].append(opt_dict["BestValCims"])
                results_dict["ValMean"].append(opt_dict['best_val_ci'])
                results_dict["TestCims"].append(opt_dict['BestTestCims'])
                results_dict["TestMean"].append(opt_dict['test_ci'])
                results_dict["OptDicList"].append(opt_dict)
            
            # Compute overall statistics
            total_mean = np.mean(results_dict['TestMean'])
            total_std = np.std(results_dict['TestMean'])
            results_dict["TotalCiMean"] = f"{total_mean:.4f}±{total_std:.4f}"
            
            logging.info(f"\n{model_name} Overall Results:")
            logging.info(f"Total C-index Mean±Std: {results_dict['TotalCiMean']}")
            logging.info(f"Results Dictionary: {results_dict}")
            
            # Save final results
            out_base = self.res_path / "GridSearch" / self.omics_data / model_name
            out_base.mkdir(parents=True, exist_ok=True)
            final_results_file = out_base / f"Final_Results.csv"
            pd.DataFrame(results_dict).to_csv(final_results_file, index=False)
            logging.info(f"Final results saved to: {final_results_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Grid Search for Base Survival Models (Cox, RSF)'
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
    grid_search = BaseModelGridSearch(
        omics_data=args.omics_data,
        device=args.device,
        repeat_time=args.repeat_time
    )
    grid_search.run()
    
    logging.info(f"\nExperiment completed. Log saved to: {exp_logger.log_file}")


if __name__ == "__main__":
    main()
