"""
HGS Training with AutoML Optimization for Prior Knowledge and Grid Search for Architecture.
Optimized version with improved readability, efficiency, and automatic logging.
"""

import os
import sys
import argparse
import logging
import warnings
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.optim as optim
import optuna
import optunahub

from models.Models import HGS
from utils.data_utils import (
    load_cohort_data, load_TCGA_data, load_TCGA_H_data,
    load_opt_data, data_split, Know_features_filt, build_hiddens, build_deep_hiddens
)
from utils.optuna_utils import KNOW_Search
from utils.hg_ops import generate_G_from_H
import pandas as pd

warnings.filterwarnings("ignore")


class ExperimentLogger:
    """Handles logging setup for experiments."""
    
    def __init__(self, script_name: str, args: Optional[argparse.Namespace] = None):
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
            arg_str = f"-{self.args.omics_data}-{self.args.prior_knowledge}-nt{self.args.n_trials}-{self.args.bayesian_optimization}"
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


class ConfigManager:
    """Manages configuration for model architecture parameters."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def load_architecture_config(self, omics_data: str) -> Dict:
        """Load model architecture configuration from YAML file."""
        config_file = self.config_dir / f"hgs_architecture_{omics_data}.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Architecture config loaded from: {config_file}")
        return config


class PriorKnowledgeConfig:
    """Configuration for prior knowledge parameters."""
    
    @staticmethod
    def get_config(knowledge_type: str, config_dict: Dict) -> Dict:
        """Get prior knowledge configuration from loaded config."""
        if knowledge_type not in config_dict['prior_knowledge']:
            raise ValueError(f"Unknown knowledge type: {knowledge_type}")
        
        pk_config = config_dict['prior_knowledge'][knowledge_type].copy()
        pk_config['type_know'] = knowledge_type
        
        # Convert list to tuple for div_know_minmax
        if 'div_know_minmax' in pk_config:
            pk_config['div_know_minmax'] = tuple(pk_config['div_know_minmax'])
        
        return pk_config


class GridSearchConfig:
    """Configuration for grid search hyperparameters."""
    
    @staticmethod
    def get_search_space(config_dict: Dict) -> Dict:
        """Get grid search space from loaded config."""
        return config_dict['grid_search_space']


class HGSOptimization:
    """Main class for HGS model optimization."""
    
    
    DATASETS = {
        'PRO': ['CCRCC', 'HaNSCC', 'LA', 'LSCC', 'UCEC', 'HCC', 'GBM', 'PDA'],
        'RNA': ['LIHC', 'STAD', 'BLCA', 'OV', 'LUSC', 'LGG', 'LUAD', 'KIRC', 'HNSC', 'BRCA']
    }
    
    def __init__(self, omics_data: str, knowledge_type: str, device: str,
                 n_trials: int, opt_type: str):
        self.omics_data = omics_data
        self.knowledge_type = knowledge_type
        self.device = f"cuda:{device}"
        self.n_trials = n_trials
        self.opt_type = opt_type
        
        self.res_path = Path("Results")
        self.data_path = Path("data")
        self.config_manager = ConfigManager()
        
        # Load configuration
        self.config = self.config_manager.load_architecture_config(omics_data)
        if omics_data=='RNA':
            if self.knowledge_type=='Reactome':
                self.config['feature_base']='ReactomeTop'
            elif self.knowledge_type=='hcluster':
                self.config['feature_base']='Reactome'
            elif self.knowledge_type=='STRING':
                self.config['feature_base']='Reactome'
        elif omics_data=='PRO':
            self.config['feature_base']='STRING'
        self.feature_base = self.config['feature_base']
        
        self._setup_device()
        
    def _setup_device(self):
        """Configure CUDA device."""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            logging.info(f"Using CUDA device: {self.device}")
        else:
            logging.warning("No CUDA device found, using CPU")
    
    def _get_base_hyperparams(self, cohort: str) -> Dict:
        """Get base hyperparameters from config."""
        hyper_params = self.config['base_hyperparams'].copy()
        
        # Add dynamic parameters
        n_hid = hyper_params['n_hid']
        hyper_params.update({
            "MLP_hiddens": [n_hid],
            "predict_hiddens": [n_hid, n_hid // 2],
            "model": "HGS",
            "dataset": cohort,
            'n_trials': self.n_trials,
            'OptType': self.opt_type,
            "feature_base": self.feature_base,
            'n_hp_ckpt': 10
        })
        
        return hyper_params
    
    def _get_data_path(self, cohort: str) -> Path:
        """Get data file path based on omics type and feature base."""
        if self.omics_data == 'PRO':
            return self.data_path / self.omics_data / cohort / "dataset.csv"
        else:  # RNA
            if self.feature_base == 'Reactome':
                return self.data_path / self.omics_data / cohort / "feature_matrix.csv"
            elif self.feature_base == 'ReactomeP12':
                return self.data_path / self.omics_data / cohort / "dataset_Reactome12_1000genes.csv"
            elif self.feature_base == 'ReactomeTop':
                return self.data_path / self.omics_data / cohort
        return None
    
    def _get_output_path(self, cohort: str, hyper_params: Dict) -> Tuple[Path, Path]:
        """Get output paths for results."""
        out_path = self.res_path / "Benchmark" / self.omics_data / f"HGS-{self.knowledge_type}" / hyper_params['OptType']  
        self.optunadb_path = self.res_path / "Benchmark" / "optuna_db"/ hyper_params['OptType']  
        # Results/Benchmark/optuna_db/Auto/RNA-STRING-STAD-layer_range.db
        results_file = out_path / cohort /  f"Results-nt{self.n_trials}.csv"
        checkpoint_path = out_path / cohort / "GridSearch"
        self.fn_final = out_path / f"Final_Results-nt{self.n_trials}.csv"
        return results_file, checkpoint_path
    
    def _create_sampler(self):
        """Create Optuna sampler based on optimization type."""
        samplers = {
            'TPE': lambda: optuna.samplers.TPESampler(seed=0),
            'GP': lambda: optuna.samplers.GPSampler(seed=0),
            'Random': lambda: optuna.samplers.RandomSampler(seed=0),
            'CmaEs': lambda: optuna.samplers.CmaEsSampler(seed=0),
            'Auto': lambda: optunahub.load_module("samplers/auto_sampler").AutoSampler(seed=0)
        }
        return samplers.get(self.opt_type, samplers['Auto'])()
    
    def _load_data(self, cohort: str, pk_config: Dict, hyper_params: Dict) -> Tuple:
        """Load data based on omics type and knowledge configuration."""
        fn_data = self._get_data_path(cohort)
        
        if self.omics_data == 'PRO':
            if self.knowledge_type == 'Reactome':
                fn_H = self.data_path / f"PriorKnow/Reactome/reactome_P{pk_config['layer_Reactome']}/H1.csv"
                data, H = load_cohort_data(str(fn_H), str(fn_data), hyper_params['cox_num'], DF=False)
            else:  # STRING or hcluster
                data, H = load_cohort_data(None, str(fn_data), hyper_params['cox_num'], DF=True)
        
        elif self.omics_data == 'RNA':
            if pk_config["type_know"] == 'hcluster':
                data = load_TCGA_data(str(fn_data), hyper_params["end_point"], Filt_num=hyper_params["cox_num"])
                H = None
            elif pk_config["type_know"] == 'Reactome':
                H_path = self.data_path / f"PriorKnow/Reactome/reactome_P{pk_config['layer_Reactome']}"
                data, H = load_opt_data(
                    data_path=str(fn_data),
                    H_path=str(H_path),
                    endpoint=hyper_params["end_point"],
                    Filt_num=hyper_params['cox_num'],
                    DF=False
                )
            elif pk_config["type_know"] == 'STRING':
                data = load_TCGA_data(str(fn_data), hyper_params["end_point"], Filt_num=-1)
                data = Know_features_filt(data)
                H = None
        
        return data, H
    
    def _create_checkpoint_name(self, hyper_params: Dict, pk_config: Dict, seed: int) -> str:
        """Create checkpoint filename from hyperparameters."""
        # Extract key hyperparameters
        n_hp = hyper_params['n_hp_ckpt']
        hp_items = list(hyper_params.items())[:n_hp]
        hp_str = str(hp_items).replace("), (", "-").replace(",", ":").replace("'", "").replace(" ", "")[2:-2]
        
        pk_str = str(list(pk_config.items())).replace("), (", "-").replace(",", ":").replace("'", "").replace(" ", "")[2:-2]
        
        return f"{hp_str}-{pk_str}-seed:{seed}"

    def _train_or_load_model(self, cohort: str, seed: int, data, H,
                            hyper_params: Dict, pk_config: Dict,
                            fn_ckpt: str, logger) -> Dict:
        """Train model or load existing checkpoint."""
        ckpt_path = Path(f"{fn_ckpt}.ckpt")
        
        if ckpt_path.exists():
            logger.info(f'Using existing checkpoint: {ckpt_path.name}')
        else:
            # Split data
            if pk_config['type_know'] != 'Reactome' and pk_config['method'] != 'multi-layer':
                data_train, data_valid, data_test, t_obs, H = data_split(
                    seed=seed, dataset=cohort, data=data, HD=pk_config, construct_H=True
                )
            else:
                data_train, data_valid, data_test, t_obs = data_split(
                    seed=seed, dataset=cohort, data=data, HD=pk_config, construct_H=False
                )
            
            # Build pooling hiddens
            if pk_config.get('divisor'):
                hyper_params['pooling_hiddens'] = build_hiddens(H.shape[1], pk_config['divisor'])
            
            logger.info(f"Pooling_hiddens: {hyper_params['pooling_hiddens']}; Data Shape: {data.shape}")
            
            # Generate graph
            G = generate_G_from_H(H.T) if hyper_params["edge_pooling"] else generate_G_from_H(H)
            logger.info(f"H: {H.shape}; data: {data.shape}; G: {G.shape}; t_obs: {t_obs}")
            
            # Create and train model
            model = HGS(
                hyper_params,
                data_train=data_train,
                data_eval=data_valid,
                data_test=data_test,
                H=H,
                fn_ckpt=fn_ckpt,
                t_obs=t_obs,
                G=G,
                seed=seed
            )
            model = model.cuda()
            
            optimizer = optim.Adam(model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['l2'])
            
            model.fit(
                optimizer=optimizer,
                logger=logger,
                num_epochs=hyper_params["epochs"],
                batch_size=hyper_params["batch_size"],
                loss_dict=hyper_params['loss_w']
            )
        
        # Load checkpoint
        device_map = {"cuda:0": self.device}
        ckpt = torch.load(f'{fn_ckpt}.ckpt', map_location=device_map)
        
        return {
            'val_ci': ckpt['final_eval_ci'],
            'val_loss': ckpt['final_eval_loss'],
            'test_ci': ckpt['final_test_ci']
        }
    
    def _run_optimization(self, cohort: str) -> Dict:
        """Run optimization for Search best Paras"""
        logger = logging.getLogger(f"HGS-{cohort}")
        
        # Initialize hyperparameters
        hyper_params = self._get_base_hyperparams(cohort)
        
        # Get output paths
        fn_results, checkpoint_path = self._get_output_path(cohort, hyper_params)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize results file
        grid_space = GridSearchConfig.get_search_space(self.config)
        with open(fn_results, 'w') as f:
            f.write(f'GridSearchSpace:\n{grid_space}\n')
            f.write(f'Hyperparameters: {hyper_params}\n')
        
        # Bayesian Optimization for Knowledge Discovery
        logger.info(f"Starting {self.opt_type} optimization for {cohort}...")
        
        pk_config = PriorKnowledgeConfig.get_config(self.knowledge_type, self.config)
        sampler = self._create_sampler()
        fn_data = self._get_data_path(cohort)
        
        study, pk_config = KNOW_Search(
            DATA=self.omics_data,
            PK_D=pk_config,
            HD=hyper_params,
            out_path=self.optunadb_path,
            fn_data=str(fn_data),
            logger=logger,
            Sampler=sampler
        )
        
        # Save optimized prior knowledge config
        with open(fn_results, 'a') as f:
            f.write(f"Optimized {self.knowledge_type} Knowledge Parameters:\n{pk_config}\n")
        
        # Load data for grid search
        logger.info(f"Loading data with {hyper_params['cox_num']} features for grid search...")
        data, H = self._load_data(cohort, pk_config, hyper_params)
        
        # Grid Search
        best_metric = -np.inf
        opt_dict = {}
        
        for n_hid in grid_space['HIDS']:
            hyper_params['n_hid'] = n_hid
            hyper_params["MLP_hiddens"] = [n_hid]
            
            for glr in grid_space['GLR']:
                hyper_params['glr'] = glr
                
                for agg in grid_space['AGG']:
                    hyper_params['AGG'] = agg
                    
                    for l2 in grid_space['L2']:
                        hyper_params["l2"] = l2
                        
                        for lr in grid_space['LR']:
                            hyper_params['lr'] = lr
                            
                            for depth in grid_space['PLD']:
                                hyper_params['depth'] = depth
                                
                                for num_hgat in grid_space['NUM_HGAT']:
                                    hyper_params['num_hgat'] = num_hgat
                                    
                                    # Build predict hiddens
                                    n = 1
                                    hyper_params['predict_hiddens'] = build_deep_hiddens(
                                        num_ini=n * hyper_params['n_hid'],
                                        divisor=2,
                                        num_min=hyper_params['num_min'],
                                        depth=hyper_params['depth']
                                    )
                                    
                                    logger.info(
                                        f"Grid Search: L2={hyper_params['l2']}, lr={hyper_params['lr']}, "
                                        f"glr={hyper_params['glr']}, agg={hyper_params['AGG']}, "
                                        f"ph={hyper_params['predict_hiddens']}"
                                    )
                                    
                                    # Write header
                                    with open(fn_results, 'a') as f:
                                        f.write(30 * "-" + '\n')
                                        f.write(f'{hyper_params}\n')
                                        f.write(f'{pk_config}\n')
                                        f.write(f'dataset,seed,L2,lr,glr,agg,pred_hids,pred_depth,pool_hids,val_ci,test_ci,val_loss\n')
                                    
                                    ci_list, loss_list, test_list = [], [], []
                                    
                                    # Run multiple seeds
                                    for seed in range(hyper_params["repeat_time"]):
                                        fn_ckpt = checkpoint_path / self._create_checkpoint_name(
                                            hyper_params, pk_config, seed
                                        )
                                        logger.info(f"Checkpoint: {fn_ckpt.name}")
                                        
                                        results = self._train_or_load_model(
                                            cohort, seed, data, H, hyper_params, pk_config,
                                            str(fn_ckpt), logger
                                        )
                                        
                                        # Write results
                                        with open(fn_results, 'a') as f:
                                            f.write(
                                                f"{cohort},{seed},{hyper_params['l2']},{hyper_params['lr']},"
                                                f"{hyper_params['glr']},{hyper_params['AGG']},"
                                                f"{hyper_params['predict_hiddens']},{hyper_params['depth']},"
                                                f"{hyper_params['pooling_hiddens']},{results['val_ci']},"
                                                f"{results['test_ci']},{results['val_loss']}\n"
                                            )
                                        
                                        ci_list.append(results['val_ci'])
                                        loss_list.append(results['val_loss'])
                                        test_list.append(results['test_ci'])
                                        
                                        logger.info(
                                            f"{cohort} seed={seed}, val_ci={results['val_ci']:.4f}, "
                                            f"val_loss={results['val_loss']:.4f}"
                                        )
                                    
                                    # Compute statistics
                                    ci_mean = np.mean(ci_list)
                                    ci_std = np.std(ci_list)
                                    test_mean = np.mean(test_list)
                                    test_std = np.std(test_list)
                                    loss_mean = np.mean(loss_list)
                                    
                                    val_ms = f'{ci_mean:.4f}±{ci_std:.4f}'
                                    test_ms = f'{test_mean:.4f}±{test_std:.4f}'
                                    
                                    with open(fn_results, 'a') as f:
                                        f.write(f"final valid {hyper_params['score']} mean±std = {val_ms}\n")
                                        f.write(f"final test {hyper_params['score']} mean±std = {test_ms}\n")
                                    
                                    # Update best parameters
                                    if ci_mean > best_metric:
                                        with open(fn_results, 'a') as f:
                                            f.write(f"--UPDATE BEST PARAM--\n")
                                        
                                        best_metric = ci_mean
                                        opt_dict = {
                                            "opt_l2": hyper_params['l2'],
                                            "opt_lr": hyper_params['lr'],
                                            "opt_glr": hyper_params['glr'],
                                            'opt_PredHids': hyper_params['predict_hiddens'],
                                            "opt_agg": hyper_params['AGG'],
                                            "best_loss": loss_mean,
                                            "best_val_mean": ci_mean,
                                            "best_test_mean": test_mean,
                                            "best_val_cims": val_ms,
                                            "best_test_cims": test_ms
                                        }
        
        # Add H shape and event rate
        if H is not None:
            opt_dict['H_shape'] = H.shape
        
        event_rate = (data[:, -1].sum() / data.shape[0] if isinstance(data, np.ndarray)
                     else data.iloc[:, -1].sum() / data.shape[0])
        opt_dict['event_rate'] = event_rate
        opt_dict['pk_config'] = pk_config
        
        return opt_dict

    def run(self):
        """Run optimization for all datasets."""
        logging.info(f"\n{'='*60}\nStarting HGS Optimization\n{'='*60}")
        
        datasets = self.DATASETS[self.omics_data]
        
        results_dict = {
            "Model": "HGS",
            "datasets": datasets,
            "TestCims": [],
            "BestValCims": [],
            "TestMean": [],
            "ValMean": [],
            "H_shape": [],
            "EventRate": [],
            "OptDicList": []
        }
        
        # Run optimization for each cohort
        for cohort in datasets:
            logging.info(f"\nProcessing cohort: {cohort}")
            opt_dict = self._run_optimization(cohort)
            
            results_dict["BestValCims"].append(opt_dict["best_val_cims"])
            results_dict["ValMean"].append(opt_dict["best_val_mean"])
            results_dict["TestCims"].append(opt_dict['best_test_cims'])
            results_dict["TestMean"].append(opt_dict['best_test_mean'])
            results_dict["H_shape"].append(opt_dict.get('H_shape', 'N/A'))
            results_dict["EventRate"].append(opt_dict['event_rate'])
            results_dict["OptDicList"].append(opt_dict)
        
        # Compute overall statistics
        total_mean = np.mean(results_dict['TestMean'])
        total_std = np.std(results_dict['TestMean'])
        results_dict["TotalCiMean"] = f"{total_mean:.4f}±{total_std:.4f}"
        
        logging.info(f"\nHGS Overall Results:")
        logging.info(f"Total C-index Mean±Std: {results_dict['TotalCiMean']}")
        logging.info(f"Results Dictionary: {results_dict}")
        
        # TDOD Save final results to last cohort's results file
        # fn_final = (self.res_path /  / self.omics_data / 
        #                 f"HGS-{self.knowledge_type}" / "Final_Results.csv")
        pd.DataFrame(results_dict).to_csv(self.fn_final, index=False)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='HGS Training with AutoML Optimization and Grid Search'
    )
    parser.add_argument(
        "--omics_data",
        type=str,
        default="PRO",
        choices=["RNA", "PRO"],
        help="Omics data type: RNA or PRO"
    )
    parser.add_argument(
        "--prior_knowledge",
        type=str,
        default="Reactome",
        choices=["Reactome", "STRING", "hcluster"],
        help="Prior knowledge type"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="1",
        help="Device to use (e.g., 1, 0)"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of Optuna trials for knowledge optimization"
    )
    parser.add_argument(
        "--bayesian_optimization",
        type=str,
        default="Auto",
        choices=["Random", "TPE", "GP", "CmaEs", "Auto"],
        help="Bayesian optimization type"
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
    logging.info(f"  Prior Knowledge: {args.prior_knowledge}")
    logging.info(f"  Device: {args.device}")
    logging.info(f"  N Trials: {args.n_trials}")
    logging.info(f"  Optimization Type: {args.bayesian_optimization}")
    
    # Run optimization
    hgs_opt = HGSOptimization(
        omics_data=args.omics_data,
        knowledge_type=args.prior_knowledge,
        device=args.device,
        n_trials=args.n_trials,
        opt_type=args.bayesian_optimization
    )
    hgs_opt.run()
    
    logging.info(f"\nExperiment completed. Log saved to: {exp_logger.log_file}")


if __name__ == "__main__":
    main()
