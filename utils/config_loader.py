import yaml
import os
import hashlib
from typing import Dict, Any, List
import logging

class ConfigLoader:
    """Configuration loader for HGS training"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize config loader
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate data consistency
        data_config = self.config['data']
        if (data_config['type'] == 'RNA' and data_config['knowledge_base'] == 'STRING') or \
           (data_config['type'] == 'PRO' and data_config['knowledge_base'] == 'Reactome'):
            raise ValueError(f"Invalid combination: DATA={data_config['type']}, KNOW={data_config['knowledge_base']}")
        
        # Validate knowledge base parameters
        knowledge_base = data_config['knowledge_base']
        if knowledge_base == 'hcluster':
            if 'criterion' not in self.config['knowledge_params']:
                raise ValueError("criterion parameter required for hcluster knowledge base")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config['logging']
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format']
        )
        
        if not log_config['include_warnings']:
            import warnings
            warnings.filterwarnings("ignore")
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration"""
        return self.config['hardware']
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config['model']
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config['data']
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameter search space"""
        return self.config['hyperparameters']
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss function weights"""
        return self.config['loss_weights']
    
    def get_knowledge_params(self) -> Dict[str, Any]:
        """Get knowledge base parameters with variable substitution"""
        knowledge_params = self.config['knowledge_params'].copy()
        data_config = self.config['data']
        
        # Set data_drive and method based on knowledge_base
        knowledge_base = data_config['knowledge_base']
        knowledge_params['DataDrive'] = knowledge_base
        knowledge_params['method'] = data_config['fcluster_method'] if knowledge_base == 'hcluster' else 'layer'
        
        # Perform variable substitution for string values
        for key, value in knowledge_params.items():
            if isinstance(value, str):
                # Replace {variable} with actual values from knowledge_params
                for var_key, var_value in knowledge_params.items():
                    if isinstance(var_value, str) and var_key != key:
                        value = value.replace(f"{{{var_key}}}", str(var_value))
                knowledge_params[key] = value
        
        return knowledge_params
    
    def get_paths(self) -> Dict[str, str]:
        """Get path configurations with variable substitution"""
        paths = self.config['paths'].copy()
        
        # Perform variable substitution (max 10 iterations to avoid infinite loops)
        for _ in range(10):
            changed = False
            for key, value in paths.items():
                if isinstance(value, str):
                    original_value = value
                    for var_key, var_value in paths.items():
                        if isinstance(var_value, str) and var_key != key:
                            value = value.replace(f"{{{var_key}}}", var_value)
                    if value != original_value:
                        paths[key] = value
                        changed = True
            if not changed:
                break
        
        return paths
    
    def get_naming_templates(self) -> Dict[str, str]:
        """Get file naming templates"""
        return self.config['naming']
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation settings"""
        return self.config['validation']
    
    def get_advanced_config(self) -> Dict[str, Any]:
        """Get advanced settings"""
        return self.config['advanced']
    
    def build_HD_dict(self, **overrides) -> Dict[str, Any]:
        """
        Build HD dictionary for model initialization
        
        Args:
            **overrides: Parameters to override from config
            
        Returns:
            HD dictionary for model
        """
        hyperparams = self.get_hyperparameters()
        loss_weights = self.get_loss_weights()
        model_config = self.get_model_config()
        
        HD = {
            # Fixed hyperparameters
            "epochs": hyperparams['epochs'],
            "print_freq": hyperparams['print_freq'],
            "batch_size": hyperparams['batch_size'],
            "n_hid": hyperparams['n_hidden'],
            "dropout": hyperparams['dropout'],
            "ACT": hyperparams['activation'],
            "MLP_hiddens": hyperparams['mlp_hiddens'],
            "predict_hiddens": hyperparams['predict_hiddens'],
            "pooling_hiddens": hyperparams['pooling_hiddens'],
            "RESNET": hyperparams['resnet'],
            "repeat_time": hyperparams['repeat_time'],
            "tfidf": hyperparams['tfidf'],
            "edge_pooling": hyperparams['edge_pooling'],
            "pooling_method": hyperparams['pooling_method'],
            "depth": hyperparams['depth'],
            "AGG": hyperparams['aggregation_methods'],
            "lr": hyperparams['learning_rates'],
            "glr": hyperparams['graph_loss_rates'],
            "l2": hyperparams['weight_decays'],
            "PLD": hyperparams['prediction_layer_depths'],
            # Loss weights
            "loss_w": loss_weights,
            
            # Model configuration
            "BayeOpt": model_config['bayesian_optimization'],
            "JM": model_config['judge_metric'],
            "n_trials": model_config['n_trials'],
            
            # Data configuration
            "end_point": self.config['data']['end_point'],
            "cox_num": self.config['data']['cox_num'],
            
            # Advanced settings
            "HG_BN": self.config['advanced']['hg_bn'],
            
        }
        
        # Apply overrides
        HD.update(overrides)
        
        return HD
    
    def build_PK_D_dict(self) -> Dict[str, Any]:
        """Build PK_D dictionary for knowledge base"""
        return self.get_knowledge_params()
    
    def get_data_path(self, dataset: str) -> str:
        """Get data file path for given dataset"""
        paths = self.get_paths()
        data_type = self.config['data']['type']
        
        if data_type == 'PRO':
            return paths['proteomic_data_template'].format(dataset=dataset)
        else:  # RNA
            return paths['rna_data_template'].format(dataset=dataset)
    
    def get_output_path(self, dataset: str) -> str:
        """Get output path for given dataset"""
        paths = self.get_paths()
        data_type = self.config['data']['type']
        knowledge_base = self.config['data']['knowledge_base']
        
        base_path = paths['output_template'].format(
            data_type=data_type,
            dataset=dataset,
            knowledge_base=knowledge_base
        )
        
        # Add edge pooling suffix if enabled
        if self.config['hyperparameters']['edge_pooling']:
            base_path += paths['edge_pooling_suffix']
        
        base_path += paths['model_suffix']
        
        return base_path
    
    def get_checkpoint_path(self, dataset: str, seed: int, params: Dict[str, Any]) -> str:
        """Get checkpoint file path"""
        naming = self.get_naming_templates()
        paths = self.get_paths()
        
        # Create a hash of knowledge parameters to avoid long filenames
        knowledge_params = self.get_knowledge_params()
        params_str = str(sorted(knowledge_params.items()))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        # Build checkpoint filename
        checkpoint_name = naming['checkpoint_template'].format(
            epochs=self.config['hyperparameters']['epochs'],
            dataset=dataset,
            seed=seed,
            l2=params.get('l2', 'default'),
            lr=params.get('lr', 'default'),
            glr=params.get('glr', 'default'),
            agg=params.get('AGG', 'default'),
            depth=params.get('depth', 'default'),
            pk_d=params_hash
        )
        
        grid_search_path = self.get_output_path(dataset) + paths['grid_search_suffix']
        return os.path.join(grid_search_path, checkpoint_name + naming['checkpoint_extension'])
    
    def get_results_file_path(self, dataset: str) -> str:
        """Get results file path"""
        naming = self.get_naming_templates()
        output_path = self.get_output_path(dataset)
        
        results_filename = naming['results_file_template'].format(
            method=self.get_knowledge_params()['method']
        )
        
        return os.path.join(output_path, results_filename)
    
    def create_directories(self, dataset: str):
        """Create necessary directories for output"""
        output_path = self.get_output_path(dataset)
        paths = self.get_paths()
        
        # Create main output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Create results files directory
        results_files_path = output_path + paths['results_files_suffix']
        os.makedirs(results_files_path, exist_ok=True)
        
        # Create grid search directory
        grid_search_path = output_path + paths['grid_search_suffix']
        os.makedirs(grid_search_path, exist_ok=True)
    
    def get_grid_search_params(self) -> List[Dict[str, Any]]:
        """Get all combinations of grid search parameters"""
        hyperparams = self.get_hyperparameters()
        
        import itertools
        
        # Grid search parameters
        param_names = ['AGG', 'lr', 'depth']
        param_values = [
            hyperparams['aggregation_methods'],
            hyperparams['learning_rates'],
            hyperparams['prediction_layer_depths'],
        ]
        
        combinations = []
        for values in itertools.product(*param_values):
            param_dict = dict(zip(param_names, values))
            combinations.append(param_dict)
        
        return combinations 