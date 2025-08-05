import functools
import math
from typing import KeysView
from typing import List
from utils.data_utils import *
import numpy as np
from models.Models_interpret import HGS
from models.Models import HGMLP
import torch.optim as optim
import optuna
from optuna.pruners import PercentilePruner,BasePruner
from optuna.study._study_direction import StudyDirection
from optuna.trial._state import TrialState
from utils.hg_ops import *
from utils.data_utils import load_data_general

class DuplicateIterationPruner(BasePruner):
    """
    DuplicatePruner

    Pruner to detect duplicate trials based on the parameters.

    This pruner is used to identify and prune trials that have the same set of parameters
    as a previously completed trial.
    """

    def prune(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> bool:
        completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

        for completed_trial in completed_trials:
            if completed_trial.params == trial.params:
                return True

        return False

def _get_top_intermediate_result_over_steps(
    trial: "optuna.trial.FrozenTrial", direction: StudyDirection, patience: int
) -> float:
    values = np.asarray(list(trial.intermediate_values.values())[:patience], dtype=float)
    return values

def _get_percentile_intermediate_result_over_trials(
    completed_trials: List["optuna.trial.FrozenTrial"],
    direction: StudyDirection,
    steps: int,
    percentile: float,
    n_min_trials: int,
    
) -> float:
    if len(completed_trials) == 0:
        raise ValueError("No trials have been completed.")

    patiences = [i for i in range(steps)]
    intermediate_values = []
    for step in patiences:
        step_intermediate_values = [
            t.intermediate_values[step] for t in completed_trials if step in t.intermediate_values
        ]
        if len(step_intermediate_values) < n_min_trials:
            return [math.nan] * steps
        intermediate_values.append(step_intermediate_values)


    if direction == StudyDirection.MAXIMIZE:
        percentile = 100 - percentile

    return [float(
        np.nanpercentile(
            np.array(intermediate_value, dtype=float),
            percentile,
        ) 
    ) for intermediate_value in intermediate_values] 


def _is_first_in_interval_step(
    step: int, intermediate_steps: KeysView[int], n_warmup_steps: int, interval_steps: int
) -> bool:
    nearest_lower_pruning_step = (
        step - n_warmup_steps
    ) // interval_steps * interval_steps + n_warmup_steps
    assert nearest_lower_pruning_step >= 0

    # `intermediate_steps` may not be sorted so we must go through all elements.
    second_last_step = functools.reduce(
        lambda second_last_step, s: s if s > second_last_step and s != step else second_last_step,
        intermediate_steps,
        -1,
    )

    return second_last_step < nearest_lower_pruning_step

class SeedPruner(PercentilePruner):
    def __init__(
        self,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        *,
        n_min_trials: int = 1,
        patience: int = 5
    ) -> None:
        super().__init__(
            50.0, n_startup_trials, n_warmup_steps, interval_steps, n_min_trials=n_min_trials
        )
        self.patience = patience
    
    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        n_trials = len(completed_trials)

        if n_trials == 0:
            return False

        if n_trials < self._n_startup_trials:
            return False

        step = trial.last_step
        if step is None:
            return False

        n_warmup_steps = self._n_warmup_steps
        if step < n_warmup_steps:
            return False

        if not _is_first_in_interval_step(
            step, trial.intermediate_values.keys(), n_warmup_steps, self._interval_steps
        ):
            return False

        direction = study.direction
        top_intermediate_result = _get_top_intermediate_result_over_steps(trial, direction, self.patience)
        if len(top_intermediate_result) < self.patience:
            return False
        if any([math.isnan(intermediate_result) for intermediate_result in top_intermediate_result]):
            return True

        ps = _get_percentile_intermediate_result_over_trials(
            completed_trials, direction, self.patience, self._percentile, self._n_min_trials
        )
        if any([math.isnan(p) for p in ps]) :
            return False

        if direction == StudyDirection.MAXIMIZE:
            return all(x <= y for x, y in zip(top_intermediate_result, ps))
        return all(x >= y for x, y in zip(top_intermediate_result, ps))


def load_params_hist(study):
    param_history = []
    for i in range(len(study.get_trials())):
        if str(study.get_trials()[i].state)=="TrialState.COMPLETE":
            param_history.append(study.get_trials()[i].params)
    return param_history

def KNOW_Opt_SC(DATA,PK_D,HD,out_path,fn_data,logger,config):
    '''
    for single chohrt or cacner dataset
    '''
    HD_cp = HD.copy()
    HD_cp['out_path']=out_path
    method = PK_D['method']
    KNOW=PK_D['DataDrive']
    JM=HD['JM']
    study_name=f"{DATA}-SC-{HD['dataset']}-{KNOW}-{method}"
    # 如果没有则自己创建
    os.makedirs(out_path, exist_ok=True)
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{out_path}/{study_name}.db")
    
    logger.info(f"Study Name:{study_name}")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=DuplicateIterationPruner(),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    logger.info(f"Study Trial Nums{len(study.get_trials())}")
    if len(study.get_trials())>=HD['n_trials']:
        logger.info(f"{study_name} Optimization Already Done...")
    else:
        def objective(trial):
            out_path=HD_cp['out_path']
            study_hist = optuna.load_study(
                study_name=study_name,
                storage=storage,
            )
            param_history = load_params_hist(study_hist)

            n=1 if HD_cp['AGG'][0]=='noAGG' else 2 
            HD['predict_hiddens']=build_deep_hiddens(num_ini=n*HD['n_hid'],divisor=2,num_min=100,depth=2)

            if KNOW=='hcluster':
                PK_D['criterion']="maxclust"

            out_path += f"/{study_name}/trail{trial.number}/"
            os.makedirs(out_path,exist_ok=True)

            fn_results = f'{out_path}/results.csv'
            #BO
            if PK_D['DataDrive'] == "STRING" or PK_D['DataDrive'] == "Reactome" :
                PK_D["divisor"] = trial.suggest_int("divisor",2,10)
                (num_buttom,num_top)=(20,104) if  PK_D['DataDrive'] == "STRING" else (2,6)
                layer_KNOW = trial.suggest_int(f"layer_{PK_D['DataDrive']}",num_buttom,num_top)
                PK_D[f"layer_{PK_D['DataDrive']}"]=layer_KNOW
                param_dic = {f"layer_{PK_D['DataDrive']}":layer_KNOW,"divisor":PK_D["divisor"]}
            elif PK_D['DataDrive'] == "hcluster":
                PK_D["divisor"] = trial.suggest_int("divisor",2,10)
                PK_D["num_clusters"] = trial.suggest_int("max_cluster_num0",2,150)
                param_dic = {"max_cluster_num0":PK_D["num_clusters"],"divisor":PK_D["divisor"]}

            if param_dic in param_history:
                raise optuna.exceptions.TrialPruned()

            # 统一数据加载
            data, H = load_data_general(DATA, PK_D, HD, fn_data, config=config)

            fres = open(fn_results, 'w')
            fres.write(f"PK_D trial {trial.number}: {PK_D}\n")
            fres.write(f'seed,eval_ci,eval_loss,test_ci\n')
            eval_list,eval_loss_list,test_list = [],[],[]
            for seed in range(HD["repeat_time"]):
                fn_ckpt = f"{out_path}/seed{seed}.ckpt"
                if os.path.isfile(f'{fn_ckpt}'):
                    logger.info(f'Using existing {fn_ckpt}.ckpt')                
                else: 
                    if PK_D['DataDrive'] =='hcluster':
                        data_train,data_valid,data_test,t_obs,H = data_split(seed=seed,dataset=HD['dataset'],data=data,HD=PK_D,construct_H=True)
                    else:
                        data_train,data_valid,data_test,t_obs= data_split(seed=seed,dataset=HD['dataset'],data=data,HD=PK_D,construct_H=False)
                    G = generate_G_from_H(H.T) if HD["edge_pooling"] else generate_G_from_H(H)

                    HD['pooling_hiddens']=build_hiddens(H.shape[1],divisor=PK_D["divisor"])
                    HD_cp['AGG']=HD_cp['AGG'][0] if isinstance(HD_cp['AGG'], list) else HD_cp['AGG']
                    HD_cp['lr']=HD_cp['lr'][0] if isinstance(HD_cp['lr'], list) else HD_cp['lr']
                    
                    model = HGS(HD_cp,data_train=data_train,data_eval=data_valid,data_test=data_test,H=H,fn_ckpt=fn_ckpt,t_obs=t_obs,G=G,seed=seed,H_list=None)                    
                    model = model.cuda()
                    optimizer = optim.Adam(model.parameters(), lr=HD_cp["lr"],weight_decay=HD_cp["l2"])
                    model = model.fit(optimizer=optimizer,logger=logger,
                    num_epochs=HD_cp['epochs'],batch_size=HD_cp["batch_size"],loss_dict=HD_cp['loss_w'])

                ckpt = torch.load(fn_ckpt)
                eval_ci     = ckpt[f'{JM}_eval_ci'] 
                eval_loss   = ckpt[f'{JM}_eval_loss']
                test_ci    = ckpt[f'test_ci']
                fres = open(fn_results, 'a')
                fres.write(f"{seed},{eval_ci},{eval_loss},{test_ci}\n")
                eval_list.append(eval_ci)
                eval_loss_list.append(eval_loss)
                test_list.append(test_ci)
                logger.info(f"{DATA} seed,val_ci,val_loss:{seed},{eval_ci},{eval_loss}")
            test_mean=np.mean(test_list)
            test_std=np.std(test_list)
            eval_mean=(np.mean(eval_list))
            eval_std=(np.std(eval_list))
            loss_mean = (np.mean(eval_loss_list))
            eval_ms=f'{round(eval_mean,4)}±{round(eval_std,4)}'
            test_ms=f'{round(test_mean,4)}±{round(test_std,4)}'
            fres.write(f"{JM} valid Cindex mean±std = {eval_ms}\n")
            fres.write(f"{JM} test Cindex mean±std = {test_ms}\n")
            fres.write(f"Test cindex mean±std = {test_ms}\n")
            trial.set_user_attr('test_ci',round(float(test_mean),4))
            trial.set_user_attr('test_std', round(float(test_std),4))
            trial.set_user_attr('val_std', round(float(eval_std),4))
            trial.set_user_attr('val_loss', round(float(loss_mean),4))
            trial.set_user_attr('test_ms', test_ms)
            trial.set_user_attr('eval_ms', eval_ms)
            trial.set_user_attr('IncdenceMatrixShape', H.shape)
            return eval_mean
        if len(study.get_trials()) == 0:
            if PK_D['DataDrive']=='hcluster':
                study.enqueue_trial({"max_cluster_num0":100,"divisor":3})
            elif PK_D['DataDrive']=='STRING':
                study.enqueue_trial({"layer_STRING":100,"divisor":5})
        study.optimize(objective, n_trials=HD['n_trials'])
    logger.info(f'Best trail:{study.best_trial}')
    logger.info(f'Best value:{study.best_value}')
    logger.info(f'Best params: {study.best_params}')

    PK_D['divisor'] = study.best_params['divisor']
    if PK_D['DataDrive'] == "hcluster":
        PK_D['num_clusters'] = study.best_params['max_cluster_num0']
    else:
        PK_D[f"layer_{PK_D['DataDrive']}"] = study.best_params[f"layer_{PK_D['DataDrive']}"]

    # fix mistake

    return study,PK_D
