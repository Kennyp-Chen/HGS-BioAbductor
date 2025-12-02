import functools
import math
from typing import KeysView
from typing import List
from utils.data_utils import *
import numpy as np
# from models.Models_interpret import HGS
# from models.HGS import HGS
from models.Models import HGS
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


def KNOW_Search(DATA,PK_D,HD,out_path,fn_data,logger,Sampler = optuna.samplers.RandomSampler(seed=0)):
    '''
    for single chohrt or cacner dataset
    '''
    HD['out_path']=str(out_path)
    method = PK_D['method']
    KNOW=PK_D['type_know']
    FEAT=HD['feature_base']
    study_name=f"{DATA}-{KNOW}-{HD['dataset']}-{method}"


    if KNOW == 'hcluster':
        storage=f"sqlite:///{out_path}/{study_name}-divminmax-{'_'.join(map(str,PK_D['div_know_minmax']))}.db"
        study_name+=f"-dkminmax-{PK_D['div_know_minmax']}"
    else:
        storage=f"sqlite:///{out_path}/{study_name}.db"

    study = optuna.create_study(
        direction='maximize',
        sampler=Sampler,
        pruner=DuplicateIterationPruner(), 
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    logger.info(f"Using {HD['OptType']} Sampler!\nStudy Trial Nums{len(study.get_trials())}")
    if len(study.get_trials())>=HD['n_trials']:
        logger.info(f"{study_name} Optimization Already Done...")

    else:
        # define objective function
        def objective(trial):
            out_path=HD['out_path']
            study_hist = optuna.load_study(
                study_name=study_name,
                storage=storage,
            )
            param_history = load_params_hist(study_hist)
            
            n=1
            HD['predict_hiddens']=build_deep_hiddens(num_ini=n*HD['n_hid'],divisor=2,num_min=HD['num_min'],depth=2)

            if KNOW=='hcluster':
                PK_D['criterion']="maxclust"

            out_path += f"/{study_name}/trail{trial.number}/"
            os.makedirs(out_path,exist_ok=True)

            fn_results = f'{out_path}/results.csv'
            
            # 知识优化空间
            if PK_D['type_know'] == "STRING" or PK_D['type_know'] == "Reactome" :
                dmin,dmax,kmin,kmax = PK_D["div_know_minmax"]# divisor min max and knowledge para min max

                if PK_D['method']=='layer_all':
                    PK_D["divisor"] = trial.suggest_int("divisor",2,20)
                    (num_buttom,num_top,step)=(20,104,1) if  PK_D['type_know'] == "STRING" else (1,12,1)
                elif PK_D['method']=='layer':
                    PK_D["divisor"] = trial.suggest_int("divisor",2,10 if PK_D['type_know'] == "STRING" else 15)
                    (num_buttom,num_top,step)=(19,104,5) if  PK_D['type_know'] == "STRING" else (1,12,1)
                elif PK_D['method']=='layer_range':
                    dmin,dmax,kmin,kmax = PK_D["div_know_minmax"]
                    PK_D["divisor"] = trial.suggest_int("divisor",dmin,dmax)
                    (num_buttom,num_top,step)=(kmin,kmax,2) if  PK_D['type_know'] == "STRING" else (kmin,kmax,1)

                logger.info(f"{KNOW} layer space (num_buttom,num_top,step):{(num_buttom,num_top,step)}")
                layer_KNOW = trial.suggest_int(f"layer_{PK_D['type_know']}",num_buttom,num_top,step=step)
                PK_D[f"layer_{PK_D['type_know']}"]=layer_KNOW
                param_dic = {f"layer_{PK_D['type_know']}":layer_KNOW,"divisor":PK_D["divisor"]}
            elif PK_D['type_know'] == "hcluster" or PK_D['type_know'] == "STRING_hcluster":
                # 由于聚类的超边数量一般较少，divisor大小可以减小
                dmin,dmax,kmin,kmax = PK_D["div_know_minmax"]
                if dmin>dmax:
                    HD['pooling_hiddens']=[50,10]
                    PK_D["divisor"] = False
                else:
                    # PK_D["divisor"] = trial.suggest_int("divisor",2,10)
                    PK_D["divisor"] = trial.suggest_int("divisor",dmin,dmax,step=10)

                # PK_D["num_clusters"] = trial.suggest_int("num_clusters",2,150)# 调整较小的聚类数可以让超边度更高，同时ward是最均匀的聚类方法
                PK_D["num_clusters"] = trial.suggest_int("num_clusters",kmin,kmax)# 调整较小的聚类数可以让超边度更高，同时ward是最均匀的聚类方法
                param_dic = {"num_clusters":PK_D["num_clusters"],"divisor":PK_D["divisor"]}
            elif PK_D['type_know'] == "BOTH":
                PK_D["divisor"] = trial.suggest_int("divisor",2,20)
                PK_D["layer_Reactome"] = trial.suggest_int("layer_Reactome",1,12)
                PK_D["layer_STRING"] = trial.suggest_int("layer_STRING",19,104,5)
                param_dic = {
                    "layer_STRING":PK_D["layer_STRING"],
                    "layer_Reactome":PK_D["layer_Reactome"],
                    "divisor":PK_D["divisor"]
                    }

            if param_dic in param_history:
                raise optuna.exceptions.TrialPruned()
            logger.info(f"PK_D trial {trial.number}: {PK_D}")
            logger.info(f"{HD['cox_num']} features are selected for Train-Valid-Test...")

            if DATA=='PRO':
                if KNOW=='Reactome':
                    fn_H = f"./data/PriorKnow/Reactome/reactome_P{PK_D['layer_Reactome']}/H1.csv" 
                    data,H = load_cohort_data(fn_H,fn_data,HD['cox_num'],DF=False) 
                else: # STRING or hcluster
                    data,H = load_cohort_data(None,fn_data,HD['cox_num'],DF=True) 

            elif DATA=='RNA':# RNA
                if PK_D["type_know"]=='hcluster':
                    if FEAT=='ReactomeP12':
                        data,H= load_cohort_data(None,fn_data,HD['cox_num'],DF=True) 
                    elif FEAT=='Reactome':
                        data = load_TCGA_data(fn_data,HD["end_point"],Filt_num=HD["cox_num"])
                    elif FEAT=='ReactomeTop':
                        data,H = load_opt_data(data_path=fn_data,H_path=H_path,endpoint=HD["end_point"],Filt_num=HD['cox_num'],DF=False) 

                elif PK_D["type_know"]=='Reactome':
                    H_path = f"./data/PriorKnow/Reactome/reactome_P{PK_D['layer_Reactome']}/" 
                    # data,H= load_opt_data(data_path=fn_data,H_path=H_path,endpoint=HD["end_point"],Filt_num=HD['cox_num'],DF=False) 
                    if FEAT=='ReactomeP12':
                        fn_H = H_path+f"H1.csv"
                        data,H= load_cohort_data(fn_H,fn_data,HD['cox_num'],DF=False) 
                    elif FEAT=='Reactome':
                        data,H = load_TCGA_H_data(H_path,fn_data,Filt_num=HD['cox_num'])
                    elif FEAT=='ReactomeTop':
                        data,H = load_opt_data(data_path=fn_data,H_path=H_path,endpoint=HD["end_point"],Filt_num=HD['cox_num'],DF=False) 
                elif PK_D["type_know"]=='STRING':
                    data = load_TCGA_data(fn_data,HD["end_point"],Filt_num=-1)
                    data = Know_features_filt(data)
            fres = open(fn_results, 'w')
            fres.write(f"PK_D trial {trial.number}: {PK_D}\n")
            fres.write(f"{HD}\n")

            fres.write(f'seed,eval_ci,eval_loss,test_ci\n')
            eval_list,eval_loss_list,test_list = [],[],[]
            for seed in range(HD["repeat_time"]):
                fn_ckpt = f"{out_path}/seed{seed}"
                if os.path.isfile(f'{fn_ckpt}.ckpt'):
                    logger.info(f'Using existing {fn_ckpt}.ckpt')                
                else: 
                    if PK_D['type_know']!='Reactome':
                        data_train,data_valid,data_test,t_obs,H = data_split(seed=seed,dataset=HD['dataset'],data=data,HD=PK_D,construct_H=True)
                    else:
                        data_train,data_valid,data_test,t_obs= data_split(seed=seed,dataset=HD['dataset'],data=data,HD=PK_D,construct_H=False)
                    G = generate_G_from_H(H.T) if HD["edge_pooling"] else generate_G_from_H(H)
                    if PK_D['divisor']:
                        HD['pooling_hiddens']=build_hiddens(H.shape[1],divisor=PK_D["divisor"])
                    logger.info(f"Pooling_hiddens:{HD['pooling_hiddens']}; Data Shape:{data.shape}\n")
                    model = HGS(HD,data_train=data_train,data_eval=data_valid,data_test=data_test,H=H,fn_ckpt=fn_ckpt,t_obs=t_obs,G=G,seed=seed,H_list=None,node_idx=[])
                    model = model.cuda()
                    optimizer = optim.Adam(model.parameters(), lr=HD["lr"],weight_decay=HD["l2"])
                    model = model.fit(optimizer=optimizer,logger=logger,
                    num_epochs=HD['epochs'],batch_size=HD["batch_size"],loss_dict=HD['loss_w'])

                ckpt = torch.load(f'{fn_ckpt}.ckpt',map_location={"cuda:0":"cuda:1"})
                eval_ci     = ckpt[f'final_eval_ci'] 
                eval_loss   = ckpt[f'final_eval_loss']
                test_ci	   = ckpt[f'final_test_ci']
                # record results
                fres = open(fn_results, 'a')
                fres.write(f"{seed},{eval_ci},{eval_loss},{test_ci}\n")
                
                # compute mean std 
                eval_list.append(eval_ci)
                eval_loss_list.append(eval_loss)
                test_list.append(test_ci)
                logger.info(f"{DATA} seed,val_ci,val_loss:{seed},{eval_ci},{eval_loss}")
            # Result Diction Write
            test_mean=np.mean(test_list)
            test_std=np.std(test_list)
            eval_mean=(np.mean(eval_list))
            eval_std=(np.std(eval_list))
            loss_mean = (np.mean(eval_loss_list))
            eval_ms=f'{round(eval_mean,4)}±{round(eval_std,4)}'
            test_ms=f'{round(test_mean,4)}±{round(test_std,4)}'
            fres.write(f"final valid Cindex mean±std = {eval_ms}\n")
            fres.write(f"final test Cindex mean±std = {test_ms}\n")
            fres.write(f"Test cindex mean±std = {test_ms}\n")
            
            trial.set_user_attr('test_ci',round(float(test_mean),4))
            trial.set_user_attr('test_std', round(float(test_std),4))
            trial.set_user_attr('val_std', round(float(eval_std),4))
            trial.set_user_attr('val_loss', round(float(loss_mean),4))
            trial.set_user_attr('test_ms', test_ms)
            trial.set_user_attr('eval_ms', eval_ms)
            trial.set_user_attr('IncdenceMatrixShape', H.shape)
            
            return eval_mean
        study.optimize(objective, n_trials=HD['n_trials'])

        logger.info(f'Best trail:{study.best_trial}')
        logger.info(f'Best value:{study.best_value}')
        logger.info(f'Best params: {study.best_params}')
    
    trials = study.get_trials()[:HD['n_trials']]
    # 尝试识别已有的最优秀，并且存储，都是fail则跳过
    try:
        best_trial = max(filter(lambda t: t.value is not None, trials), key=lambda t: t.value)
        PK_D.update(best_trial.params)
        if 'max_cluster_num0' in best_trial.params.keys():
            PK_D['num_clusters'] = best_trial.params['max_cluster_num0']
            PK_D.pop('max_cluster_num0')
    except:
        pass

    logger.info(f"Best trial params: {PK_D}")

    return study,PK_D