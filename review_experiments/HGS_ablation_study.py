import torch
import numpy as np
import os,torch
import numpy as np
import torch.optim as optim
from utils.data_utils import *
import matplotlib.pyplot as plt
from utils.optuna_utils import *
from utils.hg_ops import *
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#####
# 1. 超图层 vs GCN层
# 2. pooling消融
# 3. 损失函数消融实验
#####

def Train_Valid_Test(dataset,HD,PK_D,datar,path_GS,fn_results,OptDic,logger):
	HD['predict_hiddens']=build_deep_hiddens(num_ini=HD['n_hid'],divisor=2,num_min=HD['num_min'],depth=HD['depth'])
	logger.info(f"Grid Search Paras:L2{HD['l2']};lr{HD['lr']};glr{HD['glr']};agg{HD['AGG']};ph{HD['predict_hiddens']}")				
	fres = open(fn_results, 'a')
	fres.write(30*"-"+'\n')
	fres.write(f'{HD}\n')
	fres.write(f'{PK_D}\n')
	if HD['train_scheme']=='final_epoch':
		fres.write(f'dataset,seed,L2,lr,glr,agg,pred_hids,pred_depth,pool_hids,{JM}_val_ci,{JM}_test_ci,{JM}_val_loss,test_harmonic_score,test_all_scores\n')
	elif HD['train_scheme']=='early_stop':
		fres.write(f"dataset,seed,L2,lr,glr,agg,pred_hids,pred_depth,pool_hids,{JM}_val_{HD['score']},{JM}_test_{HD['score']},{JM}_val_{HD['score']},test_all_scores\n")
	ci_list,loss_list,test_list,hs_list = [],[],[],[]
	
	for seed in range(HD["repeat_time"]):
		fn_ckpt = f"{path_GS}/"
		fn_ckpt+= str(list(HD.items())[:HD['n_hp_ckpt']]).replace("), (","-").replace(",",":").replace("'","").replace(" ","")[2:-2]
		fn_ckpt+="-"+str(list(PK_D.items())).replace("), (","-").replace(",",":").replace("'","").replace(" ","")[2:-2]
		fn_ckpt+=f"-seed:{seed}"
		logger.info(f"ckpt:{fn_ckpt}")

		if os.path.isfile(f'{fn_ckpt}.ckpt'):
			logger.info(f'Using existing {fn_ckpt}.ckpt')  
		else: 
			# logger.info("Warning: ignore existing ckpt file and training whatever!!!!")# TODO

			if PK_D['type_know']!='Reactome':
				data_train,data_valid,data_test,t_obs,H = data_split(seed=seed,dataset=dataset,data=data,HD=PK_D,construct_H=True)
			else:
				data_train,data_valid,data_test,t_obs = data_split(seed=seed,dataset=dataset,data=data,HD=PK_D,construct_H=False)
			if PK_D['divisor']:
				HD['pooling_hiddens']=build_hiddens(H.shape[1],PK_D['divisor'])
			logger.info(f"Pooling_hiddens:{HD['pooling_hiddens']}; Data Shape:{data.shape}")

			G = generate_G_from_H(H.T) if HD["edge_pooling"] else generate_G_from_H(H)
			logger.info(f"H:{H.shape}; data:{data.shape}; G:{G.shape}; t_obs:{t_obs}")
			model = HGS(HD,data_train=data_train,data_eval=data_valid,data_test = data_test,H=H,fn_ckpt=fn_ckpt,t_obs=t_obs,G=G,seed=seed)

			model = model.cuda()
			optimizer = optim.Adam(model.parameters(), lr=HD['lr'],weight_decay=HD['l2'])
			if HD['train_scheme']=='final_epoch':
				model = model.fit(optimizer=optimizer,logger=logger,
				num_epochs=HD["epochs"],batch_size=HD["batch_size"],loss_dict=HD['loss_w'])
			elif HD['train_scheme']=='early_stop':
				scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
													factor=HD['gamma'],
													patience=HD['patience'],)
				model = model.fit_early_stop(optimizer=optimizer,scheduler=scheduler,
				logger=logger,num_epochs=HD["epochs"],batch_size=HD["batch_size"],
				loss_dict=HD['loss_w'],metric=HD['metric_update'],score=HD['score'],update_freq=HD['update_freq'])
		ckpt = torch.load(f'{fn_ckpt}.ckpt',map_location={"cuda:0":"cuda:1"})
		if HD['train_scheme']=='final_epoch':
			val_ci     = ckpt[f'{JM}_eval_ci'] 
			val_loss   = ckpt[f'{JM}_eval_loss']
			test_ci	   = ckpt[f'{JM}_test_ci']
			harmonic_score = ckpt[f'test_harmonic_score']
			test_all_scores = ckpt[f'test_all_scores']
			hs_list.append(harmonic_score)
			fres.write(f"{dataset},{seed},{HD['l2']},{HD['lr']},{HD['glr']},{HD['AGG']},{HD['predict_hiddens']},{HD['depth']},{HD['pooling_hiddens']},{val_ci},{test_ci},{val_loss},{harmonic_score},{test_all_scores}\n")
		elif HD['train_scheme']=='early_stop':
			val_ci     = ckpt[f'best_eval_score'] 
			val_loss   = ckpt[f'best_eval_loss']
			test_ci	   = ckpt[f'test_score']
			test_all_scores = ckpt[f'test_all_scores']
			fres.write(f"{dataset},{seed},{HD['l2']},{HD['lr']},{HD['glr']},{HD['AGG']},{HD['predict_hiddens']},{HD['depth']},{HD['pooling_hiddens']},{val_ci},{test_ci},{val_loss},{test_all_scores}\n")

		# compute mean std 
		ci_list.append(val_ci)
		loss_list.append(val_loss)
		test_list.append(test_ci)

		logger.info(f"{dataset} seed,val_ci,val_loss:{seed},{val_ci},{val_loss}")
	# Result Diction Write
	ci_mean=(np.mean(ci_list))
	ci_std=(np.std(ci_list))
	test_mean=np.mean(test_list)
	test_std=np.std(test_list)
	loss_mean = (np.mean(loss_list))
	val_ms=f'{round(ci_mean,4)}±{round(ci_std,4)}'
	test_ms=f'{round(test_mean,4)}±{round(test_std,4)}'

	now_metric = ci_mean 
	try:
		hs_ms = f'{round(np.mean(hs_list),4)}±{round(np.std(hs_list),4)}'
		fres.write(f"{JM} test Harmonic Score mean±std = {hs_ms}\n")
	except:
		pass
	fres.write(f"{JM} valid {HD['score']} mean±std = {val_ms}\n")
	fres.write(f"{JM} test {HD['score']} mean±std = {test_ms}\n")
	# Update

	if now_metric > best_metric:
		fres.write(f"--UPDATE BEST PARAM--\n")
		best_metric = now_metric
		OptDic["opt_l2"] = HD['l2']
		OptDic["opt_lr"] = HD['lr']
		OptDic["opt_glr"] = HD['glr']
		OptDic['opt_PredHids']=HD['predict_hiddens']
		OptDic["opt_agg"] = HD['AGG']
		OptDic["best_loss"] = round(loss_mean,4)

		best_val_mean = ci_mean
		best_test_mean = test_mean
		best_val_cims = val_ms#str
		best_test_cims = test_ms#str
	fres.close()	
	return best_val_mean,best_test_mean,best_val_cims,best_test_cims,HD

def ablation_study(data,HD,PK_D,dataset,path_weights,fn_results,logger):
	logger.info(f"{part_ablation}:{HD[part_ablation]}")
	fres = open(fn_results, 'a')
	fres.write(f"{part_ablation}:{HD[part_ablation]}\n")
	fres.write(f'part_ablation,cancer,seed,val_ci,val_loss,test_ci\n')
	test_seed_list = []
	for seed in range(10):
		if isinstance(HD[part_ablation],dict):
			fn_ckpt = f"{path_weights}/{part_ablation}-{list(HD[part_ablation].values())}_{dataset}_seed-{seed}_l2-{HD['l2']}_lr-{HD['lr']}_glr{HD['glr']}_agg{HD['AGG']}_PH{HD['predict_hiddens']}_loss_{HD['loss_w']}_{PK_D}"
		else:
			fn_ckpt = f"{path_weights}/{part_ablation}-{HD[part_ablation]}_{dataset}_seed-{seed}_l2-{HD['l2']}_lr-{HD['lr']}_glr{HD['glr']}_agg{HD['AGG']}_PH{HD['predict_hiddens']}_loss_{HD['loss_w']}_{PK_D}"
		if os.path.isfile(f'{fn_ckpt}.ckpt'):
			logger.info(f'Using existing {fn_ckpt}.ckpt')

		else: 
			if PK_D['type_know"']!='Reactome':
				data_train,data_valid,data_test,t_obs,H_df = data_split(seed=seed,dataset=dataset,data=data,HD=PK_D,construct_H=True,H_df=True)
			else:
				data_train,data_valid,data_test,t_obs = data_split(seed=seed,dataset=dataset,data=data,HD=PK_D,construct_H=False)
			H = H_df.values
			G = generate_G_from_H(H.T) if HD["edge_pooling"] else generate_G_from_H(H)
			num_ini = H.shape[0] if (not HD['edge_pooling'] or HD['graph_layer']=='GCN') else H.shape[1]
			if HD['disturb_H']>0:
				H = disturb_H(H,HD['disturb_H'],seed=seed)

			HD['pooling_hiddens']=build_hiddens(num_ini,PK_D['divisor'])
			logger.info(f"Pooling_hiddens:{HD['pooling_hiddens']}\n")
			logger.info(f"H:{H.shape}; data:{data.shape}; G:{G.shape}; t_obs:{t_obs}")
			model = HGS_ablation(HD,data_train=data_train,data_eval=data_valid,data_test = data_test,H=H,fn_ckpt=fn_ckpt,t_obs=t_obs,G=G,seed=seed)
			model = model.cuda()
			optimizer = optim.Adam(model.parameters(), lr=HD['lr'],weight_decay=HD['l2'])
			model = model.fit(optimizer=optimizer,logger=logger,
			num_epochs=HD["epochs"],batch_size=HD["batch_size"],loss_dict=HD['loss_w'])
		ckpt = torch.load(f'{fn_ckpt}.ckpt',map_location={"cuda:0":"cuda:1"})
		val_ci     = ckpt['best_eval_ci']
		val_loss   = ckpt["best_eval_loss"]
		test_ci	   = ckpt["test_ci"]
		test_seed_list.append(test_ci)
		fres.write(f"{HD[part_ablation]},{dataset},{seed},{val_ci},{val_loss},{test_ci}\n")
	
	test_mean=np.mean(test_seed_list)
	test_std=np.std(test_seed_list)
	test_ms=f'{round(test_mean,4)}±{round(test_std,4)}'
	fres.write(f"Test Cindex mean±std = {test_ms}\n")
	fres.write(f"Above Models' Hyperparameters:{HD}\n{PK_D}\n")
	test_abl_list.append(test_mean)
	tms_abl_list.append(test_ms)
	fres.close()

	# fres.write(f"Valid Cindex mean±std = {val_ms}\n")

# for part_ablation in ['pooling_method','graph_layer','disturb_H','loss_w']:#
part_ablation='loss_w'
# DATA="RNA"
DATA="PRO"
torch.cuda.set_device('cuda:1')
n_trials=20
BayeOpt=True# 获取最佳知识超参数后直接消融，模型相关的超参可以不用考虑直接设置相同尝试
Model = "HGS"
KNOW="hcluster"
fcluster_method='weighted'
cox_num=400
if DATA=='RNA':
    RNA_DATA_TYPE='TCGA'
    if KNOW=='Reactome':
        FEAT='ReactomeTop'
    elif KNOW=='hcluster':
        FEAT='Reactome'
    elif KNOW=='STRING':
        FEAT='Reactome'
PK_D={
    'type_know':KNOW,
    'method':fcluster_method if (KNOW=='hcluster' or KNOW=='STRING_hcluster') else 'layer',
    "div_know_minmax":(5,10,40,150),
    # 'method':fcluster_method if KNOW=='hcluster' else 'layer_all',# layer_all means using all string layer searching
}


if KNOW=='hcluster':
    PK_D['criterion']='maxclust'

if DATA=='RNA':
	cancers = ['LIHC','STAD','BLCA','OV','LUSC','LGG','LUAD','KIRC','HNSC','BRCA']
	batch_size=32
if DATA=='PRO':
	cancers = ['CCRCC','HaNSCC','LA','LSCC','UCEC','HCC','GBM','PDA']
	batch_size=16

for dataset in cancers:
	part_ablation= 'graph_layer'
	if part_ablation=='pooling_method':
		ablation_list=['linear','max','mean','sum']
	elif part_ablation=='loss_w':
		ablation_list=[
		{'a': 1, 'b': 1, 'c1': 1, 'c2': 0, 'd': 1},
		{'a': 1, 'b': 1, 'c1': 1, 'c2': 0, 'd': 0},
		{'a': 1, 'b': 1, 'c1': 0, 'c2': 0, 'd': 1},
		{'a': 1, 'b': 1, 'c1': 0, 'c2': 0, 'd': 0}]
	# elif part_ablation=='glr':
	# 	ablation_list=[None,0]
	elif part_ablation=='graph_layer':
		ablation_list=["HG","GCN"]
	elif part_ablation=='disturb_H':
		ablation_list=[0,0.3,0.6,0.9]


	FEAT='STRING' if DATA=='PRO' else 'Reactome'
	JM="final"
	if (DATA=='RNA' and KNOW=='STRING') or (DATA=='PRO' and KNOW=='Reactome'):
		raise ValueError
	RD={"Model":Model,
		"TestCims":[],"BestValCims":[],"TestMean":[],"ValMean":[],"H_shape":[],"EventRate":[],"OptDicList":[]}
	HD = {
		"l2"  			: 0.1,
		"lr"            : 0.01, 
		"glr"           : 0.5,
		"AGG"           : "cat",
		"predict_hiddens": [200,100],
		'depth'			: 2,
		"n_hid"         : 200, #100, # 200
		"epochs"     	: 50,
		"print_freq"    : 10,
		"batch_size"    : batch_size, 
		"cox_num"       : cox_num, # 1000
		"dropout"       : 0.5,
		"ACT"           : "LeakyReLU",
		"MLP_hiddens"   :  [200],
		"pooling_hiddens":[],
		"RESNET"        : False,
		"repeat_time"   : 10,
		"end_point"     : "OS time",
		"tfidf"			: False,
		"edge_pooling"  : True,
		"loss_w"		:{'a': 1, 'b': 1, 'c1': 1, 'c2': 0, 'd': 1},
		"pooling_method":"linear",
		'BayeOpt'		: BayeOpt,
		'JM'			: 'ci',
		'n_trials'		: 20,
		'feature_base'	: FEAT,
		'graph_layer'	: 'HG',
		'num_hgat'		: 'single',
		'disturb_H'		: 0,
	}
	out_path = f"/Backup/home/chenyupeng/Results/Abalation/{DATA}/{dataset}/{KNOW}/cox{HD['cox_num']}/"
	os.makedirs(out_path,exist_ok=True)
	logger = logging.getLogger(f"{Model} - {dataset}")
	fn_results = f"{out_path}//Results-nt{n_trials}-bs{HD['batch_size']}-{PK_D}.csv"
	if DATA=='PRO':
		fn_data = f"/Backup/home/chenyupeng/DATA/{DATA}/0.5nafilter_CoxSort/{FEAT}/{dataset}/dataset.csv"
	elif DATA=='RNA':# RNA
		if FEAT=='Reactome':
			fn_data = f"/Backup/home/chenyupeng/DATA/TCGA-{DATA}seq/COX_Selection/{dataset}/feature_matrix.csv"
		elif FEAT=='ReactomeP12':
			fn_data = f"/Backup/home/chenyupeng/DATA/TCGA-{DATA}seq/COX_Selection/{dataset}/dataset_Reactome12_1000genes.csv"
		elif FEAT == 'ReactomeTop':
			fn_data = f"/Backup/home/chenyupeng/DATA/TCGA-{DATA}seq/COX_Selection/{dataset}/"


	# Grid Search
	if HD['BayeOpt']:# Single Cohort\cancer BayeOpt
		HD['dataset']=dataset
		if HD['OptType']=='TPE':
			Sampler = optuna.samplers.TPESampler(seed=0)
		elif HD['OptType']=='GP':
			Sampler = optuna.samplers.GPSampler(seed=0)
		elif HD['OptType']=='Random':
			Sampler = optuna.samplers.RandomSampler(seed=0)
		elif HD['OptType']=='CmaEs':
			Sampler = optuna.samplers.CmaEsSampler(seed=0)
		elif HD['OptType']=='Auto':
			Sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler(seed = 0)
		logger.info(f"Start {HD['OptType']} searching for {dataset} dataset...")
		study,PK_D=KNOW_Search(DATA=DATA,PK_D=PK_D,HD=HD,out_path=out_path,fn_data=fn_data,logger=logger,Sampler=Sampler)
		fres = open(fn_results, 'a')
		fres.write(f"After {HD['OptType']} searching, {dataset} dataset BEST {KNOW} KNOWLEDGE PARAMETERS:\n{PK_D}\n")
		fres.close()
	
	# DATA LOAD
	if DATA=='PRO':
		if KNOW=='Reactome':
			fn_H = f"/Backup/home/chenyupeng/DATA/Graph/reactome_P{PK_D['layer_Reactome']}/H1.csv" 
			data,H = load_cohort_data(fn_H,fn_data,HD['cox_num'],DF=False) 
		else: # STRING or hcluster
			data,H = load_cohort_data(None,fn_data,HD['cox_num'],DF=True) 

	elif DATA=='RNA':# RNA
		if PK_D["type_know"]=='hcluster':
			if FEAT=='ReactomeP12':
				data,H= load_cohort_data(None,fn_data,HD["cox_num"],DF=True) 
			elif FEAT=='Reactome':
				data = load_TCGA_data(fn_data,HD["end_point"],Filt_num=HD["cox_num"])
		elif PK_D["type_know"]=='Reactome':
			H_path = f"/Backup/home/chenyupeng/DATA/Graph/reactome_P{PK_D['layer_Reactome']}/" 
			if FEAT=='ReactomeP12':
				fn_H = H_path+f"H1.csv"
				data,H= load_cohort_data(fn_H,fn_data,HD["cox_num"],DF=False) 
			elif FEAT=='Reactome':
				data,H = load_TCGA_H_data(H_path,fn_data,Filt_num=HD["cox_num"])
			elif FEAT=='ReactomeTop':
				data,H = load_opt_data(data_path=fn_data,H_path=H_path,endpoint=HD["end_point"],Filt_num=HD['cox_num'],DF=False) 
		elif PK_D["type_know"]=='STRING':
			data = load_TCGA_data(fn_data,HD["end_point"],Filt_num=-1)
			data = Know_features_filt(data)

	#GridSearch for Hyperparameters
	''' Grid Search Parameters Space'''
	if DATA=='PRO':
		AGG = ['cat','noAGG'] 
		LR = [0.01, 0.001] 
		GLR = [0,0.1] 
		L2 = [0.005,0.1] 
		HIDS=[100,200]


	elif DATA=='RNA':
		AGG = ['cat', 'noAGG'] 
		LR = [ 0.01,0.001] 
		GLR = [0, 0.5] 
		L2 = [0.1, 0.5] 

	# Grid Search
	best_metric= -np.inf
	OptDic={"JM":JM} # option dictionary
	path_GS=f"{out_path}/GridSearch/"
	os.makedirs(path_GS,exist_ok=True)
	for HD['n_hid'] in HIDS:
		HD["MLP_hiddens"]= [HD['n_hid']]
		for HD['glr'] in GLR:
			for HD['AGG'] in AGG:
				for HD["l2"] in L2:
					for HD['lr'] in LR:
						best_val_mean,best_test_mean,best_val_cims,best_test_cims,HD = Train_Valid_Test(dataset,HD,PK_D,data,path_GS,fn_results,OptDic,logger)
	logger.info(f"Best {dataset} dataset {Model} {KNOW} KNOWLEDGE PARAMETERS:\n{HD}\n")
	
	path_weights=f"{out_path}/10seedsModelsWeights/"
	os.makedirs(path_weights,exist_ok=True)
	os.makedirs(out_path,exist_ok=True)
	fn_results = f"{out_path}/Results_{part_ablation}.csv"
	fres = open(fn_results, 'w')
	fres.write(f"----------------Results of ablation study of {part_ablation}----------------\n")
	fres.close()


	test_abl_list = []
	tms_abl_list=[]
	# if ablation_list[0]==None and part_ablation=='glr':
	# 	ablation_list[0]=BestParas['glr']
	for HD[part_ablation] in ablation_list: 
		ablation_study(data=data,H=H,HD=HD,out_path=out_path,path_weights=path_weights,Model=Model,dataset=dataset,logger=logger,PK_D=PK_D,part_ablation=part_ablation,n_trials=n_trials,JM=JM,FEAT=FEAT,test_abl_list=test_abl_list,tms_abl_list=tms_abl_list)

	
	# 绘制直方图 并且在上方标注cindex值
	if part_ablation=='loss_w':
		ablation_list=['ours','no_log_uncensored','no_carlibration','DH_only']

	abl_list_str = [f"{x}" for x in ablation_list]

	fres = open(fn_results, 'a')
	fres.write(f"{abl_list_str}:\n{tms_abl_list}\n{test_abl_list}")
	fres.close()
	
	plt.bar(abl_list_str,test_abl_list)
	plt.xlabel(part_ablation)
	plt.ylabel('C-Index')
	plt.title(f"{dataset} {Model} {KNOW} {part_ablation} Cindex hist plot")
	plt.savefig(f"{out_path}/{dataset}_{Model}_{KNOW}_{part_ablation}_Cindex_hist_plot.png")
	plt.close()
	
	# 绘制折线图 并且在上方标注cindex值
	plt.plot(ablation_list,test_abl_list)
	plt.xlabel(part_ablation)
	plt.ylabel('C-Index')
	plt.title(f"{dataset} {Model} {KNOW} {part_ablation} Cindex curv plot")
	plt.savefig(f"{out_path}/{dataset}_{Model}_{KNOW}_{part_ablation}_Cindex_curve_plot.png")
	plt.close()