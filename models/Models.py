''' 
All models 
考虑删除

'''
# import os
# from torch import nn
# from models.layers import HGAT_embedding, Linear_pooling
# import torch.nn.functional as F
# import torch
# from torch.amp import autocast
# from collections import defaultdict
# import numpy as np
# import torch.nn.functional as F
# import torch
# from torch.autograd import Variable
# import time
# import copy
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from collections import defaultdict
# from torch.utils import data
# from utils.survfunc_utils import *
# from models.FC_Surv import FC_layers,FC_Res_layers, FC_layers1_2
# # from models.HyperGraphSurvival_Low import *
# from utils.data_utils import *
# from sksurv.ensemble import RandomSurvivalForest
# from sksurv.linear_model import CoxnetSurvivalAnalysis,CoxPHSurvivalAnalysis
# # from models.torch_utils import negative_log_likelihood/

# class HGMLP(nn.Module):
#     def __init__(self,HD,data_train,data_eval,data_test,H,fn_ckpt,t_obs,G,seed,H_list=None): 
#         super(HGMLP, self).__init__()
#         if seed is not None:
#             if torch.cuda.is_available():
#                 torch.cuda.manual_seed_all(seed)
#             torch.manual_seed(seed)
#         os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
#         ACT =HD["ACT"]
#         n_hid=HD["n_hid"]
#         pooling_nodes=HD["pooling_hiddens"]
#         dropout=HD["dropout"]
#         RC_predict=HD["RESNET"]
#         MLP_hiddens_list=HD["MLP_hiddens"]
#         edge_pooling=HD["edge_pooling"]
#         predict_layer_list=HD["predict_hiddens"]
#         self.AGG = HD["AGG"]
#         self.glr=HD["glr"]
#         self.triple_loss=False
#         self.self_rank =False
#         self.t = int(t_obs)
#         self.loss_w=None
#         data_train,data_eval=torch.Tensor(data_train).cuda(),torch.Tensor(data_eval).cuda()
#         H = torch.Tensor(H).float().cuda()
#         self.H0 = [(H.T/H.T.sum(1, keepdim=True)).float()]

#         (self.mask1_tr,self.mask2_tr),(self.mask1_va,self.mask2_va)=self.create_mask(data_train[:,-2],data_train[:,-1]),self.create_mask(data_eval[:,-2],data_eval[:,-1])
#         self.data_train, self.data_eval,self.data_test = data_train, data_eval, torch.Tensor(data_test).cuda()
#         self.data_train_batch = torch.concat([data_train,self.mask1_tr,self.mask2_tr],1)
        
#         self.fn = fn_ckpt
#         self.report = defaultdict(list)
#         self.linear = nn.Linear(1, n_hid)
#         self.hgc_encoder = nn.ModuleList()
#         for i, h in enumerate(self.H0):
#             self.hgc_encoder.append(HGAT_embedding(h,  n_hid, n_hid, dropout=dropout, alpha=0.2, transfer=True, bias=True, concat=False))
        
#         self.linear_pooling = Linear_pooling(H.shape[1], pooling_layer_nodes=pooling_nodes)
#         # self.HGEmbedding = HypergraphEmbedding(H, n_hid, G, pooling_nodes, dropout=dropout,RW=False,edge_pooling=edge_pooling,pooling_method=HD['pooling_method'],H_list=H_list)
#         if self.AGG == "cat":
#             self.Predict_Layers = FC_layers(dim_in=n_hid*2,dropout=dropout,ACT=ACT,layer_nodes=predict_layer_list,dim_out=self.t,RES=RC_predict)
#         elif self.AGG == "jump": 
#             self.Predict_Layers = FC_layers(dim_in=n_hid+H.shape[0],dropout=dropout,ACT=ACT,layer_nodes=predict_layer_list,dim_out=self.t,RES=RC_predict)
#         else:
#             self.Predict_Layers = FC_layers(dim_in=n_hid,dropout=dropout,ACT=ACT,layer_nodes=predict_layer_list,dim_out=self.t,RES=RC_predict)
#         if self.AGG !="noAGG":
#             self.MLPEmbedding = FC_layers(dim_in=H.shape[0],dropout=dropout,ACT=ACT,layer_nodes=MLP_hiddens_list,dim_out=n_hid,RES=RC_predict)
    
#     def forward(self,exp_mat):
#         with autocast(device_type='cuda'):
#             x_g1 = self.linear(exp_mat.unsqueeze(-1).float())
#             for i, layers in enumerate(self.hgc_encoder):
#                 xe = self.H0[0].matmul(x_g1)
#                 xe = layers(x_g1, xe)
#             x_g1 = self.linear_pooling(xe)
#             if self.AGG =="noAGG":
#                 return self.Predict_Layers(x_g1)
#             if self.AGG == "jump":
#                 return self.Predict_Layers(torch.cat([x_g1,exp_mat],dim=1))

#             x_g2=self.MLPEmbedding(exp_mat)
#             if self.AGG == "plus":
#                 x_g = x_g1+x_g2
#             elif self.AGG == "cat":
#                 x_g =  torch.cat([x_g1,x_g2],dim=1)
#             elif self.AGG == "mean":
#                 x_g = (x_g1+x_g2)/2
        
#             return self.Predict_Layers(x_g)

#     def fit_early_stop(self, optimizer,scheduler, num_epochs,logger,batch_size = 32,eval_full=True,metric="ci",loss_dict={"a":1,"b":1,"c1":1,"c2":0,"d":1}):
#         best_model_wts = copy.deepcopy(self.state_dict())
#         best_val_metric = -np.inf
#         train_loader = data.DataLoader(self.data_train_batch,batch_size = batch_size ,shuffle = True)
#         if eval_full:
#             batch_size = len(self.data_eval)
#         eval_loader = data.DataLoader(self.data_eval,batch_size = batch_size ,shuffle = False)
#         for epoch in range(num_epochs+1):
#             optimizer = self.train_mini_batch(train_loader,optimizer,loss_dict)

#             epoch_loss,epoch_ci = self.get_loss_ci(eval_loader,self.data_eval,False,True,True,self.mask1_va,self.mask2_va,loss_dict)
#             epoch_loss=epoch_loss.detach().cpu().numpy()
#             self.report['epoch'].append(epoch)
#             self.report['val_loss'].append(epoch_loss)
#             self.report['val_ci'].append(epoch_ci)
#             scheduler.step(epoch_loss)
#             epoch_metric = (-epoch_loss) if metric == "loss" else epoch_ci
#             if epoch_metric > best_val_metric:
#                 best_val_metric = epoch_metric
#                 best_epoch = epoch + 1
#                 best_val_ci = epoch_ci
#                 best_val_loss = epoch_loss
#                 best_model_wts = copy.deepcopy(self.state_dict())
#                 # if Valid best update Test
#                 test_ci = self.pred_ci(self.data_test)
#                 logger.info(f"{30*'-'}")
                
#                 logger.info(f'Epoch{best_epoch} Updating val ci: {best_val_ci:4f}; test ci: {test_ci:4f}')
#                 logger.info(f'Updating val loss: {best_val_loss:4f}')
        
#         # save the metric,loss,weight of training model into ckpt file
#         if self.fn is not None:
#             torch.save({'epoch': best_epoch,
#                         'state_dict': self.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'best_eval_ci': best_val_ci,
#                         'best_eval_loss': best_val_loss,
#                         'test_ci': test_ci,
#                         'report': self.report,
#                         }, f'{self.fn}.ckpt')

#         # plt loss fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_loss'], label='Train loss')
#         plt.plot(self.report['val_loss'], label='Val loss')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
#         plt.close()
        
#         # plt ci fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_ci'], label=f'Train ci')
#         plt.plot(self.report['val_ci'], label=f'Val ci')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_ci.pdf', bbox_inches='tight')
#         plt.close()

#         self.load_state_dict(best_model_wts)
#         return self


#     def fit(self, optimizer, num_epochs,logger,batch_size = 32,eval_full=True,loss_dict={"a":1,"b":1,"c1":0,"c2":0,"d":0}):
#         best_model_wts = copy.deepcopy(self.state_dict())
#         train_loader = data.DataLoader(self.data_train_batch,batch_size = batch_size ,shuffle = True)
#         if eval_full:
#             batch_size = len(self.data_eval)
#         eval_loader = data.DataLoader(self.data_eval,batch_size = batch_size ,shuffle = False)
#         for epoch in range(num_epochs+1):
#             optimizer = self.train_mini_batch(train_loader,optimizer,loss_dict)

#             if epoch % 10 == 0:#eval
#                 self.eval()
#                 logger.info('-' * 20)
#                 logger.info(f'Epoch {epoch}/{num_epochs}')
#                 eval_loss,eval_ci = self.get_loss_ci(eval_loader,self.data_eval,False,True,True,self.mask1_va,self.mask2_va,loss_dict)
#                 eval_loss=eval_loss.detach().cpu().numpy()
#                 self.report['epoch'].append(epoch)
#                 self.report['eval_loss'].append(eval_loss)
#                 self.report['eval_ci'].append(eval_ci)
#                 logger.info(f'Updating eval ci: {eval_ci:4f}; eval loss: {eval_loss:4f}')
#                 best_model_wts = copy.deepcopy(self.state_dict())
#                 best_eval_ci = eval_ci
#                 best_eval_loss = eval_loss
            
#         test_ci = self.pred_ci(self.data_test)
#         logger.info(f'Final test ci: {test_ci:4f}')

#         if self.fn is not None:
#             torch.save({
#                         'state_dict': self.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'best_eval_ci': best_eval_ci,
#                         'best_eval_loss': best_eval_loss,
#                         'test_ci':test_ci,
#                         'report': self.report,
#                         }, rf"{self.fn}.ckpt")

#         # plt loss fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_loss'], label='Train loss')
#         plt.plot(self.report['eval_loss'], label='eval loss')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
#         plt.close()
        
#         # plt ci fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_ci'], label=f'Train ci')
#         plt.plot(self.report['eval_ci'], label=f'eval ci')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_ci.pdf', bbox_inches='tight')
#         plt.close()

#         self.load_state_dict(best_model_wts)
#         return self

#     def predict(self,sgs):
#         self.eval()
#         outputs = self.forward(sgs)
#         return outputs

#     def show_report(self):
#         return pd.DataFrame(self.report)
    
#     def create_mask1(self,time,label):# time,label,
#         time,label = torch.unsqueeze(time, dim=1),torch.unsqueeze(label, dim=1)

#         mask1 = np.zeros([np.shape(time)[0], self.t]) # for the first loss function
#         for i in range(np.shape(time)[0]):
#             if label[i,0] != 0:  #not censored
#                 mask1[i,int(time[i,0])] = 1 # 死亡时间取值
#             else: #label[i,2]==0: censored
#                 mask1[i,int(time[i,0]+1):] = 1# 删失时间后的所有时间取值
#         return torch.from_numpy(mask1).float()
#     #mask2
#     def create_mask2(self,time):
#         time = torch.unsqueeze(time, dim=1)
#         meas_time = -1
#         mask2 = np.zeros([np.shape(time)[0], self.t]) # for the first loss function
#         if np.shape(meas_time):  #lonogitudinal measurements
#             for i in range(np.shape(time)[0]):
#                 t1 = int(meas_time[i, 0]) # last measurement time
#                 t2 = int(time[i, 0]) # censoring/event time
#                 mask2[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
#         else:                    #single measurement
#             for i in range(np.shape(time)[0]):
#                 t = int(time[i, 0]) # censoring/event time
#                 mask2[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time

#         return torch.from_numpy(mask2).float()
    
#     def create_mask(self,time,label):
#         return (self.create_mask1(time,label).cuda(),self.create_mask2(time).cuda())

#     def nll(self,out, I_1, mask1):
#         # I_1 = torch.sign(k)
#         tmp = torch.sum(mask1 * out, 1)
#         lz = I_1 * torch.log(tmp + 1e-08)
#         lcen = (1. - I_1) * torch.log(tmp + 1e-08)
#         return -torch.mean(lz + 1.0*lcen)
        
#     def rank_loss(self, out, e, t, mask2):#t:time
#         sigma1 = 0.1
#         one_vector = torch.unsqueeze(torch.ones_like(t, dtype = torch.float32),dim=1).cuda()
#         I_2 = torch.eq(e, 1).float()
#         I_2 = torch.diag_embed(torch.squeeze(I_2))
#         # logger.info(out[0:,e:e+1,0:].size())
#         # tmp_e = torch.reshape(out[0:,0:,0:], (-1, self.t))

#         R = torch.mm(out, mask2.t())

#         diag_R = torch.reshape(torch.diagonal(R), (-1, 1))
#         R = torch.mm(one_vector, diag_R.t()) - R
#         R = R.t()
#         t = t.unsqueeze(-1)
#         T = F.relu(torch.sign(torch.mm(one_vector, t.t().float()) - torch.mm(t.float(),one_vector.t())))# exp-relu

#         T = torch.mm(I_2, T)
#         temp_eta = torch.mean(T * torch.exp(-R/sigma1), 1, keepdim = True)

#         return torch.sum(temp_eta)
    
#     def calibration_loss(self, out,e, mask2): #Calibration loss
#         I_2 = torch.eq(e.view(e.shape[0],1), 1).float()
#         r = torch.sum(out * mask2, 0)
#         eta = torch.mean((r - I_2)**2, 1, keepdim = True)# torch.Size([62])-torch.Size([64, 1])
#         return torch.sum(eta)
    
#     def deephit_loss(self,out,e,t,mask1,mask2,print_prop=False,beta=1,alpha=1,gamma=1,logger=None):
#         with autocast(device_type='cuda'):
#             loss_nll2 = self.nll(out,e,mask1)
#             loss_rank = self.rank_loss(out,e,t,mask2)
#             loss_cali = self.calibration_loss(out,e,mask2)

#         if print_prop:
#             logger.info("proportion:",(loss_nll2/loss_rank).item())
#         return (alpha*loss_nll2) + (beta*loss_rank) + (gamma*loss_cali)
    
#     def pred_ci(self,data_test,):
#         data_loader = data.DataLoader(data_test,batch_size=data_test.shape[0],shuffle = False)
        
#         _,ci = self.get_loss_ci(
#             data_loader,data_test,train=False,ci=True,loss=False,\
#             mask1=None,mask2=None,loss_dict=None)
#         return ci
    
#     def get_triple_loss(self,pred,s_time,e,mask1,mask2,sigma=2,):
#         s_time = s_time.float().unsqueeze(-1)
#         e=e.unsqueeze(-1)
#         risk = get_risk(pred)
#         a,b,c = self.loss_w["a"],self.loss_w["b"],self.loss_w["c"]
#         like = likelihood_loss(pred,mask1)
#         if self.self_rank:
#             rank = self_ranking_loss_weight(risk,s_time,e,0.5,0.1)
#         else:   
#             rank = ranking_loss(risk,s_time,e,sigma)
#         cali = calibration_loss(pred,e,mask1,mask2,self.loss_w['nbins'])
#         return (a*like+b*rank+c*cali)
    
#     def get_loss_ci(self,data_loader,data,train,ci,loss,mask1,mask2,loss_dict):
#         if train:
#             self.train()
#         else:
#             self.eval()
#         for i,data_phase in enumerate(data_loader):
#             X_phase_batch = data_phase[:,:-2]
#             PMF_batch = self.forward(X_phase_batch)
#             if i==0:
#                 preds = PMF_batch
#             else:
#                 preds = torch.cat((preds,PMF_batch),dim=0)
#         y = data[:,-2:]
#         s_time, e = y[:, 0].long(), y[:, 1]
#         if loss:
#             # loss = self.deephit_loss(out=preds,e=e,t=s_time,alpha = 1,beta=1,gamma=0,mask1=mask1,mask2=mask2)
#             if self.triple_loss:
#                 loss=self.get_triple_loss(pred=preds,s_time=s_time,e=e,mask1=mask1,mask2=mask2)#,loss_w=loss_dict)
#             else:
#                 loss = self.get_loss(risk=preds,e=e,t=s_time,mask1=mask1,mask2=mask2,loss_dict=loss_dict)
#             loss += self.glr 
#         else:
#             loss = None
#         if ci:
#             surv = torch.clamp(1.-torch.cumsum(preds,dim=-1),0,1)
#             ci =get_ci_td(s_time=s_time,cens=1-e,f=surv)
#         else:
#             ci=0
#         return loss,ci

#     def train_mini_batch(self,train_loader,optimizer,loss_dict):
#         self.train()
#         for i,data_phase in enumerate(train_loader):
#             mask_batch,data_phase = data_phase[:,-2*self.t:],data_phase[:,:-2*self.t] 
#             # y_phase_batch = data_phase[:,-2:]
#             # X_phase_batch = data_phase[:,:-2]
#             optimizer.zero_grad()
#             with torch.set_grad_enabled(True):
#                 PMF_train_batch = self.forward(data_phase[:,:-2])
#                 s_time, e = data_phase[:, -2].long(), data_phase[:,-1]
#                 # surv mask1 == mask1??
#                 # ranking_loss(risk,s_time.float().unsqueeze(-1),e.unsqueeze(-1),sigma)
#                 # likelihood_loss(PMF_train_batch,mask_batch[:,:self.t])
#                 # calibration_loss(PMF_train_batch,e.unsqueeze(-1),mask_batch[:,:self.t],mask_batch[:,self.t:],6)
#                 if self.triple_loss:
#                     batch_loss = self.get_triple_loss(pred=PMF_train_batch,s_time=s_time,e=e,mask1=mask_batch[:,:self.t],mask2=mask_batch[:,self.t:],sigma=2)
#                 else:
#                     batch_loss = self.get_loss(risk=PMF_train_batch,e=e,t=s_time,mask1=mask_batch[:,:self.t],mask2=mask_batch[:,self.t:],loss_dict=loss_dict)
                
#                 # print(hg_loss)
#                 batch_loss.backward()
#                 optimizer.step()
#         return optimizer
#     def get_loss(self,risk,e,t,mask1,mask2,loss_dict={"a":1,"b":1,"c1":0,"c2":0,"d":0}):
#         a,b,c1,c2,d=loss_dict["a"],loss_dict["b"],loss_dict["c1"],loss_dict["c2"],loss_dict["d"]
#         surv_batch = torch.clamp(1.-torch.cumsum(risk,dim=-1),0,1)
#         log_cens_z_rank = self.deephit_loss(out=risk,e=e,t=t,alpha =a,beta=b,gamma=c1,mask1=mask1,mask2=mask2)
#         log_uncens = get_RPSorLogUncens_loss(surv_batch,t,1-e,self.t,do_brier=False,RPS_norm=False)
#         loss_RPS = get_RPSorLogUncens_loss(surv_batch,t,1-e,self.t,do_brier=True,RPS_norm=True)
#         loss = log_cens_z_rank+c2*loss_RPS+d*log_uncens
#         return loss
    
# class DeepSurv(nn.Module):
#     def __init__(self,data_train, data_eval,data_test, NumLayers,nn_seed,ACT,dropout,fn):
#         super(DeepSurv, self).__init__()  
#         if nn_seed is not None:
#             torch.manual_seed(nn_seed)
#             torch.cuda.manual_seed(nn_seed)
#         os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
#         self.fn = fn
#         self.report = defaultdict(list)
#         middle_layers_nodes=[]
#         for l in range(NumLayers):
#             middle_layers_nodes.append(20)
#             # middle_layers_nodes.append(5)
#         self.Layers = FC_layers(dim_in=data_train.shape[1]-2,dropout=dropout,ACT=ACT,layer_nodes=middle_layers_nodes,dim_out=1)
#         self.data_train = torch.Tensor(data_train).cuda()
#         self.data_eval = torch.Tensor(data_eval).cuda()
#         self.data_test = torch.Tensor(data_test).cuda()

#     def forward(self,x):
#         LRR = self.Layers(x)
#         return LRR
    
#     def fit_early_stop(self, optimizer,scheduler, num_epochs,logger,batch_size = 32,metric="ci"):
#         best_model_wts = copy.deepcopy(self.state_dict())
#         best_val_metric = -np.inf
#         train_loader = data.DataLoader(self.data_train,batch_size = batch_size ,shuffle = True)
#         for epoch in range(num_epochs):
#             epoch=epoch+1
#             optimizer = self.train_mini_batch(train_loader,optimizer)
#             # eval phase
#             self.eval()
#             # optimizer.zero_grad() 
#             # with torch.set_grad_enabled(False):
#             LRR_eval = self.predict(self.data_eval[:,:-2],)# test expression matrix
#             epoch_loss = negative_log_likelihood(LRR_eval,self.data_eval[:,-2:])
#             epoch_ci = get_ci(t=self.data_eval[:,-2],e=self.data_eval[:,-1],r=LRR_eval)
#             epoch_loss=epoch_loss.detach().cpu().numpy()
#             self.report['epoch'].append(epoch)
#             self.report['eval_loss'].append(epoch_loss)
#             self.report['eval_ci'].append(epoch_ci)
#             scheduler.step(epoch_loss)
#             epoch_metric = (-epoch_loss) if metric == "loss" else epoch_ci
#             if epoch_metric > best_val_metric:
#                 best_val_metric = epoch_metric
#                 best_epoch = epoch
#                 best_val_ci = epoch_ci
#                 best_val_loss = epoch_loss
#                 best_model_wts = copy.deepcopy(self.state_dict())
#                 # if Valid best update Test
#                 # test_ci = self.pred_ci(self.data_test)
#                 LRR_test = self.predict(self.data_test[:,:-2],)# test expression matrix
#                 test_ci = get_ci(t=self.data_test[:,-2],e=self.data_test[:,-1],r=LRR_test)
#                 logger.info(f"{10*'-'}")
#                 logger.info(f'Updating epoch {best_epoch}; val ci: {best_val_ci:4f}; test ci: {test_ci:4f};')
#                 logger.info(f'Updating val loss: {best_val_loss:4f}')
            
#             if epoch % 10 == 0:
#                 logger.info('-' * 20)
#                 logger.info(f'Epoch {epoch}/{num_epochs}')
#                 logger.info(f'Eval Loss: {epoch_loss:.4f} ci: {epoch_ci:.4f} ,')
#         # save the metric,loss,weight of training model into ckpt file
#         if self.fn is not None:
#             torch.save({'epoch': best_epoch,
#                         'state_dict': self.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'best_eval_ci': best_val_ci,
#                         'best_eval_loss': best_val_loss,
#                         'test_ci': test_ci,
#                         'report': self.report,
#                         }, f'{self.fn}.ckpt')

#         # plt loss fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_loss'], label='Train loss')
#         plt.plot(self.report['eval_loss'], label='Val loss')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
#         plt.close()
        
#         # plt ci fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_ci'], label=f'Train ci')
#         plt.plot(self.report['eval_ci'], label=f'Val ci')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_ci.pdf', bbox_inches='tight')
#         plt.close()

#         self.load_state_dict(best_model_wts)
#         return self

    
#     def fit_train_eval(self, optimizer, num_epochs,logger,batch_size = 32,eval_full=True,loss_dict={"a":1,"b":1,"c1":0,"c2":0,"d":0}):
#         best_model_wts = copy.deepcopy(self.state_dict())
#         train_loader = data.DataLoader(self.data_train,batch_size = batch_size ,shuffle = True)

#         for epoch in range(num_epochs+1):
#             optimizer = self.train_mini_batch(train_loader,optimizer)

#             if epoch % 10 == 0:#eval
#                 self.eval()
#                 logger.info('-' * 20)
#                 logger.info(f'Epoch {epoch}/{num_epochs}')

#                 LRR_eval = self.predict(self.data_eval[:,:-2],)# test expression matrix
#                 eval_loss = negative_log_likelihood(LRR_eval,self.data_eval[:,-2:])
#                 eval_ci = get_ci(t=self.data_eval[:,-2],e=self.data_eval[:,-1],r=LRR_eval)
#                 eval_loss=eval_loss.detach().cpu().numpy()
#                 self.report['epoch'].append(epoch)
#                 self.report['eval_loss'].append(eval_loss)
#                 self.report['eval_ci'].append(eval_ci)
#                 logger.info(f'Updating eval ci: {eval_ci:4f}; eval loss: {eval_loss:4f}')
#                 best_model_wts = copy.deepcopy(self.state_dict())
#                 best_eval_ci = eval_ci
#                 best_eval_loss = eval_loss
#         # last epoch test
#         LRR_test = self.predict(self.data_test[:,:-2],)# test expression matrix
#         test_ci = get_ci(t=self.data_test[:,-2],e=self.data_test[:,-1],r=LRR_test)
#         logger.info(f'Final test ci: {test_ci:4f}')
        
#         if self.fn is not None:
#             torch.save({
#                         'state_dict': self.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'best_train_ci': 0,
#                         'best_train_loss': 0,
#                         'best_eval_ci': best_eval_ci,
#                         'best_eval_loss': best_eval_loss,
#                         'test_ci'  :test_ci,
#                         'report': self.report,
#                         }, f'{self.fn}.ckpt')

#         # plt loss fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_loss'], label='Train loss')
#         plt.plot(self.report['eval_loss'], label='eval loss')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
#         plt.close()
        
#         # plt ci fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_ci'], label=f'Train ci')
#         plt.plot(self.report['eval_ci'], label=f'eval ci')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_ci.pdf', bbox_inches='tight')
#         plt.close()

#         self.load_state_dict(best_model_wts)
#         return self

#     def predict(self, exp_mat):# need to modify
#         self.eval()
#         outputs = self.forward(exp_mat)
#         return outputs

#     def show_report(self):
#         return pd.DataFrame(self.report)
    
#     def train_mini_batch(self,train_loader,optimizer):
#         self.train()
#         for i,data_phase in enumerate(train_loader):
#             optimizer.zero_grad()
#             with torch.set_grad_enabled(True):
#                 LRR_batch = self.forward(data_phase[:,:-2])
#                 batch_loss = negative_log_likelihood(LRR_batch,data_phase[:,-2:])
#                 batch_loss.backward()
#                 optimizer.step()
#         return optimizer

# class DeepHit(nn.Module):
#     def __init__(self,data_train, data_eval,data_test,nn_seed,NumLayers,t_obs,ACT,dropout,fn):
#         super(DeepHit, self).__init__()  
#         if nn_seed is not None:
#             torch.manual_seed(nn_seed)
#             torch.cuda.manual_seed(nn_seed)
#         os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
#         self.fn = fn
#         self.report = defaultdict(list)
#         self.t = int(t_obs)
#         data_train,data_eval,data_test=torch.Tensor(data_train).cuda(),torch.Tensor(data_eval).cuda(),torch.Tensor(data_test).cuda()
#         self.Layers = FC_Res_layers(dim_in=data_train.shape[1]-2,dropout=dropout,ACT=ACT,NumLayers=NumLayers,dim_out=self.t)
#         (self.mask1_tr,self.mask2_tr),(self.mask1_va,self.mask2_va),(self.mask1_te,self.mask2_te)=self.create_mask(data_train[:,-2],data_train[:,-1]),self.create_mask(data_eval[:,-2],data_eval[:,-1]),self.create_mask(data_test[:,-2],data_test[:,-1])
#         self.data_train_batch = torch.concat([data_train,self.mask1_tr,self.mask2_tr],1)
#         self.data_eval,self.data_test = data_eval,data_test
#         self.data_train = data_train
#     def forward(self,x):
#         PMF = self.Layers(x)
#         return PMF
#     def fit_early_stop(self, optimizer,scheduler, num_epochs,logger,batch_size,eval_full=False,metric="ci",loss_dict={"a":1,"b":1,"c":0}):
#         best_model_wts = copy.deepcopy(self.state_dict())
#         best_val_metric=-np.inf
#         train_loader = data.DataLoader(self.data_train_batch,batch_size = batch_size ,shuffle = True)
#         if eval_full:
#             batch_size = len(self.data_eval)
#         eval_loader = data.DataLoader(self.data_eval,batch_size = batch_size ,shuffle = False)
#         test_loader = data.DataLoader(self.data_test,batch_size = len(self.data_test) ,shuffle = False)
#         for epoch in range(num_epochs):
#             epoch=epoch+1
#             optimizer = self.train_mini_batch(train_loader,optimizer,loss_dict)
#             epoch_loss,epoch_ci = self.get_loss_ci(eval_loader,self.data_eval,False,True,True,self.mask1_va,self.mask2_va,loss_dict)
#             epoch_loss=epoch_loss.detach().cpu().numpy()
#             self.report['epoch'].append(epoch)
#             self.report['val_loss'].append(epoch_loss)
#             self.report['val_ci'].append(epoch_ci)
#             scheduler.step(epoch_loss)
#             epoch_metric = (-epoch_loss) if metric == "loss" else epoch_ci

#             if epoch_metric > best_val_metric:
#                 best_val_metric = epoch_metric
#                 best_epoch = epoch + 1
#                 best_val_ci = epoch_ci
#                 best_val_loss = epoch_loss
#                 best_model_wts = copy.deepcopy(self.state_dict())
#                 # if Valid best update Test
#                 _,test_ci = self.get_loss_ci(test_loader,self.data_test,False,True,False,self.mask1_te,self.mask2_te,loss_dict)
#                 logger.info(f"{10*'-'}")
#                 logger.info(f'Updating epoch: {best_epoch}; val ci: {best_val_ci:4f}; test ci: {test_ci:4f}')
#                 logger.info(f'Updating val loss: {best_val_loss:4f}')
#             if epoch % 10 == 0:
#                 logger.info('-' * 20)
#                 logger.info(f'Epoch {epoch}/{num_epochs}')
#                 logger.info(f'Eval Loss: {epoch_loss:.4f} ci: {epoch_ci:.4f} ,')
#         # save the metric,loss,weight of training model into ckpt file
#         if self.fn is not None:
#             torch.save({'epoch': best_epoch,
#                         'state_dict': self.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'best_eval_ci': best_val_ci,
#                         'best_eval_loss': best_val_loss,
#                         'test_ci': test_ci,
#                         'report': self.report,
#                         }, f'{self.fn}.ckpt')
       
#         # plt loss fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_loss'], label='Train loss')
#         plt.plot(self.report['val_loss'], label='Val loss')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
#         plt.close()
        
#         # plt ci fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_ci'], label=f'Train ci')
#         plt.plot(self.report['val_ci'], label=f'Val ci')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_ci.pdf', bbox_inches='tight')
#         plt.close()

#         self.load_state_dict(best_model_wts)
#         return self
#     def fit_train_eval(self, optimizer, num_epochs,logger,batch_size,eval_full=False,loss_dict={"a":1,"b":1,"c":0}):
#         best_model_wts = copy.deepcopy(self.state_dict())
#         train_loader = data.DataLoader(self.data_train_batch,batch_size = batch_size ,shuffle = True)
#         test_loader = data.DataLoader(self.data_test,batch_size = len(self.data_test) ,shuffle = False)
        
#         if eval_full:
#             batch_size = len(self.data_eval)
#         eval_loader = data.DataLoader(self.data_eval,batch_size = batch_size ,shuffle = False)
#         for epoch in range(num_epochs+1):
#             optimizer = self.train_mini_batch(train_loader,optimizer,loss_dict)

#             if epoch % 10 == 0:#eval
#                 self.eval()
#                 logger.info('-' * 20)
#                 logger.info(f'Epoch {epoch}/{num_epochs}')
#                 eval_loss,eval_ci = self.get_loss_ci(eval_loader,self.data_eval,False,True,True,self.mask1_va,self.mask2_va,loss_dict)
#                 eval_loss=eval_loss.detach().cpu().numpy()
#                 self.report['epoch'].append(epoch)
#                 self.report['eval_loss'].append(eval_loss)
#                 self.report['eval_ci'].append(eval_ci)
#                 logger.info(f'Updating eval ci: {eval_ci:4f}; eval loss: {eval_loss:4f}')
#                 best_model_wts = copy.deepcopy(self.state_dict())
#                 best_eval_ci = eval_ci
#                 best_eval_loss = eval_loss

#         _,test_ci = self.get_loss_ci(test_loader,self.data_test,False,True,False,self.mask1_te,self.mask2_te,loss_dict)

#         if self.fn is not None:
#             torch.save({
#                         'state_dict': self.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'best_train_ci': 0,
#                         'best_train_loss': 0,
#                         'best_eval_ci': best_eval_ci,
#                         'best_eval_loss': best_eval_loss,
#                         'test_ci': test_ci,
#                         'report': self.report,
#                         }, f'{self.fn}.ckpt')

#         # plt loss fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_loss'], label='Train loss')
#         plt.plot(self.report['eval_loss'], label='eval loss')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
#         plt.close()
        
#         # plt ci fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_ci'], label=f'Train ci')
#         plt.plot(self.report['eval_ci'], label=f'eval ci')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_ci.pdf', bbox_inches='tight')
#         plt.close()

#         self.load_state_dict(best_model_wts)
#         return self

#     def predict(self, exp_mat):# need to modify
#         self.eval()
#         outputs = self.forward(exp_mat)
#         return outputs

#     def show_report(self):
#         return pd.DataFrame(self.report)
    
#     #mask1
#     def create_mask1(self,time,label):# time,label,
#         time,label = torch.unsqueeze(time, dim=1),torch.unsqueeze(label, dim=1)

#         mask1 = np.zeros([np.shape(time)[0], self.t]) # for the first loss function
#         for i in range(np.shape(time)[0]):
#             if label[i,0] != 0:  #not censored
#                 mask1[i,int(time[i,0])] = 1
#             else: #label[i,2]==0: censored
#                 mask1[i,int(time[i,0]+1):] = 1
#         return torch.from_numpy(mask1).float()
#     #mask2
#     def create_mask2(self,time):
#         time = torch.unsqueeze(time, dim=1)
#         meas_time = -1
#         mask2 = np.zeros([np.shape(time)[0], self.t]) # for the first loss function
#         if np.shape(meas_time):  #lonogitudinal measurements
#             for i in range(np.shape(time)[0]):
#                 t1 = int(meas_time[i, 0]) # last measurement time
#                 t2 = int(time[i, 0]) # censoring/event time
#                 mask2[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
#         else:                    #single measurement
#             for i in range(np.shape(time)[0]):
#                 t = int(time[i, 0]) # censoring/event time
#                 mask2[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
#         return torch.from_numpy(mask2).float()
#     def create_mask(self,time,label):
#         return (self.create_mask1(time,label).cuda(),self.create_mask2(time).cuda())

#     def nll(self,out, I_1, mask1):
#         # I_1 = torch.sign(k)
#         tmp = torch.sum(mask1 * out, 1)
#         lz = I_1 * torch.log(tmp + 1e-08)
#         lcen = (1. - I_1) * torch.log(tmp + 1e-08)
#         return -torch.mean(lz + 1.0*lcen)
        
#     def rank_loss(self, out, e, t, mask2):#t:time
#         sigma1 = 0.1
#         one_vector = torch.unsqueeze(torch.ones_like(t, dtype = torch.float32),dim=1).cuda()
#         I_2 = torch.eq(e, 1).float()
#         I_2 = torch.diag_embed(torch.squeeze(I_2))
#         # print(out[0:,e:e+1,0:].size())
#         # tmp_e = torch.reshape(out[0:,0:,0:], (-1, self.t))

#         R = torch.mm(out, mask2.t())

#         diag_R = torch.reshape(torch.diagonal(R), (-1, 1))
#         R = torch.mm(one_vector, diag_R.t()) - R
#         R = R.t()
#         t = t.unsqueeze(-1)
#         T = F.relu(torch.sign(torch.mm(one_vector, t.t().float()) - torch.mm(t.float(),one_vector.t())))

#         T = torch.mm(I_2, T)
#         temp_eta = torch.mean(T * torch.exp(-R/sigma1), 1, keepdim = True)

#         return torch.sum(temp_eta)
#     def calibration_loss(self, out,e, mask2): #Calibration loss
#         I_2 = torch.eq(e.view(e.shape[0],1), 1).float()
#         r = torch.sum(out * mask2, 0)
#         eta = torch.mean((r - I_2)**2, 1, keepdim = True)# torch.Size([62])-torch.Size([64, 1])
#         return torch.sum(eta)
    
#     def deephit_loss(self,out,e,t,mask1,mask2,print_prop=False,beta=1,alpha=1,gamma=1):
#         loss_nll2 = self.nll(out,e,mask1)
#         loss_rank = self.rank_loss(out,e,t,mask2)
#         loss_cali = self.calibration_loss(out,e,mask2)

#         if print_prop:
#             print("proportion:",(loss_nll2/loss_rank).item())
#         return (alpha*loss_nll2) + (beta*loss_rank) + (gamma*loss_cali)


#     def get_loss_ci(self,data_loader,data,train,ci,loss,mask1,mask2,loss_dict):
#         if train:
#             self.train
#         else:
#             self.eval()
#         for i,data_phase in enumerate(data_loader):
#             X_phase_batch = data_phase[:,:-2]
#             PMF_batch = self.forward(X_phase_batch)
#             if i==0:
#                 preds = PMF_batch
#             else:
#                 preds = torch.cat((preds,PMF_batch),dim=0)
#         y = data[:,-2:]
#         s_time, e = y[:, 0].long(), y[:, 1]
#         if loss:
#             loss = self.deephit_loss(out=preds,e=e,t=s_time,alpha =loss_dict['a'],beta=loss_dict['b'],gamma=loss_dict['c'],mask1=mask1,mask2=mask2)
#             # loss = self.get_loss(risk=preds,e=e,t=s_time,mask1=mask1,mask2=mask2,loss_dict=loss_dict)
#         else:
#             loss = None
#         if ci:
#             surv = torch.clamp(1.-torch.cumsum(preds,dim=-1),0,1)
#             ci =get_ci_td(s_time=s_time,cens=1-e,f=surv)
#         else:
#             ci=0
#         return loss,ci

#     def train_mini_batch(self,train_loader,optimizer,loss_dict):
#         self.train()
#         for i,data_phase in enumerate(train_loader):
#             mask_batch,data_phase = data_phase[:,-2*self.t:],data_phase[:,:-2*self.t] 
#             # y_phase_batch = data_phase[:,-2:]
#             # X_phase_batch = data_phase[:,:-2]
#             optimizer.zero_grad()
#             with torch.set_grad_enabled(True):
#                 PMF_train_batch = self.forward(data_phase[:,:-2])
#                 s_time, e = data_phase[:, -2].long(), data_phase[:,-1]
#                 batch_loss = self.deephit_loss(out=PMF_train_batch,e=e,t=s_time,alpha =loss_dict['a'],beta=loss_dict['b'],gamma=loss_dict['c'],mask1=mask_batch[:,:self.t],mask2=mask_batch[:,self.t:])
#                 # batch_loss = self.get_loss(risk=PMF_train_batch,e=e,t=s_time,mask1=mask_batch[:,:self.t],mask2=mask_batch[:,self.t:],loss_dict=loss_dict)
#                 batch_loss.backward()
#                 optimizer.step()
#         return optimizer

# class DRSA(nn.Module):
#     def __init__(self,t_obs,lstm_layers,nn_seed,fn,data_train, data_eval,data_test,dropout=0.5): 
#         super(DRSA, self).__init__()
#         if nn_seed is not None:
#             if torch.cuda.is_available():
#                 torch.cuda.manual_seed_all(nn_seed)
#             torch.manual_seed(nn_seed)
#         os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
#         dim_in=data_train.shape[1]-2
#         l_hid = 2*dim_in // 3
#         self.data_train, self.data_eval, self.data_test = torch.Tensor(data_train).cuda(), torch.Tensor(data_eval).cuda(),torch.Tensor(data_test).cuda()
#         self.hidden_size = l_hid
#         self.num_layers = lstm_layers
#         self.batch_norm = nn.BatchNorm1d(dim_in)     
#         self.fc = nn.Linear(l_hid, 1)
#         self.lstm = nn.LSTM(dim_in+1, hidden_size = l_hid, batch_first = True,num_layers=self.num_layers,dropout=dropout).cuda()
#         self.sig = nn.Sigmoid()         
#         self.t = int(t_obs)
#         self.fn = fn
#         self.report = defaultdict(list)

#     def forward(self,x):
#         cov_size = x.shape[1]+1
#         cov = self.batch_norm(x) # 618,150 (batch,fea_num)
#         cov = cov.repeat(1, self.t).view(-1, self.t, cov_size-1)# torch.Size([618, 792, 150]) 直接repeat
#         time = torch.arange(1, self.t+1, 1).repeat(cov.size(0)).view(-1, self.t, 1).float().cuda()# torch.Size([618, 792, 1])离散时间*样本数
#         x = torch.cat((cov, time), dim = 2).cuda()# torch.Size([618, 792, 151])#加上的维度是离散时间（相当于是加上时间作为协变量）
#         batch_size = x.size(0)#618
#         h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).float().cuda(), requires_grad = False)# 不更新梯度参数？
#         c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).float().cuda(), requires_grad = False)
#         outs, hidden = self.lstm(x, (h0,c0))# 初始的状态和记忆细胞 outs: torch.Size([32, 62, 100])
#         outs = outs.contiguous().view(batch_size * self.t, self.hidden_size)# tensor的内存连续存储，返回一个contiguous copy；# outputs.shape = (618*183,200)
#         pdf = self.sig(self.fc(outs)).view(batch_size, self.t) # 32(batch),62 每个样本在每个事件段的h？
#         return pdf
#     def fit_early_stop(self, optimizer,scheduler, num_epochs,logger,batch_size,eval_full=False,metric="ci",loss_dict={"do_l2":0,"do_nll":1,"do_brier":0,"a":0.5,"RPS_norm":0}):
#         best_model_wts = copy.deepcopy(self.state_dict())
#         best_val_metric=-np.inf

#         train_loader = data.DataLoader(self.data_train,batch_size = batch_size ,shuffle = True)
#         if eval_full:
#             batch_size = len(self.data_eval)
#         eval_loader = data.DataLoader(self.data_eval,batch_size = batch_size ,shuffle = False)
#         test_loader = data.DataLoader(self.data_test,batch_size = len(self.data_test) ,shuffle = False)

#         for epoch in range(num_epochs):
#             epoch=epoch+1
#             optimizer = self.train_mini_batch(train_loader,optimizer,loss_dict)
#             epoch_loss,epoch_ci = self.get_loss_ci(eval_loader,self.data_eval[:,-2:],False,True,loss_dict)
    
#             epoch_loss=epoch_loss.detach().cpu().numpy()
#             self.report['epoch'].append(epoch)
#             self.report['val_loss'].append(epoch_loss)
#             self.report['val_ci'].append(epoch_ci)
#             scheduler.step(epoch_loss)
#             epoch_metric = (-epoch_loss) if metric == "loss" else epoch_ci
#             if epoch_metric > best_val_metric:
#                 best_val_metric = epoch_metric
#                 best_epoch = epoch + 1
#                 best_val_ci = epoch_ci
#                 best_val_loss = epoch_loss
#                 best_model_wts = copy.deepcopy(self.state_dict())
#                 # if Valid best update Test

#                 test_loss,test_ci = self.get_loss_ci(test_loader,self.data_test[:,-2:],False,True,loss_dict)
#                 test_loss=test_loss.detach().cpu().numpy()
#                 logger.info(f"{10*'-'}")
#                 logger.info(f'Updating epoch: {best_epoch}; val ci: {best_val_ci:4f}; test ci: {test_ci:4f}')
#                 logger.info(f'Updating val loss: {best_val_loss:4f}; test_loss:{test_loss}')
#             if epoch % 10 == 0:
#                 logger.info('-' * 20)
#                 logger.info(f'Epoch {epoch}/{num_epochs}')
#                 logger.info(f'Eval Loss: {epoch_loss:.4f} ci: {epoch_ci:.4f} ,')
#         # save the metric,loss,weight of training model into ckpt file
#         if self.fn is not None:
#             torch.save({'epoch': best_epoch,
#                         'state_dict': self.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'best_eval_ci': best_val_ci,
#                         'best_eval_loss': best_val_loss,
#                         'test_ci': test_ci,
#                         'test_loss':test_loss,
#                         'report': self.report,
#                         }, f'{self.fn}.ckpt')
       
#         # plt loss fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_loss'], label='Train loss')
#         plt.plot(self.report['val_loss'], label='Val loss')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
#         plt.close()
        
#         # plt ci fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_ci'], label=f'Train ci')
#         plt.plot(self.report['val_ci'], label=f'Val ci')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_ci.pdf', bbox_inches='tight')
#         plt.close()

#         self.load_state_dict(best_model_wts)
#         return self

#     def fit_train_eval(self, optimizer, num_epochs,logger,batch_size,eval_full=False,loss_dict={"do_l2":0,"do_nll":1,"do_brier":0,"a":0.5,"RPS_norm":0}):
#         best_model_wts = copy.deepcopy(self.state_dict())
#         train_loader = data.DataLoader(self.data_train,batch_size = batch_size ,shuffle = True)
#         test_loader = data.DataLoader(self.data_test,batch_size = len(self.data_test) ,shuffle = False)
        
#         if eval_full:
#             batch_size = len(self.data_eval)
#         eval_loader = data.DataLoader(self.data_eval,batch_size = batch_size ,shuffle = False)
#         for epoch in range(num_epochs+1):
#             optimizer = self.train_mini_batch(train_loader,optimizer,loss_dict)
#             if epoch % 10 == 0:#eval
#                 self.eval()
#                 logger.info('-' * 20)
#                 logger.info(f'Epoch {epoch}/{num_epochs}')
#                 eval_loss,eval_ci = self.get_loss_ci(eval_loader,self.data_eval[:,-2:],False,True,loss_dict)
#                 eval_loss=eval_loss.detach().cpu().numpy()
#                 self.report['epoch'].append(epoch)
#                 self.report['eval_loss'].append(eval_loss)
#                 self.report['eval_ci'].append(eval_ci)
#                 logger.info(f'Updating eval ci: {eval_ci:4f}; eval loss: {eval_loss:4f}')
#                 best_model_wts = copy.deepcopy(self.state_dict())
#                 best_eval_ci = eval_ci
#                 best_eval_loss = eval_loss
#         test_loss,test_ci = self.get_loss_ci(test_loader,self.data_test[:,-2:],False,True,loss_dict)

#         if self.fn is not None:
#             torch.save({
#                         'state_dict': self.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'best_train_ci': 0,
#                         'best_train_loss': 0,
#                         'best_eval_ci': best_eval_ci,
#                         'best_eval_loss': best_eval_loss,
#                         'test_ci':test_ci,
#                         'report': self.report,
#                         }, f'{self.fn}.ckpt')

#         # plt loss fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_loss'], label='Train loss')
#         plt.plot(self.report['eval_loss'], label='eval loss')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
#         plt.close()
        
#         # plt ci fig
#         fig = plt.figure()
#         # plt.plot(self.report['train_ci'], label=f'Train ci')
#         plt.plot(self.report['eval_ci'], label=f'eval ci')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         fig.savefig(f'{self.fn}_ci.pdf', bbox_inches='tight')
#         plt.close()

#         self.load_state_dict(best_model_wts)
#         return self

#     def predict(self,sgs):
#         self.eval()
#         outputs,_ = self.forward(sgs)
#         return outputs,_

#     def show_report(self):
#         return pd.DataFrame(self.report)
    
#     def get_loss_ci(self,data_loader,y,train,if_ci,loss_dict):
#         if train:
#             self.train
#         else:
#             self.eval()
#         for i,data_phase in enumerate(data_loader):
#             X_phase_batch = data_phase[:,:-2]
#             PDF_batch = self.forward(X_phase_batch)
#             if i ==0:
#                 preds = PDF_batch
#             else:
#                 preds = torch.cat((preds,PDF_batch),dim=0)

#         s_time, e = y[:, 0].long(), y[:, 1]

#         loss,ci = get_DRS_loss_ci(preds,loss_dict["do_l2"],loss_dict["do_brier"],loss_dict["RPS_norm"],
#         loss_dict["do_nll"],loss_dict["a"],s_time,e,self.t,if_ci)

#         return loss,ci

#     def train_mini_batch(self,train_loader,optimizer,loss_dict):
#         self.train()
#         for i,data_phase in enumerate(train_loader):
#             y_phase_batch = data_phase[:,-2:]
#             X_phase_batch = data_phase[:,:-2]
#             optimizer.zero_grad()
#             with torch.set_grad_enabled(True):
#                 PDF_train_batch = self.forward(X_phase_batch)
#                 s_time, e = y_phase_batch[:, 0].long(), y_phase_batch[:, 1]
#                 batch_loss,ci = get_DRS_loss_ci(PDF_train_batch,loss_dict["do_l2"],loss_dict["do_brier"],
#                 loss_dict["RPS_norm"],loss_dict["do_nll"],loss_dict["a"],s_time,e,self.t,False)
#                 batch_loss.backward()
#                 optimizer.step()
#         return optimizer
    

# def COX_train_eval(data_train,data_eval,data_test,HD,fn):
#     labels_train = np.array([(bool(s), t) for s, t in \
#         zip(data_train[:, -1], data_train[:, -2])], \
#         dtype=[('status', 'bool'), ('time', 'float')])
#     try:
#         model = CoxPHSurvivalAnalysis(
#             alpha=HD["alpha"],ties=HD["ties"],verbose=2
#             ).fit(data_train[:, :-2], labels_train)
#         pred_eval = model.predict(data_eval[:, :-2])
#         ci_eval = get_ci_np(data_eval[:, -2], data_eval[:, -1], pred_eval)
#         # if HD["test_ci"]:
#         pred_test = model.predict(data_test[:, :-2])
#         ci_test = get_ci_np(data_test[:, -2], data_test[:, -1], pred_test)
#     except:
#         print(f"ValueError: search direction contains NaN or infinite values. Skip it, set ci as 0.5.")
#         ci_eval=ci_test=0.5
#     print(f"ci_eval = {ci_eval}; ci_test = {ci_test}")
#     ckpt = {'eval_ci': ci_eval,'test_ci': ci_test,}
#     if fn is not None:
#         torch.save(ckpt, f'{fn}.ckpt')
#     return ckpt

# def RSF_train_eval(data_train,data_eval,data_test,seed,HD,fn):
#     labels_train = np.array([(bool(s), t) for s, t in \
#         zip(data_train[:, -1], data_train[:, -2])], \
#         dtype=[('status', 'bool'), ('time', 'float')])
#     model = RandomSurvivalForest(    
#         n_estimators=HD["n_estimators"],#max_depth=HD["max_depth"],
#         min_samples_leaf=HD["min_samples_leaf"],min_samples_split=HD["min_samples_split"],
#         max_features=HD["max_features"],#,max_leaf_nodes=HD["max_leaf_nodes"],
#         random_state=seed,n_jobs=32).fit(data_train[:, :-2], labels_train)
#     pred_eval = model.predict(data_eval[:, :-2])
#     ci_eval = get_ci_np(data_eval[:, -2], data_eval[:, -1], pred_eval)
#     # if HD["test_ci"]:
#     pred_test = model.predict(data_test[:, :-2])
#     ci_test = get_ci_np(data_test[:, -2], data_test[:, -1], pred_test)
#     print(f"ci_eval = {ci_eval}; ci_test = {ci_test}")
#     ckpt = {'eval_ci': ci_eval,'test_ci': ci_test,}
#     if fn is not None:
#         torch.save(ckpt, f'{fn}.ckpt')
#     return ckpt