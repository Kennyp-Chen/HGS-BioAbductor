''' All models '''
import os
from torch import nn
from models.layers_interpret import HGAT_embedding, Linear_pooling
import torch.nn.functional as F
import torch
from torch.cuda.amp import autocast
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import time
import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from torch.utils import data
from utils.survfunc_utils import *
from models.FC_Surv import FC_layers,FC_Res_layers, FC_layers1_2
# from models.HyperGraphSurvival_Low import *
from utils.data_utils import *
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis,CoxPHSurvivalAnalysis
# from models.torch_utils import negative_log_likelihood/

class HGS(nn.Module):
    def __init__(self,HD,data_train,data_eval,data_test,H,k,fn_ckpt,t_obs,G,seed, H_list=None,node_idx=[]): 
        super(HGS, self).__init__()
        if seed is not None:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        ACT =HD["ACT"]
        n_hid=HD["n_hid"]
        pooling_nodes=HD["pooling_hiddens"]
        dropout=HD["dropout"]
        RC_predict=HD["RESNET"]
        MLP_hiddens_list=HD["MLP_hiddens"]
        edge_pooling=HD["edge_pooling"]
        predict_layer_list=HD["predict_hiddens"]
        self.AGG = HD["AGG"]
        self.glr=HD["glr"]
        self.triple_loss=False
        self.self_rank =False
        self.t = int(t_obs)
        self.loss_w=None
        self.k = k
        data_train,data_eval=torch.Tensor(data_train).cuda(),torch.Tensor(data_eval).cuda()
        H = torch.Tensor(H).float().cuda()
        self.H0 = [(H.T/H.T.sum(1, keepdim=True)).half()]

        (self.mask1_tr,self.mask2_tr),(self.mask1_va,self.mask2_va)=self.create_mask(data_train[:,-2],data_train[:,-1]),self.create_mask(data_eval[:,-2],data_eval[:,-1])
        self.data_train, self.data_eval,self.data_test = data_train, data_eval, torch.Tensor(data_test).cuda()
        self.data_train_batch = torch.concat([data_train,self.mask1_tr,self.mask2_tr],1)
        self.fn = fn_ckpt
        self.report = defaultdict(list)
        self.linear = nn.Linear(1, n_hid)
        self.hgc_encoder = nn.ModuleList()
        for i, h in enumerate(self.H0):
            self.hgc_encoder.append(HGAT_embedding(h,  n_hid, n_hid, dropout=dropout, alpha=0.2, transfer=True, bias=True, concat=False))
        
        self.linear_pooling = Linear_pooling(H.shape[1], pooling_layer_nodes=pooling_nodes)
        # self.HGEmbedding = HypergraphEmbedding(H, n_hid, G, pooling_nodes, dropout=dropout,RW=False,edge_pooling=edge_pooling,pooling_method=HD['pooling_method'],H_list=H_list)
        if self.AGG == "cat":
            self.Predict_Layers = FC_layers(dim_in=n_hid*2,dropout=dropout,ACT=ACT,layer_nodes=predict_layer_list,dim_out=self.t,RES=RC_predict)
        elif self.AGG == "jump": 
            self.Predict_Layers = FC_layers(dim_in=n_hid+H.shape[0],dropout=dropout,ACT=ACT,layer_nodes=predict_layer_list,dim_out=self.t,RES=RC_predict)
        else:
            self.Predict_Layers = FC_layers(dim_in=n_hid,dropout=dropout,ACT=ACT,layer_nodes=predict_layer_list,dim_out=self.t,RES=RC_predict)
        if self.AGG !="noAGG":
            self.MLPEmbedding = FC_layers(dim_in=H.shape[0],dropout=dropout,ACT=ACT,layer_nodes=MLP_hiddens_list,dim_out=n_hid,RES=RC_predict)
        self.graph_loss=0
        self.G = torch.Tensor(G).half().cuda()

    def forward(self,exp_mat):
        with torch.amp.autocast(device_type='cuda'):
            # Get top k genes for each patient
              # Number of top genes to select
            topk_values, topk_indices = torch.topk(exp_mat, self.k, dim=1)
            # Create new expression matrix with only top k genes
            exp_mat = torch.gather(exp_mat, 1, topk_indices)
            x_g1 = self.linear(exp_mat.unsqueeze(-1).float())
            for i, layers in enumerate(self.hgc_encoder):

                sub_H = self.H0[0][:, topk_indices]
                # Normalize sub-adjacency matrix
                xe = sub_H.matmul(x_g1)
                
                xe = layers(x_g1, xe)#  TODO check
            
            self.graph_loss = xe/torch.norm(xe, 2, -1, keepdim=True)
            self.graph_loss = ((2-2* torch.matmul(self.graph_loss, self.graph_loss.transpose(1, 2)) )*self.G).sum()/self.G.shape[0]

            x_g1 = self.linear_pooling(xe)
            if self.AGG =="noAGG":
                return self.Predict_Layers(x_g1)
            if self.AGG == "jump":
                return self.Predict_Layers(torch.cat([x_g1,exp_mat],dim=1))
            
            x_g2=self.MLPEmbedding(exp_mat)
            if self.AGG == "plus":
                x_g = x_g1+x_g2
            elif self.AGG == "cat":
                x_g =  torch.cat([x_g1,x_g2],dim=1)
            elif self.AGG == "mean":
                x_g = (x_g1+x_g2)/2

            return self.Predict_Layers(x_g)

    def fit_early_stop(self, optimizer,scheduler, num_epochs,logger,batch_size = 32,eval_full=True,metric="ci",loss_dict={"a":1,"b":1,"c1":1,"c2":0,"d":1}):
        best_model_wts = copy.deepcopy(self.state_dict())
        best_val_metric = -np.inf
        train_loader = data.DataLoader(self.data_train_batch,batch_size = batch_size ,shuffle = True)
        if eval_full:
            batch_size = len(self.data_eval)
        eval_loader = data.DataLoader(self.data_eval,batch_size = batch_size ,shuffle = False)
        for epoch in range(num_epochs+1):
            optimizer = self.train_mini_batch(train_loader,optimizer,loss_dict)

            epoch_loss,epoch_ci = self.get_loss_ci(eval_loader,self.data_eval,False,True,True,self.mask1_va,self.mask2_va,loss_dict)
            epoch_loss=epoch_loss.detach().cpu().numpy()
            self.report['epoch'].append(epoch)
            self.report['val_loss'].append(epoch_loss)
            self.report['val_ci'].append(epoch_ci)
            scheduler.step(epoch_loss)
            epoch_metric = (-epoch_loss) if metric == "loss" else epoch_ci
            if epoch_metric > best_val_metric:
                best_val_metric = epoch_metric
                best_epoch = epoch + 1
                best_val_ci = epoch_ci
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                # if Valid best update Test
                test_ci = self.pred_ci(self.data_test)
                logger.info(f"{30*'-'}")
                
                logger.info(f'Epoch{best_epoch} Updating val ci: {best_val_ci:4f}; test ci: {test_ci:4f}')
                logger.info(f'Updating val loss: {best_val_loss:4f}')
                
        # save the metric,loss,weight of training model into ckpt file
        if self.fn is not None:
            torch.save({'epoch': best_epoch,
                        'state_dict': self.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_eval_ci': best_val_ci,
                        'best_eval_loss': best_val_loss,
                        'test_ci': test_ci,
                        'report': self.report,
                        }, f'{self.fn}.ckpt')

        # plt loss fig
        fig = plt.figure()
        # plt.plot(self.report['train_loss'], label='Train loss')
        plt.plot(self.report['val_loss'], label='Val loss')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
        plt.close()
        
        # plt ci fig
        fig = plt.figure()
        # plt.plot(self.report['train_ci'], label=f'Train ci')
        plt.plot(self.report['val_ci'], label=f'Val ci')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_ci.pdf', bbox_inches='tight')
        plt.close()

        self.load_state_dict(best_model_wts)
        return self


    def fit(self, optimizer, num_epochs,logger,batch_size = 32,eval_full=True,loss_dict={"a":1,"b":1,"c1":0,"c2":0,"d":0}):
        best_model_wts = copy.deepcopy(self.state_dict())
        train_loader = data.DataLoader(self.data_train_batch,batch_size = batch_size ,shuffle = True)
        if eval_full:
            batch_size = len(self.data_eval)
        eval_loader = data.DataLoader(self.data_eval,batch_size = batch_size ,shuffle = False)
        for epoch in range(num_epochs+1):
            optimizer = self.train_mini_batch(train_loader,optimizer,loss_dict)

            if epoch % 10 == 0:#eval
                self.eval()
                logger.info('-' * 20)
                logger.info(f'Epoch {epoch}/{num_epochs}')
                eval_loss,eval_ci = self.get_loss_ci(eval_loader,self.data_eval,False,True,True,self.mask1_va,self.mask2_va,loss_dict)
                eval_loss=eval_loss.detach().cpu().numpy()
                self.report['epoch'].append(epoch)
                self.report['eval_loss'].append(eval_loss)
                self.report['eval_ci'].append(eval_ci)
                logger.info(f'Updating eval ci: {eval_ci:4f}; eval loss: {eval_loss:4f}')
                best_model_wts = copy.deepcopy(self.state_dict())
                best_eval_ci = eval_ci
                best_eval_loss = eval_loss
            
        test_ci = self.pred_ci(self.data_test)
        logger.info(f'Final test ci: {test_ci:4f}')
        if self.fn is not None:
            torch.save({
                        'state_dict': self.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_eval_ci': best_eval_ci,
                        'best_eval_loss': best_eval_loss,
                        'test_ci':test_ci,
                        'report': self.report,
                        }, f'{self.fn}.ckpt')

        # plt loss fig
        fig = plt.figure()
        # plt.plot(self.report['train_loss'], label='Train loss')
        plt.plot(self.report['eval_loss'], label='eval loss')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
        plt.close()
        
        # plt ci fig
        fig = plt.figure()
        # plt.plot(self.report['train_ci'], label=f'Train ci')
        plt.plot(self.report['eval_ci'], label=f'eval ci')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_ci.pdf', bbox_inches='tight')
        plt.close()

        self.load_state_dict(best_model_wts)
        return self

    def predict(self,sgs):
        self.eval()
        outputs = self.forward(sgs)
        return outputs
    
    def predict_attention_mod(self, exp_mat, atn):# need to modify
        self.eval()
        self.hgc_encoder[0].hgat_layers[0].attention_node.copy_(atn)
        outputs = self.forward(exp_mat)
        return outputs
    def show_report(self):
        return pd.DataFrame(self.report)
    
    def create_mask1(self,time,label):# time,label,
        time,label = torch.unsqueeze(time, dim=1),torch.unsqueeze(label, dim=1)

        mask1 = np.zeros([np.shape(time)[0], self.t]) # for the first loss function
        for i in range(np.shape(time)[0]):
            if label[i,0] != 0:  #not censored
                mask1[i,int(time[i,0])] = 1 # 死亡时间取值
            else: #label[i,2]==0: censored
                mask1[i,int(time[i,0]+1):] = 1# 删失时间后的所有时间取值
        return torch.from_numpy(mask1).float()
    #mask2
    def create_mask2(self,time):
        time = torch.unsqueeze(time, dim=1)
        meas_time = -1
        mask2 = np.zeros([np.shape(time)[0], self.t]) # for the first loss function
        if np.shape(meas_time):  #lonogitudinal measurements
            for i in range(np.shape(time)[0]):
                t1 = int(meas_time[i, 0]) # last measurement time
                t2 = int(time[i, 0]) # censoring/event time
                mask2[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
        else:                    #single measurement
            for i in range(np.shape(time)[0]):
                t = int(time[i, 0]) # censoring/event time
                mask2[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
        return torch.from_numpy(mask2).float()
    def create_mask(self,time,label):
        return (self.create_mask1(time,label).cuda(),self.create_mask2(time).cuda())

    def nll(self,out, I_1, mask1):
        # I_1 = torch.sign(k)
        tmp = torch.sum(mask1 * out, 1)
        lz = I_1 * torch.log(tmp + 1e-08)
        lcen = (1. - I_1) * torch.log(tmp + 1e-08)
        return -torch.mean(lz + 1.0*lcen)
        
    def rank_loss(self, out, e, t, mask2):#t:time
        sigma1 = 0.1
        one_vector = torch.unsqueeze(torch.ones_like(t, dtype = torch.float32),dim=1).cuda()
        I_2 = torch.eq(e, 1).float()
        I_2 = torch.diag_embed(torch.squeeze(I_2))
        # logger.info(out[0:,e:e+1,0:].size())
        # tmp_e = torch.reshape(out[0:,0:,0:], (-1, self.t))

        R = torch.mm(out, mask2.t())

        diag_R = torch.reshape(torch.diagonal(R), (-1, 1))
        R = torch.mm(one_vector, diag_R.t()) - R
        R = R.t()
        t = t.unsqueeze(-1)
        T = F.relu(torch.sign(torch.mm(one_vector, t.t().float()) - torch.mm(t.float(),one_vector.t())))# exp-relu

        T = torch.mm(I_2, T)
        temp_eta = torch.mean(T * torch.exp(-R/sigma1), 1, keepdim = True)

        return torch.sum(temp_eta)
    def calibration_loss(self, out,e, mask2): #Calibration loss
        I_2 = torch.eq(e.view(e.shape[0],1), 1).float()
        r = torch.sum(out * mask2, 0)
        eta = torch.mean((r - I_2)**2, 1, keepdim = True)# torch.Size([62])-torch.Size([64, 1])
        return torch.sum(eta)
    
    def deephit_loss(self,out,e,t,mask1,mask2,print_prop=False,beta=1,alpha=1,gamma=1,logger=None):
        with torch.amp.autocast(device_type='cuda'):
            loss_nll2 = self.nll(out,e,mask1)
            loss_rank = self.rank_loss(out,e,t,mask2)
            loss_cali = self.calibration_loss(out,e,mask2)

        if print_prop:
            logger.info("proportion:",(loss_nll2/loss_rank).item())
        return (alpha*loss_nll2) + (beta*loss_rank) + (gamma*loss_cali)
    
    def pred_ci(self,data_test,):
        data_loader = data.DataLoader(data_test,batch_size=data_test.shape[0],shuffle = False)
        
        _,ci = self.get_loss_ci(
            data_loader,data_test,train=False,ci=True,loss=False,\
            mask1=None,mask2=None,loss_dict=None)
        return ci
    
    def get_triple_loss(self,pred,s_time,e,mask1,mask2,sigma=2,):
        s_time = s_time.float().unsqueeze(-1)
        e=e.unsqueeze(-1)
        risk = get_risk(pred)
        a,b,c = self.loss_w["a"],self.loss_w["b"],self.loss_w["c"]
        like = likelihood_loss(pred,mask1)
        if self.self_rank:
            rank = self_ranking_loss_weight(risk,s_time,e,0.5,0.1)
        else:   
            rank = ranking_loss(risk,s_time,e,sigma)
        cali = calibration_loss(pred,e,mask1,mask2,self.loss_w['nbins'])
        return (a*like+b*rank+c*cali)
    
    def get_loss_ci(self,data_loader,data,train,ci,loss,mask1,mask2,loss_dict):
        if train:
            self.train
        else:
            self.eval()
        for i,data_phase in enumerate(data_loader):
            X_phase_batch = data_phase[:,:-2]
            PMF_batch = self.forward(X_phase_batch)
            if i==0:
                preds = PMF_batch
            else:
                preds = torch.cat((preds,PMF_batch),dim=0)
        y = data[:,-2:]
        s_time, e = y[:, 0].long(), y[:, 1]
        if loss:
            # loss = self.deephit_loss(out=preds,e=e,t=s_time,alpha = 1,beta=1,gamma=0,mask1=mask1,mask2=mask2)
            if self.triple_loss:
                loss=self.get_triple_loss(pred=preds,s_time=s_time,e=e,mask1=mask1,mask2=mask2)#,loss_w=loss_dict)
            else:
                loss = self.get_loss(risk=preds,e=e,t=s_time,mask1=mask1,mask2=mask2,loss_dict=loss_dict)
            loss += self.glr * self.graph_loss
            # print(self.glr * self.graph_loss)
        else:
            loss = None
        if ci:
            surv = torch.clamp(1.-torch.cumsum(preds,dim=-1),0,1)
            ci =get_ci_td(s_time=s_time,cens=1-e,f=surv)
        else:
            ci=0
        return loss,ci

    def train_mini_batch(self,train_loader,optimizer,loss_dict):
        self.train()
        for i,data_phase in enumerate(train_loader):
            mask_batch,data_phase = data_phase[:,-2*self.t:],data_phase[:,:-2*self.t] 
            # y_phase_batch = data_phase[:,-2:]
            # X_phase_batch = data_phase[:,:-2]
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                PMF_train_batch = self.forward(data_phase[:,:-2])
                s_time, e = data_phase[:, -2].long(), data_phase[:,-1]
                # surv mask1 == mask1??
                # ranking_loss(risk,s_time.float().unsqueeze(-1),e.unsqueeze(-1),sigma)
                # likelihood_loss(PMF_train_batch,mask_batch[:,:self.t])
                # calibration_loss(PMF_train_batch,e.unsqueeze(-1),mask_batch[:,:self.t],mask_batch[:,self.t:],6)
                if self.triple_loss:
                    batch_loss = self.get_triple_loss(pred=PMF_train_batch,s_time=s_time,e=e,mask1=mask_batch[:,:self.t],mask2=mask_batch[:,self.t:],sigma=2)
                else:
                    batch_loss = self.get_loss(risk=PMF_train_batch,e=e,t=s_time,mask1=mask_batch[:,:self.t],mask2=mask_batch[:,self.t:],loss_dict=loss_dict)
                # print(self.glr * self.graph_loss)
                batch_loss+=self.glr*self.graph_loss
                # print(hg_loss)
                batch_loss.backward()
                optimizer.step()
        return optimizer
    def get_loss(self,risk,e,t,mask1,mask2,loss_dict={"a":1,"b":1,"c1":0,"c2":0,"d":0}):
        a,b,c1,c2,d=loss_dict["a"],loss_dict["b"],loss_dict["c1"],loss_dict["c2"],loss_dict["d"]
        surv_batch = torch.clamp(1.-torch.cumsum(risk,dim=-1),0,1)
        log_cens_z_rank = self.deephit_loss(out=risk,e=e,t=t,alpha =a,beta=b,gamma=c1,mask1=mask1,mask2=mask2)
        log_uncens = get_RPSorLogUncens_loss(surv_batch,t,1-e,self.t,do_brier=False,RPS_norm=False)
        # loss_RPS = get_RPSorLogUncens_loss(surv_batch,t,1-e,self.t,do_brier=True,RPS_norm=True)
        loss = log_cens_z_rank+d*log_uncens
        return loss
    
