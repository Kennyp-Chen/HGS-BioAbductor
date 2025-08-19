'''
Vanilla Layers Neural Networks version survival analysis.
'''
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch import nn
import math
import os
import sys
from torch import nn
import utils.hg_ops as hgo
import torch.nn.functional as F
import torch
import time
import copy
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
import numpy as np
from collections import defaultdict
# from utils.torch_utils import negative_log_likelihood
from utils.data_utils import get_ci
from torch.autograd import Variable
from torch.utils import data
from utils.survfunc_utils import *
from sksurv.linear_model.coxph import BreslowEstimator
class DRSA(nn.Module):
    def __init__(self,dim_in,t_obs,lstm_layers,seed,fn,data_train, data_valid, data_test,dropout=0): 
        super(DRSA, self).__init__()
        if seed is not None:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        dim_in=data_train.shape[1]-2
        l_hid = 2*dim_in // 3
        self.data_train, self.data_valid, self.data_test = data_train, data_valid, data_test
        self.hidden_size = l_hid
        self.num_layers = lstm_layers
        self.batch_norm = nn.BatchNorm1d(dim_in)     
        self.fc = nn.Linear(l_hid, 1)
        self.lstm = nn.LSTM(dim_in+1, hidden_size = l_hid, batch_first = True,num_layers=self.num_layers,dropout=dropout).cuda()
        self.sig = nn.Sigmoid()         
        self.t = int(t_obs)
        self.fn = fn
        self.report = defaultdict(list)

    def forward(self,x):
        
        cov_size = x.shape[1]+1
        cov = self.batch_norm(x) # 618,150 (batch,fea_num)
        cov = cov.repeat(1, self.t).view(-1, self.t, cov_size-1)# torch.Size([618, 792, 150]) 直接repeat
        time = torch.arange(1, self.t+1, 1).repeat(cov.size(0)).view(-1, self.t, 1).float().cuda()# torch.Size([618, 792, 1])离散时间*样本数
        x = torch.cat((cov, time), dim = 2).cuda()# torch.Size([618, 792, 151])#加上的维度是离散时间（相当于是加上时间作为协变量）
        batch_size = x.size(0)#618
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).float().cuda(), requires_grad = False)# 不更新梯度参数？
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).float().cuda(), requires_grad = False)
        outs, hidden = self.lstm(x, (h0,c0))# 初始的状态和记忆细胞 outs: torch.Size([32, 62, 100])
        outs = outs.contiguous().view(batch_size * self.t, self.hidden_size)# tensor的内存连续存储，返回一个contiguous copy；# outputs.shape = (618*183,200)
        pdf = self.sig(self.fc(outs)).view(batch_size, self.t) # 32(batch),62 每个样本在每个事件段的h？
        return pdf
 
    def fit(self, batch_size,optimizer, scheduler, num_epochs, print_freq,metric,logger,last_epoch=False,valid_full=False,loss_dict={"do_l2":0,"do_nll":1,"do_brier":0,"a":0.5,"RPS_norm":0}):
            best_model_wts = copy.deepcopy(self.state_dict())
            best_val_metric = -np.inf
            train_loader = data.DataLoader(self.data_train,batch_size = batch_size ,shuffle = True)
            if valid_full:
                batch_size = len(self.data_valid)
            valid_loader = data.DataLoader(self.data_valid,batch_size = batch_size ,shuffle = False)
            test_loader = data.DataLoader(self.data_test,batch_size = batch_size ,shuffle = False)
            for epoch in range(num_epochs):
                if epoch % print_freq == 0:
                    logger.info('-' * 20)
                    logger.info(f'Epoch {epoch}/{num_epochs - 1}')

                for phase in ['train', 'val']:
                    if phase == 'train':
                        optimizer = self.train_mini_batch(train_loader,optimizer,loss_dict)
                    else:# valid
                        if not last_epoch:
                            epoch_loss,epoch_ci = self.get_loss_ci(valid_loader,self.data_valid,False,True,loss_dict)
                            epoch_loss=epoch_loss.detach().cpu().numpy()

                        else:
                            epoch_loss,epoch_ci =0,0
                        self.report['epoch'].append(epoch)
                        self.report['val_ci'].append(epoch_ci)
                        self.report['val_loss'].append(epoch_loss)
                        scheduler.step(epoch_loss)
                        epoch_metric = (-epoch_loss) if metric == "loss" else epoch_ci
                        if epoch_metric > best_val_metric:
                            best_val_metric = epoch_metric
                            best_epoch = epoch + 1
                            best_val_ci = epoch_ci
                            best_val_loss = epoch_loss
                            best_model_wts = copy.deepcopy(self.state_dict())
                            if not last_epoch:                       
                                # TEST
                                test_loss,test_ci = self.get_loss_ci(test_loader,self.data_test,False,True,loss_dict)

                                logger.info(f'Updating val ci: {best_val_ci:4f}; test ci: {test_ci:4f}')
                                logger.info(f'Updating val loss: {best_val_loss:4f}; test loss: {test_loss:4f}')
                        if epoch % print_freq == 0:
                            logger.info(f'{phase} Loss: {epoch_loss:.4f} ci: {epoch_ci:.4f} ,')
            if last_epoch:
                test_loss,test_ci = self.get_loss_ci(test_loader,self.data_test,False,True,loss_dict)
                best_epoch=-1
                logger.info(f'Updating val ci: {best_val_ci:4f}; test ci: {test_ci:4f}')
                logger.info(f'Updating val loss: {best_val_loss:4f}; test loss: {test_loss:4f}')
            # save the metric,loss,weight of training model into ckpt file
            if self.fn is not None:
                torch.save({'epoch': best_epoch,
                            'state_dict': self.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_train_ci': 0,
                            'best_train_loss': 0,
                            'best_val_ci': best_val_ci,
                            'best_val_loss': best_val_loss,
                            'test_ci': test_ci,
                            'test_loss': test_loss,
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

    def predict(self,sgs):
        self.eval()
        outputs,_ = self.forward(sgs)
        return outputs,_

    def show_report(self):
        return pd.DataFrame(self.report)
    
    def get_loss_ci(self,data_loader,data,train,if_ci,loss_dict):
        if train:
            self.train
        else:
            self.eval()
        for i,data_phase in enumerate(data_loader):
            X_phase_batch = data_phase[:,:-2]
            PDF_batch = self.forward(X_phase_batch)
            if i ==0:
                preds = PDF_batch
            else:
                preds = torch.cat((preds,PDF_batch),dim=0)
        y = data[:,-2:]
        s_time, e = y[:, 0].long(), y[:, 1]

        loss,ci = get_DRS_loss_ci(preds,loss_dict["do_l2"],loss_dict["do_brier"],loss_dict["RPS_norm"],
        loss_dict["do_nll"],loss_dict["a"],s_time,e,self.t,if_ci)

        return loss,ci

    def train_mini_batch(self,train_loader,optimizer,loss_dict):
        self.train()
        for i,data_phase in enumerate(train_loader):
            y_phase_batch = data_phase[:,-2:]
            X_phase_batch = data_phase[:,:-2]
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                PDF_train_batch = self.forward(X_phase_batch)
                s_time, e = y_phase_batch[:, 0].long(), y_phase_batch[:, 1]
                batch_loss,ci = get_DRS_loss_ci(PDF_train_batch,loss_dict["do_l2"],loss_dict["do_brier"],
                loss_dict["RPS_norm"],loss_dict["do_nll"],loss_dict["a"],s_time,e,self.t,False)
                batch_loss.backward()
                optimizer.step()
        return optimizer
    

class FC_layers(nn.Module):
    '''DeepSurv or DeepDiscreteTimeSurv'''
    def __init__(self,dim_in,dropout=0.5,ACT = 'LeakyReLU',layer_nodes=[],dim_out=1,RES=False):
        super(FC_layers, self).__init__()   
        modules = []
        modules_res = []

        act_type = getattr(nn, ACT)
        self.RES=RES
        if RES:
            modules_res.append(nn.Linear(dim_in,dim_in))
            modules_res.append(act_type())
            modules_res.append(nn.Dropout(dropout))
            modules.append(nn.Linear(dim_in*2,dim_in))
            modules.append(act_type())
            modules.append(nn.Dropout(dropout))
        if len(layer_nodes)==0:
            modules.append(nn.Linear(dim_in,dim_out))
        else:
            modules.append(nn.Linear(dim_in,layer_nodes[0]))
            modules.append(nn.BatchNorm1d(layer_nodes[0]))
            modules.append(act_type())
            modules.append(nn.Dropout(dropout))
            for l in range(len(layer_nodes)):
                if l<len(layer_nodes)-1:
                    modules.append(nn.Linear(layer_nodes[l],layer_nodes[l+1]))
                    modules.append(nn.BatchNorm1d(layer_nodes[l+1]))
                    modules.append(act_type())
                    modules.append(nn.Dropout(dropout))
                else:
                    modules.append(nn.Linear(layer_nodes[-1],dim_out))
        self.RES_NNs = nn.Sequential(*modules_res)
        self.NNs = nn.Sequential(*modules)
    def forward(self,x):
        with autocast():
            if self.RES:
                h = self.RES_NNs(x)
                x = torch.cat((x, h),1)
            y = self.NNs(x)# log risk ratio
            if y.shape[-1] > 1 :
                y = F.softmax(y,dim=1)
            # if torch.isinf(y).sum()>0:
            #     inf_index = torch.nonzero(torch.isinf(y))
            #     y[inf_index]=y.max()*2
            return y
         
class FC_layers1_2(nn.Module):
    '''DeepSurv or DeepDiscreteTimeSurv'''
    def __init__(self,dim_in,dropout=0.5,layer_nodes=[],dim_out=1,RES=False):# if dim_out = 1 then DeepSurv
        super(FC_layers1_2, self).__init__()   
        self.RES=RES
        modules_res=[]
        modules=[]
        if len(layer_nodes)==1:
            modules.append(nn.Linear(dim_in,layer_nodes[0]))
            modules.append(nn.BatchNorm1d(layer_nodes[0]))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(layer_nodes[0],dim_out))
        elif len(layer_nodes)==2:
            modules.append(nn.Linear(dim_in,layer_nodes[0]))
            modules.append(nn.BatchNorm1d(layer_nodes[0]))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(layer_nodes[0],layer_nodes[1]))
            
            modules.append(nn.Linear(dim_in,layer_nodes[1]))
            modules.append(nn.BatchNorm1d(layer_nodes[1]))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(layer_nodes[1],dim_out))
        self.RES_NNs = nn.Sequential(*modules_res)
        self.NNs = nn.Sequential(*modules)
    def forward(self,x):
        with autocast():
            y = self.NNs(x)
            y = F.softmax(y,dim=1)
            return y
         

class FC_Res_layers(nn.Module):
    '''DeepHit'''
    def __init__(self,dim_in,dropout=0.5,ACT = 'LeakyReLU',NumLayers=0,dim_out=62):
        super(FC_Res_layers, self).__init__()  
        # DeepHit The layers order:x-fc-bn-act-dp-x
        plain=[]
        modules=[]
        ACT = eval(f"nn.{ACT}()")
        plain.append(nn.Linear(dim_in,200))#1
        plain.append(nn.BatchNorm1d(200))
        plain.append(ACT)
        plain.append(nn.Dropout(dropout))

        modules.append(nn.Linear(dim_in+200,100))#2 RESNET
        modules.append(nn.BatchNorm1d(100))
        modules.append(ACT)
        modules.append(nn.Dropout(dropout))
        # add layers
        for i in range(NumLayers):
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(100,100))
            modules.append(nn.BatchNorm1d(100))
            modules.append(ACT)
            modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(100,dim_out))#n+3
            
        self.plain = nn.Sequential(*plain)
        self.decreaseNNs = nn.Sequential(*modules)

    def forward(self,x):
        with autocast():
            shared_out = self.plain(x)
            h = torch.cat((x, shared_out),1)# 2*dim_in
            x = self.decreaseNNs(h)
            # shared_out = self.left(x)
            # h = torch.cat((x, shared_out),1)
            # x = self.right(h)
            PMF = F.softmax(x,dim=-1)#([pats,t_obs])
        return PMF
    
class DeepSurv(nn.Module):
    def __init__(self,data_train, data_valid, data_test, middle_layers_nodes:list,nn_seed,ACT,dropout,fn):
        super(DeepSurv, self).__init__()  
        if nn_seed is not None:
            torch.manual_seed(nn_seed)
            torch.cuda.manual_seed(nn_seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        self.fn = fn
        self.report = defaultdict(list)
        self.Layers = FC_layers(dim_in=data_train.shape[1]-2,dropout=dropout,ACT=ACT,layer_nodes=middle_layers_nodes,dim_out=1)
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test  = data_test
    def forward(self,x):
        LRR = self.Layers(x)
        return LRR
    
    def fit(self,batch_size, optimizer, scheduler, num_epochs, print_freq,judge_metric,logger,using_last=False):
        best_model_wts = copy.deepcopy(self.state_dict())
        best_val_metric = -np.inf
        val_batch = len(self.data_valid)
        train_batch = batch_size
        train_loader = data.DataLoader(self.data_train,batch_size = train_batch ,shuffle = True)
        valid_loader = data.DataLoader(self.data_valid,batch_size = val_batch ,shuffle = False)
        for epoch in range(num_epochs):

            if epoch % print_freq == 0:
                logger.info('-' * 20)
                logger.info(f'Epoch {epoch}/{num_epochs - 1}')
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()
                    if epoch == 0 :
                        X_test,y_test = self.data_test[:,:-2],self.data_test[:,-2:]
                        LRR_test = self.predict(X_test,)# test expression matrix
                        s_time_test, e_test = y_test[:,-2], y_test[:, -1]
                        test_loss = negative_log_likelihood(LRR_test,y_test)
                        last_ci = get_ci(t=s_time_test,e=e_test,r=LRR_test)
                        logger.info(f"epoch: {epoch} test_ci: {last_ci:.4f}")
                for i,data_phase in enumerate(train_loader) if phase == 'train' else enumerate(valid_loader):       
                    y_phase_batch = data_phase[:,-2:]
                    X_phase_batch = data_phase[:,:-2]
                    optimizer.zero_grad()  
                    with torch.set_grad_enabled(phase == 'train'):
                        LRR_batch = self.forward(X_phase_batch)
                        s_time, e = y_phase_batch[:, 0], y_phase_batch[:, 1]
                        # # ###
                        # candidate=(1/torch.cumsum(torch.exp(LRR_batch.detach()),dim=0)).reshape(-1)*e
                        # multiple_times = torch.exp(LRR_batch)
                        # base = s_time
                        # H0_b = torch.zeros([batch_size, 1])
                        # for i in range(batch_size):
                        #     bid = base[i]
                        #     H0_b[i][0] = torch.sum(candidate[base <= bid])
                        # S0_b = torch.exp(-H0_b)
                        # S_b = torch.pow(S0_b, multiple_times.cpu())
                        # W_b=1-S_b
                        # auc = roc_auc_score(e.detach().cpu(), W_b.detach())
                        # #sort time
                        # pred = LRR_batch.detach()
                        # preds = pred[base.sort()[1]]
                        # events = e[base.sort()[1]]
                        # time = base.sort()[0]
                        # c_s = (1/torch.cumsum(torch.exp(preds),dim=0)).reshape(-1)*events
                        # mt=torch.exp(pred)
                        # H0 = torch.zeros([batch_size, 1])
                        # for i in range(batch_size):
                        #     t = time[i]
                        #     H0[i][0] = torch.sum(candidate[time <= t])
                        # S0 = torch.exp(-H0)
                        # S = torch.pow(S0,mt.cpu())
                        # W=1-S
                        # auc_sorted = roc_auc_score(events.detach().cpu(), W.detach())
                        ###
                        batch_loss = negative_log_likelihood(LRR_batch,y_phase_batch)
                        if phase == "val":
                            try:
                                batch_ci = get_ci(t=s_time,e=e,r=LRR_batch)
                            except:
                                batch_ci=0.0
                            # logger.info(f"val ci:{batch_ci}")
                            # batch_citd = get_Coxph_citd(LRR=LRR_batch ,s_time=s_time,event=e)
                            # logger.info(f"val citd:{batch_citd}")
                            # surv = self.get_Cph_citd(LRR_batch)
                            # batch_citda=get_ci_td(s_time=s_time,cens=1-e,f=surv)
                            # logger.info(f"val citda:{batch_citda}")

                        else:
                            batch_ci = 0
                            batch_loss.backward()
                            optimizer.step()
            
                self.report['epoch'].append(epoch)
                if phase == 'train':
                    x_train = self.data_train[:,:-2]
                    y_train = self.data_train[:,-2:]
                    s_time_train, e_train = y_train[:, 0], y_train[:, 1]
                    LRR_train = self.predict(x_train,)# test expression matrix
                    epoch_loss = negative_log_likelihood(LRR_train,y_train)

                    try:
                        epoch_ci = get_ci(t=s_time_train,e=e_train,r=LRR_train)
                    except:
                        epoch_ci=0.0

                    self.report['train_loss'].append(epoch_loss.detach().cpu().numpy())
                    self.report['train_ci'].append(epoch_ci)
                    train_ci = epoch_ci
                    train_loss = epoch_loss
                else:
                    epoch_loss = batch_loss
                    epoch_ci = batch_ci
                    self.report['val_loss'].append(epoch_loss.detach().cpu().numpy())
                    self.report['val_ci'].append(epoch_ci)
                    scheduler.step(epoch_loss)

                if (epoch+1) % print_freq == 0:#print_freq
                    X_test = self.data_test[:,:-2]
                    y_test = self.data_test[:,-2:]
                    LRR_test = self.predict(X_test,)# test expression matrix
                    s_time_test, e_test = y_test[:,-2], y_test[:, -1]
                    test_loss = negative_log_likelihood(LRR_test,y_test)
                    try:
                        last_ci = get_ci(t=s_time_test,e=e_test,r=LRR_test)
                        last_epoch =epoch
                    except:
                        pass
                    logger.info(f'epoch{epoch} {phase} Loss: {epoch_loss:.4f} ci: {epoch_ci:.4f} ')
                    logger.info(f'last_epoch: {last_epoch} Test ci: {last_ci:.4f} ')

                # if loss using -loss to choose the largest -loss's model
                epoch_metric = (-epoch_loss) if judge_metric == "loss" else epoch_ci
                if phase == 'val' and epoch_metric > best_val_metric:
                    best_val_metric = epoch_metric
                    best_epoch = epoch + 1
                    best_train_ci = train_ci
                    best_train_loss = train_loss#.detach().cpu().numpy()
                    best_val_ci = epoch_ci
                    best_val_loss = epoch_loss.detach().cpu().numpy()
                    best_model_wts = copy.deepcopy(self.state_dict()) 
                    # if updating by val, then test 
                    X_test = self.data_test[:,:-2]
                    y_test = self.data_test[:,-2:]
                    LRR_test = self.predict(X_test,)# test expression matrix
                    s_time_test, e_test = y_test[:,-2], y_test[:, -1]
                    test_loss = negative_log_likelihood(LRR_test,y_test)

                    
                    # logger.info(30*"~")
                    try:
                        test_ci = get_ci(t=s_time_test,e=e_test,r=LRR_test)
                    except:
                        test_ci = 0.0                    
                    # logger.info(f"test ci:{test_ci}")
                    # test_citd = get_Coxph_citd(LRR=LRR_test ,s_time=s_time_test,event=e_test)
                    # logger.info(f"test citd:{test_citd}")


                    logger.info(f'Updating by {judge_metric}. Val ci: {best_val_ci:4f}; test ci: {test_ci:4f}; val loss: {best_val_loss:4f}; test loss: {test_loss:4f}')
        if using_last:
            test_ci = last_ci
            best_epoch = last_epoch  
            logger.info(f'Updating by last_epoch. test ci: {test_ci:4f}; test loss: {test_loss:4f}')
                        
        logger.info(f'\n{num_epochs} epochs training complete')

        if self.fn is not None:
            torch.save({'epoch': best_epoch,
                        'state_dict': self.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_train_ci': best_train_ci,
                        'best_train_loss': best_train_loss,
                        'best_val_ci': best_val_ci,
                        'best_val_loss': best_val_loss,
                        'test_ci': test_ci,
                        'test_loss': test_loss,
                        'report': self.report,
                        }, f'{self.fn}.ckpt')

        # plt loss fig
        fig = plt.figure()
        plt.plot(self.report['train_loss'], label='Train loss')
        plt.plot(self.report['val_loss'], label='Val loss')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
        plt.close()
        
        # plt ci fig
        fig = plt.figure()
        plt.plot(self.report['train_ci'], label=f'Train ci')
        plt.plot(self.report['val_ci'], label=f'Val ci')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_ci.pdf', bbox_inches='tight')
        plt.close()

        self.load_state_dict(best_model_wts)
        return self

    def predict(self, exp_mat):# need to modify
        self.eval()
        outputs = self.forward(exp_mat)
        return outputs

    def show_report(self):
        return pd.DataFrame(self.report)
    
    def get_Cph_citd(self,predictor):
        '''
        Input cox ph beta@x, output surv_prob at times.

        Parameters
        ----------
        prediction : tensor, shape=([n_samples,])
            Predicted risk scores.
            beta@x of Cox PH model.

        return : tensor, shape([pats,times])
        '''
        predictor = predictor.detach().cpu().numpy()
        data = torch.concat([self.data_train,self.data_test,self.data_valid],0)
        preds_all = self.predict(data[:,:-2])   
        num_pats = predictor.size
        be = BreslowEstimator()
        # fit all samples so we can get all time point results
        be.fit(linear_predictor=preds_all.detach().cpu().numpy(),event=data[:,-1].cpu().numpy(),time=data[:,-2].cpu().numpy())
        surv_f = be.get_survival_function(predictor)
        num_time = be.unique_times_.size
        surv_prob = torch.empty([num_pats,num_time])
        for i in range(num_pats):
            surv_prob[i]=torch.tensor(surv_f[i].y)
        return surv_prob#, torch.tensor(be.unique_times_)

class DeepPMFSurv(nn.Module):
    def __init__(self,data_train, data_valid, data_test, middle_layers_nodes:list,nn_seed,t_obs,ACT,dropout,fn,if_RES):
        super(DeepPMFSurv, self).__init__()  
        if nn_seed is not None:
            torch.manual_seed(nn_seed)
            torch.cuda.manual_seed(nn_seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        self.fn = fn
        self.report = defaultdict(list)
        self.t = int(t_obs)
        self.Layers = FC_layers(dim_in=data_train.shape[1]-2,dropout=dropout,ACT=ACT,layer_nodes=middle_layers_nodes,dim_out=self.t,RES=if_RES)
        (self.mask1_tr,self.mask2_tr),(self.mask1_va,self.mask2_va),(self.mask1_te,self.mask2_te)=self.create_mask(data_train[:,-2],data_train[:,-1]),self.create_mask(data_valid[:,-2],data_valid[:,-1]),self.create_mask(data_test[:,-2],data_test[:,-1])
        self.data_train_batch = torch.concat([data_train,self.mask1_tr,self.mask2_tr],1)
        self.data_valid = torch.concat([data_valid,self.mask1_va,self.mask2_va],1)
        self.data_train = data_train
        self.data_test  = data_test
    def forward(self,x):
        PMF = self.Layers(x)
        return PMF
    def fit(self,batch_size, optimizer, scheduler, num_epochs, print_freq,judge_metric,logger):
        # time calcu
        # tb_writer = SummaryWriter(log_dir=tb_fn)
        since = time.time()
        best_model_wts = copy.deepcopy(self.state_dict())
        best_val_metric = -np.inf
        val_batch = len(self.data_valid)
        train_batch = batch_size # 32
        train_loader = data.DataLoader(self.data_train_batch,batch_size = train_batch ,shuffle = True)
        valid_loader = data.DataLoader(self.data_valid,batch_size = val_batch ,shuffle = True)
        for epoch in range(num_epochs):
            if epoch % print_freq == 0:
                logger.info('-' * 20)
                logger.info(f'Epoch {epoch}/{num_epochs - 1}')
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()  
                
                for i,data_phase in enumerate(train_loader) if phase == 'train' else enumerate(valid_loader):
                    # data_phase = self.data_train_batch if phase == 'train' else self.data_valid
                    
                    mask_batch,data_phase = data_phase[:,-2*self.t:],data_phase[:,:-2*self.t]
                    y_phase_batch = data_phase[:,-2:]
                    X_phase_batch = data_phase[:,:-2]
                    optimizer.zero_grad()  
                    with torch.set_grad_enabled(phase == 'train'):
                        out = self.forward(X_phase_batch)
                        s_time, e = y_phase_batch[:, 0], y_phase_batch[:, 1]
                        surv_batch = torch.clamp(1.-torch.cumsum(out,dim=-1),0,1)
                        batch_loss = self.deephit_loss(out=out,e=e,t=s_time,alpha = 1,beta=1,gamma=0,mask1=mask_batch[:,:self.t],mask2=mask_batch[:,self.t:])
                        if phase == "val":
                            batch_ci = get_ci_td(s_time=s_time,cens=1-e,f=surv_batch)
                        else:
                            batch_ci = 0

                            batch_loss.backward()
                            optimizer.step()
            
                self.report['epoch'].append(epoch)
                if phase == 'train':
                    # x_train = self.data_train[:,:-2]
                    # y_train = self.data_train[:,-2:]
                    # s_time_train, e_train = y_train[:, 0], y_train[:, 1]
                    # PMF_train = self.predict(x_train,)# test expression matrix
                    # surv_train = torch.clamp(1.-torch.cumsum(PMF_train,dim=-1),0,1)
                    # epoch_loss = self.deephit_loss(out=PMF_train,e=e_train,t=s_time_train,beta=1,mask1=self.mask1_tr,mask2=self.mask2_tr)
                    # epoch_ci = get_ci_td(s_time=s_time_train,cens=1-e_train,f=surv_train) 
                    # self.report['train_loss'].append(epoch_loss.detach().cpu().numpy())
                    # self.report['train_ci'].append(epoch_ci)
                    # train_ci = epoch_ci
                    # train_loss = epoch_loss
                    epoch_ci=train_ci=0
                    epoch_loss=train_loss=0
                else:
                    epoch_loss = batch_loss
                    epoch_ci = batch_ci
                    self.report['val_loss'].append(epoch_loss.detach().cpu().numpy())
                    self.report['val_ci'].append(epoch_ci)
                    scheduler.step(epoch_loss)

                if epoch % print_freq == 0:#print_freq
                    logger.info(f'epoch{epoch} {phase} Loss: {epoch_loss:.4f} ci: {epoch_ci:.4f} ')

                # if loss using -loss to choose the largest -loss's model
                epoch_metric = (-epoch_loss) if judge_metric == "loss" else epoch_ci
                if phase == 'val' and epoch_metric > best_val_metric:
                    best_val_metric = epoch_metric
                    best_epoch = epoch + 1
                    best_train_ci = train_ci
                    best_train_loss = train_loss#.detach().cpu().numpy()
                    best_val_ci = epoch_ci
                    best_val_loss = epoch_loss.detach().cpu().numpy()
                    best_model_wts = copy.deepcopy(self.state_dict()) 
                    # if updating by val, then test 
                    X_test = self.data_test[:,:-2]
                    y_test = self.data_test[:,-2:]
                    PMF_test = self.predict(X_test,)# test expression matrix
                    s_time_test, e_test = y_test[:,-2], y_test[:, -1]
                    test_loss = self.deephit_loss(out=PMF_test,e=e_test,t=s_time_test,alpha = 1,beta=1,gamma=0,mask1=self.mask1_te,mask2=self.mask2_te)
                    surv_test = torch.clamp(1.-torch.cumsum(PMF_test,dim=-1),0,1)
                    test_ci = get_ci_td(s_time=s_time_test,cens=1-e_test,f=surv_test)
                    logger.info(f'Updating by {judge_metric}. Val ci: {best_val_ci:4f}; test ci: {test_ci:4f}; val loss: {best_val_loss:4f}; test loss: {test_loss:4f}')
                    
        time_elapsed = time.time() - since
        logger.info(f'\n{num_epochs} epochs training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # save the metric,loss,weight of training model into ckpt file
        if self.fn is not None:
            torch.save({'epoch': best_epoch,
                        'state_dict': self.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_train_ci': best_train_ci,
                        'best_train_loss': best_train_loss,
                        'best_val_ci': best_val_ci,
                        'best_val_loss': best_val_loss,
                        'test_ci': test_ci,
                        'test_loss': test_loss,
                        'report': self.report,
                        }, f'{self.fn}.ckpt')

        # plt loss fig
        fig = plt.figure()
        plt.plot(self.report['train_loss'], label='Train loss')
        plt.plot(self.report['val_loss'], label='Val loss')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
        plt.close()
        
        # plt ci fig
        fig = plt.figure()
        plt.plot(self.report['train_ci'], label=f'Train ci')
        plt.plot(self.report['val_ci'], label=f'Val ci')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_ci.pdf', bbox_inches='tight')
        plt.close()

        self.load_state_dict(best_model_wts)
        return self

    def predict(self, exp_mat):# need to modify
        self.eval()
        outputs = self.forward(exp_mat)
        return outputs

    def show_report(self):
        return pd.DataFrame(self.report)
    
    #mask1
    def create_mask1(self,time,label):# time,label,
        time,label = torch.unsqueeze(time, dim=1),torch.unsqueeze(label, dim=1)

        mask1 = np.zeros([np.shape(time)[0], self.t]) # for the first loss function
        for i in range(np.shape(time)[0]):
            if label[i,0] != 0:  #not censored
                mask1[i,int(time[i,0])] = 1
            else: #label[i,2]==0: censored
                mask1[i,int(time[i,0]+1):] = 1
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

    def deephit_nll(self,out, I_1, mask1):
        # I_1 = torch.sign(k)
        tmp = torch.sum(mask1 * out, 1)
        lz = I_1 * torch.log(tmp + 1e-08)
        lcen = (1. - I_1) * torch.log(tmp + 1e-08)
        return -torch.mean(lz + 1.0*lcen)
        
    def rank_loss(self, out, e, t, mask2):#t:time
        sigma1 = 0.1
        eta = []
        one_vector = torch.unsqueeze(torch.ones_like(t, dtype = torch.float32),dim=1).cuda()
        I_2 = torch.eq(e, 1).float()
        I_2 = torch.diag_embed(torch.squeeze(I_2))
        # print(out[0:,e:e+1,0:].size())
        # tmp_e = torch.reshape(out[0:,0:,0:], (-1, self.t))

        R = torch.mm(out, mask2.t())

        diag_R = torch.reshape(torch.diagonal(R), (-1, 1))
        R = torch.mm(one_vector, diag_R.t()) - R
        R = R.t()
        t = t.unsqueeze(-1)
        T = F.relu(torch.sign(torch.mm(one_vector, t.t()) - torch.mm(t,one_vector.t())))

        T = torch.mm(I_2, T)
        temp_eta = torch.mean(T * torch.exp(-R/sigma1), 1, keepdim = True)

        return torch.sum(temp_eta)
    def calibration_loss(self, out,e, mask2): #Calibration loss
        I_2 = torch.eq(e.view(e.shape[0],1), 1).float()
        r = torch.sum(out * mask2, 0)
        eta = torch.mean((r - I_2)**2, 1, keepdim = True)# torch.Size([62])-torch.Size([64, 1])
        return torch.sum(eta)
    
    def deephit_loss(self,out,e,t,mask1,mask2,print_prop=False,beta=1,alpha=1,gamma=1):
        loss_nll2 = self.deephit_nll(out,e,mask1)
        loss_rank = self.rank_loss(out,e,t,mask2)
        loss_cali = self.calibration_loss(out,e,mask2)

        if print_prop:
            print("proportion:",(loss_nll2/loss_rank).item())
        return (alpha*loss_nll2) + (beta*loss_rank) + (gamma*loss_cali)

class DeepHit(nn.Module):
    def __init__(self,data_train, data_valid, data_test, middle_layers_nodes:list,nn_seed,t_obs,ACT,dropout,fn):
        super(DeepHit, self).__init__()  
        if nn_seed is not None:
            torch.manual_seed(nn_seed)
            torch.cuda.manual_seed(nn_seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        self.fn = fn
        self.report = defaultdict(list)
        self.t = int(t_obs)
        self.Layers = FC_Res_layers(dim_in=data_train.shape[1]-2,dropout=dropout,ACT=ACT,layer_nodes=middle_layers_nodes,dim_out=self.t)
        (self.mask1_tr,self.mask2_tr),(self.mask1_va,self.mask2_va),(self.mask1_te,self.mask2_te)=self.create_mask(data_train[:,-2],data_train[:,-1]),self.create_mask(data_valid[:,-2],data_valid[:,-1]),self.create_mask(data_test[:,-2],data_test[:,-1])
        self.data_train_batch = torch.concat([data_train,self.mask1_tr,self.mask2_tr],1)
        self.data_valid = torch.concat([data_valid,self.mask1_va,self.mask2_va],1)
        self.data_train = data_train
        self.data_test  = data_test
    def forward(self,x):
        PMF = self.Layers(x)
        return PMF
    
    def fit(self,batch_size, optimizer, scheduler, num_epochs, print_freq,judge_metric,logger,last_epoch=False):
        # time calcu
        # tb_writer = SummaryWriter(log_dir=tb_fn)
        since = time.time()
        best_model_wts = copy.deepcopy(self.state_dict())
        best_val_metric = -np.inf

        val_batch = len(self.data_valid)
        train_batch = batch_size # 32
        train_loader = data.DataLoader(self.data_train_batch,batch_size = train_batch ,shuffle = True)
        valid_loader = data.DataLoader(self.data_valid,batch_size = val_batch ,shuffle = True)
        for epoch in range(num_epochs):
            if epoch % print_freq == 0:
                logger.info('-' * 20)
                logger.info(f'Epoch {epoch}/{num_epochs - 1}')
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()  
                
                for i,data_phase in enumerate(train_loader) if phase == 'train' else enumerate(valid_loader):
                    # data_phase = self.data_train_batch if phase == 'train' else self.data_valid
                    
                    mask_batch,data_phase = data_phase[:,-2*self.t:],data_phase[:,:-2*self.t]
                    y_phase_batch = data_phase[:,-2:]
                    X_phase_batch = data_phase[:,:-2]
                    optimizer.zero_grad()  
                    with torch.set_grad_enabled(phase == 'train'):
                        out = self.forward(X_phase_batch)
                        s_time, e = y_phase_batch[:, 0], y_phase_batch[:, 1]
                        surv_batch = torch.clamp(1.-torch.cumsum(out,dim=-1),0,1)
                        batch_loss = self.deephit_loss(out=out,e=e,t=s_time,alpha = 1,beta=1,gamma=0,mask1=mask_batch[:,:self.t],mask2=mask_batch[:,self.t:])
                        if last_epoch:
                            batch_ci=0
                        else:        
                            if phase == "val":
                                batch_ci = get_ci_td(s_time=s_time,cens=1-e,f=surv_batch)
                            else:
                                batch_ci = 0
                        if phase=="train":
                            batch_loss.backward()
                            optimizer.step()
            
                self.report['epoch'].append(epoch)
                if phase == 'train':
                    epoch_ci=train_ci=0
                    epoch_loss=train_loss=0
                else:
                    epoch_loss = batch_loss
                    epoch_ci = batch_ci
                    self.report['val_loss'].append(epoch_loss.detach().cpu().numpy())
                    self.report['val_ci'].append(epoch_ci)
                    scheduler.step(epoch_loss)

                if epoch % print_freq == 0:#print_freq
                    logger.info(f'epoch{epoch} {phase} Loss: {epoch_loss:.4f} ci: {epoch_ci:.4f} ')

                # if loss using -loss to choose the largest -loss's model
 
                epoch_metric = (-epoch_loss) if judge_metric == "loss" else epoch_ci
                if phase == 'val' and epoch_metric > best_val_metric:
                    best_val_metric = epoch_metric
                    best_epoch = epoch + 1
                    best_train_ci = train_ci
                    best_train_loss = train_loss#.detach().cpu().numpy()
                    best_val_ci = epoch_ci
                    best_val_loss = epoch_loss.detach().cpu().numpy()
                    best_model_wts = copy.deepcopy(self.state_dict()) 
                    # if updating by val, then test 
                    X_test = self.data_test[:,:-2]
                    y_test = self.data_test[:,-2:]
                    PMF_test = self.predict(X_test,)# test expression matrix
                    s_time_test, e_test = y_test[:,-2], y_test[:, -1]
                    test_loss = self.deephit_loss(out=PMF_test,e=e_test,t=s_time_test,alpha = 1,beta=1,gamma=0,mask1=self.mask1_te,mask2=self.mask2_te)
                    surv_test = torch.clamp(1.-torch.cumsum(PMF_test,dim=-1),0,1)
                    test_ci = get_ci_td(s_time=s_time_test,cens=1-e_test,f=surv_test)
                    logger.info(f'Updating by {judge_metric}. Val ci: {best_val_ci:4f}; test ci: {test_ci:4f}; val loss: {best_val_loss:4f}; test loss: {test_loss:4f}')
        if last_epoch:
            X_test = self.data_test[:,:-2]
            y_test = self.data_test[:,-2:]
            PMF_test = self.predict(X_test,)# test expression matrix
            s_time_test, e_test = y_test[:,-2], y_test[:, -1]
            test_loss = self.deephit_loss(out=PMF_test,e=e_test,t=s_time_test,alpha = 1,beta=1,gamma=0,mask1=self.mask1_te,mask2=self.mask2_te)
            surv_test = torch.clamp(1.-torch.cumsum(PMF_test,dim=-1),0,1)
            test_ci = get_ci_td(s_time=s_time_test,cens=1-e_test,f=surv_test)
            logger.info(f'Updating by last epoch. Val ci: {best_val_ci:4f}; test ci: {test_ci:4f}; val loss: {best_val_loss:4f}; test loss: {test_loss:4f}')
                                
        time_elapsed = time.time() - since
        logger.info(f'\n{num_epochs} epochs training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # save the metric,loss,weight of training model into ckpt file
        if self.fn is not None:
            torch.save({'epoch': best_epoch,
                        'state_dict': self.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_train_ci': best_train_ci,
                        'best_train_loss': best_train_loss,
                        'best_val_ci': best_val_ci,
                        'best_val_loss': best_val_loss,
                        'test_ci': test_ci,
                        'test_loss': test_loss,
                        'report': self.report,
                        }, f'{self.fn}.ckpt')

        # plt loss fig
        fig = plt.figure()
        plt.plot(self.report['train_loss'], label='Train loss')
        plt.plot(self.report['val_loss'], label='Val loss')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
        plt.close()
        
        # plt ci fig
        fig = plt.figure()
        plt.plot(self.report['train_ci'], label=f'Train ci')
        plt.plot(self.report['val_ci'], label=f'Val ci')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_ci.pdf', bbox_inches='tight')
        plt.close()

        self.load_state_dict(best_model_wts)
        return self

    def predict(self, exp_mat):# need to modify
        self.eval()
        outputs = self.forward(exp_mat)
        return outputs

    def show_report(self):
        return pd.DataFrame(self.report)
    
    #mask1
    def create_mask1(self,time,label):# time,label,
        time,label = torch.unsqueeze(time, dim=1),torch.unsqueeze(label, dim=1)

        mask1 = np.zeros([np.shape(time)[0], self.t]) # for the first loss function
        for i in range(np.shape(time)[0]):
            if label[i,0] != 0:  #not censored
                mask1[i,int(time[i,0])] = 1
            else: #label[i,2]==0: censored
                mask1[i,int(time[i,0]+1):] = 1
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

    def deephit_nll(self,out, I_1, mask1):
        # I_1 = torch.sign(k)
        tmp = torch.sum(mask1 * out, 1)
        lz = I_1 * torch.log(tmp + 1e-08)
        lcen = (1. - I_1) * torch.log(tmp + 1e-08)
        return -torch.mean(lz + 1.0*lcen)
        
    def rank_loss(self, out, e, t, mask2):#t:time
        sigma1 = 0.1
        eta = []
        one_vector = torch.unsqueeze(torch.ones_like(t, dtype = torch.float32),dim=1).cuda()
        I_2 = torch.eq(e, 1).float()
        I_2 = torch.diag_embed(torch.squeeze(I_2))
        # print(out[0:,e:e+1,0:].size())
        # tmp_e = torch.reshape(out[0:,0:,0:], (-1, self.t))

        R = torch.mm(out, mask2.t())

        diag_R = torch.reshape(torch.diagonal(R), (-1, 1))
        R = torch.mm(one_vector, diag_R.t()) - R
        R = R.t()
        t = t.unsqueeze(-1)
        T = F.relu(torch.sign(torch.mm(one_vector, t.t()) - torch.mm(t,one_vector.t())))

        T = torch.mm(I_2, T)
        temp_eta = torch.mean(T * torch.exp(-R/sigma1), 1, keepdim = True)

        return torch.sum(temp_eta)
    def calibration_loss(self, out,e, mask2): #Calibration loss
        I_2 = torch.eq(e.view(e.shape[0],1), 1).float()
        r = torch.sum(out * mask2, 0)
        eta = torch.mean((r - I_2)**2, 1, keepdim = True)# torch.Size([62])-torch.Size([64, 1])
        return torch.sum(eta)
    
    def deephit_loss(self,out,e,t,mask1,mask2,print_prop=False,beta=1,alpha=1,gamma=1):
        loss_nll2 = self.deephit_nll(out,e,mask1)
        loss_rank = self.rank_loss(out,e,t,mask2)
        loss_cali = self.calibration_loss(out,e,mask2)

        if print_prop:
            print("proportion:",(loss_nll2/loss_rank).item())
        return (alpha*loss_nll2) + (beta*loss_rank) + (gamma*loss_cali)