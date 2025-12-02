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

class FC_layers_Interpret(nn.Module):
    '''DeepSurv or DeepDiscreteTimeSurv'''
    def __init__(self,dim_in,dropout=0.5,ACT = 'LeakyReLU',layer_nodes=[],dim_out=1,RES=False):# if dim_out = 1 then DeepSurv
        super(FC_layers_Interpret, self).__init__()   
        modules = []
        modules_res = []
        # ACT = eval(f"nn.{ACT}()")
        # modules.append(nn.BatchNorm1d(dim_in))
        self.RES=RES
        if RES:
            modules_res.append(nn.Linear(dim_in,dim_in))#1
            modules_res.append(nn.BatchNorm1d(dim_in))
            modules_res.append(nn.LeakyReLU())
            modules_res.append(nn.Dropout(dropout))
            modules.append(nn.Linear(dim_in*2,dim_in))#2 RESNET
            modules.append(nn.BatchNorm1d(dim_in))#momentum=0.8
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout))
            self.RES_NNs = nn.Sequential(*modules_res)
        if len(layer_nodes)==0:
            modules.append(nn.Linear(dim_in,dim_out))
        else:#>=1
            modules.append(nn.Linear(dim_in,layer_nodes[0]))
            modules.append(nn.BatchNorm1d(layer_nodes[0]))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout))
            for l in range(len(layer_nodes)):
                if l<len(layer_nodes)-1:
                    exec(f"modules.append(nn.Linear(layer_nodes[{l}],layer_nodes[{l+1}]))")
                    exec(f"modules.append(nn.BatchNorm1d(layer_nodes[{l+1}]))")
                    modules.append(nn.LeakyReLU())
                    modules.append(nn.Dropout(dropout))
                else:
                    modules.append(nn.Linear(layer_nodes[-1],dim_out))
        
        self.NNs = nn.Sequential(*modules)
    def forward(self,x):
        with torch.amp.autocast(device_type='cuda'):
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

class FC_layers(nn.Module):
    '''DeepSurv or DeepDiscreteTimeSurv'''
    def __init__(self,dim_in,dropout=0.5,ACT = 'LeakyReLU',layer_nodes=[],dim_out=1,RES=False):# if dim_out = 1 then DeepSurv
        super(FC_layers, self).__init__()   
        modules = []
        modules_res = []
        ACT = eval(f"nn.{ACT}()")
        # modules.append(nn.BatchNorm1d(dim_in))
        self.RES=RES
        if RES:
            modules_res.append(nn.Linear(dim_in,dim_in))#1
            modules_res.append(nn.BatchNorm1d(dim_in))
            modules_res.append(ACT)
            modules_res.append(nn.Dropout(dropout))
            modules.append(nn.Linear(dim_in*2,dim_in))#2 RESNET
            modules.append(nn.BatchNorm1d(dim_in))#momentum=0.8
            modules.append(ACT)
            modules.append(nn.Dropout(dropout))
        if len(layer_nodes)==0:
            modules.append(nn.Linear(dim_in,dim_out))
        else:#>=1
            modules.append(nn.Linear(dim_in,layer_nodes[0]))
            modules.append(nn.BatchNorm1d(layer_nodes[0]))
            modules.append(ACT)
            modules.append(nn.Dropout(dropout))
            for l in range(len(layer_nodes)):
                if l<len(layer_nodes)-1:
                    exec(f"modules.append(nn.Linear(layer_nodes[{l}],layer_nodes[{l+1}]))")
                    exec(f"modules.append(nn.BatchNorm1d(layer_nodes[{l+1}]))")
                    modules.append(ACT)
                    modules.append(nn.Dropout(dropout))
                else:# last layer
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
    