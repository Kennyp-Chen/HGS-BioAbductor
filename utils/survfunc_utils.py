import numpy as np
import torch
from torch.utils.data import Dataset
from sksurv.linear_model.coxph import BreslowEstimator
import torch.nn.functional as F


class SurvDataset(Dataset):
    def __init__(self, X, y, expand_dims=False):
        self.X = np.expand_dims(X, -1) if expand_dims else X # N * C=1 * D
        self.y = y # N * 2

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# CoxBase NLL
def negative_log_likelihood(risk, y):
    t, e = y[:, 0], y[:, 1]#time:PFI months ;events: PFI status
    risk = torch.squeeze(risk) # 对数风险函数值
    t = torch.squeeze(t)
    e = torch.squeeze(e)

    sorted_ids = torch.argsort(t, descending=True)#根据时间降序排idx，时间越长预后越好
    sorted_risk = torch.gather(risk, 0, sorted_ids)
    sorted_e = torch.gather(e, 0, sorted_ids)

    # calculate negative likelihood
    hazard_ratio = torch.exp(sorted_risk)
    log_cumsum_hr = torch.log(torch.cumsum(hazard_ratio, 0))
    delta_risk = sorted_risk - log_cumsum_hr
    uncensored_delta_risk = delta_risk * sorted_e
    num_observed_events = torch.sum(sorted_e) + 1e-10
    neg_log_likelihood = -torch.sum(uncensored_delta_risk) / num_observed_events

    return neg_log_likelihood

# https://github.com/MLD3/Calibrated-Survival-Analysis/tree/main
# Code Release for "Estimating Calibrated Individualized Survival Curves with Deep Learning" (Kamran & Wiens), AAAI 2021. https://www.aaai.org/AAAI21Papers/AAAI-8472.KamranF.pdf
# def rps_loss(out, s_time, censored, weighting = 1,t = 52, norm = False):
#     '''Impelement brier loss
#     Incorporate normalization, censored individuals, and potential reweighting due to time to event skew'''
#     losses = 0
#     for j in range(len(out)):
#         #Do we want to normalize by length of time horizon? If so, reformulate what we divide each component by
#         if norm:
#             normed = censored[j]*(s_time[j]) + (1-censored[j]) * t
#         else:
#             normed = 1 
#         outt = out[j]
#         #Reweight due to right skew for our experiments. 
#         #Large weight on early time-points was sufficient to prevent degenerate results due to event time skew
#         #Other more complicated methods that resulted in similar performance included: Smaller weights up until time of event/censoring (decreasing as you get until event time)
#         losses += weighting*((out[j][0] - 0)**2)
#         #Following DRSA: Create surivval probabiltiesi through hazard ratios
#         survs = torch.cumprod(1-outt, 0)
#         #Up until event time -- ensure probability of survival closet o 1
#         for i in range(s_time[j] + censored[j]):
#             losses += ((1 - survs[i])**2)/normed
#         #For those who are not censored, ensure survival probabliities are lower after observed event time
#         if not censored[j]:
#             for i in range(s_time[j], t):
#                 losses += (survs[i]**2)/normed
#     return losses

def rps_loss(out, s_time, censored, weighting=1, t=52, norm=False):
    '''
    Optimized and corrected implementation of RPS loss using PyTorch batch operations,
    adapted for out.shape=[batch_size, num_time_points].
    
    s_time:long()
    '''
    
    out,s_time,censored = out.cuda(),s_time.cuda(),censored.cuda()
    batch_size, num_time_points = out.shape

    # Calculate survival probabilities
    survs = torch.cumprod(1 - out, 1) # dim=1 to calculate across time points

    # Prepare normalization factors
    if norm:
        normed = censored.float() * s_time.float() + (1 - censored.float()) * t
    else:
        normed = torch.ones(batch_size).cuda()

    # Ensure normed is broadcastable across the time points
    normed = normed.view(batch_size, 1).expand(-1, num_time_points)

    # Initialize losses
    losses = torch.tensor(0.0).cuda()

    # Loss for prediction at t=0 (redundant calculation, can be optimized)
    losses += torch.sum(weighting * (out[:, 0] - 0) ** 2)

    # Create masks for vectorized loss calculation
    time_indices = torch.arange(num_time_points).repeat(batch_size, 1).cuda() # [batch_size, num_time_points]
    event_mask = time_indices < (s_time.view(-1, 1) + censored.view(-1, 1))
    post_event_mask = time_indices >= s_time.view(-1, 1)

    # Loss calculation for survival probabilities until event time
    losses += torch.sum(((1 - survs[event_mask]) ** 2) / normed[event_mask])

    # Loss calculation for survival probabilities after event time, only for uncensored data
    non_censored_mask = (censored == 0).view(-1, 1) & post_event_mask
    losses += torch.sum((survs[non_censored_mask] ** 2) / normed[non_censored_mask])

    return losses
def rps_loss_surv(survs, s_time, censored, t=52, norm=False):
    '''
    Optimized and corrected implementation of RPS loss using PyTorch batch operations,
    adapted for out.shape=[batch_size, num_time_points].
    
    s_time:long()
    '''
    
    # out,s_time,censored = out.cuda(),s_time.cuda(),censored.cuda()
    batch_size, num_time_points = survs.shape

    # Calculate survival probabilities
    # survs = torch.cumprod(1 - out, 1) # dim=1 to calculate across time points

    # Prepare normalization factors
    if norm:
        normed = censored.float() * s_time.float() + (1 - censored.float()) * t
    else:
        normed = torch.ones(batch_size).cuda()

    # Ensure normed is broadcastable across the time points
    normed = normed.view(batch_size, 1).expand(-1, num_time_points)

    # Initialize losses
    losses = torch.tensor(0.0).cuda()
    # 未换算t=0校准项
    # # Loss for prediction at t=0 (redundant calculation, can be optimized)
    # losses += torch.sum(weighting * (out[:, 0] - 0) ** 2)

    # Create masks for vectorized loss calculation
    time_indices = torch.arange(num_time_points).repeat(batch_size, 1).cuda() # [batch_size, num_time_points]
    event_mask = time_indices < (s_time.view(-1, 1) + censored.view(-1, 1))
    post_event_mask = time_indices >= s_time.view(-1, 1)

    # Loss calculation for survival probabilities until event time
    losses += torch.sum(((1 - survs[event_mask]) ** 2) / normed[event_mask])

    # Loss calculation for survival probabilities after event time, only for uncensored data
    non_censored_mask = (censored == 0).view(-1, 1) & post_event_mask
    losses += torch.sum((survs[non_censored_mask] ** 2) / normed[non_censored_mask])

    return losses

def get_ci_td(s_time,cens,f):
    '''
    for time dependent cindex calculate
    output: ci
    f for surv_prob
    '''
    # s_time = s_time.cpu()
    s_time,cens=s_time.long(),cens.long()
    denom = 0
    num = 0
    for subj in range(len(s_time)):# 被比较者的index,即较小者
        #If people have the same event time, can incorporate into C-Index ranking by assigning 0.5 score, or can be ignored
        # nonzero返回2-D非零元素的索引；reshape（-1），所有元素放入1D;cens[subj]是被比较者，index即从小到大遍历，每一个cens
        for index in (s_time > s_time[subj]).nonzero().reshape(-1): # 用于成对的其他例子index ，用于取出所有更大时间的subj
            #Penalize discordant pairs -- only count those that are conconrdant with observed event times
            if not cens[subj]:# 较小时间者不得删失
                num += int(1-f[subj][s_time[subj]] > 1-f[index][s_time[subj]])# s_time[subj]是被比较者的发生事件时间，所有此方法可以在所有对应的时间进行比较。
                denom += 1
    if denom == 0:
        return 0.5
    else:
        return num/denom 

def get_ci_td_high_precision(s_time, cens, f):
    '''
    Calculate time-dependent C-index with higher precision.
    Parameters:
    - s_time: Survival times (1D tensor).
    - cens: Censoring status (1D tensor, 0 for uncensored, 1 for censored).
    - f: Predicted survival probabilities (2D tensor, shape: [num_samples, max_time]).

    Returns:
    - C-index (float).
    '''
    s_time, cens = s_time.long(), cens.long()
    
    # Initialize numerator and denominator
    num = torch.tensor(0.0, dtype=torch.float64).cuda()  # Use double precision
    denom = torch.tensor(0.0, dtype=torch.float64).cuda()

    # Convert s_time to a 2D tensor for broadcasting
    s_time_matrix = s_time.unsqueeze(0)  # Shape: [1, num_samples]
    s_time_diff = s_time_matrix - s_time_matrix.T  # Pairwise differences, shape: [num_samples, num_samples]

    # Create a mask for pairs where s_time[i] < s_time[j]
    valid_pairs = (s_time_diff < 0)  # Pair (i, j) is valid if s_time[i] < s_time[j]
    
    # Exclude censored subjects as "subj" (row-wise mask)
    valid_pairs &= cens.unsqueeze(1) == 0  # Only consider uncensored rows as valid

    # Extract valid pairs' indices
    valid_indices = valid_pairs.nonzero(as_tuple=True)  # Tuple of row and column indices
    
    # Compute concordance
    for i, j in zip(*valid_indices):
        concordant = (1 - f[i, s_time[i].item()]) > (1 - f[j, s_time[i].item()])
        num += concordant.to(dtype=torch.float64)  # Cast to double
        denom += 1.0

    # Handle edge case: no valid comparisons
    if denom == 0:
        return 0.5  # Default value for no valid pairs
    
    return (num / denom).item()  # Return as Python float
    
def kernel_loss(s_time,cens,surv_probs,sigma=1,do_l2=True):
    l2 = 0
    if do_l2:
        for subj in range(len(s_time)):
            #For all individuals with later observed event times
            for index in (s_time > s_time[subj]).nonzero().reshape(-1): 
                #For stability, focus only on those who are not censored. Can incorporate censored individuals here
                if not cens[subj] and not cens[index]:
                    #Penalize discordant rankings
                    l2 += torch.exp(-(((1-surv_probs[subj][s_time[subj]]) - (1-surv_probs[index][s_time[subj]]))/sigma))
    return l2

def get_surv_ana_prob(out,s_time,e,discrete_time,do_nll=1):
    out = out.double()
    #out need to be tensor of (num_samples , num_timebin)
    surv_probs = torch.cumprod(1 - out  * torch.cat([torch.zeros(1), torch.ones(discrete_time-1)]).cuda(), dim = 1)# .to(self.device)
    # cens是1-events，即 是否删失：如果删失（最后时间点仍存活），则为1
    cens = (1-e).float()
    if do_nll:
        #Gather hazard ratios and log probabilities around event time
        #Estimated Survival probability at time one before observed event time
        one_before_time = surv_probs.gather(1, (s_time-1).view(-1,1)).view(-1)
        #Hazard ratio at time of observed event
        out_at_time = out.gather(1, (s_time).view(-1,1)).view(-1)
        #Estimated survival probabilities at observed event times
        at_times = surv_probs.gather(1,(s_time).view(-1,1)).view(-1)
        #Estimated survival probabilities at final time-point
        at_end = surv_probs.gather(1, torch.tensor(discrete_time-1).cuda().repeat(1, out.shape[0]).view(out.shape[0], 1)).view(-1)#.to(self.device)
        #All uncensored indivs should have low survival probability at end
        y_uncensored = -sum(torch.log(1 - at_end).view(-1) * (1-cens))
        #All censored individuals should have high estimated times at censoring time
        y_censored = -sum(torch.log(at_times).view(-1) * cens)
        #High hazard ratio for time of event for uncensored individuals 
        uncensored_at_times = -sum((torch.log(out_at_time * one_before_time)).view(-1) * (1-cens))
        return surv_probs,y_uncensored,y_censored,uncensored_at_times,cens
    else:
        return surv_probs,cens

def get_loss_ci(predicts ,do_l2=True ,do_brier=False ,only_do_l2=False ,s_time = None , event = None,discrete_time = None,sigma=1,RPS_norm=False,if_ci=True,print_prop=False):
    '''
    sigma: Whether to use kernel loss L_kernel. Default is -1, which means not to use it
    do_l2: Control the trade-off between L_RPS and L_kernel
    Ci_td 

    '''
    surv_probs,y_uncensored,y_censored,uncensored_at_times,cens = get_surv_ana_prob(predicts,s_time,event,discrete_time)
    # loss
    #Impelement kernel loss
    l2 = kernel_loss(s_time=s_time,cens = cens,surv_probs=surv_probs,sigma=sigma,do_l2 = do_l2)
    # print("Kernel:",l2.detach().cpu())
    if do_brier:
        loss = rps_loss_surv(predicts, s_time, cens.long(), weighting = 1, t=discrete_time, norm = RPS_norm) + do_l2*(l2) 
    elif only_do_l2: 
        loss = l2
    else:
        loss = (y_uncensored + uncensored_at_times + y_censored) + do_l2*(l2)
    # print("MainLoss:",(loss-do_l2*(l2)).detach().cpu())
    if print_prop:
        print("Main/Kernel proportion:",((loss-do_l2*(l2)) / (do_l2*(l2))).detach().cpu().item())

    # loss /= len(s_time)
    if if_ci:
        ci = get_ci_td(s_time=s_time,cens=cens,f=surv_probs)
        return loss, ci
    else:
        return loss

def get_RPSorLogUncens_loss(surv_probs,s_time,cens,discrete_time,do_brier,RPS_norm):
    # l2 = kernel_loss(s_time=s_time,cens = cens,surv_probs=surv_probs,sigma=1,do_l2 = do_l2)
    s_time = s_time.long()
    if do_brier:
        loss = rps_loss_surv(surv_probs, s_time, cens.long(), t=discrete_time, norm = RPS_norm)
    else:
        at_end = surv_probs.gather(1, torch.tensor(discrete_time-1).repeat(1, surv_probs.shape[0]).cuda().view(surv_probs.shape[0], 1)).view(-1)#.to(self.device)
        #All uncensored indivs should have low survival probability at end
        y_uncensored = -sum(torch.log(1 - at_end).view(-1) * (1-cens))
        loss = y_uncensored
        # loss /= len(s_time)
    return loss
def get_DRS_loss_ci(predicts,do_l2,do_brier,RPS_norm,do_nll,a,s_time,event,t_obs,if_ci):
    '''
    sigma: Whether to use kernel loss L_kernel. Default is -1, which means not to use it
    do_l2: Control the trade-off between L_RPS and L_kernel
    Ci_td 

    '''
    if do_nll:
        surv_probs,y_uncensored,y_censored,uncensored_at_times,cens = get_surv_ana_prob(predicts,s_time,event,t_obs,do_nll)
    else:
        surv_probs,cens = get_surv_ana_prob(predicts,s_time,event,t_obs,do_nll)
    l2  = 0
    rps = 0
    if do_l2:
        l2 = kernel_loss(s_time=s_time,cens = cens,surv_probs=surv_probs,sigma=1,do_l2=do_l2)
    if do_brier:
        rps = rps_loss_surv(predicts, s_time, cens.long(), t=t_obs, norm = RPS_norm)
    loss = do_nll*((1-a)*(y_uncensored+y_censored)+a*uncensored_at_times) + do_l2*(l2) + do_brier*(rps)
    if if_ci:
        return loss, get_ci_td(s_time=s_time,cens=cens,f=surv_probs)
    else:
        return loss,0
    
def get_Cph_surv_prob(predictor,events,times,num_pats):
    '''
    Input cox ph beta@x, output surv_prob at times.

    Parameters
    ----------
    prediction : tensor, shape=([n_samples,])
        Predicted risk scores.
        beta@x of Cox PH model.

    events : tensor, shape=([n_samples,])
        Events indicators tensor, 0 for cens; 1 for event 

    times : tensor, shape=([n_samples,])
        Event time for ever patients

    return : tensor, shape([pats,times])
    '''
    predictor = predictor.to_numpy()
    times = predictor.to_numpy()
    events = predictor.to_numpy()

    be = BreslowEstimator()
    be.fit(predictor,events,times)
    surv_f = be.get_survival_function(predictor)
    surv_prob = torch.empty([num_pats,np.unique(times),np.unique(times).size])
    for i in range(num_pats):
        surv_prob[i]=torch.tensor(surv_f[i].y)
    return surv_prob


# def get_ci_td(s_time,cens,f):
#     '''
#     for time dependent cindex calculate
#     f for surv_prob
#     '''
#     denom = 0
#     num = 0
#     for subj in range(len(s_time)):
#         for index in (s_time > s_time[subj]).nonzero().reshape(-1): 
#             if not cens[subj]:
#                 num += int(1-f[subj][s_time[subj]] > 1-f[index][s_time[subj]])
#                 denom += 1
#     return num/denom 

# DeepHit

#mask1
def create_mask1(num_t,time,label):# time,label,
    mask1 = np.zeros([np.shape(time)[0], num_t]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i,0] != 0:  #not censored
            mask1[i,int(label[i,0]-1),int(time[i,0])] = 1
        else: #label[i,2]==0: censored
            mask1[i,:,int(time[i,0]+1):] = 1
    return torch.from_numpy(mask1).float()
#mask2
def create_mask2(num_t,time):
    meas_time = -1
    mask2 = np.zeros([np.shape(time)[0], num_t]) # for the first loss function
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

def deephit_nll(out, I_1, mask1):
    # I_1 = torch.sign(k)
    tmp = torch.sum(mask1 * out, 2)
    lz = I_1 * torch.log(tmp + 1e-08)
    lcen = (1. - I_1) * torch.log(tmp + 1e-08)
    return -torch.mean(lz + 1.0*lcen)
    
def rank_loss( out, e, t, mask2):#t:time
    sigma1 = 0.1
    eta = []
    one_vector = torch.ones_like(t, dtype = torch.float32)
    I_2 = torch.eq(e, 1).float()
    I_2 = torch.diag_embed(torch.squeeze(I_2))

    R = torch.mm(out, mask2.t())

    diag_R = torch.reshape(torch.diagonal(R), (-1, 1))
    R = torch.mm(one_vector, diag_R.t()) - R
    R = R.t()

    T = F.relu(torch.sign(torch.mm(one_vector, t.t()) - torch.mm(t,one_vector.t())))

    T = torch.mm(I_2, T)
    temp_eta = torch.mean(T * torch.exp(-R/sigma1), 1, keepdim = True)

    return torch.sum(temp_eta)

def deephit_loss(out,e,t,alpha,beta,mask1,mask2):
    loss_nll2 = deephit_nll(out,e,mask1)
    loss_rank = rank_loss(out,e,t,mask2)
    return (alpha*loss_nll2) + (beta*loss_rank)

def get_Coxph_citd(LRR,s_time,event):
    num_pats = LRR.shape[0]
    bx = LRR.detach().cpu().numpy()
    be = BreslowEstimator()
    time = s_time.cpu().numpy()
    events = event.cpu().numpy()
    be.fit(linear_predictor=bx,event=events,time=time)
    surv_f = be.get_survival_function(bx)
    num_time = be.unique_times_.size
    surv_prob = torch.empty([num_pats,num_time])
    for i in range(num_pats):
        surv_prob[i]=torch.tensor(surv_f[i].y)
    t_uni = s_time.unique()
    time_tmp=torch.empty([num_pats])
    t=-1
    for tu in t_uni:
        t+=1
        for i in range(num_pats):
            if s_time[i]==tu:
                time_tmp[i]=t
    ci_td = get_ci_td(s_time=time_tmp,cens=1-event,f=surv_prob)
    return ci_td


# def get_Cph_citd(self,predictor):
#     '''
#     Input cox ph beta@x, output surv_prob at times.

#     Parameters
#     ----------
#     prediction : tensor, shape=([n_samples,])
#         Predicted risk scores.
#         beta@x of Cox PH model.

#     return : tensor, shape([pats,times])
#     '''
#     predictor = predictor.detach().cpu().numpy()
#     data = torch.concat([self.data_train,self.data_test,self.data_valid],0)
#     x_tuple = self.xno,self.xe,data[:,:-2]
#     preds_all,_ = self.predict(x_tuple)   
#     num_pats = predictor.size
#     be = BreslowEstimator()
#     # fit all samples so we can get all time point results
#     be.fit(linear_predictor=preds_all.detach().cpu().numpy(),event=data[:,-1].cpu().numpy(),time=data[:,-2].cpu().numpy())
#     surv_f = be.get_survival_function(predictor)
#     num_time = be.unique_times_.size
#     surv_prob = torch.empty([num_pats,num_time])
#     for i in range(num_pats):
#         surv_prob[i]=torch.tensor(surv_f[i].y)
#     return surv_prob#, torch.tensor(be.unique_times_)

'''
Liwen Zhang, Lianzhen Zhong, et al., TripleSurv: Triplet Time-Adaptive Coordinate Learning Approach for Survival Analysis, IEEE Transactions on Knowledge and Data Engineering, 2024. doi: 10.1109/TKDE.2024.3450910.
'''
def get_risk(pred_value):
    '''
    Input:
        pred_value: batch_num * K | torch.tensor
    Output:
        risk: batch_num * 1
    '''
    K = pred_value.shape[1]
    time_line = 0.5*(torch.arange(1,K+1)+torch.arange(0,K))/K
    time_line = time_line.cuda()
    pred_value = pred_value * time_line
    risk = 1.0 - torch.sum(pred_value, dim=-1, keepdim=True)
    
    return risk
import torch


def get_surv_mask1(surv_time, dead, num_Category = 20):
    '''
    Input:
        surv_time: batch_num * 1 or 1 * batch_num or batch_num | torch.tensor
        is_alive: batch_num * 1 or or 1 * batch_num or batch_num | 0 (death) 1 (censored) | torch.tensor
        num_Category: int
    Output:
        mask1: batch_num * num_Category | torch.tensor
    '''
    cat_time = get_cat_time(surv_time, num_Category = num_Category)
    cat_time = cat_time.view(-1)
    dead = dead.view(-1)

    mask1 = torch.zeros(cat_time.shape[0], num_Category)
    for i in range(mask1.shape[0]):
        if dead[i] == 1:
            mask1[i, int(cat_time[i])] = 1
        else:
            mask1[i, int(cat_time[i] + 1):] = 1

    mask1 = mask1.cuda()

    return mask1

def get_surv_mask2(surv_time, num_Category = 20):
    '''
    Input:
        surv_time: batch_num * 1 or or 1 * batch_num or batch_num | torch.tensor
        num_Category: int
    Output:
        mask1: batch_num * num_Category | torch.tensor
    '''
    cat_time = get_cat_time(surv_time, num_Category = num_Category)
    cat_time = cat_time.view(-1)

    mask2 = torch.zeros(cat_time.shape[0], num_Category)
    for i in range(mask2.shape[0]):
        t = int(cat_time[i]) + 1
        mask2[i, :t] = 1

    mask2 = mask2.cuda()

    return mask2

def get_cat_time(surv_time, num_Category = 20):

    bin_boundaries = torch.arange(1, num_Category+1)/num_Category
    bin_boundaries = bin_boundaries.view(1, -1)
    bin_boundaries = bin_boundaries.cuda()
    tte_cat = (surv_time >= bin_boundaries).sum(dim=-1)

    return tte_cat
eps = 1e-6

def likelihood_loss(pred_value, surv_mask1):
    '''
    Input:
        pred_value: batch_num * K | torch.tensor
        surv_mask1: batch_num * K | torch.tensor
    Output:
        loss: scalar
    ''' 
    R = pred_value*surv_mask1
    logp = torch.sum(R, dim=-1).log()
    loss = -1 * torch.mean(logp)

    return loss

def ranking_loss(risk, Y_label_T, Y_label_E, sigma):
    '''
    Input:
        risk: batch_num * 1 | torch.tensor
        Y_label_E: batch_num * 1, 1 (death) 0 (censored) | torch.tensor
        Y_label_T: batch_num * 1 | torch.tensor
        sigma, scale: scalar
    Output:
        rank_loss: scalar
    '''
    B,L = risk.shape
    assert L == 1, "input risk has wrong size"
    I_2 = Y_label_E.squeeze().diag()
    # Y_label_T = torch.abs(Y_c)
    one_vector = torch.ones_like(Y_label_T, dtype=torch.float)

    R = torch.matmul(risk, one_vector.T) - torch.matmul(one_vector, risk.T)#R(i,j) = risk(i) - risk(j)
    T = torch.matmul(one_vector, Y_label_T.T) - torch.matmul(Y_label_T, one_vector.T)# T(i,j) = y(j) - y(i)
    I_3 = torch.matmul(I_2, T.sign().relu())# I_3(i,j) = e(i) * (y(j) - y(i))

    num = torch.sum(I_3)
    suma =  torch.sum(I_3 * torch.exp(-sigma*R))#-sigma*R(i,j)
    rank_loss = suma / (num + eps)

    return rank_loss

def calibration_loss(pred_value, Y_label_E, surv_mask1, surv_mask2, nbins):
    '''
    Input:
        pred_value: batch_num * K | torch.tensor
        Y_label_E: batch_num * 1, 1 (death) 0 (censored) | torch.tensor
        surv_mask1: batch_num * K
        surv_mask2: batch_num * K
        nbins: scalar
    Output:
        rank_loss: scalar
    '''

    num_time = pred_value.shape[1]
    I_2 = Y_label_E * surv_mask1

    new_mask1 = []
    new_mask2 = []
    pred_p = []
    time_step = int(num_time / nbins)

    for i in range(nbins):
        A = surv_mask2[:, i * time_step : (i + 1) * time_step]
        A = torch.max(A, dim = -1, keepdim = True).values
        B = I_2[:, i * time_step : (i + 1) * time_step]
        B = torch.max(B, dim = -1, keepdim = True).values
        C = pred_value[:, i * time_step : (i + 1) * time_step]
        C = torch.sum(C, dim = -1, keepdim=True)

        new_mask1.append(A)
        new_mask2.append(B)
        pred_p.append(C)

    new_mask1 = torch.cat(new_mask1, dim = -1)
    new_mask2 = torch.cat(new_mask2, dim = -1)
    pred_p = torch.cat(pred_p, dim = -1)
    pred_s = 1.0 + pred_p - torch.cumsum(pred_p, dim = -1)#S = 1 + p - cumsum(p)
    pred_h = pred_p / (eps+pred_s)
    r = torch.mean(pred_h, dim = 0)  # no need to divide by each individual dominator
    t_a = torch.sum(new_mask1, dim = 0)
    t_b = torch.sum(new_mask2, dim = 0)
    t = t_b / (eps + t_a)

    cal_loss = torch.mean((r - t) ** 2)

    return cal_loss   

def self_ranking_loss_weight(risk, Y_label_T, Y_label_E, sigma=0.5, scale=1.0):
    '''
    Input:
        risk: batch_num * 1 | torch.tensor
        Y_label_E: batch_num * 1, 1 (death) 0 (censored) | torch.tensor
        Y_label_T: batch_num * 1 | torch.tensor
        sigma, scale: scalar
    Output:
        rank_loss: scalar
    '''
    B,L = risk.shape
    assert L == 1, "input risk has wrong size"
    # Y_label_E = (Y_c > 0).float()
    # Y_label_T = torch.abs(Y_c)
    one_vector = torch.ones_like(Y_label_T, dtype=torch.float)

    R = torch.matmul(risk, one_vector.T) - torch.matmul(one_vector, risk.T)#R(i,j) = risk(i) - risk(j)
    T = torch.matmul(one_vector, Y_label_T.T) - torch.matmul(Y_label_T, one_vector.T)# T(i,j) = y(j) - y(i)
    mat_C = (T.sign() + 1 - Y_label_E.T).sign().relu()
    I_3 = Y_label_E*mat_C# I_3(i,j) = sign(yj - yi) + 1 - ej

    num = torch.sum(I_3)
    suma =  torch.sum(I_3 * torch.exp(sigma*(scale*T - R)))#sigma*(scale*T(i,j) - R(i,j))
    rank_loss = suma / (num + eps)

    return rank_loss