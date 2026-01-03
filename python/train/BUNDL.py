import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import scipy
from scipy.io import loadmat as loadmat
import os
import matplotlib.pyplot as plt
import sys


import torch
import torch.nn as nn

class BUNDL(nn.Module):

    def __init__(self, eps=0.001 , over=1):
        super().__init__()
        self.over = over
        self.eps = eps
       
    def forward(self, ypred, p_label,p_prior,  p_unc):

        p_theta = F.softmax(ypred, -1)
       
        with torch.no_grad():
            p_label = p_label * 0.999
            p_label[p_label==0] = 0.001
            
            term1 = p_unc*p_prior + (1-p_unc)*p_label
            term2 = self.eps * p_prior + (1-self.eps)*p_label
            
            if self.over==1:
                p_1 = p_label * term1 + (1-p_label)* term2
            elif self.over==-1:
                p_1 = p_label * term2 + (1-p_label) * term1
            else:
                p_1 = term1

        model_fit = -1*p_1*torch.log(p_theta[:, 1]) - 1*(1-p_1)*torch.log(p_theta[:, 0])
        model_fit = model_fit.mean()

        return model_fit 


def apply_TTA(x):
    shape = x.shape
    x = x.view(-1, x.shape[-1])
    dtype= x.dtype
    device = x.device
    #flip some
    flip_idx = torch.randperm(x.shape[0])[:x.shape[0]//10]
    x[flip_idx] = torch.flip(x[flip_idx], dims=[-1])

    # add zero mean random noise
    noise = (torch.rand_like(x) - 0.5) * 0.1
    x = x + noise
    
    
    # scale between [-1.1, -0.9 ] and [0.9, 1.1] 
    scale = (torch.rand_like(x)-0.5)/0.5 * 0.2
    scale[scale>=0] = scale[scale>=0] + 0.9
    scale[scale<0] = scale[scale<0] - 0.9
    x = x * scale

    x = x.view(shape).to(dtype=dtype, device=device)

    # normalize
    #x = (x - x.mean()) / (x.std() + 1e-6)
    return x

def compute_uncertainty(model, inputs, labels, type = 'MCD', ensbl_models=None):
    #find device of input
    device = inputs.device
    if type == 'MCD':
        y_samples = []
        with torch.no_grad():
            for j in range(10):
                temp_pred, _,_ = model(inputs)
                temp_proba = F.softmax(temp_pred.reshape(-1, 2), -1)[:,  1]
                y_samples.append(temp_proba.detach().cpu().numpy())

            y_samples = np.array(y_samples)
            y_samples[y_samples>=0.999] = 0.999
            y_samples[y_samples<=0.001] = 0.001
            p_avg  = np.mean(y_samples, 0)
            p_sentropy = -1*y_samples * np.log(y_samples) - (1-y_samples) * np.log(1-y_samples)
            p_sentropy = p_sentropy.mean(0)/0.6932   #p_sentropy.max()

        return p_avg, p_sentropy
    
    if type == 'TTA-MCD':
        y_samples = []
        with torch.no_grad():
            for j in range(10):
                inputs = apply_TTA(inputs)
                temp_pred, _,_ = model(inputs)
                temp_proba = F.softmax(temp_pred.reshape(-1, 2), -1)[:,  1]
                y_samples.append(temp_proba.detach().cpu().numpy())

            y_samples = np.array(y_samples)
            y_samples[y_samples>=0.999] = 0.999
            y_samples[y_samples<=0.001] = 0.001
            p_avg  = np.mean(y_samples, 0)
            p_sentropy = -1*y_samples * np.log(y_samples) - (1-y_samples) * np.log(1-y_samples)
            p_sentropy = p_sentropy.mean(0)/0.6932   #p_sentropy.max()

        return p_avg, p_sentropy

    if type == 'Loss':
        criteria = nn.CrossEntropyLoss(weight = torch.Tensor([0.2, 0.8]).double().to(device), reduction='none')
        tau = 1.0
        with torch.no_grad():
            temp_pred, _, _ = model(inputs)
            temp_proba = F.softmax(temp_pred.reshape(-1, 2), -1)[:,  1]

            p_avg = temp_proba.detach().cpu().numpy()
            loss = criteria(temp_pred.reshape(-1, 2), labels.reshape(-1) )
            p_sentropy = torch.exp(-loss/tau).detach().cpu().numpy()
            p_sentropy /= p_sentropy.sum()
            p_sentropy[p_sentropy > 0.999] = 0.999
            p_sentropy[p_sentropy<0.001] = 0.001

        return p_avg, p_sentropy
    
    if type == 'Constant':
        tau = 0.9
        with torch.no_grad():
            temp_pred, _, _ = model(inputs)
            temp_proba = F.softmax(temp_pred.reshape(-1, 2), -1)[:,  1]

            p_avg = temp_proba.detach().cpu().numpy()
            p_sentropy = np.ones(p_avg.shape[0])*tau
            p_sentropy = p_sentropy.reshape(-1)
        
        return p_avg, p_sentropy
    

    
    if type == 'Ensemble':
        y_samples = []
        with torch.no_grad():
            for ensbl_model in ensbl_models:
                temp_pred, _,_ = ensbl_model(inputs)
                temp_proba = F.softmax(temp_pred.reshape(-1, 2), -1)[:,  1]
                y_samples.append(temp_proba.detach().cpu().numpy())

            y_samples = np.array(y_samples)
            y_samples[y_samples>=0.999] = 0.999
            y_samples[y_samples<=0.001] = 0.001
            p_avg  = np.mean(y_samples, 0)
            p_sentropy = -1*y_samples * np.log(y_samples) - (1-y_samples) * np.log(1-y_samples)
            p_sentropy = p_sentropy.mean(0)/0.6932   #p_sentropy.max()            
        
        return p_avg, p_sentropy

