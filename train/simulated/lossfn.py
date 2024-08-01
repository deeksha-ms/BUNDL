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
from models import * 
from dataloader import *

class cleanCEL(nn.Module):

    def __init__(self, eps=0.001 ):
        super().__init__()

        self.cel = nn.CrossEntropyLoss()

    def forward(self, ypred, y):
        ypred = ypred.reshape(-1, 600, 2)
        y = y.reshape(-1, 600)
        N = y.shape[0]
        model_fit = 0
        for n in range(N):
            with torch.no_grad():
                s, e = self.get_szends(y[n])
                sz1 = min(s+30, e-10)       #prev s+30, e-10
                szend1 = max(s+30+10, e-30) #prev s+30+10, e-30

                len1 = szend1-sz1
            model_fit += self.cel(ypred[n, sz1:szend1], y[n, sz1:szend1])

            if len1<s:
                start0 = 0
                end0 = len1
                model_fit += self.cel(ypred[n, start0:end0], y[n, start0:end0])

            else:
                s1 = 0
                e1 = max(1, s-30)
                model_fit += self.cel(ypred[n, s1:e1], y[n, s1:e1])

                rem = len1 - (e1-s1)
                s2 = min(599, szend1+30)
                e2 = min(600, s2+rem)
                model_fit += self.cel(ypred[n, s2:e2], y[n, s2:e2])


        return model_fit/N

    def get_szends(self, y):

        delta = y[1:] - y[ :-1]

        try:
            s = np.where(delta==1)[0][0] + 1
        except:
            s=0
        try:
            e = np.where(delta==-1)[0][0] +1
        except:
            e=600
        return s, e

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

class selfadapt_loss(nn.Module):
    def __init__(self):
        super().__init__(self, alpha=0.9)
        self.t_true_dict = {}    
        self.alpha = alpha

    def forward(y_pred, y_true, epoch, fn):
        y_pred = F.softmax(y_pred, -1)
        with torch.no_grad():
            y_true = F.one_hot(y_true, num_classes=2)
            if epoch >15:
                y_true = self.alpha*self.t_true_dict[fn] + (1-self.alpha)* y_pred
            sample_weight, _ = torch.max(y_true, 1)
        
        loss = -1*y_true * torch.log(y_pred)
        loss = sample_weight * loss.sum(-1)
        loss = loss.sum()/sample_weight.sum()
    
        self.t_true_dict[fn] = y_true

        return loss

