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
from lossfn import *

def main_train_function(data_root, modelname='txlstm', cvfold=1,
        hparams = {'lr':1e-04, 'maxiter':10, 'eps':0.001},
                      traintype='cel',noise_type=None, version='data_v1_og',
                      use_cuda=False):
        lr = hparams['lr']
        maxiter = hparams['maxiter']
        eps = hparams['eps']
        device = 'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu'
        if use_cuda:
            torch.cuda.empty_cache()
        print(traintype, modelname)
        if noise_type=='None':
            noise_type=None
        if traintype=='cel':
            detloss = nn.CrossEntropyLoss(wieight = torch.Tensor([0.2, 0.8]).double().to(device))
        elif traintype=='cleancel':
            detloss = cleanCEL()
        elif traintype == 'selfadapt':
            detloss = selfadapt_loss()
        else:
            detloss = BUNDL()

        save_loc = data_root+'/models/'
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)

        valsize = 12
        #all_pts =  [str(pt) for pt in np.arange(1, 121, 1)]
        arr_pts = np.load(data_root + "pts" + str(int(cvfold-1)//10) + ".npy")
        all_pts = [str(pt) for pt in arr_pts]
        ind = int((cvfold)%10) if int((cvfold)%10) else 10
        val_pts =   all_pts[(ind-1)*valsize : ind*valsize]
        train_pts = [pt for pt in all_pts if pt not in val_pts]

        train_set =  SimDataSet_v2(data_root+'/'+version+'/',train_pts, noise_type = noise_type)
        val_set = SimDataSet_v2(data_root+'/'+version+'/',val_pts, noise_type=noise_type)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=True)
        print(modelname)
        if modelname == 'txlstm':
            model = transformer_lstm(transformer_dropout=0.2, device=device)
        elif modelname == 'tgcn':
            model = TGCN()
        else:
            model = CNN_BLSTM()
            #oldmodel = CNN_BLSTM()

        model.double()
        model.to(device)
        oldlr = {'txlstm':1e-05, 'tgcn': 1e-06, 'cnnlstm':0.01, 'cnnblstm':0.01}
        savename = modelname+'_'+traintype+'_cv'+str(cvfold)+'_'+str(lr)+'_'+str(version)+'_'+str(noise_type)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        sigmoid = torch.nn.Sigmoid()

        val_losses = []
        train_losses = []
        update_labels = False
        for epoch in range(1, maxiter+1):

            epoch_loss = 0.0
            train_len = len(train_loader)
            val_len = len(val_loader)
            val_epoch_loss = 0.0
            for batch_idx, data in enumerate(train_loader):

                optimizer.zero_grad()
                inputs = data['buffers']
                det_labels = data['sz_labels'].long().to(device)
                b, Nsz, T, _, _ = inputs.shape
                fn = data['patient numbers']
                inputs = inputs.to(torch.DoubleTensor()).to(device)
                k_pred, hc, _  = model(inputs)

                for j in range(10):
                            temp_pred, _,_ = model(inputs)
                            temp_proba = F.softmax(temp_pred.reshape(-1, 2), -1)[:,  1]
                            y_samples.append(temp_proba.detach().cpu().numpy())

                    y_samples = np.array(y_samples)
                    y_samples[y_samples>=0.999] = 0.999
                    y_samples[y_samples<=0.001] = 0.001
                    p_avg  = np.mean(y_samples, 0)
                    p_sentropy = -1*y_samples * np.log(y_samples) - (1-y_samples) * np.log(1-y_samples)
                    p_sentropy = p_sentropy.mean(0)/0.6932    #p_sentropy.max()
                    p_avg = torch.tensor(p_avg).to(device).reshape(-1)
                    p_sentropy = torch.tensor(p_sentropy).to(device).reshape(-1)
                
                if traintype == 'bundl':
                    loss = detloss(k_pred.reshape(-1, 2), det_labels.reshape(-1), p_avg, p_sentropy)
                elif traintype == 'selfadapt':
                    loss = selfadapt_loss(k_pred.reshape(-1, 2), det_labels.reshape(-1), epoch, fn[0])

                else:
                    loss = detloss(k_pred.reshape(-1,2), det_labels.reshape(-1))

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                #print('\t', batch_idx, ': Loss =', loss.item()  )
            for batch_idx, data in enumerate(val_loader):
                optimizer.zero_grad()
                with torch.no_grad():

                    inputs = data['buffers']
                    det_labels = data['sz_labels'].long().to(device)
                    b, Nsz, T, _, _ = inputs.shape
                    fn = data['patient numbers']
                    inputs = inputs.to(torch.DoubleTensor()).to(device)
                    k_pred, hc, _  = model(inputs)

                    for j in range(20):
                        with torch.no_grad():
                            temp_pred, _,_ = model(inputs)
                            temp_proba = F.softmax(temp_pred.reshape(-1, 2), -1)[:,  1]
                            y_samples.append(temp_proba.detach().cpu().numpy())

                    y_samples = np.array(y_samples)
                    y_samples[y_samples>=0.999] = 0.999
                    y_samples[y_samples<=0.001] = 0.001
                    p_avg  = np.mean(y_samples, 0)
                    p_sentropy = -1*y_samples * np.log(y_samples) - (1-y_samples) * np.log(1-y_samples)
                    p_sentropy = p_sentropy.mean(0)/0.6932    #p_sentropy.max()
                    p_avg = torch.tensor(p_avg).to(device).reshape(-1)
                    p_sentropy = torch.tensor(p_sentropy).to(device).reshape(-1)

                    if traintype == 'bundl':
                        loss = detloss(k_pred.reshape(-1, 2), det_labels.reshape(-1), p_avg, p_sentropy)
                    elif traintype == 'selfadapt':
                        loss = selfadapt_loss(k_pred.reshape(-1, 2), det_labels.reshape(-1), epoch, fn[0])
                    else:
                        loss = detloss(k_pred.reshape(-1,2), det_labels.reshape(-1))

                    val_epoch_loss += loss.item()


            epoch_loss = epoch_loss/train_len
            val_epoch_loss = val_epoch_loss/val_len
            torch.cuda.empty_cache()

            train_losses.append(epoch_loss)
            val_losses.append(val_epoch_loss)

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} '.format(epoch, epoch_loss, val_epoch_loss))

            torch.save(model.state_dict(), save_loc+savename+ '.pth.tar')
            np.save(save_loc+ savename+'_train.npy', train_losses)
            np.save(save_loc+savename+'_val.npy', val_losses)
        #del model, optimizer, train_loader, validation_loader

        return train_losses, val_losses

