from models import *
from utils import *
from dataloader import *
import numpy as np
import pandas as pd
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_cel(data_root, modelname, mn_fn,
                       maxiter = 100, lr = 1e-05, traintype='pretrain', 
                use_cuda=False):

        device = 'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu'
        if use_cuda:
            torch.cuda.empty_cache()

        det_loss = nn.CrossEntropyLoss(weight = torch.Tensor([0.2, 0.8]).double().to(device))
        manifest = read_manifest(data_root+mn_fn, ',')
        save_loc = data_root+'/pretrain_models/'
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)

        pt_list = np.load(data_root+'all_pts.npy')

        true_data = pd.read_csv('true_data.csv')
        train_set =  pretrain_data(data_root, true_data)
        train_loader = DataLoader(train_set, batch_size=300, shuffle=True)
        #val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

        if modelname == 'txlstm':
            model = transformer_lstm(transformer_dropout=0.2, device=device)
        elif modelname == 'sztrack':
            model = sztrack()
        elif modelname == 'tgcn':
            model = TGCN()
        else:
            model = PYT_encoder_pretrain(n_lstm=3)
        
        savename = modelname+'_'+traintype+'_'+str(lr)
        model.double()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        sigmoid = torch.nn.Sigmoid()

        train_losses = []
        train_loss_clean = []
        train_loss_corrupt = []
        val_losses = []
        train_len = len(train_loader)
        for epoch in range(1, maxiter+1):

            epoch_loss = 0.0
            #train_len = len(train_loader)
            #val_len = len(val_loader)
            val_epoch_loss = 0.0
            for batch_idx, data in enumerate(train_loader):

                optimizer.zero_grad()
                inputs = data['buffers']
                det_labels = data['sz_labels'].long().to(device)
                
                Nsz = inputs.shape[1]

                inputs = inputs.to(torch.DoubleTensor()).to(device)
                k_pred, _, _  = model(inputs)
               
                #loss = sens_loss_fn(k_pred, det_labels, weight[epoch], device)
                loss = det_loss(k_pred.reshape(-1, 2), det_labels.reshape(-1))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                
            
            for batch_idx, data in enumerate(val_loader):

                optimizer.zero_grad()
                with torch.no_grad():

                    inputs = data['buffers']
                    det_labels = data['sz_labels'].long().to(device)
    
                    Nsz = inputs.shape[1]
    
                    inputs = inputs.to(torch.DoubleTensor()).to(device)
                    k_pred, hc, _  = model(inputs)

                    #loss = sens_loss_fn(k_pred, det_labels, weight[epoch], device)
                    loss = det_loss(k_pred.reshape(-1, 2), det_labels.reshape(-1))
                    val_epoch_loss += loss.item()
                
            
            epoch_loss = epoch_loss/train_len
            val_epoch_loss = val_epoch_loss/val_len
            
            torch.cuda.empty_cache()
        
            
            train_losses.append(epoch_loss)
            #val_losses.append(val_epoch_loss)

            #if epoch_loss <= train_losses[-1] and epoch_val_loss <= val_losses[-1]:
            #torch.save(model.state_dict(), '/home/deeksha/EEG_Sz/GenProc/results/lstmAE_'+pt+str(ch_id)+'.pth.tar')        
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} '.format(epoch, epoch_loss, val_epoch_loss))
             
            torch.save(model.state_dict(), save_loc+savename+ '.pth.tar')
            np.save(save_loc+ savename+'_train.npy', train_losses)
            np.save(save_loc+savename+'_val.npy', val_losses)
			if len(val_losses) > 10 and val_losses[-1] > min(val_losses[-11:-1]):
    			break
            #if epoch>(maxiter-10):
            #    count = epoch%10
            #    torch.save(model.state_dict(), save_loc+savename+'_'+ str(count)+'.pth.tar')

        #del model, optimizer, train_loader, validation_loader
        
        return train_losses, val_losses
