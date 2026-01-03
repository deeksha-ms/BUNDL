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

t_true_dict = {}

def selfadapt_loss(y_pred, y_true, epoch, fn,  alpha=0.9):
    y_pred = F.softmax(y_pred, -1)
    with torch.no_grad():
        y_true = F.one_hot(y_true, num_classes=2)
        if epoch >60:
            y_true = alpha*t_true_dict[fn] + (1-alpha)* y_pred
        sample_weight, _ = torch.max(y_true, 1)
        
    loss = -1*y_true * torch.log(y_pred)
    loss = sample_weight * loss.sum(-1)
    loss = loss.sum()/sample_weight.sum()
    
    t_true_dict[fn] = y_true

    return loss



def selfadapt_train(data_root, modelname, mn_fn, cvfold, 
                    hparams = {'lr':1e-06, 'maxiter':3, 'prior':0.8, 'beta':0.25},
                    traintype='pretrain',
                    use_cuda=False):
        
        lr = hparams['lr']
        maxiter = hparams['maxiter']
        device = 'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu'
        if use_cuda:
            torch.cuda.empty_cache()

        #det_loss = nn.CrossEntropyLoss(weight = torch.Tensor([0.2, 0.8]).double().to(device))
        manifest = read_manifest(data_root+mn_fn, ',')
        save_loc = data_root+'split'+str(cvfold)+'/models/'
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)

        pt_list = np.load(data_root+'all_pts.npy')


        train_set =  myDataSet(data_root,cvfold, manifest,
                     normalize=True,train=True,
                    maxSeiz = 10)
        val_set = myDataSet(data_root,cvfold, manifest,
                     normalize=True,train=False,
                    maxSeiz = 10):
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

        if modelname == 'txlstm':
            model = transformer_lstm(transformer_dropout=0.2, device=device)
        elif modelname == 'sztrack':
            model = sztrack()
        elif modelname == 'tgcn':
            model = TGCN()
        else:
            model = CNN_BLSTM()

        savename = modelname+'_'+traintype+'_cv'+str(cvfold)+'_'+str(lr)
        model.double()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        sigmoid = torch.nn.Sigmoid()

        train_losses = []
        train_loss_clean = []
        train_loss_corrupt = []
        val_losses = []
        for epoch in range(1, maxiter+1):

            epoch_loss = 0.0
            train_len = len(train_loader)
            val_len = len(val_loader)
            val_epoch_loss = 0.0
            for batch_idx, data in enumerate(train_loader):
                fn = data['patient numbers']
                optimizer.zero_grad()
                inputs = data['buffers']
                det_labels = data['sz_labels'].long().to(device)

                Nsz = inputs.shape[1]

                inputs = inputs.to(torch.DoubleTensor()).to(device)
                k_pred, hc, _  = model(inputs)

                #loss = sens_loss_fn(k_pred, det_labels, weight[epoch], device)
                loss = selfadapt_loss(k_pred.reshape(-1, 2), det_labels.reshape(-1), epoch, fn[0])
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            for batch_idx, data in enumerate(val_loader):

                optimizer.zero_grad()
                with torch.no_grad():
                    fn = data['patient numbers']
                    inputs = data['buffers']
                    det_labels = data['sz_labels'].long().to(device)

                    Nsz = inputs.shape[1]

                    inputs = inputs.to(torch.DoubleTensor()).to(device)
                    k_pred, hc, _  = model(inputs)

                    #loss = sens_loss_fn(k_pred, det_labels, weight[epoch], device)
                    loss = selfadapt_loss(k_pred.reshape(-1, 2), det_labels.reshape(-1), epoch, fn[0])
                    val_epoch_loss += loss.item()


            epoch_loss = epoch_loss/train_len
            val_epoch_loss = val_epoch_loss/val_len

            torch.cuda.empty_cache()
            #schedule lr

            train_losses.append(epoch_loss)
            val_losses.append(val_epoch_loss)

            #if epoch_loss <= train_losses[-1] and epoch_val_loss <= val_losses[-1]:
            #torch.save(model.state_dict(), '/home/deeksha/EEG_Sz/GenProc/results/lstmAE_'+pt+str(ch_id)+'.pth.tar')        
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} '.format(epoch, epoch_loss, val_epoch_loss))
	
		    if len(val_losses) > 10 and val_losses[-1] > min(val_losses[-11:-1]):
	    		break
            torch.save(model.state_dict(), save_loc+savename+ '.pth.tar')
            np.save(save_loc+ savename+'_train.npy', train_losses)
            np.save(save_loc+savename+'_val.npy', val_losses)
	
            if epoch>(maxiter-10):
                count = epoch%10
                torch.save(model.state_dict(), save_loc+savename+'_'+ str(count)+'.pth.tar')

        #del model, optimizer, train_loader, validation_loader

        return train_losses, val_losses
