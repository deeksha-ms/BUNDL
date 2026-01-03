from models_nal import *
from utils import *
#from dataloader import *
import numpy as np
import pandas as pd
import json
import os
import scipy
import scipy.special
import torch
import torch.nn as nn
import sklearn.metrics as metrics
#from pyt_enc import *
import torch.nn.functional as F
detloss = nn.CrossEntropyLoss(weight = torch.Tensor([0.2, 0.8]).double().to('cuda:0'))

def nal_finetune(data_root, modelname, mn_fn, cvfold,
        hparams = {'lr':1e-04, 'maxiter':100, 'prior':0.8, 'beta':1.0},
                      traintype='s_model',
                      use_cuda=False):
        lr = hparams['lr']
        maxiter = hparams['maxiter']
        beta  = hparams['beta']
        prior = hparams['prior']
        device = 'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu'
        if use_cuda:
            torch.cuda.empty_cache()
        print(traintype, modelname)
        #detloss = nn.CrossEntropyLoss(weight = torch.Tensor([0.2, 0.8]).double().to(device))
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
                    maxSeiz = 10)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

        s_model = True if traintype=='s_model' else False
        cm_theta = torch.tensor(np.load(modelname.split('_')[0]+str(cvfold)+'_cmtheta_pretrain_unc.npy')).double()
        if modelname == 'txlstm_nal':
            model = transformer_lstm_nal(transformer_dropout=0.2, device=device, s_model=s_model, cm_theta = cm_theta)
        elif modelname == 'sztrack_nal':
            model = sztrack_nal( s_model=s_model, cm_theta = cm_theta)
        elif modelname == 'tgcn_nal':
            model = TGCN_nal(s_model=s_model, cm_theta = cm_theta, device=device)
        elif modelname == 'tx3lstm_nal':
            model = transformer_3lstm(transformer_dropout=0.1, device=device)
        else:
            model = CNN_BLSTM_nal(s_model=s_model, cm_theta = cm_theta, device=device)
        

        savename = modelname+'_'+traintype+'_cv'+str(cvfold)+'_'+str(lr)        
        oldlr = {'txlstm':1e-05, 'tgcn': 1e-06, 'cnnlstm':0.01, 'cnnblstm':0.01}

        ########## initialize with pretrained model, confusion matrix of pretrained or s_model
        if s_model:        
            oldsavename = modelname.split('_')[0]+'_pretrain_unc_cv'+str(cvfold)+'_'+str(oldlr[modelname.split('_')[0]])
            state_dict = torch.load(save_loc + oldsavename+'.pth.tar')
            with torch.no_grad():

                for n, p in model.named_parameters():
                    try:
                        p.data = state_dict[n].double()
                    except:
                        continue
            
        else: #c_model case
            oldsavename = modelname+'_s_model_cv'+str(cvfold)+'_'+str(oldlr[modelname.split('_')[0]])
            state_dict = torch.load(save_loc + oldsavename+'.pth.tar')
            #for n, p in state_dict.named_parameters():
            #    model[n].copy_(p)
            model.load_state_dict(state_dict)

        #print(nothing)
        model.double()
        model.to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        sigmoid = torch.nn.Sigmoid()
        train_loss_clean = []
        train_loss_corrupt = []
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

                inputs = inputs.to(torch.DoubleTensor()).to(device)
                k_pred, noise_pred, _  = model(inputs)

                y_samples = []

                loss = 0.5* detloss(k_pred.reshape(-1, 2), det_labels.reshape(-1)) + 0.5* detloss(noise_pred.reshape(-1, 2), det_labels.reshape(-1))
                
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()


            for batch_idx, data in enumerate(val_loader):
                with torch.no_grad():
                    optimizer.zero_grad()
                    inputs = data['buffers']
                    det_labels = data['sz_labels'].long().to(device)
                    b, Nsz, T, _, _ = inputs.shape

                    inputs = inputs.to(torch.DoubleTensor()).to(device)
                    k_pred, noise_pred, _  = model(inputs)

                    y_samples = []

                    loss = 0.5* detloss(k_pred.reshape(-1, 2), det_labels.reshape(-1)) + 0.5* detloss(noise_pred.reshape(-1, 2), det_labels.reshape(-1))

                    val_epoch_loss += loss.item()
            epoch_loss = epoch_loss/train_len
            val_epoch_loss = val_epoch_loss/val_len
            torch.cuda.empty_cache()
            #schedule lr

            train_losses.append(epoch_loss)
            val_losses.append(val_epoch_loss)
	    if len(val_losses) > 10 and val_losses[-1] > min(val_losses[-11:-1]):
    		break


            #if epoch_loss <= train_losses[-1] and epoch_val_loss <= val_losses[-1]:
            #torch.save(model.state_dict(), '/home/deeksha/EEG_Sz/GenProc/results/lstmAE_'+pt+str(ch_id)+'.pth.tar')        
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} '.format(epoch, epoch_loss, val_epoch_loss))

            torch.save(model.state_dict(), save_loc+savename+ '.pth.tar')
            np.save(save_loc+ savename+'_train.npy', train_losses)
            np.save(save_loc+savename+'_val.npy', val_losses)
        #del model, optimizer, train_loader, validation_loader

        return train_losses, val_losses