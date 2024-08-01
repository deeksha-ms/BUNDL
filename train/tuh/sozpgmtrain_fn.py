from txlstm_szpool import *
from baselines import *
from utils import *
from dataloader import *

import numpy as np
import pandas as pd
import json
import os

import torch
import torch.nn as nn
import sklearn.metrics as metrics

import scipy
import scipy.special

import torch.nn.functional as F

def noiseaware_finetunesoz(data_root, modelname, mn_fn, cvfold, 
        hparams = {'lr':0.01,'maxiter' : 30, 'beta':0.25, 'prior':0.8},
                traintype='ml_feclean', use_cuda=True):
        pooltype  = 'szpool'

        lr = hparams['lr']
        maxiter = hparams['maxiter']


        device = 'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu'
        sozloss = nn.BCELoss()
        detloss = nn.CrossEntropyLoss(weight = torch.Tensor([0.2, 0.8]).double().to(device))
        l1 = nn.L1Loss()


        manifest = read_manifest(data_root+mn_fn, ',')
        save_loc = data_root+'split'+str(cvfold)+'/models/'
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)

        pt_list = np.load(data_root+'all_pts.npy')

        train_set =  myDataSet(data_root,cvfold, manifest,
                     normalize=True,train=True,
                    maxSeiz = 10)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

        
        if modelname =='txmlp':
            pretrainedfn = save_loc+modelname+'_pretrained_cv' +str(cvfold)+'_'+ str(1e-04) + '_' +'.pth.tar'
            model = txlstm_szpool(transformer_dropout=0.15, pretrained=pretrainedfn, device=device, modelname=modelname, pooltype=pooltype)
        elif modelname =='ctl':
            pretrainedfn = save_loc+modelname+'_pretrained_cv' +str(cvfold)+'_'+ str(1e-04) + '_' +'.pth.tar'
            model = txlstm_szpool(transformer_dropout=0.15, pretrained=pretrainedfn, device=device, modelname=modelname, pooltype=pooltype)
        elif modelname =='tgcn':
            pretrainedfn = save_loc+modelname+'_pretrained_cv' +str(cvfold)+'_'+ str(1e-04) + '_' +'.pth.tar'
            model = txlstm_szpool(transformer_dropout=0.15, pretrained=pretrainedfn, device=device, modelname=modelname, pooltype=pooltype)
        elif modelname =='txlstm_nomask':
            pretrainedfn = save_loc+modelname+'_pretrained_cv' +str(cvfold)+'_'+ str(1e-05) + '_' +'.pth.tar'
            model = txlstm_szpool(transformer_dropout=0.2,  pretrained=pretrainedfn, device=device, modelname=modelname, pooltype=pooltype)
        else:
            if traintype== 'ml_feclean':
                print(traintype)
                pretrainedfn= save_loc+modelname+'_ml_feclean_cv'+str(cvfold)+'_'+str(1e-05) +'_'+str(0.001)+'.pth.tar'
            else:
                print(traintype)
                pretrainedfn= save_loc+modelname+'_pretrain_unc_cv'+str(cvfold)+'_'+str(1e-05) +'.pth.tar'

            #pretrainedfn = save_loc+modelname+'_pgm_fe_cv' +str(cvfold)+'_'+ str(1e-05) + '_' +'.pth.tar'
            model = txlstm_szpool(transformer_dropout=0.1, pretrained=pretrainedfn, device=device, modelname=modelname, pooltype=pooltype)


        savename = modelname+'_'+traintype+'_soztrained_cv'+str(cvfold)+'_'+str(lr)
        for n, params in model.detector.named_parameters():
            if n.startswith('multi'):
                params.requires_grad = False

        model = model.to(device)
        model.double()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        train_losses = []
        val_losses = []
        for epoch in range(1, maxiter+1):

            epoch_loss = 0.0
            train_len = len(train_loader)
            for batch_idx, data in enumerate(train_loader):

                optimizer.zero_grad()
                inputs = data['buffers']
                Nsz = inputs.shape[1]
                soz_labels = data['onset map'].to(device)
                soz_labels = soz_labels.repeat(Nsz, 1)
                det_labels = data['sz_labels'].long().to(device)


                inputs = inputs.to(torch.DoubleTensor()).to(device)
                k_pred, psoz, z, a  = model(inputs)
                del inputs
                loss1 = detloss(k_pred.reshape(-1,2), det_labels.reshape(-1))
                loss2 = sozloss(psoz.float(), soz_labels) #+ sozloss(ysoz.float(), soz_labels)
                loss = 0.1*loss1 + loss2 #+ 0.1*l1(psoz, torch.zeros_like(psoz))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()#*y.size(0)
                del det_labels
                if batch_idx%50==0:
                    print('Epoch: ', epoch, ' batch id: ', batch_idx, 'Loss: ', loss.item())
            '''
            epoch_val_loss = 0.
            val_len = len(validation_loader)
            for batch_idx,data in enumerate(validation_loader):

                optimizer.zero_grad()
                with torch.no_grad():
                    optimizer.zero_grad()
                    inputs = data['buffers']
                    Nsz = inputs.shape[1]
                    soz_labels = data['onset map'].to(device)
                    soz_labels = soz_labels.repeat(Nsz, 1)
                    det_labels = data['sz_labels'].long().to(device)
                    

                    inputs = inputs.to(torch.DoubleTensor()).to(device)
                    k_pred, psoz, z, a  = model(inputs)
                    del inputs
                    loss1 = detloss(k_pred.reshape(-1,2), det_labels.reshape(-1))
                    loss2 = sozloss(psoz.float(), soz_labels) #+ sozloss(ysoz.float(), soz_labels)
                    loss = 0.1*loss1 + loss2 #+ 0.1*l1(psoz, torch.zeros_like(psoz))
                    epoch_val_loss += loss
            '''
            epoch_loss = epoch_loss/train_len
            #epoch_val_loss = epoch_val_loss/val_len
            train_losses.append(epoch_loss)
            #val_losses.append(epoch_val_loss)

            #if epoch_loss <= train_losses[-1] and epoch_val_loss <= val_losses[-1]:
            #torch.save(model.state_dict(), '/home/deeksha/EEG_Sz/GenProc/results/lstmAE_'+pt+str(ch_id)+'.pth.tar')        
            #print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, epoch_loss, epoch_val_loss))

            torch.save(model.state_dict(), save_loc+savename +'.pth.tar')


        del model, optimizer, train_loader#, validation_loader
        return None

