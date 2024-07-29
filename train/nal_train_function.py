import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import scipy
from scipy.io import loadmat as loadmat
import os
import matplotlib.pyplot as plt

from models_nal import * 
from dataloader import *




def train_nal(data_root, modelname='txlstm', cvfold=1,
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
        detloss = nn.CrossEntropyLoss(weight = torch.Tensor([0.2, 0.8]).double().to(device))
       
        save_loc = data_root+'/models/'
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)
            
        valsize = 12
        arr_pts = np.load(data_root + "pts" + str(int(cvfold-1)//10) + ".npy")
        all_pts = [str(pt) for pt in arr_pts]
        ind = int((cvfold)%10) if int((cvfold)%10) else 10
        val_pts =   all_pts[(ind-1)*valsize : ind*valsize]               
        train_pts = [pt for pt in all_pts if pt not in val_pts]
        
        train_set =  SimDataSet_v2(data_root+'/'+version+'/',train_pts, noise_type = noise_type)
        val_set = SimDataSet_v2(data_root+'/'+version+'/',val_pts, noise_type=noise_type)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=2)
        print(modelname)
        if modelname == 'txlstm':
            model = transformer_lstm_nal(transformer_dropout=0.2, device=device)
        elif modelname == 'tgcn':
            model = TGCN_nal()
        else:
            model = CNN_BLSTM_nal()
            #oldmodel = CNN_BLSTM()

        model.double()
        model.to(device)
        
        oldlr = {'txlstm':1e-05, 'tgcn': 1e-05, 'cnnlstm':0.0001, 'cnn':0.0001}
        savename = modelname+'_'+traintype+'_cv'+str(cvfold)+'_'+str(lr)+'_'+str(version)+'_'+str(noise_type)
        oldsavename = modelname+'_cel_cv'+str(cvfold)+'_'+str(oldlr[modelname])+'_'+str(version)+'_'+str(noise_type)
        state_dict = torch.load(save_loc + oldsavename+'.pth.tar', map_location=device)
        with torch.no_grad():
            for n, p in model.named_parameters():
                try:
                    p.data = state_dict[n].double()
                except:
                    continue
            
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

                inputs = inputs.to(torch.DoubleTensor()).to(device)
                k_pred, noise_pred, _  = model(inputs)

                loss = 0.5* detloss(k_pred.reshape(-1, 2), det_labels.reshape(-1)) + 0.5* detloss(noise_pred.reshape(-1, 2), det_labels.reshape(-1))
                
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                print('\t', batch_idx, ': Loss =', loss.item()  )
            for batch_idx, data in enumerate(val_loader):
                optimizer.zero_grad()
                with torch.no_grad():

                    inputs = data['buffers']
                    det_labels = data['sz_labels'].long().to(device)
                    b, Nsz, T, _, _ = inputs.shape

                    inputs = inputs.to(torch.DoubleTensor()).to(device)
                    k_pred, noise_pred, _  = model(inputs)

                    loss = 0.5* detloss(k_pred.reshape(-1, 2), det_labels.reshape(-1)) + 0.5* detloss(noise_pred.reshape(-1, 2), det_labels.reshape(-1))
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


