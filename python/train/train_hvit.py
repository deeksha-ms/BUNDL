from dataloader import pretrainLoader

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import json
import math
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import sklearn

from hvit import HVIT
def chbtrain_dul(data_root, modelname='hvit', cvfold=1,
        hparams = {'lr':1e-04, 'maxiter':10 },
                      traintype='dul',
                      use_cuda=False):
        lr = hparams['lr']
        maxiter = hparams['maxiter']
        
        device = 'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu'
        if use_cuda:
            torch.cuda.empty_cache()
        
        detloss = nn.CrossEntropyLoss(weight = torch.Tensor([0.03, 0.97]).double().to(device))
       
        save_loc = data_root+'/hvit/'
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)
            
        #train_pts, val_pts = train_test_split(ptlist, test_size=1, random_state=42, shuffle=True)
        manifest = pd.read_csv('/project/seizuredet/data/new_seizures_only_windowed_manifest.csv')
        #ptlist = list(manifest['pt_id'].unique())
        #ptlist.remove('chb21')
        ptlist = np.load(data_root + 'ptlist' + str((cvfold-1)//10) + '.npy')
        
        count = (cvfold-1)%10 + 1
        if count<=7:
            val_pts = ptlist[(count-1)*2 : (count-1)*2+2]  #val size 2
        else:
            val_pts = ptlist[(count-3)*3-1:(count-3)*3+3-1 ] #val size 3, 
        train_pts = [pt for pt in ptlist if pt not in val_pts]
        np.save(save_loc+'trainpts'+str(cvfold)+'.npy', train_pts)
        np.save(save_loc+'valpts'+str(cvfold)+'.npy', val_pts)
        
        print(val_pts)
        print("\nStarting on fold ", cvfold, count, len(val_pts), len(train_pts))
        train_set =  pretrainLoader(train_pts, manifest)
        val_set =  pretrainLoader(val_pts, manifest)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=2)
        
      
        if modelname == 'txlstm':
            model = transformer_lstm(transformer_dropout=0.2, device=device)
        elif modelname == 'hvit':
            model = HVIT(200, 18, device=device)
        elif modelname == 'tgcn':
            model = TGCN()
        else:
            model = CNN_BLSTM()

        
        model.double()
        model.to(device)
       
        savename = modelname+'_'+traintype+'_cv'+str(cvfold)+'_'+str(lr)+'_final'
    
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        sigmoid = torch.nn.Sigmoid()

        val_losses = []
        train_losses = []
        
        for epoch in range(1, maxiter+1):

            epoch_loss = 0.0
            train_len = len(train_loader)
            val_len = len(val_loader)
            val_epoch_loss = 0.0
            for batch_idx, data in enumerate(train_loader):

                optimizer.zero_grad()
                inputs = data['buffers']
                det_labels = data['sz_labels'].long().to(device).reshape(-1)
                b, T, C, d = inputs.shape
                
                inputs = inputs.view(b*T, 1, C, d ).transpose(2, 3)
                inputs = inputs.to(torch.DoubleTensor()).to(device)
                
                logits, mu, logvar  = model(inputs, train=True)
                
                loss, ce, kl = model.compute_loss(logits, mu, logvar, det_labels.reshape(-1), weighted=False)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                #print('\t', batch_idx, ': Loss =', loss.item()  )
            for batch_idx, data in enumerate(val_loader):
                optimizer.zero_grad()
                with torch.no_grad():

                    inputs = data['buffers']
                    det_labels = data['sz_labels'].long().to(device).reshape(-1)
                    b, T, C, d = inputs.shape
                    
                    inputs = inputs.view(b*T, 1, C, d ).transpose(2,3)
                    inputs = inputs.to(torch.DoubleTensor()).to(device)
                    logits, mu, logvar  = model(inputs, train=False)
                    
                    loss, ce, kl = model.compute_loss(logits, mu, logvar, det_labels.reshape(-1), weighted=False)
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

def read_manifest(filename, d=';'):
   
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=d)
        dicts = list(reader)
    return dicts

def tuhtrain_dul(data_root, modelname='hvit', cvfold=1,
        hparams = {'lr':1e-04, 'maxiter':10 },
                      traintype='dul',
                      use_cuda=False):
        lr = hparams['lr']
        maxiter = hparams['maxiter']
        
        device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        if use_cuda:
            torch.cuda.empty_cache()
        
        detloss = nn.CrossEntropyLoss(weight = torch.Tensor([0.2, 0.8]).double().to(device))
       
        save_loc = data_root+'/hvit/'
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)
            
        #train_pts, val_pts = train_test_split(ptlist, test_size=1, random_state=42, shuffle=True)
        manifest = read_manifest(data_root + 'tuh_single_windowed_manifest.csv', ',')
        #ptlist = list(manifest['pt_id'].unique())
        #ptlist.remove('chb21')
        #ptlist = np.load(data_root + 'ptlist' + str((cvfold-1)//10) + '.npy')
        
        val_pts = np.load(data_root + 'split'+str(cvfold) + '/val_pts.npy' )
        train_pts = np.load(data_root + 'split'+str(cvfold) + '/train_pts.npy' )



        #np.save(save_loc+'trainpts'+str(cvfold)+'.npy', train_pts)
        #np.save(save_loc+'valpts'+str(cvfold)+'.npy', val_pts)
        
        print("\nStarting on fold ", cvfold, len(val_pts), len(train_pts))
        train_set = myDataSet(data_root,cvfold, manifest, train=True)
        val_set =  myDataSet(data_root,cvfold, manifest, train=False)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=2)
        
      
        if modelname == 'txlstm':
            model = transformer_lstm(transformer_dropout=0.2, device=device)
        elif modelname == 'hvit':
            model = HVIT(200, 19, device=device)
        elif modelname == 'tgcn':
            model = TGCN()
        else:
            model = CNN_BLSTM()

        
        model.double()
        model.to(device)
       
        savename = modelname+'_'+traintype+'_cv'+str(cvfold)+'_'+str(lr)+'_check'
    
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        sigmoid = torch.nn.Sigmoid()

        val_losses = []
        train_losses = []
        
        for epoch in range(1, maxiter+1):

            epoch_loss = 0.0
            train_len = len(train_loader)
            val_len = len(val_loader)
            val_epoch_loss = 0.0
            for batch_idx, data in enumerate(train_loader):

                optimizer.zero_grad()
                inputs = data['buffers']
                det_labels = data['sz_labels'].long().to(device).reshape(-1)
                b, N, T, C, d = inputs.shape
                
                inputs = inputs.view(b*N*T, 1, C, d ).transpose(2, 3)
                inputs = inputs.to(torch.DoubleTensor()).to(device)
                
                logits, mu, logvar  = model(inputs, train=True)
                
                loss, ce, kl = model.compute_loss(logits, mu, logvar, det_labels.reshape(-1), weighted=True)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                #print('\t', batch_idx, ': Loss =', ce.item(), '   ', kl.item(), '   ', model.hyperparam.item()  )
    
            for batch_idx, data in enumerate(val_loader):
                optimizer.zero_grad()
                with torch.no_grad():

                    inputs = data['buffers']
                    det_labels = data['sz_labels'].long().to(device).reshape(-1)
                    b, N, T, C, d = inputs.shape
                    
                    inputs = inputs.view(b*N*T, 1, C, d ).transpose(2,3)
                    inputs = inputs.to(torch.DoubleTensor()).to(device)
                    logits, mu, logvar  = model(inputs, train=False)
                    
                    loss, ce, kl = model.compute_loss(logits, mu, logvar, det_labels.reshape(-1), weighted=True)
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
if __name__ == "__main__":

    data_root= '/projectnb/seizuredet/bundl-chb/'

    t, v = chbtrain_dul(data_root, modelname='hvit', cvfold=1,
        hparams = {'lr':1e-04, 'maxiter':3 },
                      traintype='dul',
                      use_cuda=False)

