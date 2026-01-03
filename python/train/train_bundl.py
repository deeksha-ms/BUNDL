From BUNDL import *
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

def initialize_model_layers(model, random_seed):
    """
    Initialize all layers of a PyTorch model with a given random seed.
    Handles various layer types including transformers.
    
    Args:
        model: PyTorch model (nn.Module)
        random_seed: Integer seed for reproducibility
    """
    # Set the random seed for reproducibility
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    def init_weights(m):
        """Helper function to initialize weights based on layer type."""
        # Linear layers
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        # Convolutional layers
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        # Batch Normalization layers
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        
        # Layer Normalization
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        
        # LSTM/GRU layers
        elif isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # Embedding layers
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.01)
        
        # Multi-head Attention layers
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                nn.init.xavier_uniform_(m.in_proj_weight)
            else:
                # Separate Q, K, V projections
                nn.init.xavier_uniform_(m.q_proj_weight)
                nn.init.xavier_uniform_(m.k_proj_weight)
                nn.init.xavier_uniform_(m.v_proj_weight)
            
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
            
            nn.init.xavier_uniform_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)
        
        # Transformer Encoder Layer
        elif isinstance(m, nn.TransformerEncoderLayer):
            # The individual components (Linear, LayerNorm, MultiheadAttention)
            # will be initialized by their own isinstance checks
            pass
        
        # Transformer Decoder Layer
        elif isinstance(m, nn.TransformerDecoderLayer):
            # The individual components will be initialized by their own isinstance checks
            pass
        
        # Full Transformer Encoder/Decoder
        elif isinstance(m, (nn.TransformerEncoder, nn.TransformerDecoder)):
            # The individual layers will be initialized recursively
            pass
        
        # Full Transformer model
        elif isinstance(m, nn.Transformer):
            # The individual components will be initialized recursively
            pass
    
    # Apply initialization to all modules
    model.apply(init_weights)
    
    return model

def train_bundl(data_root, modelname, mn_fn,
                       maxiter = 100, lr = 1e-05, traintype='pretrain', load_weights=None,  
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
        
        savename = modelname+'_'+traintype+'_cv'+str(cvfold)+'_'+str(lr)+'_'+str(version)+'_'+str(noise_type)+'_'+str(uq_type)
        model.double()
        model.to(device)
	if load_weights:
  		model.load_state_dict(torch.load(load_weights, map_location=device))
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
               
		p_avg, p_sentropy = compute_uncertainty(model, inputs, det_labels, uq_type)
		p_avg = torch.tensor(p_avg).to(device).reshape(-1)
                p_sentropy = torch.tensor(p_sentropy).to(device).reshape(-1)
		loss = detloss(k_pred.reshape(-1, 2), det_labels.reshape(-1), p_avg, p_sentropy)
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

                    p_avg, p_sentropy = compute_uncertainty(model, inputs, det_labels, uq_type)
		    p_avg = torch.tensor(p_avg).to(device).reshape(-1)
                    p_sentropy = torch.tensor(p_sentropy).to(device).reshape(-1)
		    loss = detloss(k_pred.reshape(-1, 2), det_labels.reshape(-1), p_avg, p_sentropy)
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
