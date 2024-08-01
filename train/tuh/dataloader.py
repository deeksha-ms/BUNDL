import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

from utils import *
'''
###from tuh_loader import *
###from visualization import *
class pretrainLoader(Dataset):
    def __init__(self,root,  ptlist, manifest,
                 normalize=True, transform=None,
                 maxMask=10, sigma=0.2, maxSeiz = 10, addNoise=False, input_mask=False, ablate=False, permute=False):
        self.ptlist = ptlist
        self.root= root
        self.mnlist = [mnitem for mnitem in manifest if json.loads(mnitem['pt_id']) in ptlist ]
        self.transform = transform
        self.normalize = normalize
        self.maxMask = maxMask
        self.nchn = 19
        self.sigma = sigma
        self.maxSeiz = maxSeiz
        self.input_mask= input_mask
        self.addNoise = addNoise
        self.ablate = ablate
        self.permute = permute
        self.chn_neighbours = {0: [1,2,3,4],
                  1: [0,4,5,6],
                  2: [0,3,7,8],
                  3: [0,2,4,8,9],
                  4: [0,1,3,5,9],
                  5: [1,4,6,9,10],
                  6: [1,5,10,11],
                  7: [2,8,12,13,17],
                  8: [2,3,7,9,12,13,14],
                  9: [3,4,5,8,10,13,14,15],
                 10: [5,6,9,11,14,15,16],
                 11: [6, 10, 15, 16, 18],
                 12: [7, 8, 13, 17],
                 13: [7, 8, 9, 12, 14, 17],
                 14: [8,9,10,13,15,17,18],
                 15: [9,10,11,14,16,18],
                 16: [10,11,15,18],
                 17: [7,12,13,14,18],
                 18: [11, 14,15, 16, 17]}
'''
class myDataSet_withcrop(Dataset):
    def __init__(self,root,nsplit, manifest, 
                 normalize=True,train=True,
                maxSeiz = 10
                 ):
        self.root = root+'/clean_data/'
        if train:
            ptlist = np.load(root+'split'+str(nsplit)+'/train_pts.npy')
            self.crop = True
        else:
            ptlist = np.load(root+'split'+str(nsplit)+'/val_pts.npy')
            self.crop = False
        self.mnlist = [mnitem for mnitem in manifest if json.loads(mnitem['pt_id']) in ptlist ]
        self.normalize = normalize
        self.nchn = 19
        self.maxSeiz = maxSeiz 
 
    def __getitem__(self, idx):

        mnitem = self.mnlist[idx]
        fn = mnitem['fn']
        pt = int(mnitem['pt_id'])
        xloc = self.root+fn
        yloc = xloc.split('.')[0] + '_label.npy'

        X = np.load(xloc)[:self.maxSeiz, :,:,:]
        Y = np.load(yloc)[:self.maxSeiz]
        soz = self.load_onset_map(mnitem)
        if self.normalize:
            X = (X - np.mean(X))/np.std(X)
        if self.crop:
            X, Y = self.crop_data(X, Y)
        return {'patient numbers': fn,
                'buffers':torch.Tensor(X), #original
                'sz_labels':torch.Tensor(Y),  #sz
                'onset map':torch.Tensor(soz), #soz
               }  
    
    def load_onset_map(self, mnitem):
        req_chn = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 't3', 'c3', 'cz', 
                    'c4', 't4', 't5', 'p3', 'pz', 'p4', 't6', 'o1', 'o2']
        soz = np.zeros(len(req_chn))
        for i,chn in enumerate(req_chn):
            if mnitem[chn] != '':
                soz[i] = json.loads(mnitem[chn])
           
        return soz
    
    def crop_data(self, X, Y):
        #crop to 2 min
        Nsz, T , C, d = X.shape
        cropped_Y = []
        cropped_X = []
        for n in range(Nsz):
            delta = Y[n, 1:] - Y[n, :-1]
            try:
                s = int(np.where(delta==1)[0][0] + 1)
            except:
                s = 0
            try:
                e = int(np.where(delta==-1)[0][0] + 1)
            except:
                e = T
            szlen = e-s
            
            if szlen>=15:
                szt = np.random.choice(np.arange(15, min(szlen, 45)+1))
                end = min(T, s+szt)
                start = max(0, end-60)
                                
            else:
                end = e
                start  = max(0, e-60)
            reclen = end-start
            xtemp = X[n, start:end, :, :]
            ytemp = Y[n, start:end]
            if reclen<60:
                xtemp = np.concatenate((np.zeros((60-reclen, C, d)), xtemp), 0)
                ytemp = np.concatenate((np.zeros(60-reclen), ytemp), 0)
            cropped_X.append(xtemp)
            cropped_Y.append(ytemp)             
        
        return np.array(cropped_X), np.array(cropped_Y)
    
    def __len__(self):                       #gives number of recordings
        return len(self.mnlist)


class myDataSet(Dataset):
    def __init__(self,root,nsplit, manifest, 
                 normalize=True,train=True,
                maxSeiz = 10
                 ):
        self.root = root+'/clean_data/'
        if train:
            ptlist = np.load(root+'split'+str(nsplit)+'/train_pts.npy')
        else:
            ptlist = np.load(root+'split'+str(nsplit)+'/val_pts.npy')
        self.mnlist = [mnitem for mnitem in manifest if json.loads(mnitem['pt_id']) in ptlist ]
        self.normalize = normalize
        self.nchn = 19
        self.maxSeiz = maxSeiz 
 
    def __getitem__(self, idx):

        mnitem = self.mnlist[idx]
        fn = mnitem['fn']
        pt = int(mnitem['pt_id'])
        isnoisy = False
        xloc = self.root+fn
        yloc = xloc.split('.')[0] + '_label.npy'

        X = np.load(xloc)[:self.maxSeiz, :,:,:]
        Y = np.load(yloc)[:self.maxSeiz]
        soz = self.load_onset_map(mnitem)
        if self.normalize:
            X = (X - np.mean(X))/np.std(X)
            
        noise_labels =  []
        xhat = []
      
        return {'patient numbers': fn,
                'buffers':torch.Tensor(X), #original
                'sz_labels':torch.Tensor(Y),  #sz
                'onset map':torch.Tensor(soz), #soz
                'isnoisy': isnoisy
               }  
    
    def load_onset_map(self, mnitem):
        req_chn = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 't3', 'c3', 'cz', 
                    'c4', 't4', 't5', 'p3', 'pz', 'p4', 't6', 'o1', 'o2']
        soz = np.zeros(len(req_chn))
        for i,chn in enumerate(req_chn):
            if mnitem[chn] != '':
                soz[i] = json.loads(mnitem[chn])
           
        return soz
    
    def __len__(self):                       #gives number of recordings
        return len(self.mnlist)

class myDataSet_test(Dataset):
    def __init__(self,root,nsplit, manifest,
                 normalize=True,train=True,
                maxSeiz = 10
                 ):
        self.root = root+'/clean_data/'
        if train:
            ptlist = np.load(root+'split'+str(nsplit)+'/train_pts.npy')[:24]
        else:
            ptlist = np.load(root+'split'+str(nsplit)+'/val_pts.npy')
        self.mnlist = [mnitem for mnitem in manifest if json.loads(mnitem['pt_id']) in ptlist ]
        self.normalize = normalize
        self.nchn = 19
        self.maxSeiz = maxSeiz

    def __getitem__(self, idx):

        mnitem = self.mnlist[idx]
        fn = mnitem['fn']
        pt = int(mnitem['pt_id'])
        isnoisy = False
        xloc = self.root+fn
        yloc = xloc.split('.')[0] + '_label.npy'

        X = np.load(xloc)[:self.maxSeiz, :,:,:]
        Y = np.load(yloc)[:self.maxSeiz]
        soz = self.load_onset_map(mnitem)
        if self.normalize:
            X = (X - np.mean(X))/np.std(X)

        noise_labels =  []
        xhat = []

        return {'patient numbers': fn,
                'buffers':torch.Tensor(X), #original
                'sz_labels':torch.Tensor(Y),  #sz
                'onset map':torch.Tensor(soz), #soz
                'isnoisy': isnoisy
               }
    def load_onset_map(self, mnitem):
        req_chn = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 't3', 'c3', 'cz',
                    'c4', 't4', 't5', 'p3', 'pz', 'p4', 't6', 'o1', 'o2']
        soz = np.zeros(len(req_chn))
        for i,chn in enumerate(req_chn):
            if mnitem[chn] != '':
                soz[i] = json.loads(mnitem[chn])

        return soz

    def __len__(self):                       #gives number of recordings
        return len(self.mnlist)
