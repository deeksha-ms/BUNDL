import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import scipy
from scipy.io import loadmat as loadmat
import os


class SimDataSet_v3(Dataset):
    def __init__(self,
                 root,
                 pt_list, 
                 normalize=True,
                 noise_type = None
                 ):
        self.root = root
        self.ptlist = pt_list

        self.normalize = normalize
        self.nchn = 19
        self.label_prefix = '_label.mat'
        self.label_key = 'label'
        if noise_type:
            self.label_prefix = '_'+ str(noise_type)+'_label.mat'
            self.label_key = 'newlabel'
            
 
    def __getitem__(self, idx):
        
        pt = self.ptlist[idx]
        ptroot =  self.root+'pt'+str(pt)+'/'
        fnames = os.listdir(ptroot)
        xlocs = list(filter(lambda x:x.endswith("eeg.mat") , fnames) )
  
        X, Y, Ytrue = np.zeros((len(xlocs), 600, 19, 200)), np.zeros((len(xlocs), 600)),  np.zeros((len(xlocs), 600))
        for k, xloc in enumerate(xlocs):
            Z = loadmat(ptroot+xloc)['data1020']
            Z = Z.transpose((1, 2,0)).reshape(19, -1)   #because of a mistake in matlab-python compatibility
            X[k] = Z.reshape(19, 600, 200).transpose((1, 0, 2))
            
            Y[k] = loadmat(ptroot+ ("_").join(xloc.split("_")[:-1]) + self.label_prefix )[self.label_key][:,0]
            Ytrue[k] = loadmat(ptroot+ ("_").join(xloc.split("_")[:-1]) + '_label.mat' )['label'][:,0]
       
        if self.normalize:
            X = (X - np.mean(X))/np.std(X)

        return {'patient numbers': pt,
                'buffers':torch.Tensor(X), #original
                'sz_labels':torch.Tensor(Y),  #sz
                'true_labels': torch.Tensor(Ytrue)
               }  
    
    def load_onset_map(self, mnitem):
        req_chn = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 't3', 'c3', 'cz', 
                    'c4', 't4', 't5', 'p3', 'pz', 'p4', 't6', 'o1', 'o2']
        soz = np.zeros(len(req_chn))
        for i,chn in enumerate(req_chn):
            if mnitem[chn] != '':
                soz[i] = json.loads(mnitem[chn])         
        return soz
    
    def __len__(self):                   
        return len(self.ptlist)


class CHBDataset(Dataset):                 
    def __init__(self, ptlist, manifest, root='/project/seizuredet/data/'):
        self.root = root
        self.ptlist = ptlist
        self.manifest = manifest

        self.mnlist = [mnitem for  _, mnitem in manifest.iterrows() if mnitem['pt_id'] in ptlist]
    

    def __len__(self):
        return len(self.mnlist)
    
    def __getitem__(self, idx):
        mnitem = self.mnlist[idx]
        fn = mnitem['fn']
        sz_starts = np.ceil(json.loads(mnitem['sz_starts']))
        sz_ends = np.ceil(json.loads(mnitem['sz_ends']))
        xloc = mnitem['loc']
        yloc =  self.root + 'chb_training_labels/' + mnitem['pt_id'] + '/' + fn.split('dow')[0] + 'label.npy'
        # X = np.load(xloc)[:self.maxSeiz, :,:,:]
        # Y = np.load(yloc)[:self.maxSeiz] --> figure out maxSeiz 
        X = np.load(xloc)
        Y = np.load(yloc)
        # soz = self.load_onset_map(mnitem)

        return {'patient numbers': fn,
                'buffers':torch.Tensor(X), #original
                'sz_labels':torch.Tensor(Y),  #sz
               }

class SienaDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, fs = 200):
        self.df = pd.read_csv(csv_file)
        self.fs = fs
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        rec_path = self.df.iloc[idx]['recording']
        label_path = self.df.iloc[idx]['labels']
        
        X = np.load(rec_path)
        Y = np.load(label_path)

        # resample to self.fs
        if X.shape[-1] != self.fs:
            X = sig.resample(X, int(self.fs), axis =-1)
        if X.shape[0] != Y.shape[0]:
            # pad x with zeros X along axis 0 out of the three . X of shape (time, channels, samples)
            pad_width = Y.shape[0] - X.shape[0]
            X = np.pad(X, ((0,pad_width),(0,0), (0,0)), mode='constant', constant_values=0)    
       
        
        X = (X-np.mean(X))/np.std(X) 
        # add dummy axis
        X = np.expand_dims(X, axis=0)
        return {'eeg': X, 
                'sz_labels': Y}

class TUHDataset(Dataset):
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
