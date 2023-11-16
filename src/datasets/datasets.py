import torch
import pickle as pkl
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from constants import DataFoldsNew
from utils.signal_processing import pos,pos_img
from sklearn.preprocessing import minmax_scale
import h5py
import pandas as pd

class Dataset1D(Dataset):
    def __init__(self,traces_path,target,device,valid_data_ids):
        self.data_path = traces_path       
        
        self.target = target
        self.device = device

        with h5py.File(traces_path,'r') as data:
            data_lookup = {
                f"{name}_{i}":{'name':name,'split_idx':i}
                for name,sig_data in data.items()
                    if name in valid_data_ids 
                    for i in sig_data['SplitIndex']}
        
        self.data_lookup = OrderedDict(data_lookup)            
        
        

    def __len__(self):
        return len(self.data_lookup)
    
    def __getitem__(self, index):
        key = list(self.data_lookup.keys())[index]
        name,split_idx = self.data_lookup[key].values()

        with h5py.File(self.data_path,'r') as tmp_data:
            split_idx_idx = int(np.where(np.array(tmp_data[name]["SplitIndex"]) == split_idx)[0])

            split_data = tmp_data[name]['SplitData'][split_idx_idx]
            split_target = tmp_data[name][self.target][split_idx_idx]
        
        split_data = torch.from_numpy(split_data).to(self.device)
        split_data = split_data.float()

        split_target = torch.tensor(split_target).to(self.device)
        split_target = split_target.float()

        return {"data":split_data,"target":split_target,"name":key}


class DatasetCWT(Dataset):
    def __init__(self,cwt_data_path,target,device,valid_data_ids):
        self.data_path = cwt_data_path
        
        self.target = target
        self.device = device

        with h5py.File(cwt_data_path,'r') as data:
            data_lookup = {
                f"{name}_{i}":{'name':name,'split_idx':i}
                for name,sig_data in data.items()
                    if name in valid_data_ids 
                    for i in sig_data['SplitIndex']}
        
        self.data_lookup = OrderedDict(data_lookup)
        

    def __len__(self):
        return len(self.data_lookup)
    
    def __getitem__(self, index):
        key = list(self.data_lookup.keys())[index]
        name,split_idx = self.data_lookup[key].values()
        
        with h5py.File(self.data_path,'r') as tmp_data:
            split_idx_idx = int(np.where(np.array(tmp_data[name]["SplitIndex"]) == split_idx)[0])

            split_data = tmp_data[name]['SplitData'][split_idx_idx]
            split_target = tmp_data[name][self.target][split_idx_idx]
        
        split_data = np.stack([np.real(split_data),np.imag(split_data)])
        split_data = torch.from_numpy(split_data).to(self.device)
        split_data = F.interpolate(split_data.unsqueeze(0),(224,224)).squeeze(0)
        split_data = split_data.float()
                
        split_target = torch.tensor(split_target).to(self.device)
        split_target = split_target.float()

        return {"data":split_data,"target":split_target,"name":key}

class DatasetIBIS(Dataset):
    def __init__(self,ibis_data_path,target,device,valid_data_ids):
        self.data_path = ibis_data_path 
        
        self.target = target
        self.device = device
        
        with h5py.File(ibis_data_path,'r') as data:
            data_lookup = {
                f"{name}_{i}":{'name':name,'split_idx':i}
                for name,sig_data in data.items()
                    if name in valid_data_ids 
                    for i in sig_data['SplitIndex']}

        self.data_lookup = OrderedDict(data_lookup)

    def __len__(self):
        return len(self.data_lookup)
    
    def __getitem__(self, index):
        key = list(self.data_lookup.keys())[index]
        name,split_idx = self.data_lookup[key].values()
        # split_idx_idx = int(np.where(np.array(self.data[name]["SplitIndex"]) == split_idx)[0])

        with h5py.File(self.data_path,'r') as tmp_data:
            split_idx_idx = int(np.where(np.array(tmp_data[name]["SplitIndex"]) == split_idx)[0])

            split_data = tmp_data[name]['SplitData'][split_idx_idx]
            split_target = tmp_data[name][self.target][split_idx_idx]
        
        split_data = pos_img(split_data)
        if np.isnan(split_data).any().item():
            print(f"Interpolating {np.count_nonzero(np.isnan(split_data))} Nan values")
            split_data = pd.DataFrame(split_data).interpolate(
                method='linear',
                limit_direction='both',
                axis=0).to_numpy()
        split_data = minmax_scale(split_data,feature_range=(0,1))
        split_data = torch.from_numpy(split_data).to(self.device)
        split_data = split_data.unsqueeze(0)
        split_data = F.interpolate(split_data.unsqueeze(0),(224,224)).squeeze(0)
        split_data = split_data.float()

        if torch.isnan(split_data).cpu().any().item():
            print("Replacing NaN data with 0")
            split_data = torch.nan_to_num(split_data,0)

                
        split_target = torch.tensor(split_target).to(self.device)
        split_target = split_target.float()

        return {"data":split_data,"target":split_target,"name":key}