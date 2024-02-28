from collections import OrderedDict

import torch
import h5py
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import minmax_scale

from matplotlib import pyplot as plt


from ..utils.signal_processing import pos,pos_img
from ..constants import Normalization
from ..utils.pre_processing import scale_to_range

def normalize_gt(data,target):
    if target == "AUP":
        data = scale_to_range(data,Normalization.AUP_RANGE['min'],Normalization.AUP_RANGE['max'])
    elif target == "HR":
        data = scale_to_range(data,Normalization.HR_RANGE['min'],Normalization.HR_RANGE['max'])
    elif target == "RT":
        data = scale_to_range(data,Normalization.RT_RANGE['min'],Normalization.RT_RANGE['max'])
    elif target == "PWA":
        pass
    else:
        raise NotImplementedError(f"Normalization for {target} not implemented")
    return data


class Dataset1D(Dataset):
    def __init__(self,traces_path,target,valid_data_ids,normalize,flip_signal):
        self.data_path = traces_path       
        
        self.target = target        
        self.normalize = normalize
        self.flip_signal = flip_signal

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
        names = list(self.data_lookup.keys())[index]
        name,split_idx = self.data_lookup[names].values()

        with h5py.File(self.data_path,'r') as tmp_data:
            split_idx_idx = int(np.where(np.array(tmp_data[name]["SplitIndex"]) == split_idx)[0])

            split_data = tmp_data[name]['SplitData'][split_idx_idx]
            split_time = tmp_data[name]['SplitTime'][split_idx_idx]

            split_gtdata = tmp_data[name]['SplitGTData'][split_idx_idx]
            split_gtpeaks = tmp_data[name]['SplitGTPeaks'][split_idx_idx]
            split_gttime = tmp_data[name]['SplitGTTime'][split_idx_idx]
            
            split_target = {}
            if isinstance(self.target,list):
                for key in tmp_data[name].keys():
                    if any([t in key for t in self.target]):
                        split_target[key] = tmp_data[name][key][split_idx_idx]                    
                 
            else:

                split_target[self.target] = tmp_data[name][self.target][split_idx_idx]
                split_target[self.target+"_class"] = tmp_data[name][self.target+"_class"][split_idx_idx]
        
        # split_data = torch.from_numpy(split_data).to(self.device)
        split_data = torch.from_numpy(split_data)
        if self.flip_signal:
            if self.normalize:
                split_data = 1 + (split_data * -1)
            else:
                raise NotImplementedError("Flip signal only implemented for normalized data")
             
        split_data = split_data.float()

        # split_time = torch.from_numpy(split_time).to(self.device)
        split_time = torch.from_numpy(split_time)
        split_time = split_time.float()       
        

        if isinstance(self.target,list):
            if self.normalize:                
                split_regression_targets = torch.stack([torch.tensor(normalize_gt(split_target[key],key)).float() for key in self.target])
            else:                
                split_regression_targets = torch.stack([torch.tensor(split_target[key]).float() for key in self.target])
            
            split_classification_targets = torch.stack([torch.tensor(split_target[key+"_class"]).float() for key in self.target])
            
        else:            
            split_regression_targets = torch.tensor(split_target[self.target]).float()
            split_classification_targets = torch.tensor(split_target[self.target+"_class"]).float()            
            if self.normalize:
                split_regression_targets = normalize_gt(split_regression_targets,self.target)

        return {
            "data":split_data,
            "regression_target":split_regression_targets,
            "classification_target":split_classification_targets,
            "name":names,
            "time":split_time,
            "gt_data":np.stack([split_gttime,split_gtdata,split_gtpeaks])}


class DatasetCWT(Dataset):
    def __init__(self,cwt_data_path,target,valid_data_ids):
        self.data_path = cwt_data_path
        
        self.target = target        

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

            # split_time = tmp_data[name]['SplitTime'][split_idx_idx]

            split_data = tmp_data[name]['SplitData'][split_idx_idx]
            split_target = tmp_data[name][self.target][split_idx_idx]
        
        split_data = np.stack([np.real(split_data),np.imag(split_data)])
        split_data = torch.from_numpy(split_data)
        split_data = F.interpolate(split_data.unsqueeze(0),(224,224)).squeeze(0)
        split_data = split_data.float()
                
        split_target = torch.tensor(split_target)
        split_target = split_target.float()
        # split_target = normalize_gt(split_target,self.target)

        # split_time = torch.from_numpy(split_time).to(self.device)
        # split_time = split_time.float()

        return {"data":split_data,"target":split_target,"name":key, }

class DatasetIBIS(Dataset):
    def __init__(self,ibis_data_path,target,valid_data_ids):
        self.data_path = ibis_data_path 
        
        self.target = target        
        
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

            # split_time = tmp_data[name]['SplitTime'][split_idx_idx]

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
        split_data = torch.from_numpy(split_data)
        split_data = split_data.unsqueeze(0)
        split_data = F.interpolate(split_data.unsqueeze(0),(224,224)).squeeze(0)
        split_data = split_data.float()

        if torch.isnan(split_data).cpu().any().item():
            print("Replacing NaN data with 0")
            split_data = torch.nan_to_num(split_data,0)

                
        split_target = torch.tensor(split_target)
        split_target = split_target.float()
        # split_target = normalize_gt(split_target,self.target)

        # split_time = torch.from_numpy(split_time).to(self.device)
        # split_time = split_time.float()

        return {"data":split_data,"target":split_target,"name":key}