import torch
import pickle as pkl
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

class Dataset1D(Dataset):
    def __init__(self,trace_data,target,device):
        self.data = trace_data
        self.target = target
        self.device = device

        data_lookup = {f"{name}_{i}":{'name':name,'split_idx':i} for name,sig_data in self.data.items() for i in sig_data['SplitIndex']}
        self.data_lookup = OrderedDict(data_lookup)

    def __len__(self):
        return len(self.data_lookup)
    
    def __getitem__(self, index):
        key = list(self.data_lookup.keys())[index]
        name,split_idx = self.data_lookup[key].values()
        split_idx_idx = int(np.where(np.array(self.data[name]["SplitIndex"]) == split_idx)[0])

        split_data = self.data[name]['SplitData'][split_idx_idx]
        split_data = torch.from_numpy(split_data).to(self.device)

        split_target = self.data[name][self.target][split_idx_idx]
        split_target = torch.tensor(split_target).to(self.device)

        return {"data":split_data,"target":split_target}

class DatasetCWT(Dataset):
    def __init__(self,cwt_data,target,device):
        self.data = cwt_data        
        self.target = target
        self.device = device

        data_lookup = {f"{name}_{i}":{'name':name,'split_idx':i} for name,sig_data in self.data.items() for i in sig_data['SplitIndex']}
        self.data_lookup = OrderedDict(data_lookup)

    def __len__(self):
        return len(self.data_lookup)
    
    def __getitem__(self, index):
        key = list(self.data_lookup.keys())[index]
        name,split_idx = self.data_lookup[key].values()
        split_idx_idx = int(np.where(np.array(self.data[name]["SplitIndex"]) == split_idx)[0])
        
        split_data = self.data[name]['SplitData'][split_idx_idx]
        split_data = np.stack([np.real(split_data),np.imag(split_data)])
        split_data = torch.from_numpy(split_data).to(self.device)
        split_data = F.interpolate(split_data.unsqueeze(0),(224,224)).squeeze(0)
                
        split_target = self.data[name][self.target][split_idx_idx]
        split_target = torch.tensor(split_target).to(self.device)

        return {"data":split_data,"target":split_target}


def get_dataloaders(data,cfg,datset_cfg,device):

    if cfg.model.data_dim == '1d':        
        dataset_class = Dataset1D
    elif cfg.model.data_dim == '2d':
        dataset_class = DatasetCWT    

    name_to_id = lambda x: int(x.split("_")[0])
    test_names = [name for name in data.keys() if name_to_id(name) in datset_cfg['default_test_ids']]
    val_names = [name for name in data.keys() if name_to_id(name) in datset_cfg['default_val_ids']]

    test_data = {name:data.pop(name) for name in test_names}
    val_data = {name:data.pop(name) for name in val_names}
    train_data = data

    dl_train = dataset_class(train_data,cfg.model.target,device)
    dl_test = dataset_class(test_data,cfg.model.target,device)
    dl_val = dataset_class(val_data,cfg.model.target,device)

    print(f"Trainset: {len(dl_train)}, Testset: {len(dl_test)}, Trainset: {len(dl_val)}")

    train_loader = DataLoader(
        dataset=dl_train,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False)
    
    test_loader = DataLoader(
        dataset=dl_test,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False)
    
    val_loader = DataLoader(
        dataset=dl_val,        
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False)

    return train_loader,test_loader,val_loader


if __name__=="__main__":
    cfg = OmegaConf.load("x_config.yaml")
    ds_cfg = OmegaConf.load("dataset_config.yaml")
    ds_cfg = ds_cfg[cfg.dataset.name]

    traces_pkl_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\1D_Signal_Traces.pkl")
    traces_pkl_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\TrainingData\vicar\2D_Signal_CWT.pkl")

    device = 'cpu'

    with open(traces_pkl_path,'rb') as f:
            data = pkl.load(f)

    train_loader,test_loader,val_loader = get_dataloaders(data,cfg,ds_cfg,device)

    ## Pop names from list: 

    print()