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
from my_utils.signal_processing import pos,pos_img
from sklearn.preprocessing import minmax_scale
import h5py
import pandas as pd

# class Dataset1D(Dataset):
#     def __init__(self,trace_data,target,device):
#         self.data = trace_data
#         self.target = target
#         self.device = device

#         data_lookup = {f"{name}_{i}":{'name':name,'split_idx':i} for name,sig_data in self.data.items() for i in sig_data['SplitIndex']}
#         self.data_lookup = OrderedDict(data_lookup)

#     def __len__(self):
#         return len(self.data_lookup)
    
#     def __getitem__(self, index):
#         key = list(self.data_lookup.keys())[index]
#         name,split_idx = self.data_lookup[key].values()
#         split_idx_idx = int(np.where(np.array(self.data[name]["SplitIndex"]) == split_idx)[0])

#         split_data = self.data[name]['SplitData'][split_idx_idx]
#         split_data = torch.from_numpy(split_data).to(self.device)
#         split_data = split_data.float()

#         split_target = self.data[name][self.target][split_idx_idx]
#         split_target = torch.tensor(split_target).to(self.device)
#         split_target = split_target.float()

#         return {"data":split_data,"target":split_target}

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
            print()

                
        split_target = torch.tensor(split_target).to(self.device)
        split_target = split_target.float()

        return {"data":split_data,"target":split_target,"name":key}


# def get_dataloaders(data,cfg,datset_cfg,device):
def get_dataloaders(
        data_path,
        target,
        input_representation,
        test_ids,
        val_ids,
        device,      
        name_to_id_func,  
        **loader_kwargs
        ):    

    
    with h5py.File(data_path,'r') as data:
        test_names = [name for name in data.keys() if name_to_id_func(name) in test_ids]
        val_names = [name for name in data.keys() if name_to_id_func(name) in val_ids]
        train_names = [name for name in data.keys() if name not in test_names and name not in val_names]

        # test_data = {name:data.pop(name) for name in test_names}
        # val_data = {name:data.pop(name) for name in val_names}
        # train_data = data

    if input_representation == 'traces':        
        dataset_class = Dataset1D
    elif input_representation == 'cwt':
        dataset_class = DatasetCWT   
    elif input_representation == 'ibis':
        dataset_class = DatasetIBIS  
    else:
        raise NotImplementedError

    dl_train = dataset_class(data_path,target,device,train_names)
    dl_test = dataset_class(data_path,target,device,test_names)
    dl_val = dataset_class(data_path,target,device,val_names)

    print(f"Trainset: {len(dl_train)}, Testset: {len(dl_test)}, Trainset: {len(dl_val)}")

    train_loader = DataLoader(
        dataset=dl_train,
        **loader_kwargs)
    
    test_loader = DataLoader(
        dataset=dl_test,
        **loader_kwargs)
    
    val_loader = DataLoader(
        dataset=dl_val,
        **loader_kwargs)

    return train_loader,test_loader,val_loader


def get_data(root_dir,dataset,input_representation,use_gt):    
    gt_suffix = "_gt" if use_gt else ""
    data_name = f"{input_representation}_data{gt_suffix}.pkl"
    if not Path(root_dir).is_absolute():
        data_path = Path.cwd() / Path(root_dir)    
    else:
        data_path = Path(root_dir)

    data_path = data_path / dataset / data_name

    with open(data_path,'rb') as f:
        data = pkl.load(f)

    return data

def get_data_path(root_dir,dataset,input_representation,use_gt):    
    gt_suffix = "_gt" if use_gt else ""
    data_name = f"{input_representation}_data{gt_suffix}.h5"
    if not Path(root_dir).is_absolute():
        data_path = Path.cwd() / Path(root_dir)    
    else:
        data_path = Path(root_dir)

    data_path = data_path / dataset / data_name

    return data_path

def get_name_to_id_func(dataset):
    
    if dataset == 'vicar':
        name_to_id = lambda x: int(x.split("_")[0])
    elif dataset == 'vipl':
        name_to_id = lambda x: int(x.split("_")[0].replace("p",""))        
    else:
        raise NotImplementedError
    
    return name_to_id
    
if __name__=="__main__":
    cfg = OmegaConf.load("x_config.yaml")
    ds_cfg = OmegaConf.load("dataset_config.yaml")
    ds_cfg = ds_cfg[cfg.dataset.name]    

    device = 'cpu'
    
    # data = get_data(cfg.dataset.root,cfg.dataset.name,cfg.model.input_representation,cfg.dataset.use_gt)    
    data_path = get_data_path("./TrainingData",cfg.dataset.name,cfg.model.input_representation,cfg.dataset.use_gt)    

    # ds = Dataset1D(data_path,target='AUP',device='cpu',valid_data_ids=["01_01"])
    # batch = next(iter(DataLoader(ds)))



    data_folds = DataFoldsNew(cfg.dataset.use_gt,cfg.dataset.name)
    test_ids,val_ids = data_folds.get_fold(cfg.dataset.fold_number)

    loader_kwargs = {
        "batch_size":cfg.train.batch_size,
        "num_workers":cfg.dataset.num_workers,
        "shuffle":False
    }

    # train_loader,test_loader,val_loader = get_dataloaders(data,cfg,ds_cfg,device)
    train_loader,test_loader,val_loader = get_dataloaders(data_path=data_path,
                                                          target=cfg.model.target,
                                                          input_representation=cfg.model.input_representation,
                                                          test_ids=test_ids,
                                                          val_ids=val_ids,
                                                          device=device,
                                                          name_to_id_func=get_name_to_id_func(cfg.dataset.name),
                                                          **loader_kwargs)


    next(iter(train_loader))

    print()