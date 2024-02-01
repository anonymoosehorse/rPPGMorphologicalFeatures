from pathlib import Path
from omegaconf import OmegaConf
import h5py
import numpy as np
from utils.pre_processing import read_and_process,resample_data,read_h5,read_numpy,read_gt_data,detect_extrema_to_dict,create_splits,remove_faulty_splits,create_cwt,read_csv,read_pure_json
from constants import Normalization
from sklearn.preprocessing import minmax_scale

def scale_to_range(value,old_min,old_max,new_min=0,new_max=1):
    return ((new_max - new_min)*(value-old_min)/(old_max-old_min)) + new_min

def load_gt_dict(gt_path,gt_fps,dataset):
    if dataset == "vicar":
        load_func = read_h5        
    elif dataset == "vipl":
        load_func = read_numpy    
    elif dataset == "ubfc1":
        load_func = read_csv
    elif dataset == "ubfc2":
        load_func = read_csv
    elif dataset == "pure":
        load_func = read_pure_json
    
    gt_sig_dict = read_gt_data(gt_path,gt_fps,load_func)    


    return gt_sig_dict

def normalize_gt_dict(gt_sig_dict,dataset):

    if dataset not in Normalization.GT_MINMAX_DICT:
        raise NotImplementedError(f"Normalization for {dataset} not implemented")

    range_dict = Normalization.GT_MINMAX_DICT[dataset]

    for key,value in gt_sig_dict.items():
        gt_sig_dict[key][1] = scale_to_range(value[1],range_dict['min'],range_dict['max'])

    return gt_sig_dict

def normalize_signal_dict(gt_sig_dict,dataset):

    if dataset not in Normalization.SIGNAL_MINMAX_DICT:
        raise NotImplementedError(f"Normalization for {dataset} not implemented")

    range_dict = Normalization.SIGNAL_MINMAX_DICT[dataset]

    for key,value in gt_sig_dict.items():
        gt_sig_dict[key][1] = scale_to_range(value[1],range_dict['min'],range_dict['max'])

    return gt_sig_dict

if __name__ == '__main__':
    cfg = OmegaConf.load('x_preprocess_config.yaml')     
    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cmd_cfg)

    data_cfg = OmegaConf.load('x_dataset_config.yaml')
    data_cfg = data_cfg[cfg.dataset_to_run]

    traces_path = Path(data_cfg.traces_path)
    fps = data_cfg.traces_fps

    use_gt = cfg.use_gt
    use_stride = cfg.use_stride

    dataset = cfg.dataset_to_run
    gt_path = Path(data_cfg.gt_path)
    gt_fps = data_cfg.gt_fps

    ## Create the output path for the generated data
    train_data_path = Path.cwd() / cfg['output_directory'] / dataset
    train_data_path.mkdir(exist_ok=True,parents=True)

    print("Loading traces data filter and resample...", end=" ")
    if not use_gt:
        signal_dict =  read_and_process(traces_path,fps)    
        signal_dict = normalize_signal_dict(signal_dict,dataset)        
        
        if fps != 30:
            signal_dict = resample_data(signal_dict,fps,30)
            fps = 30
    print("Done")

    print("Loading Ground Truth signal data...", end=" ")
    gt_sig_dict = load_gt_dict(gt_path,gt_fps,dataset)

    gt_sig_dict = normalize_gt_dict(gt_sig_dict,dataset)

    if use_gt:
        signal_dict = resample_data(gt_sig_dict,gt_fps,30)
        fps = 30
    print("Done")
    
    print("Split data, calculate features and clean ...", end=" ")

    extrema_dict = detect_extrema_to_dict(gt_sig_dict)
    
    splits = create_splits(signal_dict,gt_sig_dict,extrema_dict,fps=fps,gt_fps=gt_fps)

    ## Normalize properties
    # for name,data in splits.items():
    #     data['HR'] = scale_to_range(np.array(data['HR']),Normalization.HR_RANGE['min'],Normalization.HR_RANGE['max'])
    #     data['RT'] = scale_to_range(np.array(data['RT']),Normalization.RT_RANGE['min'],Normalization.RT_RANGE['max'])
        

    # for name,data in splits.items():
    #     if dataset == 'vicar':
    #         data['AUP'] = list(((np.array(data['AUP'])*1000) / 30 - 10818) / 54595)
    #         data['PWA'] = list(np.array(data['PWA']) / 54595)
    #     else:
    #         data['AUP'] = list((np.array(data['AUP'])*1000) / 30)

    splits = remove_faulty_splits(splits)
    
    print("Done")    

    suffix = "_gt" if use_gt else ""

    print("Save traces training data ...",end=" ")
    with h5py.File(train_data_path / f"traces_data{suffix}.h5", 'w') as hf:
        for name,data in splits.items():
            group = hf.create_group(name)
            for key,sub_splits in data.items():
                group.create_dataset(key,data=sub_splits)
    print("Done")

    print("Generate CWT data",end=' ')
    cwt_data = create_cwt(splits)
    print("Done")

    print("Save CWT training data ...",end=" ")
    with h5py.File(train_data_path / f"cwt_data{suffix}.h5", 'w') as hf:
        for name,data in cwt_data.items():
            group = hf.create_group(name)
            for key,sub_splits in data.items():
                group.create_dataset(key,data=sub_splits)
    print("Done")
   
    
