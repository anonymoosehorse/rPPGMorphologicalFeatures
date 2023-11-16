from pathlib import Path
from omegaconf import OmegaConf
import h5py
import numpy as np
from utils.pre_processing import read_and_process,resample_data,read_h5,read_numpy,read_gt_data,detect_extrema_to_dict,create_splits,remove_faulty_splits,create_cwt

if __name__ == '__main__':
    cfg = OmegaConf.load('preprocess_config.yaml')     
    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cmd_cfg)

    data_cfg = OmegaConf.load('dataset_config.yaml')
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
        
        if fps != 30:
            signal_dict = resample_data(signal_dict,fps,30)
            fps = 30
    print("Done")

    print("Loading Ground Truth signal data...", end=" ")
    if dataset == "vicar":
        load_func = read_h5        
    elif dataset == "vipl":
        load_func = read_numpy    
    
    gt_sig_dict = read_gt_data(gt_path,gt_fps,load_func)

    if use_gt:
        signal_dict = resample_data(gt_sig_dict,gt_fps,30)
        fps = 30
    print("Done")
    
    print("Split data, calculate features and clean ...", end=" ")

    extrema_dict = detect_extrema_to_dict(gt_sig_dict)
    
    splits = create_splits(signal_dict,gt_sig_dict,extrema_dict,fps=fps,gt_fps=gt_fps)

    for name,data in splits.items():
        if dataset == 'vicar':
            data['AUP'] = list(((np.array(data['AUP'])*1000) / 30 - 10818) / 54595)
            data['PWA'] = list(np.array(data['PWA']) / 54595)
        else:
            data['AUP'] = list((np.array(data['AUP'])*1000) / 30)

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
   
    
