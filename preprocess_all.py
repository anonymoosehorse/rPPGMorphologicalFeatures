import subprocess
from pathlib import Path
from omegaconf import OmegaConf

from src.preprocess import run_preprocessing
from src.preprocess_ibis import preprocess_ibis

cfg = OmegaConf.load('x_preprocess_config.yaml')     
cmd_cfg = OmegaConf.from_cli()
cfg = OmegaConf.merge(cfg, cmd_cfg)

for dataset in ['vicar','ucla','pure']:#['vipl']: #['vicar','vipl','ubfc1','ubfc2','ucla','pure']:
    for normalize in [True]:#[True,False]:
        for use_gt in [False]:#[True,False]: #['False']:#[True,False]:        
            name = "TrainingData"
            if normalize:
                name += "Normalized"            
            
            cfg['dataset_to_run']=dataset
            cfg['use_gt']=use_gt
            cfg['normalize']=normalize
            cfg['output_directory']=name
            
            data_cfg = OmegaConf.load('x_dataset_config.yaml')
            data_cfg = data_cfg[cfg.dataset_to_run]            
            
            # run_preprocessing(cfg,data_cfg)
            preprocess_ibis(OmegaConf.merge(data_cfg,cfg))

