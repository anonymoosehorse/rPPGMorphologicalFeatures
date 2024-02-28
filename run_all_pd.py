import subprocess
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

from src.test_baseline import run_peakbased_detector

n_folds = 7

data_dim_dict={
    "traces":"1d",
    "cwt":"2d",
    "ibis":"3d",
}

name = "BaselineAnalysisNewNormalizedDataFlip"

if __name__ == "__main__":

    # for dataset in ['vicar','vipl']:
    for dataset in tqdm(['vicar']):
        for use_gt in tqdm([True,False]):
            for network in ['peakdetection1d']:
                for representation in ["traces"]:
                    for target in [["HR","RT","AUP","PWA"]]:
                        
                        batch_size = 64
                        
                        epochs = 1

                        cfg = OmegaConf.load("x_config.yaml")                                        

                        data_cfg = OmegaConf.load("x_dataset_config.yaml")
                        data_cfg = data_cfg[cfg.dataset.name]
                        
                        cfg['train']['epochs']= epochs
                        cfg['train']['batch_size']= batch_size
                        cfg['model']['name']= network
                        cfg['model']['data_dim']= data_dim_dict[representation]
                        cfg['model']['input_representation']= representation
                        cfg['model']['target']= target
                        cfg['dataset']['name']= dataset
                        cfg['dataset']['use_gt']= use_gt  
                        cfg['dataset']['normalize_data'] = True
                        cfg['dataset']['flip_signal'] = True

                        if not use_gt:
                            for fold_nr in range(n_folds):
                                cfg['dataset']['fold_number']= fold_nr                            
                                
                                run_peakbased_detector(cfg,data_cfg,name)
                                
                        else:
                            
                            run_peakbased_detector(cfg,data_cfg,name)
                            

                            

                            



