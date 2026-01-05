from pathlib import Path

from omegaconf import OmegaConf
from src.lightning_train import run_training

N_FOLDS = 7

DATA_DIM_DICT={
    "traces":"1d",
    "cwt":"2d",
    "ibis":"3d",
}

    # 'transformer1d':256,
BATCH_SIZE_DICT = {
    'resnet1d':256,
    'transformer1d':128,
    'transformer2d':64,
    'resnet2d':256
}

# checkpoint_dir = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\model_checkpoints")
checkpoint_dir = Path(r"E:\UniversityBackup\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\model_checkpoints")

DEBUG_MODE=False

def gt_experiment(checkpoint_dir:Path):
    use_gt = True
    for network in ['resnet1d','transformer1d','resnet2d','transformer2d']:#['transformer1d']: #['resnet1d','transformer1d']: #['transformer1d','transformer2d','resnet1d','resnet2d']:
        for dataset in ['pure','ucla','vipl','vicar']:
            for loss in ['L1Loss','CCCLoss']:#[True,False]:
                for representation in ["traces","cwt"]:#["traces","cwt","ibis"]:
                    for target in ["HR","RT","AUP","PWA"]:

                        if network[-2:] == "1d" and representation != "traces":
                            continue
                        if network[-2:] == "2d" and representation == "traces":
                            continue

                        if dataset == 'vipl' and representation == 'ibis':
                            continue

                        if use_gt and representation == 'ibis':
                            continue

                        batch_size = BATCH_SIZE_DICT[network]
                        # batch_size = 1

                        if network[-2:] == "1d":
                            epochs = 200
                        elif network[-2:] == "2d":
                            epochs = 200                 

                        # Load defaults and overwrite by command-line arguments
                        cfg = OmegaConf.load("x_config.yaml")
                        cmd_cfg = OmegaConf.from_cli()
                        cfg = OmegaConf.merge(cfg, cmd_cfg)

                        data_cfg = OmegaConf.load("x_dataset_config.yaml")
                        data_cfg = data_cfg[cfg.dataset.name]
                    
                        cfg["train"]["epochs"]=epochs
                        cfg["train"]["batch_size"]=batch_size
                        cfg["train"]["loss"] = loss
                        cfg["model"]["name"]=network
                        cfg["model"]["data_dim"]=DATA_DIM_DICT[representation]
                        cfg["model"]["input_representation"]=representation
                        cfg["model"]["target"]=target
                        cfg["dataset"]["name"]=dataset
                        cfg["dataset"]["use_gt"]=use_gt                    
                        cfg["comet"]["project"] = "gt-experiment"

                        # print(cmd)
                        # subprocess.run(cmd)
                        if not use_gt:
                            for fold_nr in range(N_FOLDS):
                                cfg['dataset']['fold_number'] = fold_nr
                                run_training(checkpoint_dir,cfg,data_cfg,debug_mode=DEBUG_MODE)                                                        
                                
                        else:
                            run_training(checkpoint_dir,cfg,data_cfg,debug_mode=DEBUG_MODE)

def real_experiment(checkpoint_dir:Path):
        # for dataset in ['vicar','vipl']:
    # for dataset in ['ucla']:
    use_gt = False
    for network in ['resnet1d','resnet2d','transformer2d']:#['transformer1d']: #['resnet1d','transformer1d']: #['transformer1d','transformer2d','resnet1d','resnet2d']:
        for dataset in ['pure','ucla','vipl','vicar']:
            for loss in ['L1Loss','CCCLoss']:#[True,False]:
                for representation in ["traces","cwt","ibis"]:#["traces","cwt","ibis"]:
                    for target in ["HR","RT","AUP","PWA"]:

                        if network[-2:] == "1d" and representation != "traces":
                            continue
                        if network[-2:] == "2d" and representation == "traces":
                            continue

                        if dataset == 'vipl' and representation == 'ibis':
                            continue

                        if use_gt and representation == 'ibis':
                            continue                        

                        batch_size = BATCH_SIZE_DICT[network]
                        # batch_size = 1

                        if network[-2:] == "1d":
                            epochs = 200
                        elif network[-2:] == "2d":
                            epochs = 200                 

                        # Load defaults and overwrite by command-line arguments
                        cfg = OmegaConf.load("x_config.yaml")
                        cmd_cfg = OmegaConf.from_cli()
                        cfg = OmegaConf.merge(cfg, cmd_cfg)

                        data_cfg = OmegaConf.load("x_dataset_config.yaml")
                        data_cfg = data_cfg[cfg.dataset.name]
                    
                        cfg["train"]["epochs"]=epochs
                        cfg["train"]["batch_size"]=batch_size
                        cfg["train"]["loss"] = loss
                        cfg["model"]["name"]=network
                        cfg["model"]["data_dim"]=DATA_DIM_DICT[representation]
                        cfg["model"]["input_representation"]=representation
                        cfg["model"]["target"]=target
                        cfg["dataset"]["name"]=dataset
                        cfg["dataset"]["use_gt"]=use_gt                    
                        cfg["comet"]["project"] = "real-experiment"

                        # print(cmd)
                        # subprocess.run(cmd)
                        if not use_gt:
                            for fold_nr in range(N_FOLDS):
                                cfg['dataset']['fold_number'] = fold_nr
                                run_training(checkpoint_dir,cfg,data_cfg,debug_mode=DEBUG_MODE)                                                        
                                
                        else:
                            run_training(checkpoint_dir,cfg,data_cfg,debug_mode=DEBUG_MODE)

def main(checkpoint_dir:Path):
    gt_experiment(checkpoint_dir)
    real_experiment(checkpoint_dir)
    # break

    

if __name__ == "__main__":
    checkpoint_dir = Path(__file__).parent / "model_checkpoints"
    main(checkpoint_dir)
                        

                        



