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
    'transformer1d':256,
    'transformer2d':64,
    'resnet2d':256
}

# checkpoint_dir = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\model_checkpoints")
checkpoint_dir = Path(r"E:\UniversityBackup\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\model_checkpoints")

DEBUG_MODE=False

def main(checkpoint_dir:Path):
    dataset='pure'
    use_gt=True #[True,False
    network='resnet2d'#['transformer1d']: #['resnet1d','transformer1d']: #['transformer1d'transformer2d','resnet1d','resnet2d']:
    representation="cwt"#["traces","cwt","ibis"
    target="HR"
    batch_size = BATCH_SIZE_DICT[network]    
    epochs = 200
           
    # for dataset in ['vicar','vipl']:
    # for dataset in ['ucla']:
    for loss in [['InverseCorrelationLoss','L1Loss'],'L1Loss', 'MSELoss','CCCLoss']: #['InverseCorrelationLoss']:
        for use_class_loss in [True,False]:
            for chkpt_monitor in ['val/loss_epoch', 'val/MAE','val/CCC','val/Pearson']:

                # Load defaults and overwrite by command-line arguments
                cfg = OmegaConf.load("x_config.yaml")
                cmd_cfg = OmegaConf.from_cli()
                cfg = OmegaConf.merge(cfg, cmd_cfg)

                data_cfg = OmegaConf.load("x_dataset_config.yaml")
                data_cfg = data_cfg[cfg.dataset.name]

                cfg["train"]["epochs"]=epochs
                cfg["train"]["batch_size"]=batch_size
                cfg["train"]["loss"]=loss
                cfg["train"]["do_classification"]=use_class_loss
                cfg["train"]["checkpoint"]["monitor"]=chkpt_monitor
                cfg["model"]["name"]=network
                cfg["model"]["data_dim"]=DATA_DIM_DICT[representation]
                cfg["model"]["input_representation"]=representation
                cfg["model"]["target"]=target
                cfg["dataset"]["name"]=dataset
                cfg["dataset"]["use_gt"]=use_gt                    
                cfg["comet"]["project"] = "loss-tests"
                


                # print(cmd)
                # subprocess.run(cmd)
                run_training(checkpoint_dir,cfg,data_cfg,debug_mode=DEBUG_MODE)    

    

if __name__ == "__main__":
    checkpoint_dir = Path(__file__).parent / "model_checkpoints"
    main(checkpoint_dir)
                        

                        



