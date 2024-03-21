
from pathlib import Path

from omegaconf import OmegaConf
from src.lightning_train import run_training


def main():    

    home_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation")
    checkpoint_path = home_path / "model_checkpoints"

    # Load defaults and overwrite by command-line arguments
    cfg = OmegaConf.load('x_config.yaml')
    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cmd_cfg)

    data_cfg = OmegaConf.load("x_dataset_config.yaml")
    data_cfg = data_cfg[cfg.dataset.name]
    
    run_training(checkpoint_path,cfg,data_cfg,experiment_name=None,debug_mode=False)

if __name__ == "__main__":
    main()
