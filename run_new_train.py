
from pathlib import Path

from src.lightning_train import run_training


def main():    

    home_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation")
    checkpoint_path = home_path / "model_checkpoints"
    config_path = home_path / "x_config.yaml"
    dataset_config_path = home_path / "x_dataset_config.yaml"
    
    run_training(checkpoint_path,config_path,dataset_config_path,experiment_name=None)

if __name__ == "__main__":
    main()
