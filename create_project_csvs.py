import logging

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

from src.utils.config_utils import flatten_dictionary


SKIP_EXISTING = False

def create_project_csvs(checkpoint_dir:Path,combined_results_path:Path,SKIP_EXISTING:bool=False):
    """
    Create CSV files for each project directory in the checkpoint directory.

    Args:
        checkpoint_dir (Path): The directory containing project directories.
        combined_results_path (Path): The directory where the combined CSV files will be saved.
        SKIP_EXISTING (bool, optional): Whether to skip creating CSV files for projects that already have one. Defaults to False.
    """

    if not combined_results_path.exists():
        combined_results_path.mkdir(exist_ok=True)

    for project_dir in tqdm(checkpoint_dir.iterdir()):
        data = []
        
        project_csv_file_path = combined_results_path / f"{project_dir.name}.csv"        

        if SKIP_EXISTING and project_csv_file_path.exists():
            continue

        for file in tqdm(project_dir.rglob("*.csv")):
            config_path = file.parent/ 'config.yaml'
            df = pd.read_csv(file)    
            if config_path.exists():
                cfg = OmegaConf.load(config_path)
                config = flatten_dictionary(OmegaConf.to_object(cfg))
                for k,v in config.items():
                    if k not in df.columns:
                        if isinstance(v,list):
                            v = '_'.join(v)
                        df[k] = v
            df['Experiment'] = file.parent.name
            
            data.append(df)
        data = pd.concat(data)

        data.to_csv(project_csv_file_path,index=False)

if __name__=="__main__":
    checkpoint_dir = Path(__file__).parent / "model_checkpoints"
    combined_results_path = Path(__file__).parent / "ResultCSV"
    create_project_csvs(checkpoint_dir,combined_results_path,SKIP_EXISTING)