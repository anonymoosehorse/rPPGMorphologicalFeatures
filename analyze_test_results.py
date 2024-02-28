import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

from src.utils.config_utils import flatten_dictionary

test_results_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\test_results\ruben-reproduction-new")
test_results_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\test_results\normalization_test_new")
test_results_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\test_results\normalization_test_new_shuffle")
test_results_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\test_results\flipped_multi_input_batched")
test_results_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\test_results\test_runs_new")

combined_results_path = Path.cwd() / (test_results_path.stem + ".csv")

redo = True

if not combined_results_path.exists() or redo:
    data = []
    
    for file in tqdm(test_results_path.rglob("*.csv")):
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

    data.to_csv(combined_results_path,index=False)
else:
    data = pd.read_csv(combined_results_path)


print(data)