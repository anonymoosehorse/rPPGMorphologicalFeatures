import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

test_results_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\test_results\ruben-reproduction-new")
test_results_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\test_results\normalization_test_new")
test_results_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\test_results\normalization_test_new_shuffle")

combined_results_path = Path.cwd() / (test_results_path.stem + ".csv")

redo = True

if not combined_results_path.exists() or redo:
    data = []
    for file in tqdm(test_results_path.rglob("*.csv")):
        df = pd.read_csv(file)    
        data.append(df)
    data = pd.concat(data)

    data.to_csv(combined_results_path,index=False)
else:
    data = pd.read_csv(combined_results_path)


print(data)