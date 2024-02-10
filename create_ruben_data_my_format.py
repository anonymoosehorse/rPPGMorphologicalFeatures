from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

ruben_results_dir = Path(r"D:\Projects\Waveform\Data\RubenData\vicar")
output_dir = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\TrainingDataRuben\vicar")
output_dir.mkdir(parents=True, exist_ok=True)

mode = "_gt"

traces_dir = ruben_results_dir / f"split_traces5_1D{mode}"

data_dirct = {}

for file in tqdm(traces_dir.glob("*.npy")):
    name = file.stem
    vid_name = '_'.join(name.split("_")[:-1])
    split_index = int(name.split("_")[-1])
    split_data = np.loadtxt(file)

    if vid_name not in data_dirct:
        data_dirct[vid_name] = {}
        data_dirct[vid_name]['SplitIndex'] = [split_index]
        data_dirct[vid_name]['SplitData'] = [split_data[1,:]]
        data_dirct[vid_name]['SplitTime'] = [split_data[0,:]]
        for target in ['HR','RT','PWA','AUP']:
            target_dir = str(traces_dir) + f"_{target.lower().replace('aup','area')}"
            target_data = np.loadtxt(Path(target_dir) / f"{vid_name}_{split_index}.npy")
            data_dirct[vid_name][target] = [target_data]
    else:
        data_dirct[vid_name]['SplitIndex'].append(split_index)
        data_dirct[vid_name]['SplitData'].append(split_data[1,:])
        data_dirct[vid_name]['SplitTime'].append(split_data[0,:])
        for target in ['HR','RT','PWA','AUP']:
            target_dir = Path(str(traces_dir) + f"_{target.lower().replace('aup','area')}")
            target_data = np.loadtxt(Path(target_dir) / f"{vid_name}_{split_index}.npy")
            data_dirct[vid_name][target].append(target_data)
     
    

f = h5py.File(output_dir / f"traces_data{mode}.h5", "w")
for key in tqdm(data_dirct.keys()):
    grp = f.create_group(key)
    for sub_key,data in data_dirct[key].items():
        grp.create_dataset(sub_key,data=np.asarray(data))
        
# for key in f.keys():
#     split_index = f[key]['SplitIndex']
#     split_data = f[key]['SplitData']
#     for idx, split in zip(split_index, split_data):
#         print(f"Split {idx} has {split} traces")
#         ruben_result_path = traces_dir / f"{key}_{idx}.npy"
#         ruben_result_trace = np.loadtxt(ruben_result_path)

#         ruben_result_target_path = Path(str(traces_dir) + "_hr") / f"{key}_{idx}.npy"
#         ruben_result_target = np.loadtxt(ruben_result_target_path)

#         ## Recreate Ruben output as h5 py file and try to train on it!

#         print()


        # plt.plot(ruben_result[1,:])
        # plt.plot(split)
        # plt.title(f"Name {key} Split {idx}")
        # plt.show()
        
print()

