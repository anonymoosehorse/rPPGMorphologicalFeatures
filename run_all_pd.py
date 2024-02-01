import subprocess
from pathlib import Path
from tqdm import tqdm

n_folds = 7

data_dim_dict={
    "traces":"1d",
    "cwt":"2d",
    "ibis":"3d",
}

# script_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\src\lightning_main.py")
script_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\src\test_baseline.py")

cmd_list = []

# for dataset in ['vicar','vipl']:
for dataset in tqdm(['vicar','vipl']):
    for use_gt in tqdm([True,False]):
        for network in ['peakdetection1d']:
            for representation in ["traces"]:
                for target in ["all"]:
                    
                    batch_size = 64

                    ##HACK: To test if everything is running smooth just train two epochs
                    epochs = 1
                    
                    cmd = [
                        r"python",str(script_path),
                        f"train.epochs={epochs}",
                        f"train.batch_size={batch_size}",
                        f"model.name={network}",
                        f"model.data_dim={data_dim_dict[representation]}",
                        f"model.input_representation={representation}",
                        f"model.target={target}",
                        f"dataset.name={dataset}",
                        f"dataset.use_gt={use_gt}"
                    ]

                    if not use_gt:
                        for fold_nr in range(n_folds):
                            fold_cmd = cmd + [f"dataset.fold_number={fold_nr}"]
                            print(fold_cmd)
                            subprocess.run(fold_cmd)                            
                            
                    else:
                        print(cmd)
                        subprocess.run(cmd)

                        

                        



