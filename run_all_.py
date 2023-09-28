import subprocess
from pathlib import Path

n_folds = 7

data_dim_dict={
    "traces":"1d",
    "cwt":"2d",
    "ibis":"3d",
}

script_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\src\lightning_main.py")

for dataset in ['vicar','vipl']:
    for use_gt in [True,False]:
        for network in ['transformer1d','resnet1d','resnet2d','transformer2d']:
            for representation in ["cwt", "traces", "ibis"]:
                for target in ["AUP", "RT","PWA","HR"]:
                    
                    if network[-2:] == "1d" and representation != "traces":
                        continue
                    if network[-2:] == "2d" and representation == "traces":
                        continue

                    if dataset == 'vipl' and representation == 'ibis':
                        continue

                    if use_gt and representation == 'ibis':
                        continue

                    batch_size = 32
                    if network == 'resnet1d':
                        batch_size = 1

                    if network[-2:] == "1d":
                        epochs = 200
                    elif network[-2:] == "2d":
                        epochs = 60    

                    ##HACK: To test if everything is running smooth just train two epochs
                    epochs = 30
                    
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

                    print(cmd)
                    subprocess.run(cmd)
                    # if not use_gt:
                    #     for fold_nr in range(n_folds):
                    #         fold_cmd = cmd + [f"dataset.fold_number={fold_nr}"]
                    #         print(fold_cmd)
                    #         subprocess.run(fold_cmd)                            
                    # else:
                    #     print(cmd)
                    #     subprocess.run(cmd)
                        
                        



