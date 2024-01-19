import subprocess
from pathlib import Path

script_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\src\preprocess.py")

for dataset in ['vicar','ubfc1','ubfc2','pure']:
    for use_gt in [True,False]:                    
        cmd = [
            r"python",str(script_path),
            f"dataset_to_run={dataset}",
            f"use_gt={use_gt}",
            f"output_directory=TrainingDataNormalized"
        ]


        print(cmd)
        subprocess.run(cmd)
                    

                        

                        



