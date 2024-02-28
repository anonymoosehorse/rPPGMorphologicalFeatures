from pathlib import Path
import shutil
# from tqdm import tqdm

model_checkpoint_dir_path = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\model_checkpoints")
test_output_dir = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\test_results")

test_output_dir.mkdir(exist_ok=True)

test_result_paths = list(model_checkpoint_dir_path.rglob("*.csv"))

for i,result_path in enumerate(test_result_paths):
    print(f"\r {(i/len(test_result_paths)) * 100}% Done", end=" ")
    rel_path = result_path.relative_to(model_checkpoint_dir_path)
    
    out_path = test_output_dir / rel_path    
    out_path.parent.mkdir(exist_ok=True,parents=True)
    shutil.copy(result_path,out_path)

    cfg_path = result_path.parent / 'config.yaml'
    if cfg_path.exists():        
        shutil.copy(cfg_path,out_path.parent / 'config.yaml')
    else:
        print(f"Couldn't find {cfg_path}")