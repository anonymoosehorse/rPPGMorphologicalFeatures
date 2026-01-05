from pathlib import Path
import shutil


if __name__ == "__main__":
    checkpoints_dir = Path(r"E:\UniversityBackup\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\model_checkpoints\complete_run")
    for run_dir in checkpoints_dir.iterdir():
        if not (run_dir / 'TestResults.csv').exists():
            # print(run_dir.stem)
            shutil.rmtree(run_dir)