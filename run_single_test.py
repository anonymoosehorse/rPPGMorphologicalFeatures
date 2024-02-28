import argparse
from pathlib import Path

from src.lightning_test import run_test


def main():

    checkpoint_dir = Path(r"D:\Projects\Waveform\Code\AlternativeRubenCode\waveform_feature_estimation\model_checkpoints\test_runs_new\reasonable_datum_1677")
    
    run_test(checkpoint_dir)

if __name__ == "__main__":
    main()
