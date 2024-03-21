import argparse
from pathlib import Path

from src.lightning_test import run_test


def main():
    parser = argparse.ArgumentParser(description='Run test on a directory of config files')
    parser.add_argument('config_dir', type=str, help='Path to the directory containing the config files')
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    for config_path in config_dir.rglob("*.yaml"):
        try:
            run_test(config_path.parent)
        except Exception as e:
            print(f"Error in {config_path}")
            print(e)

if __name__ == "__main__":
    main()
