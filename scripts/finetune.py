"""Run one weak-supervision fine-tuning experiment from a YAML config."""
import argparse

import yaml

from semdepth.finetune import run_finetune


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    row = run_finetune(cfg)
    print(row)


if __name__ == "__main__":
    main()
