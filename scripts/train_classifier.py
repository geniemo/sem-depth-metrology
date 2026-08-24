"""Train the 4-way bucket/case classifier on real train images."""
import argparse

import yaml

from semdepth.classify import train_classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(train_classifier(cfg))


if __name__ == "__main__":
    main()
