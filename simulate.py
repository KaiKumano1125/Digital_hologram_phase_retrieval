"""
simulate.py — CLI entry point for forward hologram simulation.

Usage:
    python simulate.py
    python simulate.py --config config/config.yaml
"""

import argparse
import torch

from config.config_loader import load_config
from pipeline.simulator import GaborSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a Gabor inline hologram.")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to YAML config file."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {cfg.mode} | Reference: {cfg.reference_type}")

    GaborSimulator(cfg, device).run()


if __name__ == "__main__":
    main()
