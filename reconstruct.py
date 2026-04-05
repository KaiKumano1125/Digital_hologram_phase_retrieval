"""
reconstruct.py — CLI entry point for holographic phase retrieval.

Usage:
    python reconstruct.py
    python reconstruct.py --config config/config.yaml
"""

import argparse
import torch

from config.config_loader import load_config
from pipeline.optimizer import ReconstructionLoop


def main() -> None:
    parser = argparse.ArgumentParser(description="Holographic phase retrieval via gradient descent.")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to YAML config file."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {cfg.mode} | Iterations: {cfg.max_iter} | LR: {cfg.learning_rate}")

    ReconstructionLoop(cfg, device).run()


if __name__ == "__main__":
    main()
