#!/usr/bin/env python3
"""Main training script for AlignVLM stage 2 training."""

import argparse
import yaml
from pathlib import Path

from core.trainer import train_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description='Train AlignVLM stage 2')
    parser.add_argument('--config',
                        type=str,
                        default='configs/train_config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Train model
    train_model(config)


if __name__ == '__main__':
    main()
