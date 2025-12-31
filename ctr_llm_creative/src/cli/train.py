"""Stage 1: Training entry point."""

import argparse
import logging
from typing import Optional


logger = logging.getLogger(__name__)


def train(config_path: str, output_dir: Optional[str] = None) -> None:
    """
    Train a CTR/multitask model.
    
    Args:
        config_path: Path to training config YAML
        output_dir: Output directory for run artifacts
    """
    logger.info(f"Training with config: {config_path}")
    # Implementation will be added
    pass


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train CTR/multitask model")
    parser.add_argument("--config", required=True, help="Config path")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    
    args = parser.parse_args()
    train(config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
