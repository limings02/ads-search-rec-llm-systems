#!/bin/bash

# Run experiment pipeline: train -> simulate -> evaluate

echo "Running experiment pipeline..."

# Stage 1: Training
echo "Stage 1: Training..."
python -m src.cli.train --config configs/experiments/avazu_infra_deepfm.yaml

# Stage 2: Simulation (optional, for closed-loop datasets)
# echo "Stage 2: Simulation..."
# python -m src.cli.simulate --run-id RUN_ID --auction-stream data/processed/auction_stream.parquet

# Stage 3: Evaluation
# echo "Stage 3: Evaluation..."
# python -m src.cli.evaluate --run-id RUN_ID

echo "Done!"
