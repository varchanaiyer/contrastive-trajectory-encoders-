#!/bin/bash
# Setup script for the project

set -e

echo "Setting up Contrastive Trajectory Encoders project..."

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.py

# Create experiment directories
echo "Creating experiment directories..."
mkdir -p experiments/{data,checkpoints/{encoder,policy},logs/{encoder,policy},visualizations,results}

# Create .gitkeep files
touch experiments/.gitkeep
touch experiments/data/.gitkeep
touch experiments/checkpoints/.gitkeep
touch experiments/logs/.gitkeep
touch experiments/visualizations/.gitkeep
touch experiments/results/.gitkeep

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Collect data: python scripts/collect_data.py --config configs/encoder_config.yaml --split train"
echo "3. Train encoder: python scripts/train_encoder.py --config configs/encoder_config.yaml --data-path experiments/data/pendulum_train_segments.pkl"
echo ""
echo "See QUICKSTART.md for detailed instructions."
