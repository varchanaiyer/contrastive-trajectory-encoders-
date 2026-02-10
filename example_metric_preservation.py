#!/usr/bin/env python3
"""
Example: Analyze metric preservation after training an encoder.

This script demonstrates the complete workflow:
1. Collect trajectory data
2. Train a small encoder
3. Analyze metric preservation

This is a simplified version for demonstration purposes.
For production use, use the scripts in scripts/ directory.
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import TrajectoryCollector, TrajectoryDataset, get_context_distributions
from src.models import LSTMEncoder, SupConLoss
from src.training import EncoderTrainer

# Configuration
ENV_NAME = "pendulum"
SEGMENT_LENGTH = 32
NUM_SEGMENTS_PER_CONTEXT = 20  # Fewer for quick demo
LATENT_DIM = 32  # Smaller for quick demo
NUM_EPOCHS = 10  # Fewer for quick demo
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TEMPERATURE = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*80)
print("METRIC PRESERVATION ANALYSIS - QUICK DEMO")
print("="*80)
print(f"\nDevice: {DEVICE}")
print("Note: This is a minimal demo. For better results, use longer training.")

# ============================================================================
# STEP 1: Collect Data
# ============================================================================
print("\n" + "="*80)
print("STEP 1: COLLECTING TRAJECTORY DATA")
print("="*80)

# Get a few training contexts
train_contexts = get_context_distributions(ENV_NAME, split='train')
train_contexts = train_contexts[:5]  # Just 5 contexts for quick demo
print(f"Using {len(train_contexts)} contexts:")
for i, ctx in enumerate(train_contexts):
    print(f"  Context {i}: gravity = {ctx['g']:.2f}")

collector = TrajectoryCollector(
    env_name=ENV_NAME,
    contexts=train_contexts,
    segment_length=SEGMENT_LENGTH,
    num_segments_per_context=NUM_SEGMENTS_PER_CONTEXT,
    policy='random',
    seed=42
)

segments = collector.collect(verbose=True)
print(f"\nCollected {len(segments)} trajectory segments")

# Save data
data_path = Path("demo_outputs/demo_segments.pkl")
data_path.parent.mkdir(exist_ok=True)
collector.save(segments, str(data_path))

# ============================================================================
# STEP 2: Train Encoder
# ============================================================================
print("\n" + "="*80)
print("STEP 2: TRAINING ENCODER (QUICK DEMO)")
print("="*80)

# Infer dimensions
first_seg = segments[0]
obs_dim = first_seg.observations.shape[-1]
action_dim = first_seg.actions.shape[-1] if len(first_seg.actions.shape) > 1 else 1
input_dim = obs_dim + action_dim

print(f"Input dimension: {input_dim}")

# Create dataset
dataset = TrajectoryDataset(segments, augmentation="noise")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# Create encoder
encoder = LSTMEncoder(
    input_dim=input_dim,
    hidden_dim=128,
    num_layers=2,
    latent_dim=LATENT_DIM,
    dropout=0.1,
    bidirectional=True
)

print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

# Train
loss_fn = SupConLoss(temperature=TEMPERATURE)
trainer = EncoderTrainer(
    encoder=encoder,
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=LEARNING_RATE,
    weight_decay=1e-4,
    device=DEVICE,
    log_dir=None,
    checkpoint_dir=None
)

print(f"\nTraining for {NUM_EPOCHS} epochs...")
trainer.train(num_epochs=NUM_EPOCHS, eval_every=2)

# Save encoder
encoder_path = Path("demo_outputs/demo_encoder.pt")
checkpoint = {
    'encoder_state_dict': encoder.state_dict(),
    'epoch': NUM_EPOCHS,
    'config': {
        'architecture': 'lstm',
        'hidden_dim': 128,
        'num_layers': 2,
        'latent_dim': LATENT_DIM,
        'dropout': 0.1,
        'bidirectional': True
    }
}
torch.save(checkpoint, encoder_path)
print(f"\nEncoder saved to: {encoder_path}")

# ============================================================================
# STEP 3: Analyze Metric Preservation
# ============================================================================
print("\n" + "="*80)
print("STEP 3: ANALYZING METRIC PRESERVATION")
print("="*80)

print("\nNow run the analysis script:")
print(f"\npython scripts/analyze_metric_preservation.py \\")
print(f"    --encoder-path {encoder_path} \\")
print(f"    --data-path {data_path} \\")
print(f"    --results-dir demo_outputs/results")

print("\nOr import and use directly:")
print("-" * 80)

# Import the analysis functions
from scripts.analyze_metric_preservation import (
    load_encoder,
    compute_mean_embeddings_per_context,
    compute_pairwise_distances,
    analyze_metric_preservation,
    plot_metric_preservation,
    save_statistics
)

# Load encoder
print("Loading encoder...")
encoder_loaded = load_encoder(str(encoder_path), input_dim, DEVICE)

# Compute mean embeddings
print("Computing mean embeddings...")
mean_embeddings, context_params = compute_mean_embeddings_per_context(
    encoder_loaded, segments, DEVICE
)

# Compute distances
print("Computing pairwise distances...")
param_distances, embedding_distances = compute_pairwise_distances(
    mean_embeddings, context_params, param_key='g'
)

# Analyze
print("Analyzing metric preservation...")
stats = analyze_metric_preservation(param_distances, embedding_distances)

# Print results
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Pearson Correlation:  {stats['pearson_correlation']:.4f}")
print(f"Spearman Correlation: {stats['spearman_correlation']:.4f}")
print(f"R² Score:             {stats['r2_score']:.4f}")

if stats['pearson_correlation'] > 0.8:
    print("\n✓ EXCELLENT metric preservation!")
elif stats['pearson_correlation'] > 0.6:
    print("\n✓ GOOD metric preservation")
else:
    print("\n✗ WEAK metric preservation - consider training longer")

# Save results
print("\nSaving results...")
results_dir = Path("demo_outputs/results")
plot_metric_preservation(
    param_distances,
    embedding_distances,
    stats,
    str(results_dir / "metric_preservation_correlation.png"),
    "Gravity"
)
save_statistics(stats, str(results_dir / "metric_preservation_stats.txt"))

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print(f"\nResults saved to: {results_dir}/")
print(f"  - Plot: {results_dir}/metric_preservation_correlation.png")
print(f"  - Stats: {results_dir}/metric_preservation_stats.txt")
print("\nNote: This demo uses minimal training for speed.")
print("For production, use the full pipeline with more epochs and data.")
print("="*80)
