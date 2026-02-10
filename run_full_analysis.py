#!/usr/bin/env python3
"""
Complete workflow: Train encoder and analyze metric preservation.

This script runs the full pipeline:
1. Collect trajectory data
2. Train encoder
3. Analyze metric preservation
4. Generate plots and statistics

Use this for a complete end-to-end analysis.
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

# Import metric preservation analysis
from scripts.analyze_metric_preservation import (
    compute_mean_embeddings_per_context,
    compute_pairwise_distances,
    analyze_metric_preservation,
    plot_metric_preservation,
    save_statistics
)

# =============================================================================
# CONFIGURATION
# =============================================================================
ENV_NAME = "pendulum"
SEGMENT_LENGTH = 32
NUM_SEGMENTS_PER_CONTEXT = 100  # More data = better results
LATENT_DIM = 64
NUM_EPOCHS = 50  # More epochs = better convergence
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TEMPERATURE = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
DATA_DIR = Path("experiments/data")
ENCODER_DIR = Path("experiments/encoder")
RESULTS_DIR = Path("results")

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
ENCODER_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPLETE WORKFLOW: ENCODER TRAINING + METRIC PRESERVATION ANALYSIS")
print("="*80)
print(f"\nConfiguration:")
print(f"  Environment: {ENV_NAME}")
print(f"  Device: {DEVICE}")
print(f"  Segment length: {SEGMENT_LENGTH}")
print(f"  Segments per context: {NUM_SEGMENTS_PER_CONTEXT}")
print(f"  Latent dim: {LATENT_DIM}")
print(f"  Training epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")

# =============================================================================
# PHASE 1: COLLECT TRAJECTORY DATA
# =============================================================================
print("\n" + "="*80)
print("PHASE 1: COLLECTING TRAJECTORY DATA")
print("="*80)

train_contexts = get_context_distributions(ENV_NAME, split='train')
print(f"Available training contexts: {len(train_contexts)}")

# Use subset for faster training (you can use all for production)
train_contexts = train_contexts[:10]
print(f"Using {len(train_contexts)} contexts")
print("Gravity values:")
for i, ctx in enumerate(train_contexts):
    print(f"  Context {i}: g = {ctx['g']:.2f}")

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
data_path = DATA_DIR / "pendulum_train_segments.pkl"
collector.save(segments, str(data_path))

# =============================================================================
# PHASE 2: TRAIN TRAJECTORY ENCODER
# =============================================================================
print("\n" + "="*80)
print("PHASE 2: TRAINING TRAJECTORY ENCODER")
print("="*80)

# Infer input dimension
first_seg = segments[0]
obs_dim = first_seg.observations.shape[-1]
action_dim = first_seg.actions.shape[-1] if len(first_seg.actions.shape) > 1 else 1
input_dim = obs_dim + action_dim

print(f"Input dimension: {input_dim} (obs: {obs_dim}, action: {action_dim})")

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

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# Create encoder
encoder = LSTMEncoder(
    input_dim=input_dim,
    hidden_dim=256,
    num_layers=2,
    latent_dim=LATENT_DIM,
    dropout=0.2,
    bidirectional=True
)

print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

# Create loss function
loss_fn = SupConLoss(temperature=TEMPERATURE)

# Train
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

print("\nTraining encoder...")
trainer.train(num_epochs=NUM_EPOCHS, eval_every=5)

# Save encoder
encoder_path = ENCODER_DIR / "best_encoder.pt"
checkpoint = {
    'encoder_state_dict': encoder.state_dict(),
    'epoch': NUM_EPOCHS,
    'config': {
        'architecture': 'lstm',
        'hidden_dim': 256,
        'num_layers': 2,
        'latent_dim': LATENT_DIM,
        'dropout': 0.2,
        'bidirectional': True
    }
}
torch.save(checkpoint, encoder_path)
print(f"\nEncoder saved to: {encoder_path}")

# =============================================================================
# PHASE 3: ANALYZE METRIC PRESERVATION
# =============================================================================
print("\n" + "="*80)
print("PHASE 3: ANALYZING METRIC PRESERVATION")
print("="*80)

# Compute mean embeddings
print("\nComputing mean embeddings for each context...")
mean_embeddings, context_params = compute_mean_embeddings_per_context(
    encoder, segments, DEVICE
)

# Compute pairwise distances
print("Computing pairwise distances...")
param_distances, embedding_distances = compute_pairwise_distances(
    mean_embeddings, context_params, param_key='g'
)
print(f"Computed {len(param_distances)} pairwise distances")

# Analyze metric preservation
print("\nAnalyzing metric preservation...")
stats = analyze_metric_preservation(param_distances, embedding_distances)

# Print results
print("\n" + "="*80)
print("METRIC PRESERVATION ANALYSIS RESULTS")
print("="*80)

# Best model information
best_model = stats.get('best_model', 'linear')
best_r2 = stats.get('best_model_r2', stats['r2_score'])
best_eq = stats.get('best_model_equation', 'N/A')

print(f"\nBest Fit Model:       {best_model.upper()}")
print(f"Best Model R²:        {best_r2:.4f}")
print(f"Equation:             {best_eq}")

print(f"\nSpearman Correlation: {stats['spearman_correlation']:.4f} (p={stats['spearman_pvalue']:.2e})")
print(f"Pearson Correlation:  {stats['pearson_correlation']:.4f} (p={stats['pearson_pvalue']:.2e})")

# Print all model R² scores
all_models = stats.get('all_models', {})
if all_models:
    print("\nAll Models Tested (R² scores):")
    sorted_models = sorted(all_models.items(), key=lambda x: x[1]['r2'], reverse=True)
    for name, model in sorted_models:
        if model['r2'] > -float('inf'):
            marker = " <-- BEST" if name == best_model else ""
            print(f"  {name.capitalize():12s}: {model['r2']:.4f}{marker}")

# Interpretation (use Spearman for non-linear relationships)
print("\n" + "-"*80)
print("Interpretation:")
spearman = stats['spearman_correlation']
if spearman > 0.8:
    print("  [OK] EXCELLENT: Strong metric preservation (Spearman > 0.8)")
    print("  -> Encoder successfully learned meaningful context representations")
    print("  -> Ready for Phase 2 policy training!")
elif spearman > 0.6:
    print("  [OK] GOOD: Moderate metric preservation (0.6 < Spearman < 0.8)")
    print("  -> Encoder learned reasonable context representations")
    print("  -> Consider training longer for better results")
else:
    print("  [!!] WEAK: Limited metric preservation (Spearman < 0.6)")
    print("  -> Encoder may not have learned meaningful representations")

if best_model != 'linear':
    print(f"\nNote: Best fit is {best_model.upper()}, not linear.")
    print("  This is normal - embeddings often have non-linear relationships")
    print("  with parameters (e.g., saturation effects, compression).")
print("="*80)

# Create visualization
print(f"\nGenerating visualization...")
plot_path = RESULTS_DIR / 'metric_preservation_correlation.png'
plot_metric_preservation(
    param_distances,
    embedding_distances,
    stats,
    str(plot_path),
    "Gravity"
)

# Save statistics
print(f"Saving statistics...")
stats_path = RESULTS_DIR / 'metric_preservation_stats.txt'
save_statistics(stats, str(stats_path))

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("WORKFLOW COMPLETE!")
print("="*80)

print(f"\nGenerated Files:")
print(f"  Data:       {data_path}")
print(f"  Encoder:    {encoder_path}")
print(f"  Plot:       {plot_path}")
print(f"  Stats:      {stats_path}")

print(f"\nResults Summary:")
print(f"  Contexts analyzed: {len(mean_embeddings)}")
print(f"  Pairwise distances: {len(param_distances)}")
print(f"  Best model: {best_model.upper()} (R2={best_r2:.4f})")
print(f"  Spearman correlation: {spearman:.4f}")

if spearman > 0.8:
    print(f"\n[SUCCESS] Your encoder shows excellent metric preservation!")
    print(f"   Next step: Train a policy (Phase 2) using this encoder")
elif spearman > 0.6:
    print(f"\n[OK] Good results. Consider training longer for improvement.")
else:
    print(f"\n[NOTE] Moderate results. Consider training longer or adjusting hyperparameters.")

print("\n" + "="*80)
