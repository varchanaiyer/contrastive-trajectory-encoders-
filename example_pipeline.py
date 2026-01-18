#!/usr/bin/env python3
"""
Complete example pipeline demonstrating the full workflow.

This script shows how to:
1. Collect trajectory data
2. Train a contrastive encoder
3. Visualize embeddings
4. Train a context-conditional policy
5. Evaluate zero-shot performance

Note: This is a simplified example. For production use, run the individual
scripts in the scripts/ directory.
"""

import torch
import numpy as np
from pathlib import Path

# Import project modules
from src.data import (
    TrajectoryCollector,
    TrajectoryDataset,
    get_context_distributions,
    make_carl_env
)
from src.models import LSTMEncoder, InfoNCELoss
from src.training import EncoderTrainer
from src.evaluation import EmbeddingVisualizer, compute_embedding_quality

print("="*80)
print("CONTRASTIVE TRAJECTORY ENCODERS - EXAMPLE PIPELINE")
print("="*80)

# Configuration
ENV_NAME = "pendulum"
SEGMENT_LENGTH = 32
NUM_SEGMENTS_PER_CONTEXT = 50  # Reduced for quick demo
LATENT_DIM = 64
NUM_EPOCHS = 10  # Reduced for quick demo
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\nConfiguration:")
print(f"  Environment: {ENV_NAME}")
print(f"  Device: {DEVICE}")
print(f"  Segment length: {SEGMENT_LENGTH}")
print(f"  Latent dim: {LATENT_DIM}")

# Step 1: Collect trajectory data
print("\n" + "="*80)
print("STEP 1: COLLECTING TRAJECTORY DATA")
print("="*80)

train_contexts = get_context_distributions(ENV_NAME, split='train')
print(f"Training contexts: {len(train_contexts)}")
print(f"Example contexts: {train_contexts[:3]}")

# Use subset for demo
train_contexts = train_contexts[:5]

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

# Step 2: Create dataset and train encoder
print("\n" + "="*80)
print("STEP 2: TRAINING TRAJECTORY ENCODER")
print("="*80)

# Infer input dimension
first_seg = segments[0]
obs_dim = first_seg.observations.shape[-1]
action_dim = first_seg.actions.shape[-1] if len(first_seg.actions.shape) > 1 else 1
input_dim = obs_dim + action_dim

print(f"Input dimension: {input_dim} (obs: {obs_dim}, action: {action_dim})")

# Create dataset
dataset = TrajectoryDataset(segments)

# Split train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False
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

print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")

# Create loss function
loss_fn = InfoNCELoss(temperature=0.07)

# Create trainer
trainer = EncoderTrainer(
    encoder=encoder,
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=3e-4,
    weight_decay=1e-5,
    device=DEVICE,
    log_dir=None,  # Disable logging for demo
    checkpoint_dir=None
)

# Train
print("\nTraining encoder...")
trainer.train(num_epochs=NUM_EPOCHS, eval_every=2)

# Step 3: Evaluate embedding quality
print("\n" + "="*80)
print("STEP 3: EVALUATING EMBEDDING QUALITY")
print("="*80)

quality_metrics = compute_embedding_quality(
    encoder=trainer.encoder,
    segments=segments,
    device=DEVICE
)

# Step 4: Visualize embeddings
print("\n" + "="*80)
print("STEP 4: VISUALIZING EMBEDDINGS")
print("="*80)

visualizer = EmbeddingVisualizer(save_dir='demo_outputs')

print("Generating t-SNE visualization...")
reduced, context_ids = visualizer.visualize_embeddings(
    encoder=trainer.encoder,
    segments=segments,
    method='tsne',
    title='Learned Trajectory Embeddings',
    color_by='context',
    save_name='demo_embeddings.png'
)

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)

print("\nSummary:")
print(f"  Segments collected: {len(segments)}")
print(f"  Contexts: {len(train_contexts)}")
print(f"  Embedding quality:")
print(f"    - Intra-context similarity: {quality_metrics['intra_similarity']:.3f}")
print(f"    - Inter-context similarity: {quality_metrics['inter_similarity']:.3f}")
print(f"    - Separation: {quality_metrics['separation']:.3f}")
print(f"    - Silhouette score: {quality_metrics['silhouette_score']:.3f}")

print("\nNext steps:")
print("1. Train for more epochs (100+) for better embeddings")
print("2. Collect more data (100+ segments per context)")
print("3. Use the trained encoder for Phase 2 policy training")
print("4. See QUICKSTART.md for the full pipeline")

print("\nVisualization saved to: demo_outputs/demo_embeddings.png")
