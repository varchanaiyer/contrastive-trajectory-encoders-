#!/usr/bin/env python3
"""
Complete pipeline script for Contrastive Trajectory Encoders.

This script:
1. Collects trajectory data from CARL-Pendulum environment
2. Trains a contrastive encoder (LSTM-based)
3. Evaluates embedding quality
4. Visualizes embeddings with t-SNE

Run with: python3 run_pipeline.py
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configuration
ENV_NAME = "pendulum"
SEGMENT_LENGTH = 32
NUM_SEGMENTS_PER_CONTEXT = 100  # More data per context
LATENT_DIM = 64
NUM_EPOCHS = 50  # More epochs for convergence
BATCH_SIZE = 64  # Larger batch for better contrastive learning
LEARNING_RATE = 1e-3  # Higher LR with warmup
TEMPERATURE = 0.1  # Slightly higher temp for stability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*80)
print("CONTRASTIVE TRAJECTORY ENCODERS - COMPLETE PIPELINE")
print("="*80)

print(f"\nConfiguration:")
print(f"  Environment: {ENV_NAME}")
print(f"  Device: {DEVICE}")
print(f"  Segment length: {SEGMENT_LENGTH}")
print(f"  Segments per context: {NUM_SEGMENTS_PER_CONTEXT}")
print(f"  Latent dim: {LATENT_DIM}")
print(f"  Training epochs: {NUM_EPOCHS}")

# ============================================================================
# STEP 1: Collect Trajectory Data
# ============================================================================
print("\n" + "="*80)
print("STEP 1: COLLECTING TRAJECTORY DATA")
print("="*80)

from src.data import TrajectoryCollector, TrajectoryDataset, get_context_distributions

train_contexts = get_context_distributions(ENV_NAME, split='train')
print(f"Available training contexts: {len(train_contexts)}")

# Use more contexts for better learning
train_contexts = train_contexts[:10]  # 10 contexts instead of 5
print(f"Using {len(train_contexts)} contexts")

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

# Verify segment shapes
first_seg = segments[0]
print(f"Segment shape - obs: {first_seg.observations.shape}, actions: {first_seg.actions.shape}")

# ============================================================================
# STEP 2: Create Dataset and Train Encoder
# ============================================================================
print("\n" + "="*80)
print("STEP 2: TRAINING TRAJECTORY ENCODER")
print("="*80)

# Infer input dimension
obs_dim = first_seg.observations.shape[-1]
action_dim = first_seg.actions.shape[-1] if len(first_seg.actions.shape) > 1 else 1
input_dim = obs_dim + action_dim

print(f"Input dimension: {input_dim} (obs: {obs_dim}, action: {action_dim})")

# Create dataset with augmentation for better positive pairs
dataset = TrajectoryDataset(segments, augmentation="noise")

# Split train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Create data loaders with more workers
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# Create encoder
from src.models import LSTMEncoder, SupConLoss

encoder = LSTMEncoder(
    input_dim=input_dim,
    hidden_dim=256,  # Larger hidden dim
    num_layers=2,
    latent_dim=LATENT_DIM,
    dropout=0.2,  # More dropout for regularization
    bidirectional=True
)

print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

# Create loss function - use SupConLoss for better multi-positive handling
loss_fn = SupConLoss(temperature=TEMPERATURE)

from src.training import EncoderTrainer

trainer = EncoderTrainer(
    encoder=encoder,
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=LEARNING_RATE,
    weight_decay=1e-4,  # More regularization
    device=DEVICE,
    log_dir=None,
    checkpoint_dir=None
)

# Train
print("\nTraining encoder...")
trainer.train(num_epochs=NUM_EPOCHS, eval_every=2)

# ============================================================================
# STEP 3: Evaluate Embedding Quality
# ============================================================================
print("\n" + "="*80)
print("STEP 3: EVALUATING EMBEDDING QUALITY")
print("="*80)

def compute_embedding_quality_fixed(encoder, segments, device='cpu'):
    """Compute embedding quality metrics with fixed observation handling."""
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    encoder.eval()
    encoder.to(device)

    embeddings = []
    context_ids = []

    with torch.no_grad():
        for seg in segments:
            # Prepare trajectory - use all observations (fixed)
            obs = seg.observations
            actions = seg.actions
            if len(actions.shape) == 1:
                actions = actions[:, None]

            trajectory = np.concatenate([obs, actions], axis=-1)
            traj_tensor = torch.FloatTensor(trajectory).unsqueeze(0).to(device)

            # Encode
            emb = encoder(traj_tensor).cpu().numpy()
            embeddings.append(emb)
            context_ids.append(seg.context_id)

    embeddings = np.concatenate(embeddings, axis=0)
    context_ids = np.array(context_ids)

    # Compute metrics
    unique_contexts = np.unique(context_ids)

    # 1. Intra-context similarity (should be high)
    intra_similarities = []
    for ctx in unique_contexts:
        ctx_mask = context_ids == ctx
        ctx_embeddings = embeddings[ctx_mask]
        if len(ctx_embeddings) > 1:
            # Compute pairwise cosine similarity
            norms = np.linalg.norm(ctx_embeddings, axis=1, keepdims=True)
            normalized = ctx_embeddings / (norms + 1e-8)
            sim_matrix = normalized @ normalized.T
            # Get upper triangle (excluding diagonal)
            upper_tri = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
            intra_similarities.extend(upper_tri)

    intra_sim = np.mean(intra_similarities) if intra_similarities else 0.0

    # 2. Inter-context similarity (should be low)
    inter_similarities = []
    for i, ctx1 in enumerate(unique_contexts):
        for ctx2 in unique_contexts[i+1:]:
            ctx1_embeddings = embeddings[context_ids == ctx1]
            ctx2_embeddings = embeddings[context_ids == ctx2]

            # Normalize
            norms1 = np.linalg.norm(ctx1_embeddings, axis=1, keepdims=True)
            norms2 = np.linalg.norm(ctx2_embeddings, axis=1, keepdims=True)
            normalized1 = ctx1_embeddings / (norms1 + 1e-8)
            normalized2 = ctx2_embeddings / (norms2 + 1e-8)

            # Cross similarity
            cross_sim = normalized1 @ normalized2.T
            inter_similarities.extend(cross_sim.flatten())

    inter_sim = np.mean(inter_similarities) if inter_similarities else 0.0

    # 3. Separation score
    separation = intra_sim - inter_sim

    # 4. Silhouette score
    try:
        silhouette = silhouette_score(embeddings, context_ids)
    except:
        silhouette = 0.0

    # 5. Davies-Bouldin score (lower is better)
    try:
        db_score = davies_bouldin_score(embeddings, context_ids)
    except:
        db_score = float('inf')

    return {
        'intra_similarity': float(intra_sim),
        'inter_similarity': float(inter_sim),
        'separation': float(separation),
        'silhouette_score': float(silhouette),
        'davies_bouldin_score': float(db_score)
    }

quality_metrics = compute_embedding_quality_fixed(
    encoder=trainer.encoder,
    segments=segments,
    device=DEVICE
)

print("\nEmbedding Quality Metrics:")
print(f"  Intra-context similarity: {quality_metrics['intra_similarity']:.4f} (should be high)")
print(f"  Inter-context similarity: {quality_metrics['inter_similarity']:.4f} (should be low)")
print(f"  Separation score: {quality_metrics['separation']:.4f} (should be positive)")
print(f"  Silhouette score: {quality_metrics['silhouette_score']:.4f} (should be high)")
print(f"  Davies-Bouldin score: {quality_metrics['davies_bouldin_score']:.4f} (should be low)")

# ============================================================================
# STEP 4: Visualize Embeddings
# ============================================================================
print("\n" + "="*80)
print("STEP 4: VISUALIZING EMBEDDINGS")
print("="*80)

from sklearn.manifold import TSNE

# Collect embeddings for visualization
encoder.eval()
encoder.to(DEVICE)

all_embeddings = []
all_context_ids = []
all_context_params = []

with torch.no_grad():
    for seg in segments:
        obs = seg.observations
        actions = seg.actions
        if len(actions.shape) == 1:
            actions = actions[:, None]

        trajectory = np.concatenate([obs, actions], axis=-1)
        traj_tensor = torch.FloatTensor(trajectory).unsqueeze(0).to(DEVICE)

        emb = encoder(traj_tensor).cpu().numpy()
        all_embeddings.append(emb)
        all_context_ids.append(seg.context_id)
        all_context_params.append(list(seg.context_params.values())[0])

all_embeddings = np.concatenate(all_embeddings, axis=0)
all_context_ids = np.array(all_context_ids)
all_context_params = np.array(all_context_params)

print(f"Embeddings shape: {all_embeddings.shape}")
print("Running t-SNE dimensionality reduction...")

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
embeddings_2d = tsne.fit_transform(all_embeddings)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Color by context ID
scatter1 = axes[0].scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1],
    c=all_context_ids, cmap='tab10', alpha=0.7, s=50
)
axes[0].set_title('t-SNE Embeddings by Context ID')
axes[0].set_xlabel('t-SNE Dimension 1')
axes[0].set_ylabel('t-SNE Dimension 2')
plt.colorbar(scatter1, ax=axes[0], label='Context ID')

# Plot 2: Color by gravity parameter
scatter2 = axes[1].scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1],
    c=all_context_params, cmap='viridis', alpha=0.7, s=50
)
axes[1].set_title('t-SNE Embeddings by Gravity (g)')
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
plt.colorbar(scatter2, ax=axes[1], label='Gravity (g)')

plt.tight_layout()

# Save visualization
output_dir = Path('demo_outputs')
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'embeddings_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)

print("\nSummary:")
print(f"  Segments collected: {len(segments)}")
print(f"  Contexts used: {len(train_contexts)}")
print(f"  Encoder trained for: {NUM_EPOCHS} epochs")
print(f"  Final embedding quality:")
print(f"    - Intra-context similarity: {quality_metrics['intra_similarity']:.4f}")
print(f"    - Inter-context similarity: {quality_metrics['inter_similarity']:.4f}")
print(f"    - Separation: {quality_metrics['separation']:.4f}")
print(f"    - Silhouette score: {quality_metrics['silhouette_score']:.4f}")

print("\nNext steps:")
print("1. Train for more epochs (100+) for better embeddings")
print("2. Collect more data (100+ segments per context)")
print("3. Use the trained encoder for Phase 2 policy training")
print("4. See QUICKSTART.md for the full pipeline")

print(f"\nVisualization saved to: {output_path}")
print("="*80)
