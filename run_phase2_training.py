#!/usr/bin/env python3
"""
Phase 2: Context-Conditional Policy Training

This script trains a policy that uses the Phase 1 encoder to:
1. Encode trajectory history into context embeddings
2. Condition policy actions on [observation, context_embedding]
3. Adapt zero-shot to unseen contexts (interpolation and extrapolation)

Prerequisites:
- Run Phase 1 first: python run_full_analysis.py
- Or have a trained encoder at experiments/encoder/best_encoder.pt
"""

# Fix OpenMP conflict (must be before importing numpy/torch)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import get_context_distributions, make_carl_env
from src.models import LSTMEncoder, ContextEncoder
from src.training import PolicyTrainer

# =============================================================================
# CONFIGURATION
# =============================================================================
ENV_NAME = "pendulum"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Encoder settings (must match Phase 1)
ENCODER_PATH = Path("experiments/encoder/best_encoder.pt")
LATENT_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True
BUFFER_LENGTH = 32

# Policy settings
ALGORITHM = "sac"  # Options: "ppo", "sac" - SAC is more sample efficient
TOTAL_TIMESTEPS = 100_000  # Reduced for faster iteration
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5

# Paths
CHECKPOINT_DIR = Path("experiments/policy")
LOG_DIR = Path("experiments/logs/policy")
RESULTS_DIR = Path("results")

# Create directories
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PHASE 2: CONTEXT-CONDITIONAL POLICY TRAINING")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Environment: {ENV_NAME}")
print(f"  Device: {DEVICE}")
print(f"  Algorithm: {ALGORITHM.upper()}")
print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"  Encoder path: {ENCODER_PATH}")

# =============================================================================
# PHASE 2.1: LOAD TRAINED ENCODER
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 2.1: LOADING TRAINED ENCODER FROM PHASE 1")
print("=" * 80)

# Check if encoder exists
if not ENCODER_PATH.exists():
    print(f"\nERROR: Encoder not found at {ENCODER_PATH}")
    print("Please run Phase 1 first: python run_full_analysis.py")
    sys.exit(1)

# Load checkpoint
checkpoint = torch.load(ENCODER_PATH, map_location=DEVICE)
print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

# Get encoder config from checkpoint
enc_config = checkpoint.get('config', {})
print(f"Encoder config: {enc_config}")

# Infer input dimension from saved trajectory data (more reliable than CARL env)
import pickle
data_path = Path("experiments/data/pendulum_train_segments.pkl")
if data_path.exists():
    with open(data_path, 'rb') as f:
        segments = pickle.load(f)
    first_seg = segments[0]
    obs_dim = first_seg.observations.shape[-1]
    action_dim = first_seg.actions.shape[-1] if len(first_seg.actions.shape) > 1 else 1
    input_dim = obs_dim + action_dim
else:
    # Fallback: hardcoded for pendulum (obs=3, action=1)
    obs_dim = 3
    action_dim = 1
    input_dim = 4
    print("  Warning: Using hardcoded dimensions for pendulum")

print(f"Input dimension: {input_dim} (obs: {obs_dim}, action: {action_dim})")

# Create encoder with same architecture as Phase 1
base_encoder = LSTMEncoder(
    input_dim=input_dim,
    hidden_dim=enc_config.get('hidden_dim', HIDDEN_DIM),
    num_layers=enc_config.get('num_layers', NUM_LAYERS),
    latent_dim=enc_config.get('latent_dim', LATENT_DIM),
    dropout=enc_config.get('dropout', DROPOUT),
    bidirectional=enc_config.get('bidirectional', BIDIRECTIONAL)
)

# Load weights
base_encoder.load_state_dict(checkpoint['encoder_state_dict'])
print("Encoder weights loaded successfully!")

# Wrap in ContextEncoder (freezes encoder for policy training)
encoder = ContextEncoder(
    trajectory_encoder=base_encoder,
    freeze_encoder=True  # Important: freeze encoder during policy training
)
encoder.eval()

print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

# =============================================================================
# PHASE 2.2: SET UP TRAINING AND TEST CONTEXTS
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 2.2: SETTING UP TRAINING AND TEST CONTEXTS")
print("=" * 80)

# Training contexts (same as Phase 1)
train_contexts = get_context_distributions(ENV_NAME, split='train')
train_contexts = train_contexts[:10]  # Use same subset as Phase 1

# Get training gravity values to determine interpolation range
train_g_values = [c['g'] for c in train_contexts]
min_train_g, max_train_g = min(train_g_values), max(train_g_values)

# Custom test contexts with BOTH interpolation and extrapolation
# Interpolation: values between min and max training gravity (but not in training set)
# Extrapolation: values outside training range
test_contexts = [
    # INTERPOLATION (within training range 5.0-9.74, but not trained on)
    {"g": 5.25},   # Between training points
    {"g": 6.3},    # Between training points
    {"g": 7.0},    # Between training points
    {"g": 7.9},    # Between training points
    {"g": 8.5},    # Between training points
    {"g": 9.0},    # Between training points
    # EXTRAPOLATION (outside training range)
    {"g": 3.0},    # Below training range
    {"g": 4.0},    # Below training range
    {"g": 12.0},   # Above training range (moderate)
    {"g": 15.0},   # Above training range (harder)
]

print(f"\nTraining contexts ({len(train_contexts)}):")
for i, ctx in enumerate(train_contexts):
    print(f"  {i}: g = {ctx['g']:.2f}")

print(f"\nTest contexts for zero-shot evaluation ({len(test_contexts)}):")
for i, ctx in enumerate(test_contexts):
    # Determine if interpolation or extrapolation (using already computed range)
    if min_train_g <= ctx['g'] <= max_train_g:
        eval_type = "INTERPOLATION"
    else:
        eval_type = "EXTRAPOLATION"

    print(f"  {i}: g = {ctx['g']:.2f} ({eval_type})")

# =============================================================================
# PHASE 2.3: TRAIN CONTEXT-CONDITIONAL POLICY
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 2.3: TRAINING CONTEXT-CONDITIONAL POLICY")
print("=" * 80)

# Create policy trainer
trainer = PolicyTrainer(
    env_name=ENV_NAME,
    encoder=encoder,
    contexts=train_contexts,
    algorithm=ALGORITHM,
    buffer_length=BUFFER_LENGTH,
    device=DEVICE,
    log_dir=str(LOG_DIR),
    checkpoint_dir=str(CHECKPOINT_DIR)
)

# Policy network architecture
policy_kwargs = {
    'net_arch': [256, 256],
    'activation_fn': torch.nn.ReLU
}

print(f"\nStarting {ALGORITHM.upper()} training...")
print(f"This will take a while. Progress updates every {EVAL_FREQ:,} steps.")
print("-" * 80)

# Train the policy
trainer.train(
    total_timesteps=TOTAL_TIMESTEPS,
    eval_contexts=test_contexts[:1],  # Use first test context for progress eval
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    policy_kwargs=policy_kwargs
)

# =============================================================================
# PHASE 2.4: EVALUATE ZERO-SHOT ADAPTATION
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 2.4: EVALUATING ZERO-SHOT ADAPTATION")
print("=" * 80)

print("\nEvaluating on all test contexts (zero-shot)...")
print("These are contexts the policy has NEVER seen during training!\n")

results = trainer.evaluate(
    contexts=test_contexts,
    n_episodes=N_EVAL_EPISODES,
    deterministic=True
)

# Categorize results
train_g_values = [c['g'] for c in train_contexts]
min_train_g, max_train_g = min(train_g_values), max(train_g_values)

interpolation_results = []
extrapolation_results = []

for ctx_id, ctx_result in results.items():
    ctx = ctx_result['context']
    if min_train_g <= ctx['g'] <= max_train_g:
        interpolation_results.append(ctx_result)
    else:
        extrapolation_results.append(ctx_result)

# =============================================================================
# PHASE 2.5: SUMMARY AND RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 2 RESULTS SUMMARY")
print("=" * 80)

# Calculate overall statistics
all_rewards = [r['mean_reward'] for r in results.values()]
overall_mean = np.mean(all_rewards)
overall_std = np.std(all_rewards)

print(f"\nOverall Performance:")
print(f"  Mean reward: {overall_mean:.2f} +/- {overall_std:.2f}")
print(f"  Number of test contexts: {len(test_contexts)}")

if interpolation_results:
    interp_rewards = [r['mean_reward'] for r in interpolation_results]
    print(f"\nInterpolation (within training range [{min_train_g:.1f}, {max_train_g:.1f}]):")
    print(f"  Mean reward: {np.mean(interp_rewards):.2f} +/- {np.std(interp_rewards):.2f}")
    print(f"  Number of contexts: {len(interpolation_results)}")

if extrapolation_results:
    extrap_rewards = [r['mean_reward'] for r in extrapolation_results]
    print(f"\nExtrapolation (outside training range):")
    print(f"  Mean reward: {np.mean(extrap_rewards):.2f} +/- {np.std(extrap_rewards):.2f}")
    print(f"  Number of contexts: {len(extrapolation_results)}")

# Interpretation
print("\n" + "-" * 80)
print("Interpretation:")

# For pendulum, higher (less negative) rewards are better
# Typical random policy gets around -1000 to -1500
# Good policy gets around -200 to -500
if overall_mean > -500:
    print("  [EXCELLENT] Strong zero-shot adaptation!")
    print("  -> Policy successfully generalizes to unseen contexts")
elif overall_mean > -800:
    print("  [GOOD] Reasonable zero-shot adaptation")
    print("  -> Policy shows generalization, room for improvement")
else:
    print("  [MODERATE] Limited zero-shot adaptation")
    print("  -> Consider training longer or tuning hyperparameters")

if interpolation_results and extrapolation_results:
    interp_mean = np.mean([r['mean_reward'] for r in interpolation_results])
    extrap_mean = np.mean([r['mean_reward'] for r in extrapolation_results])

    if interp_mean > extrap_mean:
        print("\n  Note: Interpolation performs better than extrapolation (expected)")
    else:
        print("\n  Note: Extrapolation performs comparably to interpolation (good sign!)")

print("=" * 80)

# Save results
results_path = RESULTS_DIR / 'phase2_results.json'
summary = {
    'overall': {
        'mean_reward': float(overall_mean),
        'std_reward': float(overall_std),
        'num_contexts': len(test_contexts)
    },
    'interpolation': {
        'mean_reward': float(np.mean([r['mean_reward'] for r in interpolation_results])) if interpolation_results else None,
        'num_contexts': len(interpolation_results)
    },
    'extrapolation': {
        'mean_reward': float(np.mean([r['mean_reward'] for r in extrapolation_results])) if extrapolation_results else None,
        'num_contexts': len(extrapolation_results)
    },
    'per_context': results,
    'config': {
        'algorithm': ALGORITHM,
        'total_timesteps': TOTAL_TIMESTEPS,
        'train_contexts': len(train_contexts),
        'test_contexts': len(test_contexts),
        'device': DEVICE
    }
}

with open(results_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nResults saved to: {results_path}")
print(f"Policy saved to: {CHECKPOINT_DIR / 'final_policy.zip'}")
print(f"Logs saved to: {LOG_DIR}")

print("\n" + "=" * 80)
print("PHASE 2 COMPLETE!")
print("=" * 80)
print("\nYour context-conditional policy has been trained and evaluated.")
print("The policy uses the Phase 1 encoder to infer context from trajectory history,")
print("enabling zero-shot adaptation to new environments without retraining.")
