#!/usr/bin/env python3
"""Evaluate the best saved model checkpoint on test contexts"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from pathlib import Path
import json
from stable_baselines3 import PPO

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data import get_context_distributions, make_carl_env
from src.models import LSTMEncoder, ContextEncoder
from src.training.policy_trainer import ContextBufferWrapper
from stable_baselines3.common.monitor import Monitor
import pickle

print("=" * 80)
print("EVALUATING BEST MODEL CHECKPOINT")
print("=" * 80)

# Configuration
ENV_NAME = "pendulum"
DEVICE = 'cpu'
BEST_MODEL_PATH = Path("experiments/policy/best_model.zip")
ENCODER_PATH = Path("experiments/encoder/best_encoder.pt")
N_EVAL_EPISODES = 10  # More episodes for better statistics

# Load encoder
print("\nLoading encoder...")
checkpoint = torch.load(ENCODER_PATH, map_location=DEVICE)
enc_config = checkpoint.get('config', {})

# Get dimensions from saved data
data_path = Path("experiments/data/pendulum_train_segments.pkl")
with open(data_path, 'rb') as f:
    segments = pickle.load(f)
first_seg = segments[0]
obs_dim = first_seg.observations.shape[-1]
action_dim = first_seg.actions.shape[-1] if len(first_seg.actions.shape) > 1 else 1
input_dim = obs_dim + action_dim

# Create encoder
base_encoder = LSTMEncoder(
    input_dim=input_dim,
    hidden_dim=enc_config.get('hidden_dim', 256),
    num_layers=enc_config.get('num_layers', 2),
    latent_dim=enc_config.get('latent_dim', 64),
    dropout=enc_config.get('dropout', 0.2),
    bidirectional=enc_config.get('bidirectional', True)
)
base_encoder.load_state_dict(checkpoint['encoder_state_dict'])

encoder = ContextEncoder(trajectory_encoder=base_encoder, freeze_encoder=True)
encoder.eval()

# Load best policy
print(f"Loading best model from {BEST_MODEL_PATH}...")
policy = PPO.load(BEST_MODEL_PATH, device=DEVICE)

# Test contexts (same as training script)
train_contexts = get_context_distributions(ENV_NAME, split='train')[:10]
train_g_values = [c['g'] for c in train_contexts]
min_train_g, max_train_g = min(train_g_values), max(train_g_values)

test_contexts = [
    {"g": 5.25}, {"g": 6.3}, {"g": 7.0}, {"g": 7.9}, {"g": 8.5}, {"g": 9.0},
    {"g": 3.0}, {"g": 4.0}, {"g": 12.0}, {"g": 15.0},
]

print(f"\nEvaluating on {len(test_contexts)} test contexts ({N_EVAL_EPISODES} episodes each)...")
print("-" * 80)

results = {}
interpolation_results = []
extrapolation_results = []

for ctx_id, context in enumerate(test_contexts):
    # Create environment with context wrapper
    base_env = make_carl_env(ENV_NAME, context)
    base_env = Monitor(base_env)
    env = ContextBufferWrapper(base_env, encoder, buffer_length=32, device=DEVICE)

    episode_rewards = []

    for ep in range(N_EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            if truncated:
                break

        episode_rewards.append(episode_reward)

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    # Categorize
    if min_train_g <= context['g'] <= max_train_g:
        eval_type = "INTERP"
        interpolation_results.append(mean_reward)
    else:
        eval_type = "EXTRAP"
        extrapolation_results.append(mean_reward)

    results[f'context_{ctx_id}'] = {
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'context': context
    }

    print(f"  g={context['g']:5.2f} ({eval_type}): {mean_reward:8.2f} +/- {std_reward:6.2f}")

# Summary
print("\n" + "=" * 80)
print("BEST MODEL RESULTS SUMMARY")
print("=" * 80)

all_rewards = [r['mean_reward'] for r in results.values()]
overall_mean = np.mean(all_rewards)
overall_std = np.std(all_rewards)

print(f"\nOverall: {overall_mean:.2f} +/- {overall_std:.2f}")

if interpolation_results:
    print(f"Interpolation ({len(interpolation_results)} contexts): {np.mean(interpolation_results):.2f} +/- {np.std(interpolation_results):.2f}")

if extrapolation_results:
    print(f"Extrapolation ({len(extrapolation_results)} contexts): {np.mean(extrapolation_results):.2f} +/- {np.std(extrapolation_results):.2f}")

# Save results
results_path = Path("results/best_model_results.json")
summary = {
    'model': 'best_model.zip',
    'overall': {'mean_reward': float(overall_mean), 'std_reward': float(overall_std)},
    'interpolation': {'mean_reward': float(np.mean(interpolation_results)) if interpolation_results else None},
    'extrapolation': {'mean_reward': float(np.mean(extrapolation_results)) if extrapolation_results else None},
    'per_context': results
}

with open(results_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to: {results_path}")
