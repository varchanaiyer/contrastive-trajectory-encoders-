#!/usr/bin/env python3
"""
Baseline Training: Vanilla SAC and Oracle SAC

Trains two baselines for comparison against the CTE-augmented SAC:
1. Vanilla SAC: No context information (3D observation only)
2. Oracle SAC: Ground-truth gravity appended (3D obs + 1D gravity = 4D)

Both are trained on the same 10 contexts and evaluated on the same test contexts
as the CTE-augmented policy for fair comparison.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from pathlib import Path
import sys
import json
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from src.data import get_context_distributions, make_carl_env

# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Same training contexts as CTE
train_contexts = get_context_distributions("pendulum", split='train')[:10]
train_g_values = [c['g'] for c in train_contexts]
min_train_g, max_train_g = min(train_g_values), max(train_g_values)

# Same test contexts as CTE
test_contexts = [
    {"g": 5.25}, {"g": 6.3}, {"g": 7.0}, {"g": 7.9}, {"g": 8.5}, {"g": 9.0},
    {"g": 3.0}, {"g": 4.0}, {"g": 12.0}, {"g": 15.0},
]

policy_kwargs = {
    'net_arch': [256, 256],
    'activation_fn': torch.nn.ReLU
}


# =============================================================================
# WRAPPERS
# =============================================================================

class CARLObsWrapper(gym.Wrapper):
    """Wrapper that handles CARL dict observations, returning flat arrays."""

    def __init__(self, env):
        super().__init__(env)
        obs, _ = env.reset()
        obs = self._extract(obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    def _extract(self, obs):
        if isinstance(obs, dict):
            obs = obs.get('obs', list(obs.values())[0])
        return np.asarray(obs, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._extract(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._extract(obs), reward, done, truncated, info


class OracleWrapper(gym.Wrapper):
    """Wrapper that appends ground-truth gravity to the observation."""

    def __init__(self, env, gravity_value):
        super().__init__(env)
        self.gravity_value = np.float32(gravity_value)
        obs, _ = env.reset()
        obs = self._extract(obs)
        augmented = np.append(obs, self.gravity_value)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=augmented.shape, dtype=np.float32
        )

    def _extract(self, obs):
        if isinstance(obs, dict):
            obs = obs.get('obs', list(obs.values())[0])
        return np.asarray(obs, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._extract(obs)
        return np.append(obs, self.gravity_value), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self._extract(obs)
        return np.append(obs, self.gravity_value), reward, done, truncated, info


# =============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# =============================================================================

def train_and_evaluate(name, make_env_fn, train_contexts, test_contexts):
    """Train a SAC agent and evaluate on test contexts."""

    print(f"\n{'=' * 80}")
    print(f"TRAINING: {name}")
    print(f"{'=' * 80}")

    # Create training env (sample a random context)
    train_ctx = train_contexts[np.random.randint(len(train_contexts))]
    train_env = make_env_fn(train_ctx)

    # Create eval env
    eval_ctx = test_contexts[0]
    eval_env = make_env_fn(eval_ctx)

    log_dir = Path(f"experiments/logs/baselines/{name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(f"experiments/baselines/{name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train
    model = SAC(
        'MlpPolicy',
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir),
        device=DEVICE
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        best_model_save_path=str(checkpoint_dir),
        log_path=str(log_dir),
        deterministic=True
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(checkpoint_dir / "final_policy.zip")
    print(f"\n{name} training complete!")

    # Evaluate on all test contexts
    print(f"\nEvaluating {name} on test contexts...")
    results = {}

    for ctx_id, context in enumerate(test_contexts):
        env = make_env_fn(context)
        episode_rewards = []

        for _ in range(N_EVAL_EPISODES):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                if truncated:
                    break
            episode_rewards.append(episode_reward)

        env.close()
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        results[f'context_{ctx_id}'] = {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'context': context
        }
        print(f"  Context {ctx_id} {context}: {mean_reward:.2f} +/- {std_reward:.2f}")

    train_env.close()
    eval_env.close()
    return results


# =============================================================================
# RUN BASELINES
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BASELINE TRAINING: Vanilla SAC & Oracle SAC")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Training contexts: {len(train_contexts)} (g = {min_train_g:.2f} to {max_train_g:.2f})")
    print(f"Test contexts: {len(test_contexts)}")

    # --- Vanilla SAC ---
    def make_vanilla_env(context):
        env = make_carl_env("pendulum", context)
        env = Monitor(CARLObsWrapper(env))
        return env

    vanilla_results = train_and_evaluate(
        "vanilla_sac", make_vanilla_env, train_contexts, test_contexts
    )

    # --- Oracle SAC ---
    def make_oracle_env(context):
        env = make_carl_env("pendulum", context)
        env = Monitor(OracleWrapper(env, context['g']))
        return env

    oracle_results = train_and_evaluate(
        "oracle_sac", make_oracle_env, train_contexts, test_contexts
    )

    # ==========================================================================
    # COMPARE ALL RESULTS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("BASELINE RESULTS COMPARISON")
    print("=" * 80)

    # Load CTE results if available
    cte_results_path = RESULTS_DIR / 'phase2_results.json'
    cte_results = None
    if cte_results_path.exists():
        with open(cte_results_path) as f:
            cte_data = json.load(f)
            cte_results = cte_data.get('per_context', {})

    def compute_stats(results, test_contexts):
        interp, extrap = [], []
        for ctx_id, context in enumerate(test_contexts):
            key = f'context_{ctx_id}'
            if key in results:
                r = float(results[key]['mean_reward'])
                if min_train_g <= context['g'] <= max_train_g:
                    interp.append(r)
                else:
                    extrap.append(r)
        return interp, extrap

    print(f"\n{'Method':<20} {'Interp Mean':>15} {'Extrap Mean':>15}")
    print("-" * 50)

    for name, res in [("Vanilla SAC", vanilla_results), ("Oracle SAC", oracle_results)]:
        interp, extrap = compute_stats(res, test_contexts)
        interp_mean = np.mean(interp) if interp else float('nan')
        extrap_mean = np.mean(extrap) if extrap else float('nan')
        print(f"{name:<20} {interp_mean:>15.2f} {extrap_mean:>15.2f}")

    if cte_results:
        interp, extrap = compute_stats(cte_results, test_contexts)
        interp_mean = np.mean(interp) if interp else float('nan')
        extrap_mean = np.mean(extrap) if extrap else float('nan')
        print(f"{'CTE SAC':<20} {interp_mean:>15.2f} {extrap_mean:>15.2f}")

    # Save baseline results
    baseline_summary = {
        'vanilla_sac': vanilla_results,
        'oracle_sac': oracle_results,
        'config': {
            'total_timesteps': TOTAL_TIMESTEPS,
            'train_contexts': len(train_contexts),
            'test_contexts': len(test_contexts),
            'train_range': [float(min_train_g), float(max_train_g)]
        }
    }
    results_path = RESULTS_DIR / 'baseline_results.json'
    with open(results_path, 'w') as f:
        json.dump(baseline_summary, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
