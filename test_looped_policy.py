#!/usr/bin/env python3
"""
Test: Does iterating z (looped architecture) improve policy performance?

This script compares the standard single-pass policy against the LoopedContextPolicy
at varying loop depths (K=1,2,3,5). It runs two phases:

Phase A - Diagnostic (no training): Measures how each architecture uses z by
    checking context sensitivity and hidden state evolution across loops.

Phase B - RL comparison: Trains SAC policies with each architecture and
    evaluates zero-shot adaptation on held-out test contexts.

Usage:
    python test_looped_policy.py                  # Full run (diagnostic + training)
    python test_looped_policy.py --diagnostic     # Diagnostic only (fast, ~1 min)
    python test_looped_policy.py --timesteps 50000  # Shorter training runs
"""

# Fix OpenMP conflict (must be before importing numpy/torch)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from src.data import get_context_distributions
from src.models import LSTMEncoder, ContextEncoder
from src.training.policy_trainer import PolicyTrainer, make_contextual_env

# =============================================================================
# SB3-COMPATIBLE FEATURE EXTRACTORS
# =============================================================================
# SB3 calls forward(observations) with a single augmented [obs | z] tensor.
# These extractors split it back into obs and z before processing.


class BaselineMLP(BaseFeaturesExtractor):
    """Standard MLP baseline — processes [obs | z] as a flat vector."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 256,
                 hidden_dim: int = 256):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class LoopedExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible wrapper for the looped architecture.

    Receives augmented observation [obs | z], splits them, then applies
    K iterations of a shared reasoning block that integrates z into an
    evolving hidden state before producing features.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256,
                 raw_obs_dim: int = 3, hidden_dim: int = 256, num_loops: int = 3):
        super().__init__(observation_space, features_dim)
        self.raw_obs_dim = raw_obs_dim
        context_dim = observation_space.shape[0] - raw_obs_dim
        self.num_loops = num_loops
        self.hidden_dim = hidden_dim

        # Project raw observation into hidden space
        self.obs_proj = nn.Sequential(
            nn.Linear(raw_obs_dim, hidden_dim),
            nn.ReLU(),
        )

        # Shared looped block: [h_k, z] -> residual update
        self.loop_block = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # LayerNorm stabilizes the recurrence
        self.loop_norm = nn.LayerNorm(hidden_dim)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations[:, :self.raw_obs_dim]
        z = observations[:, self.raw_obs_dim:]
        z = nn.functional.normalize(z, p=2, dim=-1)

        h = self.obs_proj(obs)
        for _ in range(self.num_loops):
            h = h + self.loop_block(torch.cat([h, z], dim=-1))
            h = self.loop_norm(h)

        return self.output_head(h)

    def forward_with_intermediates(self, observations: torch.Tensor):
        """Return hidden states at each loop iteration (for diagnostics)."""
        obs = observations[:, :self.raw_obs_dim]
        z = observations[:, self.raw_obs_dim:]
        z = nn.functional.normalize(z, p=2, dim=-1)

        h = self.obs_proj(obs)
        intermediates = [h.detach().clone()]

        for _ in range(self.num_loops):
            h = h + self.loop_block(torch.cat([h, z], dim=-1))
            h = self.loop_norm(h)
            intermediates.append(h.detach().clone())

        return self.output_head(h), intermediates


# =============================================================================
# ENCODER LOADING
# =============================================================================

def load_encoder(device='cpu'):
    """Load Phase 1 pre-trained encoder."""
    encoder_path = Path("experiments/encoder/best_encoder.pt")
    if not encoder_path.exists():
        print(f"ERROR: No encoder at {encoder_path}. Run Phase 1 first.")
        sys.exit(1)

    checkpoint = torch.load(encoder_path, map_location=device)
    cfg = checkpoint.get('config', {})

    # Infer input dim from saved data
    data_path = Path("experiments/data/pendulum_train_segments.pkl")
    if data_path.exists():
        with open(data_path, 'rb') as f:
            segments = pickle.load(f)
        seg = segments[0]
        obs_dim = seg.observations.shape[-1]
        act_dim = seg.actions.shape[-1] if len(seg.actions.shape) > 1 else 1
        input_dim = obs_dim + act_dim
    else:
        obs_dim, act_dim, input_dim = 3, 1, 4

    base = LSTMEncoder(
        input_dim=input_dim,
        hidden_dim=cfg.get('hidden_dim', 256),
        num_layers=cfg.get('num_layers', 2),
        latent_dim=cfg.get('latent_dim', 64),
        dropout=cfg.get('dropout', 0.2),
        bidirectional=cfg.get('bidirectional', True),
    )
    base.load_state_dict(checkpoint['encoder_state_dict'])

    encoder = ContextEncoder(base, freeze_encoder=True)
    encoder.eval()
    return encoder, obs_dim


# =============================================================================
# PHASE A: DIAGNOSTIC — How well does each architecture use z?
# =============================================================================

def run_diagnostic(encoder, obs_dim, device='cpu'):
    """
    Quick diagnostic (no RL training) that measures:
    1. Context sensitivity: How much do features change when z changes?
    2. Hidden state convergence: Do loop iterations converge or keep changing?
    3. Discriminability: Can the architecture distinguish different contexts via z?
    """
    print("\n" + "=" * 80)
    print("PHASE A: DIAGNOSTIC — Context sensitivity analysis")
    print("=" * 80)

    latent_dim = encoder.encoder.latent_dim
    aug_dim = obs_dim + latent_dim

    # Build a fake observation space matching the augmented dim
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                               shape=(aug_dim,), dtype=np.float32)

    # Collect real z embeddings from different contexts
    print("\nCollecting real z embeddings from encoder...")
    data_path = Path("experiments/data/pendulum_train_segments.pkl")
    if data_path.exists():
        with open(data_path, 'rb') as f:
            segments = pickle.load(f)

        # Group segments by context and encode
        from collections import defaultdict
        by_context = defaultdict(list)
        for seg in segments:
            by_context[seg.context_id].append(seg)

        z_by_context = {}
        for ctx_id, segs in sorted(by_context.items())[:5]:  # Use 5 contexts
            trajs = []
            for seg in segs[:10]:
                obs = seg.observations
                act = seg.actions
                if len(act.shape) == 1:
                    act = act[:, None]
                traj = np.concatenate([obs[:len(act)], act], axis=-1)
                trajs.append(traj)
            traj_tensor = torch.FloatTensor(np.stack(trajs)).to(device)
            with torch.no_grad():
                z_embeddings = encoder(traj_tensor)
            z_by_context[ctx_id] = z_embeddings.cpu()
        print(f"  Encoded {len(z_by_context)} contexts, {z_embeddings.shape[-1]}-dim embeddings")
    else:
        # Fallback: use random z vectors
        print("  No segment data found, using synthetic z vectors")
        z_by_context = {
            i: torch.randn(10, latent_dim) for i in range(5)
        }

    # Create a batch of fixed observations with varying z
    fixed_obs = torch.randn(1, obs_dim).expand(10, -1)  # Same obs, repeated

    configs = {
        "Baseline MLP":     {"class": BaselineMLP, "kwargs": {"hidden_dim": 256}},
        "Looped (K=1)":     {"class": LoopedExtractor, "kwargs": {"raw_obs_dim": obs_dim, "hidden_dim": 256, "num_loops": 1}},
        "Looped (K=2)":     {"class": LoopedExtractor, "kwargs": {"raw_obs_dim": obs_dim, "hidden_dim": 256, "num_loops": 2}},
        "Looped (K=3)":     {"class": LoopedExtractor, "kwargs": {"raw_obs_dim": obs_dim, "hidden_dim": 256, "num_loops": 3}},
        "Looped (K=5)":     {"class": LoopedExtractor, "kwargs": {"raw_obs_dim": obs_dim, "hidden_dim": 256, "num_loops": 5}},
    }

    diag_results = {}

    for name, cfg in configs.items():
        print(f"\n--- {name} ---")
        extractor = cfg["class"](obs_space, features_dim=256, **cfg["kwargs"]).to(device)
        extractor.eval()

        n_params = sum(p.numel() for p in extractor.parameters())
        print(f"  Parameters: {n_params:,}")

        # --- Test 1: Context sensitivity ---
        # Feed same obs with z from different contexts, measure feature variance
        all_features = []
        ctx_ids = sorted(z_by_context.keys())

        for ctx_id in ctx_ids:
            z_samples = z_by_context[ctx_id][:10]
            obs_batch = fixed_obs[:len(z_samples)].to(device)
            aug = torch.cat([obs_batch, z_samples.to(device)], dim=-1)
            with torch.no_grad():
                feats = extractor(aug)
            all_features.append(feats.cpu())

        # Mean feature per context
        ctx_means = torch.stack([f.mean(0) for f in all_features])
        # Between-context variance (high = architecture distinguishes contexts well)
        between_var = ctx_means.var(dim=0).mean().item()
        # Within-context variance (low = stable representation)
        within_var = np.mean([f.var(dim=0).mean().item() for f in all_features])
        sensitivity = between_var / (within_var + 1e-8)

        print(f"  Context sensitivity:  between_var={between_var:.4f}  within_var={within_var:.4f}  ratio={sensitivity:.2f}")

        # --- Test 2: z perturbation sensitivity ---
        # How much do features change when z is perturbed by a small amount?
        z_base = z_by_context[ctx_ids[0]][:1].to(device)
        obs_single = fixed_obs[:1].to(device)
        perturbation_magnitudes = [0.01, 0.05, 0.1, 0.5]
        deltas = []
        for eps in perturbation_magnitudes:
            z_pert = z_base + torch.randn_like(z_base) * eps
            aug_base = torch.cat([obs_single, z_base], dim=-1)
            aug_pert = torch.cat([obs_single, z_pert], dim=-1)
            with torch.no_grad():
                f_base = extractor(aug_base)
                f_pert = extractor(aug_pert)
            delta = (f_base - f_pert).norm().item()
            deltas.append(delta)
        print(f"  z-perturbation response: {dict(zip(perturbation_magnitudes, [f'{d:.3f}' for d in deltas]))}")

        # --- Test 3: Hidden state evolution (looped only) ---
        if hasattr(extractor, 'forward_with_intermediates'):
            aug = torch.cat([obs_single, z_base], dim=-1)
            with torch.no_grad():
                _, intermediates = extractor.forward_with_intermediates(aug)

            norms = [h.norm().item() for h in intermediates]
            # How much does h change between iterations?
            step_deltas = []
            for i in range(1, len(intermediates)):
                d = (intermediates[i] - intermediates[i-1]).norm().item()
                step_deltas.append(d)

            print(f"  Hidden state norms per iteration:  {['%.3f' % n for n in norms]}")
            print(f"  Step-to-step change (convergence): {['%.3f' % d for d in step_deltas]}")

        diag_results[name] = {
            "params": n_params,
            "between_context_var": between_var,
            "within_context_var": within_var,
            "sensitivity_ratio": sensitivity,
            "z_perturbation_response": dict(zip(
                [str(m) for m in perturbation_magnitudes],
                deltas
            )),
        }

    # Summary table
    print("\n" + "-" * 80)
    print(f"{'Architecture':<20} {'Params':>8} {'Sensitivity':>12} {'Between Var':>12} {'Within Var':>12}")
    print("-" * 80)
    for name, r in diag_results.items():
        print(f"{name:<20} {r['params']:>8,} {r['sensitivity_ratio']:>12.2f} "
              f"{r['between_context_var']:>12.4f} {r['within_context_var']:>12.4f}")
    print("-" * 80)

    return diag_results


# =============================================================================
# PHASE B: RL TRAINING COMPARISON
# =============================================================================

def run_rl_comparison(encoder, obs_dim, device='cpu', total_timesteps=100_000):
    """Train SAC with each architecture and compare zero-shot adaptation."""
    print("\n" + "=" * 80)
    print("PHASE B: RL TRAINING COMPARISON")
    print("=" * 80)
    print(f"Training each variant for {total_timesteps:,} timesteps\n")

    latent_dim = encoder.encoder.latent_dim
    train_contexts = get_context_distributions("pendulum", split='train')[:10]
    train_g_values = [c['g'] for c in train_contexts]
    min_g, max_g = min(train_g_values), max(train_g_values)

    test_contexts = [
        {"g": 5.25}, {"g": 6.3}, {"g": 7.0},    # Interpolation
        {"g": 7.9},  {"g": 8.5}, {"g": 9.0},     # Interpolation
        {"g": 3.0},  {"g": 4.0},                  # Extrapolation
        {"g": 12.0}, {"g": 15.0},                 # Extrapolation
    ]

    configs = {
        "Baseline MLP": {
            "class": BaselineMLP,
            "kwargs": {"hidden_dim": 256},
        },
        "Looped (K=1)": {
            "class": LoopedExtractor,
            "kwargs": {"raw_obs_dim": obs_dim, "hidden_dim": 256, "num_loops": 1},
        },
        "Looped (K=3)": {
            "class": LoopedExtractor,
            "kwargs": {"raw_obs_dim": obs_dim, "hidden_dim": 256, "num_loops": 3},
        },
        "Looped (K=5)": {
            "class": LoopedExtractor,
            "kwargs": {"raw_obs_dim": obs_dim, "hidden_dim": 256, "num_loops": 5},
        },
    }

    all_results = {}

    for name, cfg in configs.items():
        print(f"\n{'=' * 60}")
        print(f"Training: {name}")
        print(f"{'=' * 60}")

        tag = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        exp_dir = Path(f"experiments/looped_test/{tag}")
        log_dir = exp_dir / "logs"
        ckpt_dir = exp_dir / "policy"
        exp_dir.mkdir(parents=True, exist_ok=True)

        trainer = PolicyTrainer(
            env_name="pendulum",
            encoder=encoder,
            contexts=train_contexts,
            algorithm='sac',
            buffer_length=32,
            device=device,
            log_dir=str(log_dir),
            checkpoint_dir=str(ckpt_dir),
        )

        extractor_cls = cfg["class"]
        extractor_kwargs = cfg["kwargs"]

        policy_kwargs = {
            'features_extractor_class': extractor_cls,
            'features_extractor_kwargs': {**extractor_kwargs, 'features_dim': 256},
            'net_arch': [256, 256],
            'activation_fn': nn.ReLU,
        }

        t0 = time.time()
        trainer.train(
            total_timesteps=total_timesteps,
            eval_contexts=test_contexts[:1],
            eval_freq=max(total_timesteps // 5, 5000),
            n_eval_episodes=3,
            policy_kwargs=policy_kwargs,
        )
        train_time = time.time() - t0

        # Evaluate
        print(f"\nEvaluating {name} on test contexts...")
        results = trainer.evaluate(
            contexts=test_contexts,
            n_episodes=5,
            deterministic=True,
        )

        # Split interpolation / extrapolation
        interp_rewards, extrap_rewards = [], []
        for ctx_key, ctx_result in results.items():
            g = ctx_result['context']['g']
            if min_g <= g <= max_g:
                interp_rewards.append(ctx_result['mean_reward'])
            else:
                extrap_rewards.append(ctx_result['mean_reward'])

        all_rewards = [r['mean_reward'] for r in results.values()]

        all_results[name] = {
            "overall_mean": float(np.mean(all_rewards)),
            "overall_std": float(np.std(all_rewards)),
            "interpolation_mean": float(np.mean(interp_rewards)) if interp_rewards else None,
            "extrapolation_mean": float(np.mean(extrap_rewards)) if extrap_rewards else None,
            "train_time_sec": train_time,
            "per_context": {k: {"mean": float(v["mean_reward"]),
                                "std": float(v["std_reward"]),
                                "g": v["context"]["g"]}
                           for k, v in results.items()},
        }

        print(f"\n  {name} — Overall: {np.mean(all_rewards):.1f} | "
              f"Interp: {np.mean(interp_rewards):.1f} | "
              f"Extrap: {np.mean(extrap_rewards):.1f} | "
              f"Time: {train_time:.0f}s")

    # ==========================================================================
    # Summary table
    # ==========================================================================
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(f"\n{'Architecture':<20} {'Overall':>10} {'Interp':>10} {'Extrap':>10} {'Time (s)':>10}")
    print("-" * 60)
    for name, r in all_results.items():
        interp = f"{r['interpolation_mean']:.1f}" if r['interpolation_mean'] is not None else "N/A"
        extrap = f"{r['extrapolation_mean']:.1f}" if r['extrapolation_mean'] is not None else "N/A"
        print(f"{name:<20} {r['overall_mean']:>10.1f} {interp:>10} {extrap:>10} {r['train_time_sec']:>10.0f}")
    print("-" * 60)

    # Identify best
    best = max(all_results, key=lambda k: all_results[k]['overall_mean'])
    print(f"\nBest architecture: {best} (mean reward {all_results[best]['overall_mean']:.1f})")

    baseline_mean = all_results.get("Baseline MLP", {}).get("overall_mean", None)
    if baseline_mean is not None:
        print("\nImprovement over baseline MLP:")
        for name, r in all_results.items():
            if name == "Baseline MLP":
                continue
            diff = r['overall_mean'] - baseline_mean
            sign = "+" if diff >= 0 else ""
            print(f"  {name}: {sign}{diff:.1f}")

    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test looped z architecture")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Run diagnostic only (no RL training)")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total SAC timesteps per variant (default: 100k)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load encoder
    encoder, obs_dim = load_encoder(device)
    print(f"Encoder loaded — latent_dim={encoder.encoder.latent_dim}, obs_dim={obs_dim}")

    # Phase A: Diagnostic
    diag_results = run_diagnostic(encoder, obs_dim, device)

    if args.diagnostic:
        print("\n(Skipping RL training — run without --diagnostic for full comparison)")
        return

    # Phase B: RL comparison
    rl_results = run_rl_comparison(encoder, obs_dim, device, args.timesteps)

    # Save everything
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out = {
        "diagnostic": diag_results,
        "rl_comparison": rl_results,
        "config": {
            "timesteps": args.timesteps,
            "device": device,
        },
    }
    out_path = results_dir / "looped_policy_test.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
