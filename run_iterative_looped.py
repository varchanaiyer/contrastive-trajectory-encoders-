#!/usr/bin/env python3
"""
Iterative Co-Training with Looped Architecture

Tests whether recursive self-improvement produces better z embeddings when
the policy uses the LoopedExtractor (iterative z refinement) instead of
a standard MLP.

Each round:
  1. Collect trajectories (random for round 0, trained policy otherwise)
  2. Train a fresh encoder on the collected data
  3. Measure z quality: separation score, intra/inter similarity
  4. Train a policy using the LoopedExtractor with the new encoder
  5. Evaluate on held-out test contexts

After all rounds, output a comparison table showing whether z improved.

Usage:
    python run_iterative_looped.py                     # Full run (3 rounds)
    python run_iterative_looped.py --rounds 2           # Fewer rounds
    python run_iterative_looped.py --policy-steps 25000  # Faster training
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import time
import pickle
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from src.data import (
    TrajectoryCollector, TrajectoryDataset,
    get_context_distributions, make_carl_env
)
from src.data.trajectory_collector import TrajectorySegment
from src.models import LSTMEncoder, SupConLoss, ContextEncoder
from src.training import EncoderTrainer, PolicyTrainer
from src.training.policy_trainer import ContextBufferWrapper


# =============================================================================
# LOOPED FEATURE EXTRACTOR (SB3-compatible)
# =============================================================================

class LoopedExtractor(BaseFeaturesExtractor):
    """SB3-compatible looped architecture that splits [obs|z] and iterates."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 256,
                 raw_obs_dim: int = 3, hidden_dim: int = 256, num_loops: int = 3):
        super().__init__(observation_space, features_dim)
        self.raw_obs_dim = raw_obs_dim
        context_dim = observation_space.shape[0] - raw_obs_dim
        self.num_loops = num_loops

        self.obs_proj = nn.Sequential(nn.Linear(raw_obs_dim, hidden_dim), nn.ReLU())
        self.loop_block = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.loop_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(nn.Linear(hidden_dim, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations[:, :self.raw_obs_dim]
        z = observations[:, self.raw_obs_dim:]
        z = nn.functional.normalize(z, p=2, dim=-1)

        h = self.obs_proj(obs)
        for _ in range(self.num_loops):
            h = h + self.loop_block(torch.cat([h, z], dim=-1))
            h = self.loop_norm(h)
        return self.output_head(h)


# =============================================================================
# CONFIGURATION
# =============================================================================

ENV_NAME = "pendulum"
SEGMENT_LENGTH = 32
LATENT_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True
ENCODER_EPOCHS = 50
ENCODER_LR = 1e-3
ENCODER_BATCH_SIZE = 64
TEMPERATURE = 0.1


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _extract_obs(obs):
    if isinstance(obs, dict):
        obs = obs.get('obs', list(obs.values())[0])
    return np.asarray(obs, dtype=np.float32)


def collect_with_random_policy(contexts, num_segments_per_context, seed=42):
    collector = TrajectoryCollector(
        env_name=ENV_NAME, contexts=contexts,
        segment_length=SEGMENT_LENGTH,
        num_segments_per_context=num_segments_per_context,
        policy='random', seed=seed
    )
    return collector.collect(verbose=True)


def collect_with_trained_policy(contexts, policy_path, encoder_path,
                                num_segments_per_context, device, seed=42):
    checkpoint = torch.load(encoder_path, map_location=device)
    base_encoder = LSTMEncoder(
        input_dim=4,
        hidden_dim=checkpoint['config'].get('hidden_dim', HIDDEN_DIM),
        num_layers=checkpoint['config'].get('num_layers', NUM_LAYERS),
        latent_dim=checkpoint['config'].get('latent_dim', LATENT_DIM),
        dropout=checkpoint['config'].get('dropout', DROPOUT),
        bidirectional=checkpoint['config'].get('bidirectional', BIDIRECTIONAL),
    )
    base_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    ctx_encoder = ContextEncoder(base_encoder, freeze_encoder=True)
    ctx_encoder.eval()

    policy = SAC.load(str(policy_path), device=device)
    all_segments = []

    for ctx_id, context in enumerate(contexts):
        collected = 0
        while collected < num_segments_per_context:
            base_env = make_carl_env(ENV_NAME, context)
            wrapped = ContextBufferWrapper(base_env, ctx_encoder, SEGMENT_LENGTH, device)
            aug_obs, _ = wrapped.reset()
            done = truncated = False

            ep_obs = [wrapped.obs_buffer[-1]]
            ep_actions, ep_rewards = [], []

            while not (done or truncated):
                action, _ = policy.predict(aug_obs, deterministic=False)
                aug_obs, reward, done, truncated, _ = wrapped.step(action)
                ep_obs.append(wrapped.obs_buffer[-1])
                ep_actions.append(action)
                ep_rewards.append(reward)
            wrapped.close()

            ep_obs = np.array(ep_obs)
            ep_actions = np.array(ep_actions)
            ep_rewards = np.array(ep_rewards)

            n_segs = len(ep_actions) // SEGMENT_LENGTH
            for i in range(n_segs):
                if collected >= num_segments_per_context:
                    break
                s, e = i * SEGMENT_LENGTH, (i + 1) * SEGMENT_LENGTH
                all_segments.append(TrajectorySegment(
                    observations=ep_obs[s:e], actions=ep_actions[s:e],
                    rewards=ep_rewards[s:e], context_id=ctx_id,
                    context_params=context
                ))
                collected += 1
        print(f"    Context {ctx_id} (g={context['g']:.2f}): {collected} segments")

    return all_segments


def compute_embedding_metrics(encoder, segments, device):
    """Compute z quality: separation score, intra/inter similarity."""
    encoder.eval()
    from scipy.spatial.distance import cosine

    ctx_embs = {}
    for seg in segments:
        obs = seg.observations
        act = seg.actions
        if len(act.shape) == 1:
            act = act[:, None]
        traj = np.concatenate([obs[:len(act)], act], axis=-1)
        t = torch.FloatTensor(traj).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = encoder(t).squeeze(0).cpu().numpy()
        ctx_embs.setdefault(seg.context_id, []).append(emb)

    # Intra-context similarity
    intra = []
    for embs in ctx_embs.values():
        for i in range(len(embs)):
            for j in range(i + 1, min(i + 10, len(embs))):
                intra.append(1 - cosine(embs[i], embs[j]))

    # Inter-context similarity
    inter = []
    ids = list(ctx_embs.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            for a in ctx_embs[ids[i]][:5]:
                for b in ctx_embs[ids[j]][:5]:
                    inter.append(1 - cosine(a, b))

    intra_mean = float(np.mean(intra)) if intra else 0
    inter_mean = float(np.mean(inter)) if inter else 0
    return {
        'intra_similarity': intra_mean,
        'inter_similarity': inter_mean,
        'separation_score': intra_mean - inter_mean,
    }


def train_encoder_round(segments, round_dir, round_num, device):
    """Train a fresh encoder on segments."""
    seg = segments[0]
    obs_dim = seg.observations.shape[-1]
    act_dim = seg.actions.shape[-1] if len(seg.actions.shape) > 1 else 1
    input_dim = obs_dim + act_dim

    dataset = TrajectoryDataset(segments, augmentation="noise")
    n_train = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(42 + round_num)
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=ENCODER_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=ENCODER_BATCH_SIZE, shuffle=False)

    encoder = LSTMEncoder(
        input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        latent_dim=LATENT_DIM, dropout=DROPOUT, bidirectional=BIDIRECTIONAL
    )
    trainer = EncoderTrainer(
        encoder=encoder, loss_fn=SupConLoss(temperature=TEMPERATURE),
        train_loader=train_loader, val_loader=val_loader,
        learning_rate=ENCODER_LR, weight_decay=1e-4,
        device=device, log_dir=None, checkpoint_dir=None
    )
    trainer.train(num_epochs=ENCODER_EPOCHS, eval_every=10)

    metrics = compute_embedding_metrics(encoder, segments, device)

    enc_path = round_dir / "encoder.pt"
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'epoch': ENCODER_EPOCHS,
        'config': {
            'architecture': 'lstm', 'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS, 'latent_dim': LATENT_DIM,
            'dropout': DROPOUT, 'bidirectional': BIDIRECTIONAL
        }
    }, enc_path)

    return encoder, enc_path, metrics


def train_policy_looped(encoder_path, round_dir, round_num, train_contexts,
                        test_contexts, policy_steps, num_loops, device):
    """Train a policy with the LoopedExtractor."""
    checkpoint = torch.load(encoder_path, map_location=device)
    base_encoder = LSTMEncoder(
        input_dim=4,
        hidden_dim=checkpoint['config'].get('hidden_dim', HIDDEN_DIM),
        num_layers=checkpoint['config'].get('num_layers', NUM_LAYERS),
        latent_dim=checkpoint['config'].get('latent_dim', LATENT_DIM),
        dropout=checkpoint['config'].get('dropout', DROPOUT),
        bidirectional=checkpoint['config'].get('bidirectional', BIDIRECTIONAL),
    )
    base_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    ctx_encoder = ContextEncoder(base_encoder, freeze_encoder=True)
    ctx_encoder.eval()

    policy_dir = round_dir / "policy"
    log_dir = round_dir / "logs"
    policy_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    trainer = PolicyTrainer(
        env_name=ENV_NAME, encoder=ctx_encoder, contexts=train_contexts,
        algorithm='sac', buffer_length=SEGMENT_LENGTH, device=device,
        log_dir=str(log_dir), checkpoint_dir=str(policy_dir),
    )

    policy_kwargs = {
        'features_extractor_class': LoopedExtractor,
        'features_extractor_kwargs': {
            'features_dim': 256, 'raw_obs_dim': 3,
            'hidden_dim': 256, 'num_loops': num_loops,
        },
        'net_arch': [256, 256],
        'activation_fn': nn.ReLU,
    }

    trainer.train(
        total_timesteps=policy_steps,
        eval_contexts=test_contexts[:1],
        eval_freq=max(policy_steps // 5, 5000),
        n_eval_episodes=3,
        policy_kwargs=policy_kwargs,
    )

    results = trainer.evaluate(contexts=test_contexts, n_episodes=5, deterministic=True)

    train_g = [c['g'] for c in train_contexts]
    min_g, max_g = min(train_g), max(train_g)
    interp, extrap = [], []
    for r in results.values():
        g = r['context']['g']
        (interp if min_g <= g <= max_g else extrap).append(r['mean_reward'])

    all_rewards = [r['mean_reward'] for r in results.values()]
    eval_metrics = {
        'overall_mean': float(np.mean(all_rewards)),
        'interp_mean': float(np.mean(interp)) if interp else float('nan'),
        'extrap_mean': float(np.mean(extrap)) if extrap else float('nan'),
    }

    policy_path = policy_dir / "final_policy.zip"
    return policy_path, eval_metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Iterative co-training with looped architecture")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--policy-steps", type=int, default=50_000)
    parser.add_argument("--segments-per-context", type=int, default=50)
    parser.add_argument("--num-loops", type=int, default=3, help="Loop iterations K in LoopedExtractor")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    train_contexts = get_context_distributions(ENV_NAME, split='train')[:10]
    test_contexts = [
        {"g": 5.25}, {"g": 6.3}, {"g": 7.0}, {"g": 7.9}, {"g": 8.5}, {"g": 9.0},
        {"g": 3.0}, {"g": 4.0}, {"g": 12.0}, {"g": 15.0},
    ]

    base_dir = Path("experiments/iterative_looped")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ITERATIVE CO-TRAINING WITH LOOPED ARCHITECTURE")
    print("=" * 80)
    print(f"  Rounds:          {args.rounds}")
    print(f"  Policy steps:    {args.policy_steps:,}")
    print(f"  Segments/ctx:    {args.segments_per_context}")
    print(f"  Loop depth K:    {args.num_loops}")
    print(f"  Device:          {device}")

    all_results = []
    current_policy_path = None
    current_encoder_path = None

    for rnd in range(args.rounds):
        t0 = time.time()
        round_dir = base_dir / f"round_{rnd}"
        round_dir.mkdir(parents=True, exist_ok=True)

        header = "BOOTSTRAP (Random)" if rnd == 0 else f"SELF-IMPROVEMENT (policy from round {rnd-1})"
        print(f"\n{'=' * 80}")
        print(f"ROUND {rnd}: {header}")
        print(f"{'=' * 80}")

        # Step 1: Collect
        print(f"\n--- Step 1: Data Collection ---")
        if rnd == 0:
            segments = collect_with_random_policy(
                train_contexts, args.segments_per_context, seed=42
            )
        else:
            segments = collect_with_trained_policy(
                train_contexts, current_policy_path, current_encoder_path,
                args.segments_per_context, device, seed=42 + rnd
            )

        with open(round_dir / "segments.pkl", 'wb') as f:
            pickle.dump(segments, f)

        seg_rewards = [seg.rewards.sum() for seg in segments]
        traj_stats = {
            'mean_segment_reward': float(np.mean(seg_rewards)),
            'std_segment_reward': float(np.std(seg_rewards)),
            'num_segments': len(segments),
        }
        print(f"  Segments: {len(segments)}, mean_reward: {traj_stats['mean_segment_reward']:.2f}")

        # Step 2: Train encoder
        print(f"\n--- Step 2: Encoder Training ---")
        encoder, enc_path, enc_metrics = train_encoder_round(segments, round_dir, rnd, device)
        current_encoder_path = enc_path
        print(f"  z quality — separation: {enc_metrics['separation_score']:.4f} "
              f"(intra: {enc_metrics['intra_similarity']:.4f}, "
              f"inter: {enc_metrics['inter_similarity']:.4f})")

        # Step 3: Train looped policy
        print(f"\n--- Step 3: Looped Policy Training (K={args.num_loops}) ---")
        policy_path, eval_metrics = train_policy_looped(
            enc_path, round_dir, rnd, train_contexts, test_contexts,
            args.policy_steps, args.num_loops, device
        )
        current_policy_path = policy_path

        elapsed = time.time() - t0
        round_result = {
            'round': rnd,
            'data_source': 'random' if rnd == 0 else f'looped_policy_round_{rnd-1}',
            'trajectory_stats': traj_stats,
            'encoder_metrics': enc_metrics,
            'eval_metrics': eval_metrics,
            'elapsed_sec': elapsed,
        }
        all_results.append(round_result)

        with open(round_dir / "results.json", 'w') as f:
            json.dump(round_result, f, indent=2, default=str)

        print(f"\n  Round {rnd} — Overall: {eval_metrics['overall_mean']:.1f} | "
              f"Interp: {eval_metrics['interp_mean']:.1f} | "
              f"Extrap: {eval_metrics['extrap_mean']:.1f} | "
              f"z-sep: {enc_metrics['separation_score']:.4f} | "
              f"Time: {elapsed:.0f}s")

    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("RESULTS: z QUALITY AND POLICY PERFORMANCE ACROSS ROUNDS")
    print(f"{'=' * 80}")

    print(f"\n{'Round':<7} {'Source':<22} {'z-Sep':>8} {'Intra':>8} {'Inter':>8} "
          f"{'Interp':>9} {'Extrap':>9} {'Overall':>9}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['round']:<7} {r['data_source']:<22} "
              f"{r['encoder_metrics']['separation_score']:>8.4f} "
              f"{r['encoder_metrics']['intra_similarity']:>8.4f} "
              f"{r['encoder_metrics']['inter_similarity']:>8.4f} "
              f"{r['eval_metrics']['interp_mean']:>9.1f} "
              f"{r['eval_metrics']['extrap_mean']:>9.1f} "
              f"{r['eval_metrics']['overall_mean']:>9.1f}")
    print("-" * 90)

    # Improvement analysis
    if len(all_results) >= 2:
        r0, rN = all_results[0], all_results[-1]
        sep_d = rN['encoder_metrics']['separation_score'] - r0['encoder_metrics']['separation_score']
        interp_d = rN['eval_metrics']['interp_mean'] - r0['eval_metrics']['interp_mean']
        extrap_d = rN['eval_metrics']['extrap_mean'] - r0['eval_metrics']['extrap_mean']
        overall_d = rN['eval_metrics']['overall_mean'] - r0['eval_metrics']['overall_mean']

        print(f"\nRound {len(all_results)-1} vs Round 0:")
        print(f"  z separation:  {sep_d:+.4f} ({'improved' if sep_d > 0 else 'degraded'})")
        print(f"  Interpolation: {interp_d:+.1f} ({'improved' if interp_d > 0 else 'degraded'})")
        print(f"  Extrapolation: {extrap_d:+.1f} ({'improved' if extrap_d > 0 else 'degraded'})")
        print(f"  Overall:       {overall_d:+.1f} ({'improved' if overall_d > 0 else 'degraded'})")

        # Per-round z trajectory
        print(f"\n  z separation trajectory: ", end="")
        for r in all_results:
            print(f"  round {r['round']}: {r['encoder_metrics']['separation_score']:.4f}", end="")
        print()

    # Save
    out_path = results_dir / "iterative_looped_results.json"
    with open(out_path, 'w') as f:
        json.dump({
            'config': {
                'num_rounds': args.rounds, 'policy_steps': args.policy_steps,
                'segments_per_context': args.segments_per_context,
                'num_loops': args.num_loops, 'device': device,
            },
            'rounds': all_results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print(f"{'=' * 80}")
    print("ITERATIVE LOOPED CO-TRAINING COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
