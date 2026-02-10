#!/usr/bin/env python3
"""
Iterative Co-Training: Recursive Self-Improvement for Context-Conditional RL

This script implements a recursive self-improvement loop where:
  Round 0: Random policy → trajectories → encoder₀ → policy₀
  Round 1: policy₀ → better trajectories → encoder₁ → policy₁
  Round 2: policy₁ → even better trajectories → encoder₂ → policy₂
  ...

Each round, the improved policy generates more informative trajectories,
which train a better encoder, which produces a better policy.

The agent literally improves its own training signal.
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
import pickle
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import (
    TrajectoryCollector, TrajectoryDataset,
    get_context_distributions, make_carl_env
)
from src.data.trajectory_collector import TrajectorySegment
from src.models import LSTMEncoder, SupConLoss, ContextEncoder
from src.training import EncoderTrainer, PolicyTrainer
from src.training.policy_trainer import ContextBufferWrapper

from stable_baselines3 import SAC

# =============================================================================
# CONFIGURATION
# =============================================================================
ENV_NAME = "pendulum"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Iterative settings
NUM_ROUNDS = 4          # Round 0 = random baseline, rounds 1-3 = self-improvement

# Encoder settings
SEGMENT_LENGTH = 32
NUM_SEGMENTS_PER_CONTEXT = 100
LATENT_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True
ENCODER_EPOCHS = 50
ENCODER_LR = 1e-3
ENCODER_BATCH_SIZE = 64
TEMPERATURE = 0.1

# Policy settings
ALGORITHM = "sac"
POLICY_TIMESTEPS = 100_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5

# Paths
BASE_DIR = Path("experiments/iterative")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Contexts
train_contexts = get_context_distributions(ENV_NAME, split='train')[:10]
train_g_values = [c['g'] for c in train_contexts]
min_train_g, max_train_g = min(train_g_values), max(train_g_values)

test_contexts = [
    # INTERPOLATION (within training range 5.0-9.74)
    {"g": 5.25}, {"g": 6.3}, {"g": 7.0},
    {"g": 7.9}, {"g": 8.5}, {"g": 9.0},
    # EXTRAPOLATION (outside training range)
    {"g": 3.0}, {"g": 4.0}, {"g": 12.0}, {"g": 15.0},
]


# =============================================================================
# DATA COLLECTION FUNCTIONS
# =============================================================================

def _extract_obs(obs):
    """Extract observation array from potentially dict observation."""
    if isinstance(obs, dict):
        obs = obs.get('obs', list(obs.values())[0])
    return np.asarray(obs, dtype=np.float32)


def collect_with_random_policy(contexts, segment_length, num_segments_per_context, seed=42):
    """Collect trajectory segments using a random policy (Round 0)."""
    print("  Collecting with RANDOM policy...")
    collector = TrajectoryCollector(
        env_name=ENV_NAME,
        contexts=contexts,
        segment_length=segment_length,
        num_segments_per_context=num_segments_per_context,
        policy='random',
        seed=seed
    )
    return collector.collect(verbose=True)


def collect_with_trained_policy(
    contexts, policy_path, encoder_path,
    segment_length, num_segments_per_context, seed=42
):
    """
    Collect trajectory segments using a trained context-conditional policy.

    The policy expects augmented observations [obs | context_embedding] from the
    ContextBufferWrapper. We run the policy in this wrapped environment but save
    the RAW (obs, action) pairs for the next round of encoder training.
    """
    print(f"  Collecting with TRAINED policy from {policy_path}")

    # Load encoder
    checkpoint = torch.load(encoder_path, map_location=DEVICE)
    base_encoder = LSTMEncoder(
        input_dim=4,  # pendulum: obs=3, action=1
        hidden_dim=checkpoint['config'].get('hidden_dim', HIDDEN_DIM),
        num_layers=checkpoint['config'].get('num_layers', NUM_LAYERS),
        latent_dim=checkpoint['config'].get('latent_dim', LATENT_DIM),
        dropout=checkpoint['config'].get('dropout', DROPOUT),
        bidirectional=checkpoint['config'].get('bidirectional', BIDIRECTIONAL)
    )
    base_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    context_encoder = ContextEncoder(trajectory_encoder=base_encoder, freeze_encoder=True)
    context_encoder.eval()

    # Load trained policy
    policy = SAC.load(str(policy_path), device=DEVICE)

    all_segments = []

    for context_id, context in enumerate(contexts):
        segments_collected = 0

        while segments_collected < num_segments_per_context:
            # Create wrapped environment (policy needs augmented obs)
            base_env = make_carl_env(ENV_NAME, context)
            wrapped_env = ContextBufferWrapper(
                base_env, context_encoder, segment_length, DEVICE
            )

            augmented_obs, _ = wrapped_env.reset()
            done = False
            truncated = False

            # Store RAW observations and actions (not augmented)
            ep_obs = [wrapped_env.obs_buffer[-1]]  # raw obs from buffer
            ep_actions = []
            ep_rewards = []

            while not (done or truncated):
                # Policy acts on augmented obs
                action, _ = policy.predict(augmented_obs, deterministic=False)
                augmented_obs, reward, done, truncated, _ = wrapped_env.step(action)

                # Save the RAW observation (first 3 dims, not the 67-dim augmented)
                raw_obs = wrapped_env.obs_buffer[-1]
                ep_obs.append(raw_obs)
                ep_actions.append(action)
                ep_rewards.append(reward)

            wrapped_env.close()

            # Convert to arrays
            ep_obs = np.array(ep_obs)
            ep_actions = np.array(ep_actions)
            ep_rewards = np.array(ep_rewards)

            # Split into segments
            episode_length = len(ep_actions)
            if episode_length < segment_length:
                continue

            num_segs = episode_length // segment_length
            for i in range(num_segs):
                if segments_collected >= num_segments_per_context:
                    break
                start = i * segment_length
                end = start + segment_length
                segment = TrajectorySegment(
                    observations=ep_obs[start:end],
                    actions=ep_actions[start:end],
                    rewards=ep_rewards[start:end],
                    context_id=context_id,
                    context_params=context
                )
                all_segments.append(segment)
                segments_collected += 1

        print(f"    Context {context_id} (g={context['g']:.2f}): {segments_collected} segments")

    return all_segments


# =============================================================================
# ENCODER TRAINING FUNCTION
# =============================================================================

def train_encoder(segments, round_dir, round_num):
    """Train a fresh encoder on the given trajectory segments."""
    print(f"\n  Training encoder (round {round_num})...")

    # Infer dimensions
    first_seg = segments[0]
    obs_dim = first_seg.observations.shape[-1]
    action_dim = first_seg.actions.shape[-1] if len(first_seg.actions.shape) > 1 else 1
    input_dim = obs_dim + action_dim

    # Create dataset
    dataset = TrajectoryDataset(segments, augmentation="noise")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42 + round_num)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=ENCODER_BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=ENCODER_BATCH_SIZE, shuffle=False
    )

    # Create fresh encoder
    encoder = LSTMEncoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL
    )

    loss_fn = SupConLoss(temperature=TEMPERATURE)

    trainer = EncoderTrainer(
        encoder=encoder,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=ENCODER_LR,
        weight_decay=1e-4,
        device=DEVICE,
        log_dir=None,
        checkpoint_dir=None
    )

    trainer.train(num_epochs=ENCODER_EPOCHS, eval_every=10)

    # Compute embedding quality metrics
    metrics = compute_embedding_metrics(encoder, segments)

    # Save encoder
    encoder_path = round_dir / "encoder.pt"
    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'epoch': ENCODER_EPOCHS,
        'config': {
            'architecture': 'lstm',
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'latent_dim': LATENT_DIM,
            'dropout': DROPOUT,
            'bidirectional': BIDIRECTIONAL
        }
    }
    torch.save(checkpoint, encoder_path)
    print(f"  Encoder saved to {encoder_path}")

    return encoder, encoder_path, metrics


def compute_embedding_metrics(encoder, segments):
    """Compute embedding quality metrics (separation score, etc.)."""
    encoder.eval()

    # Group segments by context
    context_embeddings = {}
    for seg in segments:
        ctx_id = seg.context_id
        obs = seg.observations
        actions = seg.actions
        if len(actions.shape) == 1:
            actions = actions[:, None]
        trajectory = np.concatenate([obs, actions], axis=-1)
        traj_tensor = torch.FloatTensor(trajectory).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = encoder(traj_tensor).squeeze(0).cpu().numpy()

        if ctx_id not in context_embeddings:
            context_embeddings[ctx_id] = []
        context_embeddings[ctx_id].append(emb)

    # Compute intra and inter context similarity
    from scipy.spatial.distance import cosine

    intra_sims = []
    for ctx_id, embs in context_embeddings.items():
        for i in range(len(embs)):
            for j in range(i + 1, min(i + 10, len(embs))):  # sample pairs
                sim = 1 - cosine(embs[i], embs[j])
                intra_sims.append(sim)

    inter_sims = []
    ctx_ids = list(context_embeddings.keys())
    for i in range(len(ctx_ids)):
        for j in range(i + 1, len(ctx_ids)):
            # Compare a few embeddings from each context
            for k in range(min(5, len(context_embeddings[ctx_ids[i]]))):
                for l in range(min(5, len(context_embeddings[ctx_ids[j]]))):
                    sim = 1 - cosine(
                        context_embeddings[ctx_ids[i]][k],
                        context_embeddings[ctx_ids[j]][l]
                    )
                    inter_sims.append(sim)

    intra_mean = np.mean(intra_sims) if intra_sims else 0
    inter_mean = np.mean(inter_sims) if inter_sims else 0
    separation = intra_mean - inter_mean

    return {
        'intra_similarity': float(intra_mean),
        'inter_similarity': float(inter_mean),
        'separation_score': float(separation)
    }


# =============================================================================
# POLICY TRAINING FUNCTION
# =============================================================================

def train_policy(encoder_path, round_dir, round_num):
    """Train a context-conditional policy using the given encoder."""
    print(f"\n  Training policy (round {round_num})...")

    # Load encoder
    checkpoint = torch.load(encoder_path, map_location=DEVICE)
    base_encoder = LSTMEncoder(
        input_dim=4,
        hidden_dim=checkpoint['config'].get('hidden_dim', HIDDEN_DIM),
        num_layers=checkpoint['config'].get('num_layers', NUM_LAYERS),
        latent_dim=checkpoint['config'].get('latent_dim', LATENT_DIM),
        dropout=checkpoint['config'].get('dropout', DROPOUT),
        bidirectional=checkpoint['config'].get('bidirectional', BIDIRECTIONAL)
    )
    base_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    context_encoder = ContextEncoder(trajectory_encoder=base_encoder, freeze_encoder=True)
    context_encoder.eval()

    # Create policy trainer
    policy_dir = round_dir / "policy"
    log_dir = round_dir / "logs"
    policy_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    trainer = PolicyTrainer(
        env_name=ENV_NAME,
        encoder=context_encoder,
        contexts=train_contexts,
        algorithm=ALGORITHM,
        buffer_length=SEGMENT_LENGTH,
        device=DEVICE,
        log_dir=str(log_dir),
        checkpoint_dir=str(policy_dir)
    )

    policy_kwargs = {
        'net_arch': [256, 256],
        'activation_fn': torch.nn.ReLU
    }

    trainer.train(
        total_timesteps=POLICY_TIMESTEPS,
        eval_contexts=test_contexts[:1],
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        policy_kwargs=policy_kwargs
    )

    # Evaluate on all test contexts
    print(f"\n  Evaluating policy (round {round_num}) on test contexts...")
    results = trainer.evaluate(
        contexts=test_contexts,
        n_episodes=N_EVAL_EPISODES,
        deterministic=True
    )

    # Compute interpolation/extrapolation means
    interp_rewards, extrap_rewards = [], []
    for ctx_id, ctx_result in results.items():
        ctx = ctx_result['context']
        r = ctx_result['mean_reward']
        if min_train_g <= ctx['g'] <= max_train_g:
            interp_rewards.append(r)
        else:
            extrap_rewards.append(r)

    eval_metrics = {
        'interp_mean': float(np.mean(interp_rewards)) if interp_rewards else float('nan'),
        'extrap_mean': float(np.mean(extrap_rewards)) if extrap_rewards else float('nan'),
        'overall_mean': float(np.mean([r['mean_reward'] for r in results.values()])),
        'per_context': {k: {'mean_reward': float(v['mean_reward']),
                            'std_reward': float(v['std_reward']),
                            'context': v['context']}
                        for k, v in results.items()}
    }

    policy_path = policy_dir / "final_policy.zip"
    return policy_path, eval_metrics


# =============================================================================
# MAIN ITERATIVE LOOP
# =============================================================================

def main():
    print("=" * 80)
    print("ITERATIVE CO-TRAINING: RECURSIVE SELF-IMPROVEMENT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Rounds:              {NUM_ROUNDS}")
    print(f"  Encoder epochs/round: {ENCODER_EPOCHS}")
    print(f"  Policy steps/round:  {POLICY_TIMESTEPS:,}")
    print(f"  Segments/context:    {NUM_SEGMENTS_PER_CONTEXT}")
    print(f"  Training contexts:   {len(train_contexts)}")
    print(f"  Test contexts:       {len(test_contexts)}")
    print(f"  Device:              {DEVICE}")

    all_round_results = []
    current_policy_path = None
    current_encoder_path = None

    for round_num in range(NUM_ROUNDS):
        round_start = time.time()
        round_dir = BASE_DIR / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 80}")
        if round_num == 0:
            print(f"ROUND {round_num}: BOOTSTRAP (Random Policy)")
        else:
            print(f"ROUND {round_num}: SELF-IMPROVEMENT (Using policy from round {round_num - 1})")
        print(f"{'=' * 80}")

        # ----- Step 1: Collect Trajectories -----
        print(f"\n--- Step 1: Data Collection ---")
        if round_num == 0:
            segments = collect_with_random_policy(
                train_contexts, SEGMENT_LENGTH, NUM_SEGMENTS_PER_CONTEXT,
                seed=42
            )
        else:
            segments = collect_with_trained_policy(
                train_contexts, current_policy_path, current_encoder_path,
                SEGMENT_LENGTH, NUM_SEGMENTS_PER_CONTEXT,
                seed=42 + round_num
            )

        # Save segments
        data_path = round_dir / "segments.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(segments, f)
        print(f"  Saved {len(segments)} segments to {data_path}")

        # Compute trajectory statistics
        all_rewards = [seg.rewards.sum() for seg in segments]
        traj_stats = {
            'mean_segment_reward': float(np.mean(all_rewards)),
            'std_segment_reward': float(np.std(all_rewards)),
            'num_segments': len(segments)
        }
        print(f"  Trajectory stats: mean_reward={traj_stats['mean_segment_reward']:.2f}, "
              f"std={traj_stats['std_segment_reward']:.2f}")

        # ----- Step 2: Train Encoder -----
        print(f"\n--- Step 2: Encoder Training ---")
        encoder, encoder_path, encoder_metrics = train_encoder(segments, round_dir, round_num)
        current_encoder_path = encoder_path
        print(f"  Embedding metrics: separation={encoder_metrics['separation_score']:.4f} "
              f"(intra={encoder_metrics['intra_similarity']:.4f}, "
              f"inter={encoder_metrics['inter_similarity']:.4f})")

        # ----- Step 3: Train Policy -----
        print(f"\n--- Step 3: Policy Training ---")
        policy_path, eval_metrics = train_policy(encoder_path, round_dir, round_num)
        current_policy_path = policy_path
        print(f"\n  Round {round_num} Results:")
        print(f"    Interpolation mean: {eval_metrics['interp_mean']:.2f}")
        print(f"    Extrapolation mean: {eval_metrics['extrap_mean']:.2f}")
        print(f"    Overall mean:       {eval_metrics['overall_mean']:.2f}")

        # ----- Save Round Results -----
        round_elapsed = time.time() - round_start
        round_result = {
            'round': round_num,
            'data_source': 'random' if round_num == 0 else f'policy_round_{round_num - 1}',
            'trajectory_stats': traj_stats,
            'encoder_metrics': encoder_metrics,
            'eval_metrics': eval_metrics,
            'elapsed_seconds': round_elapsed
        }
        all_round_results.append(round_result)

        # Save per-round results
        with open(round_dir / "results.json", 'w') as f:
            json.dump(round_result, f, indent=2, default=str)

        print(f"\n  Round {round_num} completed in {round_elapsed/60:.1f} minutes")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("ITERATIVE CO-TRAINING RESULTS")
    print(f"{'=' * 80}")

    # Comparison table
    print(f"\n{'Round':<8} {'Data Source':<25} {'Separation':<12} "
          f"{'Interp Mean':<13} {'Extrap Mean':<13} {'Overall':<10}")
    print("-" * 81)

    for r in all_round_results:
        print(f"{r['round']:<8} {r['data_source']:<25} "
              f"{r['encoder_metrics']['separation_score']:<12.4f} "
              f"{r['eval_metrics']['interp_mean']:<13.2f} "
              f"{r['eval_metrics']['extrap_mean']:<13.2f} "
              f"{r['eval_metrics']['overall_mean']:<10.2f}")

    # Improvement analysis
    if len(all_round_results) >= 2:
        r0 = all_round_results[0]
        r_last = all_round_results[-1]

        interp_improvement = r_last['eval_metrics']['interp_mean'] - r0['eval_metrics']['interp_mean']
        extrap_improvement = r_last['eval_metrics']['extrap_mean'] - r0['eval_metrics']['extrap_mean']
        sep_improvement = r_last['encoder_metrics']['separation_score'] - r0['encoder_metrics']['separation_score']

        print(f"\nImprovement (Round {len(all_round_results)-1} vs Round 0):")
        print(f"  Interpolation: {interp_improvement:+.2f} "
              f"({'better' if interp_improvement > 0 else 'worse'})")
        print(f"  Extrapolation: {extrap_improvement:+.2f} "
              f"({'better' if extrap_improvement > 0 else 'worse'})")
        print(f"  Separation:    {sep_improvement:+.4f} "
              f"({'better' if sep_improvement > 0 else 'worse'})")

    # Save comprehensive results
    results_path = RESULTS_DIR / "iterative_cotraining_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'num_rounds': NUM_ROUNDS,
                'encoder_epochs': ENCODER_EPOCHS,
                'policy_timesteps': POLICY_TIMESTEPS,
                'segments_per_context': NUM_SEGMENTS_PER_CONTEXT,
                'segment_length': SEGMENT_LENGTH,
                'train_contexts': len(train_contexts),
                'test_contexts': len(test_contexts),
                'train_range': [float(min_train_g), float(max_train_g)]
            },
            'rounds': all_round_results
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")
    print(f"Round artifacts in: {BASE_DIR}/round_*/")

    print(f"\n{'=' * 80}")
    print("ITERATIVE CO-TRAINING COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
