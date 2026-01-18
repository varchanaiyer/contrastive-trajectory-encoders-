"""Evaluation metrics and baseline comparisons"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from ..data.carl_envs import make_carl_env


def compute_zero_shot_metrics(
    policy,
    encoder,
    contexts: List[Dict[str, float]],
    env_name: str,
    n_episodes: int = 10,
    buffer_length: int = 32,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate zero-shot performance on new contexts.

    Args:
        policy: Trained policy
        encoder: Trajectory encoder
        contexts: List of test contexts (OOD)
        env_name: Name of environment
        n_episodes: Number of episodes per context
        buffer_length: Buffer length for context inference
        device: Device for inference

    Returns:
        Dictionary with metrics per context
    """
    from ..training.policy_trainer import make_contextual_env

    results = {}
    all_rewards = []

    for ctx_id, context in enumerate(contexts):
        env = make_contextual_env(env_name, context, encoder, buffer_length, device)

        episode_rewards = []
        episode_lengths = []
        context_inference_times = []

        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            while not (done or truncated):
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        env.close()

        # Compute statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        median_reward = np.median(episode_rewards)
        mean_length = np.mean(episode_lengths)

        results[f'context_{ctx_id}'] = {
            'context': context,
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'median_reward': float(median_reward),
            'mean_length': float(mean_length),
            'all_rewards': episode_rewards
        }

        all_rewards.extend(episode_rewards)

        print(f"Context {ctx_id} {context}: {mean_reward:.2f} ± {std_reward:.2f}")

    # Overall statistics
    results['overall'] = {
        'mean_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'median_reward': float(np.median(all_rewards))
    }

    return results


def compare_baselines(
    results_dict: Dict[str, Dict],
    save_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple baseline methods.

    Args:
        results_dict: Dictionary mapping method name to results
        save_path: Path to save comparison results

    Returns:
        Comparison summary
    """
    comparison = {}

    for method_name, method_results in results_dict.items():
        all_rewards = []

        # Collect all rewards across contexts
        for key, value in method_results.items():
            if key.startswith('context_'):
                all_rewards.extend(value['all_rewards'])

        comparison[method_name] = {
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'median_reward': float(np.median(all_rewards)),
            'min_reward': float(np.min(all_rewards)),
            'max_reward': float(np.max(all_rewards)),
            'num_episodes': len(all_rewards)
        }

    # Print comparison table
    print("\n" + "="*80)
    print("BASELINE COMPARISON")
    print("="*80)
    print(f"{'Method':<25} {'Mean':<12} {'Std':<12} {'Median':<12}")
    print("-"*80)

    for method_name, stats in comparison.items():
        print(f"{method_name:<25} {stats['mean_reward']:<12.2f} "
              f"{stats['std_reward']:<12.2f} {stats['median_reward']:<12.2f}")

    print("="*80)

    # Save to file
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"\nSaved comparison to {save_path}")

    return comparison


def compute_embedding_quality(
    encoder: torch.nn.Module,
    segments: List,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Compute embedding quality metrics.

    Args:
        encoder: Trained encoder
        segments: List of trajectory segments
        device: Device for inference

    Returns:
        Dictionary with quality metrics
    """
    encoder.eval()
    encoder = encoder.to(device)

    embeddings = []
    context_ids = []

    with torch.no_grad():
        for seg in segments:
            # Prepare trajectory
            obs = seg.observations[:-1]
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
            similarity_matrix = np.dot(ctx_embeddings, ctx_embeddings.T)
            mask = ~np.eye(len(ctx_embeddings), dtype=bool)
            intra_sim = similarity_matrix[mask].mean()
            intra_similarities.append(intra_sim)

    avg_intra_similarity = np.mean(intra_similarities) if intra_similarities else 0.0

    # 2. Inter-context similarity (should be low)
    inter_similarities = []
    for i, ctx1 in enumerate(unique_contexts):
        for ctx2 in unique_contexts[i+1:]:
            ctx1_mask = context_ids == ctx1
            ctx2_mask = context_ids == ctx2

            ctx1_embeddings = embeddings[ctx1_mask]
            ctx2_embeddings = embeddings[ctx2_mask]

            similarity = np.dot(ctx1_embeddings, ctx2_embeddings.T).mean()
            inter_similarities.append(similarity)

    avg_inter_similarity = np.mean(inter_similarities) if inter_similarities else 0.0

    # 3. Silhouette score (measures clustering quality)
    from sklearn.metrics import silhouette_score

    try:
        silhouette = silhouette_score(embeddings, context_ids)
    except:
        silhouette = 0.0

    # 4. Davies-Bouldin Index (lower is better)
    from sklearn.metrics import davies_bouldin_score

    try:
        davies_bouldin = davies_bouldin_score(embeddings, context_ids)
    except:
        davies_bouldin = 0.0

    metrics = {
        'intra_similarity': float(avg_intra_similarity),
        'inter_similarity': float(avg_inter_similarity),
        'separation': float(avg_intra_similarity - avg_inter_similarity),
        'silhouette_score': float(silhouette),
        'davies_bouldin_index': float(davies_bouldin)
    }

    print("\nEmbedding Quality Metrics:")
    print(f"  Intra-context similarity: {metrics['intra_similarity']:.4f}")
    print(f"  Inter-context similarity: {metrics['inter_similarity']:.4f}")
    print(f"  Separation: {metrics['separation']:.4f}")
    print(f"  Silhouette score: {metrics['silhouette_score']:.4f}")
    print(f"  Davies-Bouldin index: {metrics['davies_bouldin_index']:.4f}")

    return metrics


def analyze_context_distribution(
    encoder: torch.nn.Module,
    segments: List,
    device: str = 'cpu',
    save_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Analyze the distribution of embeddings in latent space.

    Args:
        encoder: Trained encoder
        segments: List of trajectory segments
        device: Device for inference
        save_path: Path to save analysis results

    Returns:
        Dictionary with analysis results
    """
    encoder.eval()
    encoder = encoder.to(device)

    embeddings = []
    context_ids = []
    context_params = []

    with torch.no_grad():
        for seg in segments:
            obs = seg.observations[:-1]
            actions = seg.actions
            if len(actions.shape) == 1:
                actions = actions[:, None]

            trajectory = np.concatenate([obs, actions], axis=-1)
            traj_tensor = torch.FloatTensor(trajectory).unsqueeze(0).to(device)

            emb = encoder(traj_tensor).cpu().numpy()
            embeddings.append(emb)
            context_ids.append(seg.context_id)
            context_params.append(seg.context_params)

    embeddings = np.concatenate(embeddings, axis=0)
    context_ids = np.array(context_ids)

    # Compute per-context statistics
    unique_contexts = np.unique(context_ids)
    context_stats = {}

    for ctx in unique_contexts:
        ctx_mask = context_ids == ctx
        ctx_embeddings = embeddings[ctx_mask]

        context_stats[int(ctx)] = {
            'mean': ctx_embeddings.mean(axis=0),
            'std': ctx_embeddings.std(axis=0),
            'count': int(ctx_mask.sum())
        }

    analysis = {
        'embeddings': embeddings,
        'context_ids': context_ids,
        'context_stats': context_stats
    }

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as numpy
        np.savez(
            save_path,
            embeddings=embeddings,
            context_ids=context_ids
        )
        print(f"Saved analysis to {save_path}")

    return analysis
