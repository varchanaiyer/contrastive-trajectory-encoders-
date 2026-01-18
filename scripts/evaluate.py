#!/usr/bin/env python3
"""Script to evaluate trained policies and compare baselines"""

import argparse
import sys
from pathlib import Path
import torch
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_context_distributions
from src.models import LSTMEncoder, TransformerEncoder, ContextEncoder
from src.evaluation import compute_zero_shot_metrics, compare_baselines
from src.utils import load_config
from stable_baselines3 import PPO, SAC


def main():
    parser = argparse.ArgumentParser(description="Evaluate policies")
    parser.add_argument('--policy-path', type=str, required=True,
                        help='Path to trained policy')
    parser.add_argument('--encoder-path', type=str, required=True,
                        help='Path to trained encoder')
    parser.add_argument('--env-name', type=str, required=True,
                        help='Environment name')
    parser.add_argument('--n-episodes', type=int, default=20,
                        help='Number of evaluation episodes per context')
    parser.add_argument('--output', type=str, default='experiments/results',
                        help='Output directory for results')
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get test contexts
    test_contexts = get_context_distributions(args.env_name, split='test')
    print(f"Evaluating on {len(test_contexts)} test contexts")

    # Load encoder
    print(f"\nLoading encoder from {args.encoder_path}")
    encoder_dir = Path(args.encoder_path).parent
    encoder_config_path = encoder_dir / 'config.yaml'

    if not encoder_config_path.exists():
        print(f"Encoder config not found at {encoder_config_path}")
        return

    encoder_cfg = load_config(str(encoder_config_path))['encoder']

    # Infer dimensions
    from src.data import make_carl_env
    temp_env = make_carl_env(args.env_name)
    obs_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0] if hasattr(temp_env.action_space, 'shape') else 1
    input_dim = obs_dim + action_dim
    temp_env.close()

    # Create encoder
    if encoder_cfg['architecture'] == 'lstm':
        base_encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_cfg['hidden_dim'],
            num_layers=encoder_cfg['num_layers'],
            latent_dim=encoder_cfg['latent_dim'],
            dropout=encoder_cfg['dropout'],
            bidirectional=encoder_cfg['bidirectional']
        )
    elif encoder_cfg['architecture'] == 'transformer':
        base_encoder = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_cfg['hidden_dim'],
            num_heads=encoder_cfg['num_heads'],
            num_layers=encoder_cfg['num_layers'],
            latent_dim=encoder_cfg['latent_dim'],
            dropout=encoder_cfg['dropout']
        )

    # Load encoder weights
    checkpoint = torch.load(args.encoder_path, map_location=device)
    base_encoder.load_state_dict(checkpoint['encoder_state_dict'])

    encoder = ContextEncoder(base_encoder, freeze_encoder=True)
    encoder = encoder.to(device)

    # Load policy
    print(f"Loading policy from {args.policy_path}")
    if args.policy_path.endswith('.zip'):
        # Try to detect algorithm from config
        policy_dir = Path(args.policy_path).parent
        policy_config_path = policy_dir / 'config.yaml'

        if policy_config_path.exists():
            policy_cfg = load_config(str(policy_config_path))
            algorithm = policy_cfg['policy']['algorithm']
        else:
            algorithm = 'ppo'  # Default

        if algorithm == 'ppo':
            policy = PPO.load(args.policy_path, device=device)
        elif algorithm == 'sac':
            policy = SAC.load(args.policy_path, device=device)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    else:
        print("Policy should be a .zip file saved by Stable-Baselines3")
        return

    # Evaluate
    print("\n" + "="*80)
    print("ZERO-SHOT EVALUATION ON TEST CONTEXTS")
    print("="*80)

    results = compute_zero_shot_metrics(
        policy=policy,
        encoder=encoder,
        contexts=test_contexts,
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        buffer_length=32,
        device=device
    )

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / 'zero_shot_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {results_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Overall Mean Reward: {results['overall']['mean_reward']:.2f} ± {results['overall']['std_reward']:.2f}")
    print(f"Overall Median Reward: {results['overall']['median_reward']:.2f}")
    print("="*80)


if __name__ == "__main__":
    main()
