#!/usr/bin/env python3
"""Script to train context-conditional policy (Phase 2)"""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_context_distributions
from src.models import LSTMEncoder, TransformerEncoder, ContextEncoder
from src.training import PolicyTrainer
from src.utils import load_config, save_config


def main():
    parser = argparse.ArgumentParser(description="Train context-conditional policy")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--encoder-path', type=str, required=True,
                        help='Path to trained encoder checkpoint')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print("Configuration:")
    print(config)

    # Set device
    device = config['experiment']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Load encoder
    print(f"\nLoading encoder from {args.encoder_path}")
    checkpoint = torch.load(args.encoder_path, map_location=device)

    # Load encoder config to reconstruct architecture
    encoder_checkpoint_dir = Path(args.encoder_path).parent
    encoder_config_path = encoder_checkpoint_dir / 'config.yaml'

    if encoder_config_path.exists():
        encoder_cfg = load_config(str(encoder_config_path))
        enc_cfg = encoder_cfg['encoder']

        # Infer input dimension (you may need to adjust this)
        # For now, we'll need to know the environment
        env_name = config['environment']['name']
        from src.data import make_carl_env
        temp_env = make_carl_env(env_name)
        obs_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.shape[0] if hasattr(temp_env.action_space, 'shape') else 1
        input_dim = obs_dim + action_dim
        temp_env.close()

        # Create encoder
        if enc_cfg['architecture'] == 'lstm':
            base_encoder = LSTMEncoder(
                input_dim=input_dim,
                hidden_dim=enc_cfg['hidden_dim'],
                num_layers=enc_cfg['num_layers'],
                latent_dim=enc_cfg['latent_dim'],
                dropout=enc_cfg['dropout'],
                bidirectional=enc_cfg['bidirectional']
            )
        elif enc_cfg['architecture'] == 'transformer':
            base_encoder = TransformerEncoder(
                input_dim=input_dim,
                hidden_dim=enc_cfg['hidden_dim'],
                num_heads=enc_cfg['num_heads'],
                num_layers=enc_cfg['num_layers'],
                latent_dim=enc_cfg['latent_dim'],
                dropout=enc_cfg['dropout']
            )
        else:
            raise ValueError(f"Unknown architecture: {enc_cfg['architecture']}")

    else:
        raise ValueError(f"Encoder config not found at {encoder_config_path}")

    # Load encoder weights
    base_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    print(f"Loaded encoder from epoch {checkpoint['epoch']}")

    # Wrap in ContextEncoder
    encoder = ContextEncoder(
        trajectory_encoder=base_encoder,
        freeze_encoder=config['encoder']['freeze']
    )

    # Get training and test contexts
    env_name = config['environment']['name']
    train_contexts = get_context_distributions(env_name, split=config['environment']['train_split'])
    test_contexts = get_context_distributions(env_name, split=config['environment']['test_split'])

    print(f"\nEnvironment: {env_name}")
    print(f"Training contexts: {len(train_contexts)}")
    print(f"Test contexts: {len(test_contexts)}")

    # Create policy trainer
    trainer = PolicyTrainer(
        env_name=env_name,
        encoder=encoder,
        contexts=train_contexts,
        algorithm=config['policy']['algorithm'],
        buffer_length=config['encoder']['buffer_length'],
        device=device,
        log_dir=config['paths']['log_dir'],
        checkpoint_dir=config['paths']['checkpoint_dir']
    )

    # Policy kwargs
    policy_kwargs = {
        'net_arch': config['policy']['net_arch'],
        'activation_fn': torch.nn.ReLU
    }

    # Save config
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(checkpoint_dir / 'config.yaml'))

    # Train
    print("\nStarting policy training...")
    trainer.train(
        total_timesteps=config['training']['total_timesteps'],
        eval_contexts=test_contexts,
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        policy_kwargs=policy_kwargs
    )

    # Evaluate on test contexts
    print("\n" + "="*80)
    print("EVALUATING ON TEST CONTEXTS (ZERO-SHOT)")
    print("="*80)

    results = trainer.evaluate(
        contexts=test_contexts,
        n_episodes=config['training']['n_eval_episodes'],
        deterministic=True
    )

    # Save results
    import json
    results_path = checkpoint_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
