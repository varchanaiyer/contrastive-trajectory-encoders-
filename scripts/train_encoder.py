#!/usr/bin/env python3
"""Script to train the trajectory encoder (Phase 1)"""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import TrajectoryCollector, TrajectoryDataset
from src.models import LSTMEncoder, TransformerEncoder, InfoNCELoss, TripletLoss, SupConLoss
from src.training import EncoderTrainer
from src.utils import load_config, save_config


def main():
    parser = argparse.ArgumentParser(description="Train trajectory encoder")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data-path', type=str, help='Path to pre-collected data (optional)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
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

    # Load or collect data
    if args.data_path:
        print(f"\nLoading data from {args.data_path}")
        segments = TrajectoryCollector.load(args.data_path)
    else:
        print("\nData path not provided. Please run collect_data.py first or provide --data-path")
        return

    # Infer input dimension from data
    first_seg = segments[0]
    obs_dim = first_seg.observations.shape[-1]
    action_dim = first_seg.actions.shape[-1] if len(first_seg.actions.shape) > 1 else 1
    input_dim = obs_dim + action_dim - 1  # -1 because we use obs[:-1]
    print(f"\nInferred input dimension: {input_dim} (obs: {obs_dim}, action: {action_dim})")

    # Create dataset
    full_dataset = TrajectoryDataset(segments, augmentation=config['data']['augmentation'])

    # Train/val split
    train_size = int(config['data']['train_val_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['experiment']['seed'])
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )

    # Create encoder
    encoder_config = config['encoder']
    if encoder_config['architecture'] == 'lstm':
        encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_config['hidden_dim'],
            num_layers=encoder_config['num_layers'],
            latent_dim=encoder_config['latent_dim'],
            dropout=encoder_config['dropout'],
            bidirectional=encoder_config['bidirectional']
        )
    elif encoder_config['architecture'] == 'transformer':
        encoder = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_config['hidden_dim'],
            num_heads=encoder_config['num_heads'],
            num_layers=encoder_config['num_layers'],
            latent_dim=encoder_config['latent_dim'],
            dropout=encoder_config['dropout']
        )
    else:
        raise ValueError(f"Unknown encoder architecture: {encoder_config['architecture']}")

    print(f"\nEncoder architecture: {encoder_config['architecture']}")
    print(f"Number of parameters: {sum(p.numel() for p in encoder.parameters())}")

    # Create loss function
    loss_config = config['loss']
    if loss_config['type'] == 'infonce':
        loss_fn = InfoNCELoss(temperature=loss_config['temperature'])
    elif loss_config['type'] == 'triplet':
        loss_fn = TripletLoss(margin=loss_config['margin'])
    elif loss_config['type'] == 'supcon':
        loss_fn = SupConLoss(temperature=loss_config['temperature'])
    else:
        raise ValueError(f"Unknown loss type: {loss_config['type']}")

    # Create trainer
    trainer = EncoderTrainer(
        encoder=encoder,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        device=device,
        log_dir=config['paths']['log_dir'],
        checkpoint_dir=config['paths']['checkpoint_dir']
    )

    # Resume from checkpoint if provided
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Save config to experiment directory
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(checkpoint_dir / 'config.yaml'))

    # Train
    print("\nStarting training...")
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        eval_every=config['training']['eval_every']
    )

    print("\nTraining complete! Best model saved to:")
    print(checkpoint_dir / 'best_encoder.pt')


if __name__ == "__main__":
    main()
