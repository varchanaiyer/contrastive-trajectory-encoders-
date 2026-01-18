#!/usr/bin/env python3
"""Script to visualize learned embeddings"""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import TrajectoryCollector
from src.models import LSTMEncoder, TransformerEncoder
from src.evaluation import EmbeddingVisualizer, compute_embedding_quality
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings")
    parser.add_argument('--encoder-path', type=str, required=True,
                        help='Path to trained encoder')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to trajectory data')
    parser.add_argument('--save-dir', type=str, default='experiments/visualizations',
                        help='Directory to save plots')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca'],
                        help='Dimensionality reduction method')
    args = parser.parse_args()

    # Load encoder config
    encoder_dir = Path(args.encoder_path).parent
    config_path = encoder_dir / 'config.yaml'

    if not config_path.exists():
        print(f"Config not found at {config_path}")
        return

    config = load_config(str(config_path))
    enc_cfg = config['encoder']

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    print(f"Loading data from {args.data_path}")
    segments = TrajectoryCollector.load(args.data_path)
    print(f"Loaded {len(segments)} segments")

    # Infer input dimension
    first_seg = segments[0]
    obs_dim = first_seg.observations.shape[-1]
    action_dim = first_seg.actions.shape[-1] if len(first_seg.actions.shape) > 1 else 1
    input_dim = obs_dim + action_dim - 1

    # Create encoder
    if enc_cfg['architecture'] == 'lstm':
        encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=enc_cfg['hidden_dim'],
            num_layers=enc_cfg['num_layers'],
            latent_dim=enc_cfg['latent_dim'],
            dropout=enc_cfg['dropout'],
            bidirectional=enc_cfg['bidirectional']
        )
    elif enc_cfg['architecture'] == 'transformer':
        encoder = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=enc_cfg['hidden_dim'],
            num_heads=enc_cfg['num_heads'],
            num_layers=enc_cfg['num_layers'],
            latent_dim=enc_cfg['latent_dim'],
            dropout=enc_cfg['dropout']
        )
    else:
        raise ValueError(f"Unknown architecture: {enc_cfg['architecture']}")

    # Load weights
    print(f"Loading encoder from {args.encoder_path}")
    checkpoint = torch.load(args.encoder_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()

    print(f"Encoder loaded from epoch {checkpoint['epoch']}")

    # Create visualizer
    visualizer = EmbeddingVisualizer(save_dir=args.save_dir)

    # Compute embedding quality
    print("\nComputing embedding quality metrics...")
    quality_metrics = compute_embedding_quality(encoder, segments, device)

    # Visualize embeddings
    print(f"\nGenerating {args.method.upper()} visualization...")
    reduced, context_ids = visualizer.visualize_embeddings(
        encoder=encoder,
        segments=segments,
        method=args.method,
        title=f'Trajectory Embeddings ({args.method.upper()})',
        color_by='context',
        save_name=f'embeddings_{args.method}_by_context.png'
    )

    # Visualize by reward
    visualizer.visualize_embeddings(
        encoder=encoder,
        segments=segments,
        method=args.method,
        title=f'Trajectory Embeddings colored by Reward ({args.method.upper()})',
        color_by='reward',
        save_name=f'embeddings_{args.method}_by_reward.png'
    )

    # Plot context separation
    print("\nPlotting context separation...")
    import numpy as np

    # Get embeddings
    embeddings = []
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

    embeddings = np.concatenate(embeddings, axis=0)
    context_params = [seg.context_params for seg in segments]

    visualizer.plot_context_separation(
        embeddings=embeddings,
        context_ids=context_ids,
        context_params=context_params,
        save_name='context_separation.png'
    )

    # Plot parameter correlation if possible
    if context_params and len(context_params[0]) > 0:
        param_name = list(context_params[0].keys())[0]
        print(f"\nPlotting correlation with parameter: {param_name}")
        visualizer.plot_context_parameter_correlation(
            embeddings=embeddings,
            context_params=context_params,
            param_name=param_name,
            save_name=f'correlation_{param_name}.png'
        )

    print(f"\nVisualization complete! Plots saved to {args.save_dir}")


if __name__ == "__main__":
    main()
