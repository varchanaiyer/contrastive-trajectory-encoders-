#!/usr/bin/env python3
"""
Analyze metric preservation in learned trajectory embeddings.

This script evaluates whether the encoder preserves the metric structure of the
context parameter space. Specifically, it checks if distances between contexts
in the parameter space correlate with distances in the embedding space.

For the pendulum environment, this means checking if contexts with similar
gravity values produce similar embeddings, and if the embedding distances
are proportional to the gravity differences.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from itertools import combinations
from typing import List, Dict, Tuple, Optional
import pickle
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import TrajectoryCollector, get_context_distributions
from src.models import LSTMEncoder, TransformerEncoder
from src.data.trajectory_collector import TrajectorySegment


def load_encoder(encoder_path: str, input_dim: int, device: str = 'cpu'):
    """
    Load trained encoder from checkpoint.

    Args:
        encoder_path: Path to saved encoder checkpoint
        input_dim: Input dimension (obs_dim + action_dim)
        device: Device to load model on

    Returns:
        Loaded encoder model in eval mode
    """
    print(f"Loading encoder from {encoder_path}")
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)

    # Try to get config from checkpoint or use defaults
    if 'config' in checkpoint:
        enc_cfg = checkpoint['config']
    else:
        # Use default config
        print("Warning: No config found in checkpoint, using defaults")
        enc_cfg = {
            'architecture': 'lstm',
            'hidden_dim': 256,
            'num_layers': 2,
            'latent_dim': 64,
            'dropout': 0.2,
            'bidirectional': True
        }

    # Create encoder based on architecture
    arch = enc_cfg.get('architecture', 'lstm')
    if arch == 'lstm':
        encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=enc_cfg.get('hidden_dim', 256),
            num_layers=enc_cfg.get('num_layers', 2),
            latent_dim=enc_cfg.get('latent_dim', 64),
            dropout=enc_cfg.get('dropout', 0.2),
            bidirectional=enc_cfg.get('bidirectional', True)
        )
    elif arch == 'transformer':
        encoder = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=enc_cfg.get('hidden_dim', 128),
            num_heads=enc_cfg.get('num_heads', 4),
            num_layers=enc_cfg.get('num_layers', 3),
            latent_dim=enc_cfg.get('latent_dim', 64),
            dropout=enc_cfg.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Load state dict
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Encoder loaded (epoch {epoch}, architecture: {arch})")

    return encoder


def encode_trajectory(encoder: torch.nn.Module, segment: TrajectorySegment, device: str) -> np.ndarray:
    """
    Encode a single trajectory segment.

    Args:
        encoder: Trained encoder model
        segment: Trajectory segment to encode
        device: Device to run on

    Returns:
        Embedding vector as numpy array
    """
    # Concatenate observations and actions
    obs = segment.observations
    actions = segment.actions

    # Handle action shape
    if len(actions.shape) == 1:
        actions = actions[:, None]

    # Combine obs and actions
    trajectory = np.concatenate([obs, actions], axis=-1)

    # Convert to tensor and add batch dimension
    traj_tensor = torch.FloatTensor(trajectory).unsqueeze(0).to(device)

    # Encode
    with torch.no_grad():
        embedding = encoder(traj_tensor).cpu().numpy()

    return embedding[0]  # Remove batch dimension


def compute_mean_embeddings_per_context(
    encoder: torch.nn.Module,
    segments: List[TrajectorySegment],
    device: str
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, float]]]:
    """
    Compute mean embedding for each context.

    Args:
        encoder: Trained encoder model
        segments: List of trajectory segments
        device: Device to run on

    Returns:
        - Dictionary mapping context_id to mean embedding
        - Dictionary mapping context_id to context parameters
    """
    # Group segments by context
    context_embeddings = {}
    context_params = {}

    print("Computing embeddings for all trajectories...")
    for segment in segments:
        ctx_id = segment.context_id

        # Store context parameters
        if ctx_id not in context_params:
            context_params[ctx_id] = segment.context_params

        # Encode segment
        embedding = encode_trajectory(encoder, segment, device)

        # Add to list for this context
        if ctx_id not in context_embeddings:
            context_embeddings[ctx_id] = []
        context_embeddings[ctx_id].append(embedding)

    # Compute mean embedding for each context
    mean_embeddings = {}
    for ctx_id, embeddings in context_embeddings.items():
        mean_embeddings[ctx_id] = np.mean(embeddings, axis=0)

    print(f"Computed mean embeddings for {len(mean_embeddings)} contexts")
    return mean_embeddings, context_params


def compute_pairwise_distances(
    mean_embeddings: Dict[int, np.ndarray],
    context_params: Dict[int, Dict[str, float]],
    param_key: str = 'g'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise distances in parameter space and embedding space.

    Args:
        mean_embeddings: Mean embeddings per context
        context_params: Context parameters per context
        param_key: Key for the parameter to use (e.g., 'g' for gravity)

    Returns:
        - Array of parameter distances
        - Array of embedding distances
    """
    context_ids = sorted(mean_embeddings.keys())

    param_distances = []
    embedding_distances = []

    # Compute all pairwise distances
    for ctx_i, ctx_j in combinations(context_ids, 2):
        # Parameter distance
        param_i = context_params[ctx_i][param_key]
        param_j = context_params[ctx_j][param_key]
        param_dist = abs(param_i - param_j)

        # Embedding distance (L2 norm)
        emb_i = mean_embeddings[ctx_i]
        emb_j = mean_embeddings[ctx_j]
        emb_dist = np.linalg.norm(emb_i - emb_j)

        param_distances.append(param_dist)
        embedding_distances.append(emb_dist)

    return np.array(param_distances), np.array(embedding_distances)


def analyze_metric_preservation(
    param_distances: np.ndarray,
    embedding_distances: np.ndarray
) -> Dict[str, float]:
    """
    Compute correlation metrics to assess metric preservation.
    Tests multiple relationship types (linear, logarithmic, sqrt, polynomial, power)
    and selects the best fit.

    Args:
        param_distances: Parameter space distances
        embedding_distances: Embedding space distances

    Returns:
        Dictionary of statistics including best fit model
    """
    # Pearson correlation (linear relationship)
    pearson_corr, pearson_pval = pearsonr(param_distances, embedding_distances)

    # Spearman correlation (monotonic relationship)
    spearman_corr, spearman_pval = spearmanr(param_distances, embedding_distances)

    X = param_distances.reshape(-1, 1)
    y = embedding_distances

    # Store all model results
    models = {}

    # 1. Linear: y = a*x + b
    reg_linear = LinearRegression()
    reg_linear.fit(X, y)
    y_pred_linear = reg_linear.predict(X)
    r2_linear = reg_linear.score(X, y)
    models['linear'] = {
        'r2': r2_linear,
        'params': {'slope': reg_linear.coef_[0], 'intercept': reg_linear.intercept_},
        'y_pred': y_pred_linear,
        'equation': f"y = {reg_linear.coef_[0]:.4f}x + {reg_linear.intercept_:.4f}"
    }

    # 2. Logarithmic: y = a*log(x) + b
    # Avoid log(0) by adding small epsilon
    x_log = np.log(param_distances + 1e-8).reshape(-1, 1)
    reg_log = LinearRegression()
    reg_log.fit(x_log, y)
    y_pred_log = reg_log.predict(x_log)
    r2_log = reg_log.score(x_log, y)
    models['logarithmic'] = {
        'r2': r2_log,
        'params': {'a': reg_log.coef_[0], 'b': reg_log.intercept_},
        'y_pred': y_pred_log,
        'equation': f"y = {reg_log.coef_[0]:.4f}*log(x) + {reg_log.intercept_:.4f}"
    }

    # 3. Square root: y = a*sqrt(x) + b
    x_sqrt = np.sqrt(param_distances).reshape(-1, 1)
    reg_sqrt = LinearRegression()
    reg_sqrt.fit(x_sqrt, y)
    y_pred_sqrt = reg_sqrt.predict(x_sqrt)
    r2_sqrt = reg_sqrt.score(x_sqrt, y)
    models['sqrt'] = {
        'r2': r2_sqrt,
        'params': {'a': reg_sqrt.coef_[0], 'b': reg_sqrt.intercept_},
        'y_pred': y_pred_sqrt,
        'equation': f"y = {reg_sqrt.coef_[0]:.4f}*sqrt(x) + {reg_sqrt.intercept_:.4f}"
    }

    # 4. Polynomial (quadratic): y = a*x^2 + b*x + c
    poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_model.fit(X, y)
    y_pred_poly = poly_model.predict(X)
    r2_poly = poly_model.score(X, y)
    poly_coefs = poly_model.named_steps['linearregression'].coef_
    poly_intercept = poly_model.named_steps['linearregression'].intercept_
    models['polynomial'] = {
        'r2': r2_poly,
        'params': {'a': poly_coefs[2], 'b': poly_coefs[1], 'c': poly_intercept},
        'y_pred': y_pred_poly,
        'equation': f"y = {poly_coefs[2]:.4f}x² + {poly_coefs[1]:.4f}x + {poly_intercept:.4f}"
    }

    # 5. Power law: y = a * x^b
    def power_func(x, a, b):
        return a * np.power(x + 1e-8, b)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(power_func, param_distances, embedding_distances,
                               p0=[1.0, 0.5], maxfev=5000)
        y_pred_power = power_func(param_distances, *popt)
        ss_res = np.sum((embedding_distances - y_pred_power) ** 2)
        ss_tot = np.sum((embedding_distances - np.mean(embedding_distances)) ** 2)
        r2_power = 1 - (ss_res / ss_tot)
        models['power'] = {
            'r2': r2_power,
            'params': {'a': popt[0], 'b': popt[1]},
            'y_pred': y_pred_power,
            'equation': f"y = {popt[0]:.4f}*x^{popt[1]:.4f}"
        }
    except (RuntimeError, ValueError):
        # Power law fitting failed, skip it
        models['power'] = {
            'r2': -np.inf,
            'params': {},
            'y_pred': None,
            'equation': "fitting failed"
        }

    # 6. Exponential saturation: y = a * (1 - exp(-b*x)) + c
    def saturation_func(x, a, b, c):
        return a * (1 - np.exp(-b * x)) + c

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt_sat, _ = curve_fit(saturation_func, param_distances, embedding_distances,
                                    p0=[1.0, 0.1, 0.0], maxfev=5000)
        y_pred_sat = saturation_func(param_distances, *popt_sat)
        ss_res = np.sum((embedding_distances - y_pred_sat) ** 2)
        ss_tot = np.sum((embedding_distances - np.mean(embedding_distances)) ** 2)
        r2_sat = 1 - (ss_res / ss_tot)
        models['saturation'] = {
            'r2': r2_sat,
            'params': {'a': popt_sat[0], 'b': popt_sat[1], 'c': popt_sat[2]},
            'y_pred': y_pred_sat,
            'equation': f"y = {popt_sat[0]:.4f}*(1 - exp(-{popt_sat[1]:.4f}*x)) + {popt_sat[2]:.4f}"
        }
    except (RuntimeError, ValueError):
        models['saturation'] = {
            'r2': -np.inf,
            'params': {},
            'y_pred': None,
            'equation': "fitting failed"
        }

    # Find the best model (highest R²)
    best_model_name = max(models.keys(), key=lambda k: models[k]['r2'])
    best_model = models[best_model_name]

    stats = {
        'pearson_correlation': pearson_corr,
        'pearson_pvalue': pearson_pval,
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_pval,
        'r2_score': best_model['r2'],
        'linear_slope': models['linear']['params']['slope'],
        'linear_intercept': models['linear']['params']['intercept'],
        'best_model': best_model_name,
        'best_model_r2': best_model['r2'],
        'best_model_equation': best_model['equation'],
        'best_model_y_pred': best_model['y_pred'],
        'all_models': models
    }

    return stats


def plot_metric_preservation(
    param_distances: np.ndarray,
    embedding_distances: np.ndarray,
    stats: Dict[str, float],
    save_path: str,
    param_name: str = "Gravity"
):
    """
    Create scatter plot showing metric preservation with best-fit model.

    Args:
        param_distances: Parameter space distances
        embedding_distances: Embedding space distances
        stats: Statistics dictionary
        save_path: Path to save the plot
        param_name: Name of the parameter for labels
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Sort data for smooth curve plotting
    sort_idx = np.argsort(param_distances)
    x_sorted = param_distances[sort_idx]
    y_sorted = embedding_distances[sort_idx]

    # === Left plot: Best fit model ===
    ax1 = axes[0]

    # Scatter plot
    ax1.scatter(param_distances, embedding_distances, alpha=0.6, s=50,
                color='steelblue', edgecolors='black', linewidth=0.5,
                label='Data points')

    # Plot best fit curve
    best_model = stats['best_model']
    y_pred = stats['best_model_y_pred']
    if y_pred is not None:
        y_pred_sorted = y_pred[sort_idx]
        ax1.plot(x_sorted, y_pred_sorted, 'r-', linewidth=2.5,
                label=f'Best fit: {best_model.capitalize()}')

    # Also plot linear for comparison (dashed)
    y_linear = stats['linear_slope'] * x_sorted + stats['linear_intercept']
    if best_model != 'linear':
        ax1.plot(x_sorted, y_linear, 'g--', linewidth=1.5, alpha=0.7,
                label='Linear fit')

    ax1.set_xlabel(f'Parameter Distance (Δ{param_name.lower()})', fontsize=12)
    ax1.set_ylabel('Embedding Distance (L2)', fontsize=12)
    ax1.set_title('Metric Preservation: Best Fit Model', fontsize=14, fontweight='bold')

    # Statistics text box
    textstr = '\n'.join([
        f"Best Model: {best_model.capitalize()}",
        f"R² = {stats['best_model_r2']:.4f}",
        f"",
        f"Spearman ρ = {stats['spearman_correlation']:.4f}",
        f"Pearson r = {stats['pearson_correlation']:.4f}",
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
             fontsize=11, verticalalignment='top', bbox=props)

    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # === Right plot: Model comparison ===
    ax2 = axes[1]

    # Get R² scores for all models
    all_models = stats.get('all_models', {})
    model_names = []
    r2_scores = []

    for name, model in all_models.items():
        if model['r2'] > -np.inf:
            model_names.append(name.capitalize())
            r2_scores.append(model['r2'])

    # Sort by R² score
    sorted_idx = np.argsort(r2_scores)[::-1]
    model_names = [model_names[i] for i in sorted_idx]
    r2_scores = [r2_scores[i] for i in sorted_idx]

    # Bar colors: best model in green, others in blue
    colors = ['forestgreen' if name.lower() == best_model else 'steelblue'
              for name in model_names]

    bars = ax2.barh(model_names, r2_scores, color=colors, edgecolor='black')

    # Add R² values as text
    for bar, r2 in zip(bars, r2_scores):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{r2:.4f}', va='center', fontsize=10)

    ax2.set_xlabel('R² Score', fontsize=12)
    ax2.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, max(r2_scores) * 1.15 if r2_scores else 1)
    ax2.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Good threshold (0.8)')
    ax2.axvline(x=0.6, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold (0.6)')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # Save plot
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def save_statistics(stats: Dict[str, float], save_path: str):
    """
    Save statistics to text file.

    Args:
        stats: Statistics dictionary
        save_path: Path to save the statistics
    """
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("METRIC PRESERVATION ANALYSIS RESULTS\n")
        f.write("=" * 60 + "\n\n")

        # Best Model Section
        f.write("BEST FIT MODEL:\n")
        f.write("-" * 60 + "\n")
        best_model = stats.get('best_model', 'linear')
        best_r2 = stats.get('best_model_r2', stats['r2_score'])
        best_eq = stats.get('best_model_equation', 'N/A')
        f.write(f"Model Type:           {best_model.upper()}\n")
        f.write(f"R² Score:             {best_r2:>8.4f}\n")
        f.write(f"Equation:             {best_eq}\n\n")

        # Correlation Metrics
        f.write("\nCORRELATION METRICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Spearman Correlation: {stats['spearman_correlation']:>8.4f}")
        f.write(f"  (p = {stats['spearman_pvalue']:.2e})\n")
        f.write(f"Pearson Correlation:  {stats['pearson_correlation']:>8.4f}")
        f.write(f"  (p = {stats['pearson_pvalue']:.2e})\n\n")

        f.write("Note: Spearman measures monotonic relationship (rank-based).\n")
        f.write("      Pearson measures linear relationship.\n")
        f.write("      High Spearman + Low Pearson = non-linear monotonic relationship.\n\n")

        # Model Comparison
        f.write("\nMODEL COMPARISON (R² Scores):\n")
        f.write("-" * 60 + "\n")

        all_models = stats.get('all_models', {})
        if all_models:
            # Sort by R² descending
            sorted_models = sorted(all_models.items(),
                                   key=lambda x: x[1]['r2'], reverse=True)
            for name, model in sorted_models:
                r2 = model['r2']
                eq = model.get('equation', 'N/A')
                marker = " <-- BEST" if name == best_model else ""
                if r2 > -np.inf:
                    f.write(f"  {name.capitalize():12s}  R² = {r2:>7.4f}  {eq}{marker}\n")
        else:
            f.write(f"  Linear:     R² = {stats['r2_score']:.4f}\n")

        # Interpretation
        f.write("\n\nINTERPRETATION:\n")
        f.write("-" * 60 + "\n")

        # Use Spearman for overall assessment (works for non-linear)
        spearman = stats['spearman_correlation']
        if spearman > 0.8:
            f.write("✓ EXCELLENT: Strong metric preservation (ρ > 0.8)\n")
            f.write("  The encoder successfully preserves distance relationships.\n")
        elif spearman > 0.6:
            f.write("✓ GOOD: Moderate metric preservation (0.6 < ρ < 0.8)\n")
            f.write("  The encoder shows reasonable distance preservation.\n")
        else:
            f.write("✗ WEAK: Limited metric preservation (ρ < 0.6)\n")
            f.write("  Consider training longer or adjusting hyperparameters.\n")

        # Note about relationship type
        f.write(f"\nRelationship Type: {best_model.upper()}\n")
        if best_model != 'linear':
            f.write("  Note: The relationship is non-linear. This is normal and\n")
            f.write("  can indicate that the encoder has learned a compressed or\n")
            f.write("  saturating representation of the parameter space.\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("WHAT THIS MEANS:\n")
        f.write("-" * 60 + "\n")
        f.write("Metric preservation measures whether the encoder maintains\n")
        f.write("the neighborhood structure: contexts with similar parameters\n")
        f.write("should produce similar embeddings.\n\n")
        f.write("A strong monotonic correlation (Spearman > 0.8) indicates\n")
        f.write("successful learning, even if the relationship is non-linear.\n")
        f.write("=" * 60 + "\n")

    print(f"Statistics saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze metric preservation in trajectory embeddings"
    )
    parser.add_argument(
        '--encoder-path',
        type=str,
        required=True,
        help='Path to trained encoder checkpoint'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to trajectory data (pickle file with segments)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results/)'
    )
    parser.add_argument(
        '--param-key',
        type=str,
        default='g',
        help='Context parameter key to analyze (default: g for gravity)'
    )
    parser.add_argument(
        '--param-name',
        type=str,
        default='Gravity',
        help='Human-readable parameter name for plots (default: Gravity)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use: cuda, cpu, or auto (default: auto)'
    )

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load trajectory data
    print(f"\nLoading trajectory data from {args.data_path}")
    segments = TrajectoryCollector.load(args.data_path)
    print(f"Loaded {len(segments)} trajectory segments")

    # Infer input dimension from data
    first_seg = segments[0]
    obs_dim = first_seg.observations.shape[-1]
    action_dim = first_seg.actions.shape[-1] if len(first_seg.actions.shape) > 1 else 1
    input_dim = obs_dim + action_dim
    print(f"Input dimension: {input_dim} (obs: {obs_dim}, action: {action_dim})")

    # Load encoder
    encoder = load_encoder(args.encoder_path, input_dim, device)

    # Compute mean embeddings per context
    print(f"\nStep 1: Computing mean embeddings for each context...")
    mean_embeddings, context_params = compute_mean_embeddings_per_context(
        encoder, segments, device
    )

    # Verify all contexts have the parameter
    if args.param_key not in list(context_params.values())[0]:
        available_keys = list(list(context_params.values())[0].keys())
        raise ValueError(
            f"Parameter key '{args.param_key}' not found in context params. "
            f"Available keys: {available_keys}"
        )

    # Compute pairwise distances
    print(f"\nStep 2: Computing pairwise distances...")
    param_distances, embedding_distances = compute_pairwise_distances(
        mean_embeddings, context_params, args.param_key
    )
    print(f"Computed {len(param_distances)} pairwise distances")

    # Analyze metric preservation
    print(f"\nStep 3: Analyzing metric preservation...")
    stats = analyze_metric_preservation(param_distances, embedding_distances)

    # Print results
    print("\n" + "="*60)
    print("METRIC PRESERVATION ANALYSIS RESULTS")
    print("="*60)
    print(f"\nPearson Correlation:  {stats['pearson_correlation']:.4f} (p={stats['pearson_pvalue']:.6f})")
    print(f"Spearman Correlation: {stats['spearman_correlation']:.4f} (p={stats['spearman_pvalue']:.6f})")
    print(f"R² Score:             {stats['r2_score']:.4f}")
    print(f"Linear Slope:         {stats['linear_slope']:.4f}")
    print(f"Linear Intercept:     {stats['linear_intercept']:.4f}")

    # Interpretation
    print("\n" + "-"*60)
    print("Interpretation:")
    if stats['pearson_correlation'] > 0.8:
        print("  ✓ EXCELLENT: Strong metric preservation (r > 0.8)")
    elif stats['pearson_correlation'] > 0.6:
        print("  ✓ GOOD: Moderate metric preservation (0.6 < r < 0.8)")
    else:
        print("  ✗ WEAK: Limited metric preservation (r < 0.6)")
    print("="*60)

    # Create visualization
    print(f"\nStep 4: Creating visualization...")
    plot_path = Path(args.results_dir) / 'metric_preservation_correlation.png'
    plot_metric_preservation(
        param_distances,
        embedding_distances,
        stats,
        str(plot_path),
        args.param_name
    )

    # Save statistics
    print(f"\nStep 5: Saving statistics...")
    stats_path = Path(args.results_dir) / 'metric_preservation_stats.txt'
    save_statistics(stats, str(stats_path))

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.results_dir}/")
    print(f"  - Plot: {plot_path}")
    print(f"  - Stats: {stats_path}")
    print()


if __name__ == "__main__":
    main()
