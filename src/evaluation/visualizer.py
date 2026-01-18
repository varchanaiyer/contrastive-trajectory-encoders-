"""Visualization tools for embeddings and results"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class EmbeddingVisualizer:
    """Visualize trajectory embeddings using dimensionality reduction"""

    def __init__(self, save_dir: Optional[str] = None):
        """
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 8)

    @torch.no_grad()
    def visualize_embeddings(
        self,
        encoder: torch.nn.Module,
        segments: List,
        method: str = 'tsne',
        title: str = 'Trajectory Embeddings',
        color_by: str = 'context',
        save_name: Optional[str] = None
    ):
        """
        Visualize embeddings in 2D using t-SNE or PCA.

        Args:
            encoder: Trained encoder model
            segments: List of TrajectorySegment objects
            method: Dimensionality reduction method ('tsne' or 'pca')
            title: Plot title
            color_by: How to color points ('context' or 'reward')
            save_name: Filename to save plot
        """
        encoder.eval()
        device = next(encoder.parameters()).device

        # Extract embeddings
        embeddings = []
        context_ids = []
        avg_rewards = []

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
            avg_rewards.append(seg.rewards.mean())

        embeddings = np.concatenate(embeddings, axis=0)
        context_ids = np.array(context_ids)
        avg_rewards = np.array(avg_rewards)

        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            reduced = reducer.fit_transform(embeddings)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))

        if color_by == 'context':
            scatter = ax.scatter(
                reduced[:, 0],
                reduced[:, 1],
                c=context_ids,
                cmap='tab20',
                alpha=0.6,
                s=50
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Context ID', fontsize=12)

        elif color_by == 'reward':
            scatter = ax.scatter(
                reduced[:, 0],
                reduced[:, 1],
                c=avg_rewards,
                cmap='viridis',
                alpha=0.6,
                s=50
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Average Reward', fontsize=12)

        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

        return reduced, context_ids

    def plot_context_separation(
        self,
        embeddings: np.ndarray,
        context_ids: np.ndarray,
        context_params: List[Dict[str, float]],
        save_name: Optional[str] = None
    ):
        """
        Plot similarity matrix to show context separation.

        Args:
            embeddings: Embedding vectors (N, latent_dim)
            context_ids: Context IDs (N,)
            context_params: List of context parameter dictionaries
            save_name: Filename to save plot
        """
        unique_contexts = np.unique(context_ids)

        # Compute mean embedding per context
        mean_embeddings = []
        for ctx in unique_contexts:
            ctx_mask = context_ids == ctx
            mean_emb = embeddings[ctx_mask].mean(axis=0)
            mean_embeddings.append(mean_emb)

        mean_embeddings = np.array(mean_embeddings)

        # Compute similarity matrix
        similarity_matrix = np.dot(mean_embeddings, mean_embeddings.T)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            ax=ax,
            cbar_kws={'label': 'Cosine Similarity'}
        )

        ax.set_xlabel('Context ID', fontsize=12)
        ax.set_ylabel('Context ID', fontsize=12)
        ax.set_title('Context Embedding Similarity Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        metrics: Optional[Dict[str, List[float]]] = None,
        save_name: Optional[str] = None
    ):
        """
        Plot training and validation curves.

        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch
            metrics: Additional metrics to plot
            save_name: Filename to save plot
        """
        n_plots = 1 + (1 if val_losses else 0) + (len(metrics) if metrics else 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

        if n_plots == 1:
            axes = [axes]

        # Training loss
        axes[0].plot(train_losses, label='Train Loss', linewidth=2)
        if val_losses:
            axes[0].plot(val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Curves')
        axes[0].legend()
        axes[0].grid(True)

        # Additional metrics
        if metrics:
            for idx, (metric_name, values) in enumerate(metrics.items(), start=1):
                axes[idx].plot(values, linewidth=2, color='green')
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel(metric_name)
                axes[idx].set_title(f'{metric_name} over Training')
                axes[idx].grid(True)

        plt.tight_layout()

        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def plot_baseline_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        save_name: Optional[str] = None
    ):
        """
        Plot comparison between different baselines.

        Args:
            results: Dictionary mapping method name to metrics
            save_name: Filename to save plot
        """
        methods = list(results.keys())
        mean_rewards = [results[m]['mean_reward'] for m in methods]
        std_rewards = [results[m]['std_reward'] for m in methods]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(methods))
        bars = ax.bar(x, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)

        # Color bars
        colors = ['blue', 'orange', 'green', 'red']
        for bar, color in zip(bars, colors[:len(bars)]):
            bar.set_color(color)

        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Mean Reward', fontsize=12)
        ax.set_title('Baseline Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def plot_context_parameter_correlation(
        self,
        embeddings: np.ndarray,
        context_params: List[Dict[str, float]],
        param_name: str,
        save_name: Optional[str] = None
    ):
        """
        Plot correlation between embedding dimensions and context parameters.

        Args:
            embeddings: Embedding vectors (N, latent_dim)
            context_params: List of context parameter dictionaries
            param_name: Name of parameter to correlate (e.g., 'gravity')
            save_name: Filename to save plot
        """
        # Extract parameter values
        param_values = np.array([ctx[param_name] for ctx in context_params])

        # Use PCA to get top 2 components
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx in range(2):
            scatter = axes[idx].scatter(
                param_values,
                reduced[:, idx],
                alpha=0.6,
                s=50,
                c=param_values,
                cmap='viridis'
            )
            axes[idx].set_xlabel(param_name, fontsize=12)
            axes[idx].set_ylabel(f'PC{idx+1}', fontsize=12)
            axes[idx].set_title(f'PC{idx+1} vs {param_name}', fontsize=14)
            plt.colorbar(scatter, ax=axes[idx])

        plt.tight_layout()

        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()
