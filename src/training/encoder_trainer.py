"""Phase 1: Contrastive training for trajectory encoder"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available")


class EncoderTrainer:
    """Trainer for Phase 1: Learning trajectory embeddings via contrastive learning"""

    def __init__(
        self,
        encoder: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Args:
            encoder: Trajectory encoder model
            loss_fn: Contrastive loss function (e.g., InfoNCELoss)
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to train on
            log_dir: Directory for tensorboard logs
            checkpoint_dir: Directory to save checkpoints
        """
        self.encoder = encoder.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            encoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 100,  # Assuming max 100 epochs
            eta_min=learning_rate * 0.01
        )

        # Logging
        self.log_dir = Path(log_dir) if log_dir else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = None
        if TENSORBOARD_AVAILABLE and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.encoder.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            # Move to device
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negative'].to(self.device)

            # Forward pass
            anchor_emb = self.encoder(anchor)
            positive_emb = self.encoder(positive)
            negative_emb = self.encoder(negative)

            # Compute loss
            if hasattr(self.loss_fn, 'forward'):
                # InfoNCE with explicit negatives
                loss = self.loss_fn(anchor_emb, positive_emb, negative_emb.unsqueeze(1))
            else:
                loss = self.loss_fn(anchor_emb, positive_emb)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

            if self.writer:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        if self.val_loader is None:
            return {}

        self.encoder.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negative'].to(self.device)

            anchor_emb = self.encoder(anchor)
            positive_emb = self.encoder(positive)
            negative_emb = self.encoder(negative)

            if hasattr(self.loss_fn, 'forward'):
                loss = self.loss_fn(anchor_emb, positive_emb, negative_emb.unsqueeze(1))
            else:
                loss = self.loss_fn(anchor_emb, positive_emb)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if self.writer:
            self.writer.add_scalar('val/loss', avg_loss, self.epoch)

        return {'val_loss': avg_loss}

    @torch.no_grad()
    def compute_embedding_quality_metrics(self) -> Dict[str, float]:
        """
        Compute metrics to assess embedding quality:
        - Intra-context similarity (should be high)
        - Inter-context similarity (should be low)
        """
        if self.val_loader is None:
            return {}

        self.encoder.eval()

        embeddings = []
        context_ids = []

        for batch in self.val_loader:
            anchor = batch['anchor'].to(self.device)
            ctx_id = batch['context_id'].cpu().numpy()

            emb = self.encoder(anchor)
            embeddings.append(emb.cpu().numpy())
            context_ids.append(ctx_id)

        embeddings = np.concatenate(embeddings, axis=0)
        context_ids = np.concatenate(context_ids, axis=0)

        # Compute intra-context similarity
        unique_contexts = np.unique(context_ids)
        intra_similarities = []

        for ctx in unique_contexts:
            ctx_mask = context_ids == ctx
            ctx_embeddings = embeddings[ctx_mask]

            if len(ctx_embeddings) > 1:
                # Compute pairwise cosine similarity
                similarity_matrix = np.dot(ctx_embeddings, ctx_embeddings.T)
                # Exclude diagonal
                mask = ~np.eye(len(ctx_embeddings), dtype=bool)
                intra_sim = similarity_matrix[mask].mean()
                intra_similarities.append(intra_sim)

        avg_intra_similarity = np.mean(intra_similarities) if intra_similarities else 0.0

        # Compute inter-context similarity
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

        # Separation metric (higher is better)
        separation = avg_intra_similarity - avg_inter_similarity

        metrics = {
            'intra_similarity': avg_intra_similarity,
            'inter_similarity': avg_inter_similarity,
            'separation': separation
        }

        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'metrics/{key}', value, self.epoch)

        return metrics

    def save_checkpoint(self, filename: str = 'checkpoint.pt', is_best: bool = False):
        """Save training checkpoint"""
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)

        if is_best:
            best_path = self.checkpoint_dir / 'best_encoder.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from epoch {self.epoch}")

    def train(self, num_epochs: int, eval_every: int = 1):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
            eval_every: Evaluate every N epochs
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Validation batches: {len(self.val_loader)}")

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}")

            # Validate
            if self.val_loader and (epoch % eval_every == 0):
                val_metrics = self.validate()
                quality_metrics = self.compute_embedding_quality_metrics()

                print(f"Validation Loss = {val_metrics['val_loss']:.4f}")
                print(f"Embedding Quality: Intra={quality_metrics['intra_similarity']:.4f}, "
                      f"Inter={quality_metrics['inter_similarity']:.4f}, "
                      f"Separation={quality_metrics['separation']:.4f}")

                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint(is_best=True)

            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(filename=f'checkpoint_epoch_{epoch}.pt')

        print("Training complete!")
        if self.writer:
            self.writer.close()
