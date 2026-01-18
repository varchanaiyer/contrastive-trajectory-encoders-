"""Contrastive loss functions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""

    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        """
        Args:
            temperature: Temperature parameter for softmax
            reduction: Reduction mode ('mean', 'sum', 'none')
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchor: Anchor embeddings (batch_size, latent_dim)
            positive: Positive embeddings (batch_size, latent_dim)
            negatives: Negative embeddings (batch_size, num_negatives, latent_dim)
                      If None, uses in-batch negatives

        Returns:
            Loss value
        """
        batch_size = anchor.shape[0]

        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)

        # Compute positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # (batch_size,)

        if negatives is not None:
            # Explicit negatives provided
            negatives = F.normalize(negatives, p=2, dim=-1)
            # anchor: (batch_size, latent_dim)
            # negatives: (batch_size, num_negatives, latent_dim)
            neg_sim = torch.bmm(
                anchor.unsqueeze(1),
                negatives.transpose(1, 2)
            ).squeeze(1) / self.temperature  # (batch_size, num_negatives)

            # Concatenate positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch_size, 1 + num_negatives)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        else:
            # Use in-batch negatives
            # Compute all pairwise similarities
            all_embeddings = torch.cat([anchor, positive], dim=0)  # (2*batch_size, latent_dim)
            sim_matrix = torch.mm(
                all_embeddings, all_embeddings.t()
            ) / self.temperature  # (2*batch_size, 2*batch_size)

            # Create labels: anchor[i] matches with positive[i]
            labels = torch.arange(batch_size, device=anchor.device)
            labels = torch.cat([labels + batch_size, labels], dim=0)  # (2*batch_size,)

            # Mask out self-similarity
            mask = torch.eye(2 * batch_size, dtype=torch.bool, device=anchor.device)
            sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

            logits = sim_matrix

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard negative mining"""

    def __init__(self, margin: float = 1.0, mining: str = 'hard'):
        """
        Args:
            margin: Margin for triplet loss
            mining: Mining strategy ('hard', 'semi-hard', 'none')
        """
        super().__init__()
        self.margin = margin
        self.mining = mining

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings (batch_size, latent_dim)
            positive: Positive embeddings (batch_size, latent_dim)
            negative: Negative embeddings (batch_size, latent_dim)

        Returns:
            Loss value
        """
        # Normalize
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)

        # Compute distances
        pos_dist = torch.sum((anchor - positive) ** 2, dim=-1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=-1)

        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (can handle multiple positives per anchor)"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (batch_size, latent_dim)
            labels: (batch_size,) context IDs

        Returns:
            Loss value
        """
        batch_size = features.shape[0]
        features = F.normalize(features, p=2, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.mm(features, features.t()) / self.temperature

        # Create mask for positives (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()

        # Remove self-similarity
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        # For numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = -mean_log_prob_pos.mean()

        return loss
