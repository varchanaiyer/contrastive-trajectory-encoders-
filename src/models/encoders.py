"""Trajectory encoder architectures"""

import torch
import torch.nn as nn
from typing import Optional
import math


class LSTMEncoder(nn.Module):
    """LSTM-based trajectory encoder"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 64,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        """
        Args:
            input_dim: Dimension of input (obs_dim + action_dim)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            latent_dim: Dimension of output embedding
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Output projection
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

        # L2 normalization for contrastive learning
        self.normalize = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            x: Input trajectory (batch_size, seq_len, input_dim)

        Returns:
            Latent embedding (batch_size, latent_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state (or concatenate forward/backward)
        if self.bidirectional:
            # Concatenate final hidden states from both directions
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h_final = torch.cat([h_forward, h_backward], dim=-1)
        else:
            h_final = h_n[-1]

        # Project to latent space
        z = self.output_proj(h_final)

        # L2 normalize for contrastive learning
        if self.normalize:
            z = nn.functional.normalize(z, p=2, dim=-1)

        return z


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    """Transformer-based trajectory encoder"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        latent_dim: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        """
        Args:
            input_dim: Dimension of input (obs_dim + action_dim)
            hidden_dim: Hidden dimension of transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            latent_dim: Dimension of output embedding
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection with pooling
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

        # L2 normalization
        self.normalize = True

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer encoder.

        Args:
            x: Input trajectory (batch_size, seq_len, input_dim)
            mask: Attention mask (batch_size, seq_len)

        Returns:
            Latent embedding (batch_size, latent_dim)
        """
        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Global average pooling over sequence
        if mask is not None:
            # Masked average pooling
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)

        # Project to latent space
        z = self.output_proj(x)

        # L2 normalize
        if self.normalize:
            z = nn.functional.normalize(z, p=2, dim=-1)

        return z


class EncoderEnsemble(nn.Module):
    """Ensemble of encoders for robust embeddings"""

    def __init__(self, encoders: list):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.latent_dim = encoders[0].latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average embeddings from all encoders"""
        embeddings = [encoder(x) for encoder in self.encoders]
        z = torch.stack(embeddings, dim=0).mean(dim=0)
        return nn.functional.normalize(z, p=2, dim=-1)
