"""Context-conditional policy architectures"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class ContextConditionalPolicy(BaseFeaturesExtractor):
    """
    Feature extractor that concatenates observations with context embeddings.
    Compatible with Stable-Baselines3 policies.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        context_dim: int,
        features_dim: int = 256,
        normalize_context: bool = True
    ):
        """
        Args:
            observation_space: Observation space from the environment
            context_dim: Dimension of context embedding
            features_dim: Dimension of output features
            normalize_context: Whether to L2-normalize context embeddings
        """
        # The features_dim is what SB3 will see as the feature output
        super().__init__(observation_space, features_dim)

        self.context_dim = context_dim
        self.normalize_context = normalize_context

        # Get observation dimension
        if isinstance(observation_space, gym.spaces.Box):
            obs_dim = observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported observation space: {observation_space}")

        # Network that processes [obs, context] concatenation
        self.net = nn.Sequential(
            nn.Linear(obs_dim + context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (batch_size, obs_dim)
            context: (batch_size, context_dim) or (context_dim,) for single context

        Returns:
            Features (batch_size, features_dim)
        """
        # Handle single context (broadcast to batch)
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(observations.shape[0], -1)

        # Normalize context if needed
        if self.normalize_context:
            context = nn.functional.normalize(context, p=2, dim=-1)

        # Concatenate and process
        combined = torch.cat([observations, context], dim=-1)
        return self.net(combined)


class ContextualMLPExtractor(BaseFeaturesExtractor):
    """
    MLP feature extractor with context conditioning.
    More flexible version that can be used with custom policies.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        context_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        features_dim: int = 256,
        activation: str = 'relu'
    ):
        super().__init__(observation_space, features_dim)

        self.context_dim = context_dim

        if isinstance(observation_space, gym.spaces.Box):
            obs_dim = observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported observation space: {observation_space}")

        # Build network
        layers = []
        input_dim = obs_dim + context_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, features_dim))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(observations.shape[0], -1)

        combined = torch.cat([observations, context], dim=-1)
        return self.net(combined)


class ContextEncoder(nn.Module):
    """
    Wrapper that combines trajectory encoder with policy.
    Used during Phase 2 training.
    """

    def __init__(
        self,
        trajectory_encoder: nn.Module,
        freeze_encoder: bool = True
    ):
        """
        Args:
            trajectory_encoder: Pre-trained encoder from Phase 1
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()

        self.encoder = trajectory_encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.freeze_encoder = freeze_encoder

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Encode trajectory to context embedding.

        Args:
            trajectory: (batch_size, seq_len, input_dim) or (seq_len, input_dim)

        Returns:
            Context embedding (batch_size, context_dim) or (context_dim,)
        """
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(trajectory)
        else:
            return self.encoder(trajectory)

    def train(self, mode: bool = True):
        """Override train mode to keep encoder frozen if specified"""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self


class OraclePolicy(BaseFeaturesExtractor):
    """
    Policy that receives ground-truth context parameters (upper bound baseline).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        context_param_names: Tuple[str, ...],
        features_dim: int = 256
    ):
        """
        Args:
            observation_space: Environment observation space
            context_param_names: Names of context parameters (e.g., ('gravity', 'friction'))
            features_dim: Output feature dimension
        """
        super().__init__(observation_space, features_dim)

        if isinstance(observation_space, gym.spaces.Box):
            obs_dim = observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported observation space: {observation_space}")

        context_dim = len(context_param_names)

        self.net = nn.Sequential(
            nn.Linear(obs_dim + context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor, context_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (batch_size, obs_dim)
            context_params: (batch_size, num_params) ground-truth parameters

        Returns:
            Features (batch_size, features_dim)
        """
        if context_params.dim() == 1:
            context_params = context_params.unsqueeze(0).expand(observations.shape[0], -1)

        combined = torch.cat([observations, context_params], dim=-1)
        return self.net(combined)
