"""Trajectory collection and dataset for contrastive learning"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path
from tqdm import tqdm
import gymnasium as gym

from .carl_envs import make_carl_env, ContextSampler


def _extract_obs(obs):
    """Extract observation array from CARL environment output.

    CARL environments may return dict-based observations with 'obs' key.
    This function handles both dict and array formats.
    """
    if isinstance(obs, dict):
        return obs.get('obs', obs.get('observation', obs))
    return obs


class TrajectorySegment:
    """A single trajectory segment with metadata"""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        context_id: int,
        context_params: Dict[str, float]
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.context_id = context_id
        self.context_params = context_params

    def __len__(self):
        return len(self.observations)


class TrajectoryCollector:
    """Collect trajectory segments from CARL environments for contrastive learning"""

    def __init__(
        self,
        env_name: str,
        contexts: List[Dict[str, float]],
        segment_length: int = 32,
        num_segments_per_context: int = 100,
        policy: str = "random",
        seed: int = 42
    ):
        """
        Args:
            env_name: Name of CARL environment
            contexts: List of context dictionaries to sample from
            segment_length: Length of each trajectory segment
            num_segments_per_context: Number of segments to collect per context
            policy: Policy for action selection ('random', 'trained', or path to model)
            seed: Random seed
        """
        self.env_name = env_name
        self.contexts = contexts
        self.segment_length = segment_length
        self.num_segments_per_context = num_segments_per_context
        self.policy = policy
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

    def _get_action(self, env: gym.Env, obs: np.ndarray) -> np.ndarray:
        """Get action from policy"""
        if self.policy == "random":
            return env.action_space.sample()
        elif self.policy == "trained":
            # TODO: Load and use trained policy
            raise NotImplementedError("Trained policy not yet implemented")
        else:
            # Assume it's a path to a saved model
            raise NotImplementedError("Custom policy loading not yet implemented")

    def collect(self, verbose: bool = True) -> List[TrajectorySegment]:
        """
        Collect trajectory segments from all contexts.

        Returns:
            List of TrajectorySegment objects
        """
        all_segments = []

        iterator = enumerate(self.contexts)
        if verbose:
            iterator = tqdm(
                iterator,
                total=len(self.contexts),
                desc="Collecting trajectories"
            )

        for context_id, context in iterator:
            # Create environment with this context
            env = make_carl_env(self.env_name, context)

            segments_collected = 0
            while segments_collected < self.num_segments_per_context:
                # Run episode and collect segments
                obs_raw, _ = env.reset(seed=self.seed + context_id * 10000 + segments_collected)
                obs = _extract_obs(obs_raw)
                done = False
                truncated = False

                # Storage for current episode
                ep_obs = [obs]
                ep_actions = []
                ep_rewards = []

                while not (done or truncated):
                    action = self._get_action(env, obs_raw)
                    obs_raw, reward, done, truncated, _ = env.step(action)
                    obs = _extract_obs(obs_raw)

                    ep_obs.append(obs)
                    ep_actions.append(action)
                    ep_rewards.append(reward)

                # Convert to numpy arrays
                ep_obs = np.array(ep_obs)
                ep_actions = np.array(ep_actions)
                ep_rewards = np.array(ep_rewards)

                # Split episode into segments
                episode_length = len(ep_actions)
                if episode_length < self.segment_length:
                    # Episode too short, skip or pad
                    continue

                # Create non-overlapping segments
                num_segments = episode_length // self.segment_length
                for i in range(num_segments):
                    if segments_collected >= self.num_segments_per_context:
                        break

                    start_idx = i * self.segment_length
                    end_idx = start_idx + self.segment_length

                    segment = TrajectorySegment(
                        observations=ep_obs[start_idx:end_idx],
                        actions=ep_actions[start_idx:end_idx],
                        rewards=ep_rewards[start_idx:end_idx],
                        context_id=context_id,
                        context_params=context
                    )

                    all_segments.append(segment)
                    segments_collected += 1

            env.close()

        return all_segments

    def save(self, segments: List[TrajectorySegment], path: str):
        """Save collected segments to disk"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(segments, f)

        print(f"Saved {len(segments)} segments to {path}")

    @staticmethod
    def load(path: str) -> List[TrajectorySegment]:
        """Load segments from disk"""
        with open(path, 'rb') as f:
            segments = pickle.load(f)
        print(f"Loaded {len(segments)} segments from {path}")
        return segments


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for contrastive learning on trajectory segments"""

    def __init__(
        self,
        segments: List[TrajectorySegment],
        augmentation: Optional[str] = None
    ):
        """
        Args:
            segments: List of trajectory segments
            augmentation: Type of data augmentation ('noise', 'dropout', None)
        """
        self.segments = segments
        self.augmentation = augmentation

        # Group segments by context for positive pair sampling
        self.context_to_segments = {}
        for idx, seg in enumerate(segments):
            if seg.context_id not in self.context_to_segments:
                self.context_to_segments[seg.context_id] = []
            self.context_to_segments[seg.context_id].append(idx)

    def __len__(self):
        return len(self.segments)

    def _augment(self, obs: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        if self.augmentation == "noise":
            obs = obs + np.random.normal(0, 0.01, obs.shape)
            actions = actions + np.random.normal(0, 0.01, actions.shape)
        elif self.augmentation == "dropout":
            # Randomly zero out some timesteps
            mask = np.random.binomial(1, 0.9, obs.shape[0])
            obs = obs * mask[:, None]
            actions = actions * mask[:, None] if len(actions.shape) > 1 else actions * mask
        return obs, actions

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample with anchor, positive, and negative segments.

        Returns:
            Dictionary with:
                - anchor: (segment_length, obs_dim + action_dim)
                - positive: (segment_length, obs_dim + action_dim)
                - negative: (segment_length, obs_dim + action_dim)
                - context_id: int
        """
        # Anchor segment
        anchor_seg = self.segments[idx]

        # Sample positive: different segment from SAME context
        positive_candidates = self.context_to_segments[anchor_seg.context_id]
        positive_candidates = [i for i in positive_candidates if i != idx]
        if len(positive_candidates) == 0:
            # If only one segment for this context, use the same one
            positive_idx = idx
        else:
            positive_idx = np.random.choice(positive_candidates)
        positive_seg = self.segments[positive_idx]

        # Sample negative: segment from DIFFERENT context
        negative_context = np.random.choice(
            [c for c in self.context_to_segments.keys() if c != anchor_seg.context_id]
        )
        negative_idx = np.random.choice(self.context_to_segments[negative_context])
        negative_seg = self.segments[negative_idx]

        # Concatenate observations and actions
        def segment_to_tensor(seg: TrajectorySegment) -> torch.Tensor:
            obs = seg.observations
            actions = seg.actions
            if self.augmentation:
                obs, actions = self._augment(obs, actions)

            # Handle different action shapes
            if len(actions.shape) == 1:
                actions = actions[:, None]

            combined = np.concatenate([obs, actions], axis=-1)
            return torch.FloatTensor(combined)

        return {
            "anchor": segment_to_tensor(anchor_seg),
            "positive": segment_to_tensor(positive_seg),
            "negative": segment_to_tensor(negative_seg),
            "context_id": anchor_seg.context_id
        }
