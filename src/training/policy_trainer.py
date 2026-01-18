"""Phase 2: Training context-conditional policies with PPO/SAC"""

import torch
import numpy as np
from typing import Dict, Optional, Callable, List
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import pickle

from ..data.carl_envs import make_carl_env, ContextSampler
from ..models.policies import ContextEncoder


class ContextBufferWrapper(gym.Wrapper):
    """
    Wrapper that maintains a buffer of recent transitions and computes
    context embedding on-the-fly using the trajectory encoder.
    """

    def __init__(
        self,
        env: gym.Env,
        context_encoder: ContextEncoder,
        buffer_length: int = 32,
        device: str = 'cpu'
    ):
        """
        Args:
            env: Base environment
            context_encoder: Encoder to compute context from trajectory
            buffer_length: Length of trajectory buffer for encoding
            device: Device for encoder inference
        """
        super().__init__(env)

        self.context_encoder = context_encoder
        self.buffer_length = buffer_length
        self.device = device

        # Buffer for observations and actions
        self.obs_buffer = []
        self.action_buffer = []

        # Cached context embedding
        self.current_context = None

        # Modify observation space to include context
        obs_dim = env.observation_space.shape[0]
        context_dim = context_encoder.encoder.latent_dim
        self.original_obs_space = env.observation_space

        # New observation space is [obs, context]
        low = np.concatenate([env.observation_space.low, np.full(context_dim, -np.inf)])
        high = np.concatenate([env.observation_space.high, np.full(context_dim, np.inf)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        """Reset environment and buffers"""
        obs, info = self.env.reset(**kwargs)

        self.obs_buffer = [obs]
        self.action_buffer = []
        self.current_context = None

        # Return obs with zero context initially
        context_dim = self.context_encoder.encoder.latent_dim
        zero_context = np.zeros(context_dim, dtype=np.float32)
        return np.concatenate([obs, zero_context]), info

    def _compute_context(self) -> np.ndarray:
        """Compute context embedding from current buffer"""
        if len(self.action_buffer) == 0:
            # No actions yet, return zero context
            context_dim = self.context_encoder.encoder.latent_dim
            return np.zeros(context_dim, dtype=np.float32)

        # Prepare trajectory tensor
        obs = np.array(self.obs_buffer[:-1])  # All but last obs
        actions = np.array(self.action_buffer)

        # Handle action shape
        if len(actions.shape) == 1:
            actions = actions[:, None]

        # Concatenate
        trajectory = np.concatenate([obs, actions], axis=-1)

        # Convert to tensor and encode
        traj_tensor = torch.FloatTensor(trajectory).unsqueeze(0).to(self.device)

        with torch.no_grad():
            context = self.context_encoder(traj_tensor)

        return context.squeeze(0).cpu().numpy()

    def step(self, action):
        """Step environment and update context"""
        obs, reward, done, truncated, info = self.env.step(action)

        # Update buffers
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)

        # Trim buffers to max length
        if len(self.obs_buffer) > self.buffer_length:
            self.obs_buffer.pop(0)
            self.action_buffer.pop(0)

        # Compute context if buffer is full
        if len(self.action_buffer) >= self.buffer_length:
            self.current_context = self._compute_context()
        elif self.current_context is None:
            context_dim = self.context_encoder.encoder.latent_dim
            self.current_context = np.zeros(context_dim, dtype=np.float32)

        # Return augmented observation
        augmented_obs = np.concatenate([obs, self.current_context])

        return augmented_obs, reward, done, truncated, info


def make_contextual_env(
    env_name: str,
    context: Dict[str, float],
    encoder: ContextEncoder,
    buffer_length: int = 32,
    device: str = 'cpu'
) -> gym.Env:
    """Create environment with context wrapper"""
    base_env = make_carl_env(env_name, context)
    base_env = Monitor(base_env)
    wrapped_env = ContextBufferWrapper(base_env, encoder, buffer_length, device)
    return wrapped_env


class PolicyTrainer:
    """Trainer for Phase 2: Learning context-conditional policies"""

    def __init__(
        self,
        env_name: str,
        encoder: ContextEncoder,
        contexts: List[Dict[str, float]],
        algorithm: str = 'ppo',
        buffer_length: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Args:
            env_name: Name of CARL environment
            encoder: Pre-trained trajectory encoder
            contexts: List of training contexts
            algorithm: RL algorithm ('ppo' or 'sac')
            buffer_length: Length of trajectory buffer for context encoding
            device: Device for training
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.env_name = env_name
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.contexts = contexts
        self.algorithm = algorithm.lower()
        self.buffer_length = buffer_length
        self.device = device

        self.log_dir = Path(log_dir) if log_dir else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Will be initialized in train()
        self.policy = None
        self.context_sampler = ContextSampler(contexts, mode='uniform')

    def make_env(self, context: Dict[str, float]) -> Callable:
        """Create environment factory for vectorized environments"""
        def _init():
            return make_contextual_env(
                self.env_name,
                context,
                self.encoder,
                self.buffer_length,
                self.device
            )
        return _init

    def train(
        self,
        total_timesteps: int = 1_000_000,
        eval_contexts: Optional[List[Dict[str, float]]] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        policy_kwargs: Optional[Dict] = None
    ):
        """
        Train the context-conditional policy.

        Args:
            total_timesteps: Total training timesteps
            eval_contexts: Contexts for evaluation (OOD test set)
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of episodes per evaluation
            policy_kwargs: Additional policy arguments
        """
        print(f"Training {self.algorithm.upper()} with context-conditional policy")
        print(f"Training contexts: {len(self.contexts)}")
        print(f"Device: {self.device}")

        # Create training environment (sample contexts randomly)
        # For simplicity, use a single environment for now
        train_context = self.context_sampler.sample()[1]
        train_env = self.make_env(train_context)()

        # Create evaluation environment if provided
        eval_env = None
        if eval_contexts:
            eval_context = eval_contexts[0]
            eval_env = self.make_env(eval_context)()

        # Default policy kwargs
        if policy_kwargs is None:
            policy_kwargs = {
                'net_arch': [256, 256],
                'activation_fn': torch.nn.ReLU
            }

        # Create RL algorithm
        if self.algorithm == 'ppo':
            self.policy = PPO(
                'MlpPolicy',
                train_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=str(self.log_dir) if self.log_dir else None,
                device=self.device
            )
        elif self.algorithm == 'sac':
            self.policy = SAC(
                'MlpPolicy',
                train_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=str(self.log_dir) if self.log_dir else None,
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        # Setup callbacks
        callbacks = []

        if self.checkpoint_dir:
            checkpoint_callback = CheckpointCallback(
                save_freq=50000,
                save_path=str(self.checkpoint_dir),
                name_prefix='policy'
            )
            callbacks.append(checkpoint_callback)

        if eval_env:
            eval_callback = EvalCallback(
                eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                best_model_save_path=str(self.checkpoint_dir) if self.checkpoint_dir else None,
                log_path=str(self.log_dir) if self.log_dir else None,
                deterministic=True
            )
            callbacks.append(eval_callback)

        # Train
        self.policy.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )

        print("Training complete!")

        # Save final model
        if self.checkpoint_dir:
            save_path = self.checkpoint_dir / 'final_policy.zip'
            self.policy.save(save_path)
            print(f"Saved final policy to {save_path}")

    def evaluate(
        self,
        contexts: List[Dict[str, float]],
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate policy on given contexts.

        Args:
            contexts: List of contexts to evaluate on
            n_episodes: Number of episodes per context
            deterministic: Use deterministic policy

        Returns:
            Dictionary with evaluation metrics
        """
        if self.policy is None:
            raise ValueError("Policy not trained yet")

        results = {}

        for ctx_id, context in enumerate(contexts):
            env = self.make_env(context)()
            episode_rewards = []

            for _ in range(n_episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0

                while not done:
                    action, _ = self.policy.predict(obs, deterministic=deterministic)
                    obs, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward
                    if truncated:
                        break

                episode_rewards.append(episode_reward)

            env.close()

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)

            results[f'context_{ctx_id}'] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'context': context
            }

            print(f"Context {ctx_id} {context}: {mean_reward:.2f} ± {std_reward:.2f}")

        return results
