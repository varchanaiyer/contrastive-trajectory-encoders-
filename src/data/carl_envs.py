"""CARL environment wrappers and context distributions"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import gymnasium as gym

# Import CARL environments with robust error handling for Colab compatibility
# Note: We import directly from submodules to avoid CARL's __init__.py which may
# have broken imports (e.g., CARLMarioEnv) that prevent importing anything
try:
    # Try the most specific path first (CARL v1.1+)
    from carl.envs.gymnasium.classic_control.carl_pendulum import CARLPendulum
    from carl.envs.gymnasium.classic_control.carl_cartpole import CARLCartPole
except ImportError:
    try:
        # Try older module structure
        from carl.envs.gymnasium.classic_control import CARLPendulum, CARLCartPole
    except ImportError:
        try:
            # Last resort - may fail due to mario import issue
            from carl.envs import CARLPendulum, CARLCartPole
        except ImportError as e:
            raise ImportError(
                f"Failed to import CARL environments. "
                f"Try: pip install carl-bench gymnasium. Error: {e}"
            )

MUJOCO_AVAILABLE = False
CARLAnt = None
CARLHalfCheetah = None
try:
    # Try direct module imports to avoid broken __init__.py
    from carl.envs.gymnasium.mujoco.carl_ant import CARLAnt
    from carl.envs.gymnasium.mujoco.carl_halfcheetah import CARLHalfCheetah
    MUJOCO_AVAILABLE = True
except ImportError:
    try:
        from carl.envs.gymnasium.mujoco import CARLAnt, CARLHalfCheetah
        MUJOCO_AVAILABLE = True
    except ImportError:
        try:
            from carl.envs import CARLAnt, CARLHalfCheetah
            MUJOCO_AVAILABLE = True
        except ImportError:
            pass  # MuJoCo environments not available


def get_context_distributions(
    env_name: str,
    split: str = "train"
) -> List[Dict[str, float]]:
    """
    Get context distributions for train/test splits.

    Args:
        env_name: Name of the environment (ant, pendulum, cartpole)
        split: 'train' or 'test' (OOD contexts)

    Returns:
        List of context dictionaries with environment parameters
    """
    if env_name.lower() == "pendulum":
        if split == "train":
            # Train on moderate gravity range
            return [
                {"g": g}
                for g in np.linspace(5.0, 15.0, 20)
            ]
        else:  # test
            # Test on extreme gravity (OOD)
            return [
                {"g": g}
                for g in [3.0, 4.0, 18.0, 20.0, 25.0]
            ]

    elif env_name.lower() == "cartpole":
        if split == "train":
            # Train on varied pole lengths and masses
            contexts = []
            for length in np.linspace(0.3, 0.7, 5):
                for gravity in np.linspace(7.0, 13.0, 4):
                    contexts.append({
                        "length": float(length),
                        "gravity": float(gravity)
                    })
            return contexts
        else:  # test
            return [
                {"length": 0.2, "gravity": 9.8},
                {"length": 0.9, "gravity": 9.8},
                {"length": 0.5, "gravity": 5.0},
                {"length": 0.5, "gravity": 15.0},
            ]

    elif env_name.lower() == "ant":
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo environments not available")

        if split == "train":
            contexts = []
            for gravity in np.linspace(-15.0, -5.0, 5):
                for friction in np.linspace(0.5, 1.5, 4):
                    contexts.append({
                        "gravity": float(gravity),
                        "friction": float(friction)
                    })
            return contexts
        else:  # test
            return [
                {"gravity": -20.0, "friction": 0.8},
                {"gravity": -3.0, "friction": 0.8},
                {"gravity": -10.0, "friction": 0.2},
                {"gravity": -10.0, "friction": 2.0},
            ]

    else:
        raise ValueError(f"Unknown environment: {env_name}")


def make_carl_env(
    env_name: str,
    context: Optional[Dict[str, float]] = None,
    render_mode: Optional[str] = None
) -> gym.Env:
    """
    Create a CARL environment with specified context.

    Args:
        env_name: Name of the environment
        context: Dictionary of context parameters (e.g., {"g": 10.0})
        render_mode: Rendering mode for gymnasium

    Returns:
        Configured CARL environment
    """
    env_name = env_name.lower()

    # CARL v1.1+ requires contexts as a dict with context_id keys
    contexts = {0: context} if context else None
    kwargs = {"obs_context_features": [], "obs_context_as_dict": False}
    if render_mode:
        kwargs["render_mode"] = render_mode

    if env_name == "pendulum":
        env = CARLPendulum(contexts=contexts, **kwargs)

    elif env_name == "cartpole":
        env = CARLCartPole(contexts=contexts, **kwargs)

    elif env_name == "ant":
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo not available. Install with: pip install mujoco-py")
        env = CARLAnt(contexts=contexts, **kwargs)

    elif env_name == "halfcheetah":
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo not available")
        env = CARLHalfCheetah(contexts=contexts, **kwargs)

    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    return env


class ContextSampler:
    """Sample contexts for curriculum learning or random sampling"""

    def __init__(self, contexts: List[Dict[str, float]], mode: str = "uniform"):
        """
        Args:
            contexts: List of context dictionaries
            mode: Sampling mode ('uniform', 'curriculum', 'adaptive')
        """
        self.contexts = contexts
        self.mode = mode
        self.weights = np.ones(len(contexts)) / len(contexts)

    def sample(self) -> Tuple[int, Dict[str, float]]:
        """
        Sample a context based on the current strategy.

        Returns:
            Tuple of (context_id, context_dict)
        """
        if self.mode == "uniform":
            idx = np.random.randint(len(self.contexts))
        elif self.mode == "curriculum":
            # Sample easier contexts early in training
            idx = np.random.choice(len(self.contexts), p=self.weights)
        else:
            idx = np.random.randint(len(self.contexts))

        return idx, self.contexts[idx]

    def update_weights(self, performances: Dict[int, float]):
        """
        Update sampling weights based on performance (for adaptive sampling).

        Args:
            performances: Dict mapping context_id to recent performance metric
        """
        if self.mode == "adaptive":
            # Sample harder contexts more frequently
            for ctx_id, perf in performances.items():
                if ctx_id < len(self.weights):
                    # Lower performance = higher weight
                    self.weights[ctx_id] = 1.0 / (perf + 0.1)

            # Normalize
            self.weights = self.weights / self.weights.sum()
