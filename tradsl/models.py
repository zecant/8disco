"""
Model Interfaces for TradSL

Section 10: BaseTrainableModel and BaseAgentArchitecture interfaces.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from dataclasses import dataclass
from enum import Enum


class TradingAction(Enum):
    """Discrete action space actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    FLATTEN = "flatten"


@dataclass
class ModelSpec:
    """Specification for a registered model."""
    cls: type
    description: str


class BaseTrainableModel(ABC):
    """
    Section 10.4.2: Supervised model interface.
    
    Trainable models fit to historical data and produce
    intermediate outputs consumed by agent architectures.
    """
    
    @abstractmethod
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit model to historical features and labels.
        
        Args:
            features: Array of shape (n_samples, n_features)
            labels: Array of shape (n_samples,)
        
        Constraints:
            - Must complete synchronously
            - Must not access data outside provided arrays
        """
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> float:
        """
        Produce output for current bar.
        
        Args:
            features: Array of shape (n_features,)
        
        Returns:
            Scalar float (probability, value, etc.)
            0.0 if not fitted
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save model state atomically."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model state."""
        pass
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """True after at least one successful fit()."""
        pass


class BaseAgentArchitecture(ABC):
    """
    Section 10.6.2: RL agent architecture interface.
    
    Agent architectures receive observations and produce
    trading actions with conviction.
    """
    
    @abstractmethod
    def observe(
        self,
        observation: np.ndarray,
        portfolio_state: Dict[str, float],
        bar_index: int
    ) -> Tuple[TradingAction, float]:
        """
        Generate action and conviction from observation.
        
        Args:
            observation: Feature vector from DAG
            portfolio_state: Dict of portfolio metrics
            bar_index: Bar counter since block start
        
        Returns:
            Tuple of (action, conviction)
            action: TradingAction enum
            conviction: float in [0, 1]
        """
        pass
    
    @abstractmethod
    def record_experience(
        self,
        observation: np.ndarray,
        portfolio_state: Dict[str, float],
        action: TradingAction,
        conviction: float,
        reward: float,
        next_observation: np.ndarray,
        next_portfolio_state: Dict[str, float],
        done: bool
    ) -> None:
        """Record experience in replay buffer."""
        pass
    
    @abstractmethod
    def update_policy(self, bar_index: int) -> None:
        """Update policy using replay buffer."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save agent state."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load agent state."""
        pass
    
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """True after at least one policy update."""
        pass
    
    def should_update(self, bar_index: int, recent_performance: float = 0.0) -> bool:
        """
        Determine if policy should update.
        
        Override for custom update schedules.
        Default: every_n_bars behavior.
        """
        return False


class ReplayBuffer(ABC):
    """Section 10.7: Experience replay buffer interface."""
    
    @abstractmethod
    def add(
        self,
        observation: np.ndarray,
        portfolio_state: Dict[str, float],
        action: TradingAction,
        conviction: float,
        reward: float,
        next_observation: np.ndarray,
        next_portfolio_state: Dict[str, float],
        done: bool
    ) -> None:
        """Add experience to buffer."""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample batch of experiences."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear buffer (for block boundaries)."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Current buffer size."""
        pass


# Model registry
MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(name: str, model_cls: type, description: str = "") -> None:
    """Register a model class."""
    MODEL_REGISTRY[name] = ModelSpec(cls=model_cls, description=description)


def get_model(name: str) -> Optional[type]:
    """Get model class by name."""
    spec = MODEL_REGISTRY.get(name)
    return spec.cls if spec else None
