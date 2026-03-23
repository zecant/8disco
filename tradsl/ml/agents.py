"""
Agents for TradSL.

Example usage in DSL:
    agent:
      type=function
      function=ml.agents.dummy
      inputs=[state]
      warmup=100
"""
import pandas as pd
from tradsl.mlfunctions import Agent


class DummyAgent(Agent):
    """Dummy agent that returns random actions. Placeholder for RL agents."""
    
    def __init__(
        self,
        warmup: int = 50,
        n_actions: int = 3,
        model=None,
        default_asset: str = "AAPL",
        default_confidence: float = 0.5,
    ):
        import numpy as np
        self.n_actions = n_actions
        self.rng = np.random.default_rng()
        self.default_asset = default_asset
        self.default_confidence = default_confidence
        super().__init__(model=model, warmup=warmup)
    
    def _default_predict(self, data):
        action = self.rng.integers(0, self.n_actions)
        return self._format_output(action, self.default_confidence, self.default_asset)


class TabularQAgent(Agent):
    """Tabular Q-learning agent wrapper."""
    
    def __init__(
        self,
        warmup: int = 50,
        n_states: int = 100,
        n_actions: int = 3,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        model=None,
        default_asset: str = "AAPL",
    ):
        import numpy as np
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.q_table = None
        self.rng = np.random.default_rng()
        self.default_asset = default_asset
        super().__init__(model=model, warmup=warmup)
    
    def _default_predict(self, data):
        if self.q_table is None:
            self.q_table = self.rng.random((self.n_states, self.n_actions))
        state = self._state_from_data(data)
        action = int(self.rng.integers(0, self.n_actions))
        confidence = float(self.q_table[state % self.n_states, action])
        return self._format_output(action, confidence, self.default_asset)
    
    def _state_from_data(self, data):
        """Convert input data to state index. Override for custom discretization."""
        return hash(tuple(data.values.flatten())) % self.n_states


dummy = DummyAgent
tabular_q = TabularQAgent
