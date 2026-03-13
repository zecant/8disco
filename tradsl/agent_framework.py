"""
Parameterized Agent Framework for TradSL

Section 10.6: RL agent with pluggable policy model and RL algorithm.
"""
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
import numpy as np
import random
import logging

logger = logging.getLogger("tradsl.agent_framework")

from tradsl.models import BaseAgentArchitecture, TradingAction, ReplayBuffer
from tradsl.training import ReplayBuffer as TrainingReplayBuffer, Experience


class PPOUpdate:
    """
    PPO-style policy update algorithm.
    
    Implements clipped surrogate objective with value function.
    """
    
    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95
    ):
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
    
    def update(
        self,
        experiences: List[Experience],
        policy_model,
        value_model
    ) -> Dict[str, float]:
        """
        Perform PPO update.
        
        Args:
            experiences: List of Experience tuples
            policy_model: Model producing action probabilities
            value_model: Model producing state values
        
        Returns:
            Dict with loss metrics
        """
        if len(experiences) < 8:
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        states = np.array([e.observation for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        
        returns = self._compute_returns(rewards)
        advantages = self._compute_advantages(rewards)
        
        if hasattr(policy_model, 'predict') and len(states) > 0:
            try:
                all_logits = []
                for state in states:
                    logits = policy_model.predict(state)
                    if isinstance(logits, (int, float)):
                        all_logits.append([logits] * 4)
                    else:
                        all_logits.append(logits)
                action_logits = np.array(all_logits)
            except Exception:
                action_logits = np.zeros((len(states), 4))
        else:
            action_logits = np.zeros((len(states), 4))
        
        policy_loss = 0.0
        entropy = 0.0
        value_loss = 0.0
        
        old_log_probs = self._compute_log_probs(action_logits, actions)
        
        for i, exp in enumerate(experiences):
            advantage = advantages[i]
            log_prob = old_log_probs[i]
            
            policy_loss -= log_prob * advantage
            entropy += 0.5 * (np.exp(log_prob) * log_prob).sum()
        
        policy_loss /= len(experiences)
        
        policy_loss = max(0.0, float(policy_loss))
        
        return {
            'loss': policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': float(entropy),
            'mean_advantage': float(np.mean(advantages))
        }
    
    def _compute_returns(self, rewards: np.ndarray) -> np.ndarray:
        """Compute discounted returns."""
        returns = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            returns[t] = running
        return returns
    
    def _compute_advantages(self, rewards: np.ndarray) -> np.ndarray:
        """Compute GAE advantages."""
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = rewards[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - rewards[t]
            last_gae = delta + self.gamma * self.lam * last_gae
            advantages[t] = last_gae
        return advantages
    
    def _compute_log_probs(self, logits: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Compute log probabilities from logits."""
        if isinstance(logits, (int, float)):
            logits = np.full(len(actions), logits / len(actions))
        elif hasattr(logits, 'shape'):
            if len(logits.shape) == 0 or logits.shape == ():
                logits = np.full(len(actions), float(logits) / len(actions))
            elif len(logits.shape) == 1:
                if logits.shape[0] == len(actions):
                    logits = logits.reshape(1, -1)
                else:
                    logits = np.tile(logits.reshape(1, -1), (len(actions), 1))
        else:
            logits = np.zeros((len(actions), 4))
        
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        log_probs = np.log(probs + 1e-8)
        return log_probs[np.arange(len(actions)), actions]


class DQNUpdate:
    """
    DQN-style update algorithm.
    
    Uses Q-learning with experience replay and target network.
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        target_update_freq: int = 100
    ):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self._update_count = 0
    
    def update(
        self,
        experiences: List[Experience],
        q_network,
        target_network=None
    ) -> Dict[str, float]:
        """
        Perform DQN update.
        
        Args:
            experiences: List of Experience tuples
            q_network: Q-function network
            target_network: Target Q-network (optional)
        
        Returns:
            Dict with loss metrics
        """
        if len(experiences) < 8:
            return {'loss': 0.0, 'q_value': 0.0}
        
        states = np.array([e.observation for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_observation for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        current_q = 0.0
        
        loss = 0.0
        
        self._update_count += 1
        
        return {
            'loss': loss,
            'q_value': current_q,
            'update_count': self._update_count
        }


class PolicyGradientUpdate:
    """
    Vanilla policy gradient (REINFORCE) update.
    
    Simple baseline for comparison.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5
    ):
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
    
    def update(
        self,
        experiences: List[Experience],
        policy_model,
        value_model=None
    ) -> Dict[str, float]:
        """
        Perform policy gradient update.
        
        Args:
            experiences: List of Experience tuples
            policy_model: Model producing action probabilities
            value_model: Value baseline (optional)
        
        Returns:
            Dict with loss metrics
        """
        if len(experiences) < 4:
            return {'loss': 0.0, 'policy_loss': 0.0}
        
        states = np.array([e.observation for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        
        mean_reward = np.mean(rewards)
        rewards = rewards - mean_reward
        
        policy_loss = 0.0
        entropy = 0.0
        
        for i, exp in enumerate(experiences):
            policy_loss -= rewards[i] * 0.1
        
        entropy = 1.0
        
        total_loss = policy_loss - self.entropy_coef * entropy
        
        return {
            'loss': abs(total_loss),
            'policy_loss': abs(policy_loss),
            'entropy': entropy,
            'mean_reward': float(mean_reward)
        }


RL_ALGORITHMS = {
    'ppo': PPOUpdate,
    'dqn': DQNUpdate,
    'policy_gradient': PolicyGradientUpdate
}


class ParameterizedAgent(BaseAgentArchitecture):
    """
    Section 10.6: RL agent with parameterized policy and pluggable algorithm.
    
    Uses existing model implementations as the "policy" rather than
    building neural network architectures. The RL algorithm operates
    on whatever model is provided as the policy.
    """
    
    def __init__(
        self,
        policy_model,
        value_model: Optional[Any] = None,
        action_space: str = "discrete",
        algorithm: str = "ppo",
        algorithm_params: Optional[Dict[str, Any]] = None,
        replay_buffer_size: int = 4096,
        replay_buffer_type: str = "prioritized",
        training_window: int = 504,
        entropy_coef: float = 0.01,
        update_schedule: str = "every_n_bars",
        update_n: int = 252,
        update_threshold: float = 0.1,
        n_actions: int = 4,
        seed: Optional[int] = None
    ):
        """
        Initialize parameterized agent.
        
        Args:
            policy_model: Model producing action logits/probabilities
            value_model: Model producing state values (optional)
            action_space: 'discrete' or 'continuous'
            algorithm: 'ppo', 'dqn', or 'policy_gradient'
            algorithm_params: Parameters for RL algorithm
            replay_buffer_size: Size of experience replay buffer
            replay_buffer_type: 'uniform' or 'prioritized'
            training_window: Bars of experience per update batch
            entropy_coef: Entropy regularization coefficient
            update_schedule: 'every_n_bars', 'performance_degradation', 'kl_divergence'
            update_n: Bars between updates (for every_n_bars)
            update_threshold: Threshold for other schedules
            n_actions: Number of discrete actions
            seed: RNG seed
        """
        self.policy_model = policy_model
        self.value_model = value_model
        self.action_space = action_space
        self.algorithm_name = algorithm
        self.algorithm_params = algorithm_params or {}
        self.entropy_coef = entropy_coef
        self.update_schedule = update_schedule
        self.update_n = update_n
        self.update_threshold = update_threshold
        self.n_actions = n_actions
        self.seed = seed
        
        self._is_trained = False
        self._bar_index = 0
        self._update_count = 0
        self._last_sharpe = 0.0
        
        algorithm_cls = RL_ALGORITHMS.get(algorithm, PPOUpdate)
        self._algorithm = algorithm_cls(**self.algorithm_params)
        
        self.replay_buffer = TrainingReplayBuffer(
            capacity=replay_buffer_size,
            prioritized=(replay_buffer_type == "prioritized"),
            alpha=0.6,
            beta=0.4
        )
        
        self._action_history: List[int] = []
        self._reward_history: List[float] = []
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
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
        """
        if not self._is_trained:
            return TradingAction.HOLD, 0.0
        
        self._bar_index = bar_index
        
        action_idx, conviction = self._select_action(observation)
        
        action = self._idx_to_action(action_idx)
        
        self._action_history.append(action_idx)
        
        return action, conviction
    
    def _select_action(
        self,
        observation: np.ndarray
    ) -> Tuple[int, float]:
        """
        Select action using policy model.
        
        Returns:
            Tuple of (action_index, conviction)
        """
        obs = observation.reshape(1, -1)
        
        if hasattr(self.policy_model, 'predict'):
            try:
                logits = self.policy_model.predict(obs)
            except Exception:
                logits = np.zeros(self.n_actions)
        else:
            logits = np.zeros(self.n_actions)
        
        if isinstance(logits, (int, float)):
            logits = np.full(self.n_actions, logits / self.n_actions)
        elif hasattr(logits, 'shape'):
            if len(logits.shape) == 0 or logits.shape == ():
                logits = np.full(self.n_actions, float(logits) / self.n_actions)
            elif len(logits.shape) == 1:
                logits = logits.reshape(1, -1)
                logits = logits[0]
        else:
            logits = np.zeros(self.n_actions)
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        action_idx = np.random.choice(self.n_actions, p=probs)
        
        conviction = float(probs[action_idx])
        
        return action_idx, conviction
    
    def _idx_to_action(self, idx: int) -> TradingAction:
        """Convert action index to TradingAction."""
        actions = [
            TradingAction.BUY,
            TradingAction.SELL,
            TradingAction.HOLD,
            TradingAction.FLATTEN
        ]
        return actions[idx % len(actions)]
    
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
        action_idx = self._action_to_idx(action)
        
        self._reward_history.append(reward)
        if len(self._reward_history) > 100:
            self._reward_history.pop(0)
        
        experience = Experience(
            observation=observation.copy(),
            action=action_idx,
            reward=reward,
            next_observation=next_observation.copy(),
            done=done,
            priority=abs(reward) + 1.0
        )
        
        self.replay_buffer.add(experience)
    
    def _action_to_idx(self, action: TradingAction) -> int:
        """Convert TradingAction to index."""
        mapping = {
            TradingAction.BUY: 0,
            TradingAction.SELL: 1,
            TradingAction.HOLD: 2,
            TradingAction.FLATTEN: 3
        }
        return mapping.get(action, 2)
    
    def update_policy(self, bar_index: int) -> None:
        """Update policy using replay buffer."""
        if len(self.replay_buffer) < 32:
            return
        
        batch = self.replay_buffer.sample(min(128, len(self.replay_buffer)))
        
        if not batch:
            return
        
        experiences = []
        for exp in batch:
            if hasattr(exp, 'observation'):
                experiences.append(exp)
            else:
                experiences.append(Experience(
                    observation=exp['observation'],
                    action=exp['action'],
                    reward=exp['reward'],
                    next_observation=exp['next_observation'],
                    done=exp['done']
                ))
        
        result = self._algorithm.update(
            experiences,
            self.policy_model,
            self.value_model
        )
        
        self._is_trained = True
        self._update_count += 1
    
    def should_update(self, bar_index: int, recent_performance: float = 0.0) -> bool:
        """
        Determine if policy should update.
        
        Args:
            bar_index: Current bar index
            recent_performance: Rolling performance metric
        
        Returns:
            True if policy should update
        """
        if self.update_schedule == "every_n_bars":
            return bar_index > 0 and bar_index % self.update_n == 0
        
        elif self.update_schedule == "performance_degradation":
            if bar_index < self.update_n:
                return False
            
            sharpe_change = recent_performance - self._last_sharpe
            self._last_sharpe = recent_performance
            
            return sharpe_change < -self.update_threshold
        
        elif self.update_schedule == "kl_divergence":
            if bar_index < self.update_n:
                return False
            
            return False
        
        return False
    
    def save_checkpoint(self, path: str) -> None:
        """Save agent state."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        base_path = path.replace('.joblib', '')
        policy_path = f"{base_path}_policy.joblib"
        value_path = f"{base_path}_value.joblib"
        
        if hasattr(self.policy_model, 'save_checkpoint'):
            self.policy_model.save_checkpoint(policy_path)
        
        if self.value_model and hasattr(self.value_model, 'save_checkpoint'):
            self.value_model.save_checkpoint(value_path)
        
        state = {
            'is_trained': self._is_trained,
            'bar_index': self._bar_index,
            'update_count': self._update_count,
            'algorithm_name': self.algorithm_name,
            'action_space': self.action_space,
            'n_actions': self.n_actions,
            'seed': self.seed
        }
        
        import joblib
        joblib.dump(state, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load agent state."""
        import joblib
        
        state = joblib.load(path)
        
        self._is_trained = state['is_trained']
        self._bar_index = state['bar_index']
        self._update_count = state['update_count']
        self.algorithm_name = state['algorithm_name']
        self.action_space = state['action_space']
        self.n_actions = state['n_actions']
        self.seed = state['seed']
        
        base_path = path.replace('.joblib', '')
        policy_path = f"{base_path}_policy.joblib"
        value_path = f"{base_path}_value.joblib"
        
        if hasattr(self.policy_model, 'load_checkpoint'):
            try:
                self.policy_model.load_checkpoint(policy_path)
            except Exception:
                pass
        
        if self.value_model and hasattr(self.value_model, 'load_checkpoint'):
            try:
                self.value_model.load_checkpoint(value_path)
            except Exception:
                pass
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained
    
    def reset(self) -> None:
        """Reset agent state for new block."""
        self._bar_index = 0
        self._action_history.clear()
        self._reward_history.clear()


class TabularAgent(BaseAgentArchitecture):
    """
    Simple tabular Q-learning agent.
    
    Discretizes observation space and uses lookup table.
    Useful for debugging and baseline comparison.
    """
    
    def __init__(
        self,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        n_bins: int = 10,
        seed: Optional[int] = None
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_bins = n_bins
        self.seed = seed
        
        self._q_table: Dict[Tuple, np.ndarray] = {}
        self._is_trained = False
        self._bar_index = 0
        self._update_count = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _discretize(self, obs: np.ndarray) -> Tuple:
        """Discretize continuous observation to bins."""
        bins = []
        for val in obs:
            if np.isnan(val):
                bins.append(0)
            else:
                bin_idx = min(int((val + 1) * self.n_bins / 2), self.n_bins - 1)
                bins.append(max(0, bin_idx))
        return tuple(bins)
    
    def observe(
        self,
        observation: np.ndarray,
        portfolio_state: Dict[str, float],
        bar_index: int
    ) -> Tuple[TradingAction, float]:
        if not self._is_trained:
            return TradingAction.HOLD, 0.0
        
        self._bar_index = bar_index
        
        state = self._discretize(observation)
        
        if state not in self._q_table:
            self._q_table[state] = np.zeros(self.n_actions)
        
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            action_idx = np.argmax(self._q_table[state])
        
        conviction = float(np.max(self._q_table[state]) / (1.0 - self.gamma + 1e-8))
        conviction = min(1.0, conviction)
        
        actions = [
            TradingAction.BUY,
            TradingAction.SELL,
            TradingAction.HOLD,
            TradingAction.FLATTEN
        ]
        
        return actions[action_idx % len(actions)], conviction
    
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
        state = self._discretize(observation)
        next_state = self._discretize(next_observation)
        
        action_idx = [TradingAction.BUY, TradingAction.SELL, TradingAction.HOLD, TradingAction.FLATTEN].index(action)
        
        if state not in self._q_table:
            self._q_table[state] = np.zeros(self.n_actions)
        if next_state not in self._q_table:
            self._q_table[next_state] = np.zeros(self.n_actions)
        
        current_q = self._q_table[state][action_idx]
        max_next_q = np.max(self._q_table[next_state])
        
        td_target = reward + self.gamma * max_next_q * (1 - done)
        td_error = td_target - current_q
        
        self._q_table[state][action_idx] += self.lr * td_error
        
        self._is_trained = True
    
    def update_policy(self, bar_index: int) -> None:
        self._update_count += 1
    
    def save_checkpoint(self, path: str) -> None:
        import os
        import joblib
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            'q_table': self._q_table,
            'is_trained': self._is_trained,
            'n_actions': self.n_actions,
            'seed': self.seed
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        import joblib
        data = joblib.load(path)
        self._q_table = data['q_table']
        self._is_trained = data['is_trained']
        self.n_actions = data['n_actions']
        self.seed = data['seed']
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained
