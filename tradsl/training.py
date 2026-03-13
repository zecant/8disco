"""
Training Framework for TradSL

Section 14: Randomized block training, walk-forward testing.
"""
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import random
import logging

logger = logging.getLogger("tradsl.training")


@dataclass
class Block:
    """A training block with start/end indices."""
    start_idx: int
    end_idx: int
    start_date: str = ""
    end_date: str = ""
    
    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx


@dataclass 
class TrainingConfig:
    """Configuration for training."""
    training_window: int
    retrain_schedule: str = "every_n_bars"
    retrain_n: int = 252
    n_training_blocks: int = 40
    block_size_min: int = 30
    block_size_max: int = 120
    seed: int = 42
    
    def validate(self) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []
        
        if self.block_size_min <= 0:
            errors.append("block_size_min must be positive")
        if self.block_size_max <= 0:
            errors.append("block_size_max must be positive")
        if self.block_size_min >= self.block_size_max:
            errors.append("block_size_min must be less than block_size_max")
        if self.n_training_blocks <= 0:
            errors.append("n_training_blocks must be positive")
        
        total_coverage = self.n_training_blocks * self.block_size_max
        max_coverage = int(self.training_window * 0.8)
        if total_coverage > max_coverage:
            errors.append(
                f"Block coverage {total_coverage} exceeds 80% of training window {max_coverage}"
            )
        
        return errors


class BlockSampler:
    """
    Section 14.4: Sample non-overlapping blocks from training pool.
    
    Algorithm:
    1. Set RNG seed
    2. Sample block sizes uniformly in [block_size_min, block_size_max]
    3. Verify total coverage ≤ 80% of pool
    4. Sample start positions, maintaining non-overlap
    5. Return sorted block list
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def sample_blocks(
        self,
        pool_start_idx: int,
        pool_end_idx: int,
        exclude_range: Optional[Tuple[int, int]] = None
    ) -> List[Block]:
        """
        Sample n_blocks non-overlapping blocks.
        
        Args:
            pool_start_idx: Start of training pool
            pool_end_idx: End of training pool
            exclude_range: (start, end) of test set to exclude
        
        Returns:
            List of Block objects, sorted by start_idx
        """
        random.seed(self.config.seed)
        
        pool_length = pool_end_idx - pool_start_idx
        max_coverage = int(pool_length * 0.8)
        
        blocks: List[Block] = []
        available_start = pool_start_idx
        
        for _ in range(self.config.n_training_blocks):
            block_size = random.randint(
                self.config.block_size_min,
                self.config.block_size_max
            )
            
            if available_start + block_size > pool_end_idx:
                break
            
            if exclude_range:
                excl_start, excl_end = exclude_range
                if available_start < excl_end and available_start + block_size > excl_start:
                    available_start = excl_end
                    if available_start + block_size > pool_end_idx:
                        break
            
            block = Block(
                start_idx=available_start,
                end_idx=available_start + block_size
            )
            blocks.append(block)
            
            available_start += block_size
            
            total_covered = sum(b.length for b in blocks)
            if total_covered >= max_coverage:
                break
        
        blocks.sort(key=lambda b: b.start_idx)
        
        return blocks


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool
    priority: float = 1.0


class ReplayBuffer:
    """
    Experience replay buffer with configurable sampling strategy.
    
    Supports:
    - Uniform sampling
    - Prioritized sampling (via SumTree)
    """
    
    def __init__(
        self,
        capacity: int,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4
    ):
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        
        self._buffer: List[Experience] = []
        self._priorities: List[float] = []
        self._max_priority = 1.0
    
    def add(self, experience: Experience) -> None:
        """Add experience to buffer."""
        priority = self._max_priority if self.prioritized else 1.0
        
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
            self._priorities.pop(0)
        
        self._buffer.append(experience)
        self._priorities.append(priority)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        if not self._buffer:
            return []
        
        if self.prioritized:
            return self._sample_prioritized(batch_size)
        else:
            indices = random.sample(range(len(self._buffer)), min(batch_size, len(self._buffer)))
            return [self._buffer[i] for i in indices]
    
    def _sample_prioritized(self, batch_size: int) -> List[Experience]:
        """Sample using priorities."""
        priorities = np.array(self._priorities) ** self.alpha
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(
            len(self._buffer),
            size=min(batch_size, len(self._buffer)),
            p=probs,
            replace=False
        )
        
        weights = (len(self._buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return [self._buffer[i] for i in indices]
    
    def clear(self) -> None:
        """Clear buffer (for block boundaries)."""
        self._buffer.clear()
        self._priorities.clear()
        self._max_priority = 1.0
    
    def __len__(self) -> int:
        return len(self._buffer)


@dataclass
class PolicyUpdateResult:
    """Result of a policy update."""
    loss: float
    gradient_norm: float
    batch_size: int
    learning_rate: float


class BlockTrainer:
    """
    Section 14.7: Train agent on shuffled blocks.
    
    Algorithm:
    1. Shuffle block order
    2. For each block:
       - Reset portfolio, bar_index
       - Flush replay buffer
       - Wait for warmup
       - Run bars in temporal order
       - On each bar: execute full on_bar protocol
    3. Save checkpoint
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        agent,
        reward_function,
        block_sampler: Optional[BlockSampler] = None
    ):
        self.config = config
        self.agent = agent
        self.reward_function = reward_function
        self.block_sampler = block_sampler or BlockSampler(config)
        
        self._block_results: List[Dict[str, Any]] = []
    
    def train(
        self,
        data: np.ndarray,
        warmup_bars: int,
        execute_bar_fn: Callable[[int, np.ndarray], Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Train agent on randomized blocks.
        
        Args:
            data: Price/feature data array
            warmup_bars: Bars to wait before generating signals
            execute_bar_fn: Function to execute on each bar
                          Signature: (bar_idx, bar_data) -> portfolio_state
        
        Returns:
            Training results dict
        """
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid training config: {errors}")
        
        blocks = self.block_sampler.sample_blocks(
            pool_start_idx=warmup_bars,
            pool_end_idx=len(data)
        )
        
        if not blocks:
            raise ValueError("No training blocks generated")
        
        random.seed(self.config.seed)
        random.shuffle(blocks)
        
        total_bars_processed = 0
        
        for block_idx, block in enumerate(blocks):
            self._train_block(
                block=block,
                data=data,
                warmup_bars=warmup_bars,
                execute_bar_fn=execute_bar_fn,
                block_idx=block_idx
            )
            total_bars_processed += block.length
        
        return {
            'n_blocks': len(blocks),
            'total_bars': total_bars_processed,
            'block_results': self._block_results
        }
    
    def _train_block(
        self,
        block: Block,
        data: np.ndarray,
        warmup_bars: int,
        execute_bar_fn: Callable,
        block_idx: int
    ) -> None:
        """Train on a single block."""
        if hasattr(self.agent, 'replay_buffer'):
            self.agent.replay_buffer.clear()
        
        if hasattr(self.agent, 'reset'):
            self.agent.reset()
        
        portfolio_value = 100000.0
        
        for bar_idx in range(block.start_idx, block.end_idx):
            relative_idx = bar_idx - block.start_idx
            
            portfolio_state = execute_bar_fn(bar_idx, data[bar_idx])
            
            portfolio_value = portfolio_state.get('portfolio_value', portfolio_value)
            
            if relative_idx >= warmup_bars and hasattr(self.agent, 'should_update'):
                if self.agent.should_update(
                    relative_idx,
                    recent_performance=portfolio_state.get('rolling_sharpe', 0.0)
                ):
                    self.agent.update_policy(relative_idx)
        
        self._block_results.append({
            'block_idx': block_idx,
            'start_idx': block.start_idx,
            'end_idx': block.end_idx,
            'length': block.length
        })


class WalkForwardTester:
    """
    Section 14.8: Sequential walk-forward test on held-out data.
    """
    
    def __init__(self, config: TrainingConfig, agent):
        self.config = config
        self.agent = agent
    
    def test(
        self,
        data: np.ndarray,
        test_start_idx: int,
        warmup_bars: int,
        execute_bar_fn: Callable
    ) -> Dict[str, Any]:
        """
        Run walk-forward test.
        
        Args:
            data: Price/feature data
            test_start_idx: Start of test period
            warmup_bars: Bars for warmup
            execute_bar_fn: Function to execute on each bar
        
        Returns:
            Test results with equity curve, trades, metrics
        """
        equity_curve = []
        trades = []
        portfolio_value = 100000.0
        
        for bar_idx in range(test_start_idx, len(data)):
            relative_idx = bar_idx - test_start_idx
            
            portfolio_state = execute_bar_fn(bar_idx, data[bar_idx])
            
            portfolio_value = portfolio_state.get('portfolio_value', portfolio_value)
            equity_curve.append(portfolio_value)
            
            if relative_idx >= warmup_bars and hasattr(self.agent, 'should_update'):
                if self.agent.should_update(
                    relative_idx,
                    portfolio_state.get('rolling_sharpe', 0.0)
                ):
                    self.agent.update_policy(relative_idx)
        
        return {
            'equity_curve': np.array(equity_curve),
            'trades': trades,
            'n_bars': len(data) - test_start_idx,
            'final_value': portfolio_value
        }
