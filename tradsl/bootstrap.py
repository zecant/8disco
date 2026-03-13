"""
Bootstrap Ensemble for Performance Validation

Section 14.9: Dual bootstrap system for statistical significance
and real-world behavior estimation.
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
import random

from tradsl.training import TrainingConfig


@dataclass
class BootstrapResult:
    """Results from bootstrap analysis."""
    n_samples: int
    
    sharpe_samples: np.ndarray
    sortino_samples: np.ndarray
    max_drawdown_samples: np.ndarray
    calmar_samples: np.ndarray
    total_return_samples: np.ndarray
    win_rate_samples: np.ndarray
    profit_factor_samples: np.ndarray
    
    @property
    def sharpe_median(self) -> float:
        return float(np.median(self.sharpe_samples))
    
    @property
    def sharpe_p5(self) -> float:
        return float(np.percentile(self.sharpe_samples, 5))
    
    @property
    def sharpe_p95(self) -> float:
        return float(np.percentile(self.sharpe_samples, 95))
    
    @property
    def sortino_median(self) -> float:
        return float(np.median(self.sortino_samples))
    
    @property
    def max_drawdown_median(self) -> float:
        return float(np.median(self.max_drawdown_samples))
    
    @property
    def calmar_median(self) -> float:
        return float(np.median(self.calmar_samples))
    
    @property
    def total_return_median(self) -> float:
        return float(np.median(self.total_return_samples))
    
    @property
    def win_rate_median(self) -> float:
        return float(np.median(self.win_rate_samples))
    
    @property
    def profit_factor_median(self) -> float:
        return float(np.median(self.profit_factor_samples))
    
    def is_significant(self, metric: str = "sharpe", alpha: float = 0.05) -> bool:
        """Check if CI excludes zero for the given metric."""
        samples = getattr(self, f"{metric}_samples")
        p5 = np.percentile(samples, alpha * 100)
        p95 = np.percentile(samples, (1 - alpha) * 100)
        return (p5 > 0) or (p95 < 0)
    
    def summary(self) -> str:
        """Generate summary table."""
        lines = [
            "Bootstrap Results",
            "=" * 60,
            f"Samples: {self.n_samples}",
            "",
            f"{'Metric':<20} {'Median':>12} {'5th %':>12} {'95th %':>12}",
            "-" * 60,
            f"{'Sharpe':<20} {self.sharpe_median:>12.4f} {self.sharpe_p5:>12.4f} {self.sharpe_p95:>12.4f}",
            f"{'Max Drawdown':<20} {np.median(self.max_drawdown_samples):>12.4f} {np.percentile(self.max_drawdown_samples, 5):>12.4f} {np.percentile(self.max_drawdown_samples, 95):>12.4f}",
            f"{'Total Return':<20} {np.median(self.total_return_samples):>12.4f} {np.percentile(self.total_return_samples, 5):>12.4f} {np.percentile(self.total_return_samples, 95):>12.4f}",
            f"{'Win Rate':<20} {np.median(self.win_rate_samples):>12.4f} {np.percentile(self.win_rate_samples, 5):>12.4f} {np.percentile(self.win_rate_samples, 95):>12.4f}",
        ]
        return "\n".join(lines)


class IIDBootstrap:
    """
    IID bootstrap for statistical significance testing.
    
    Used to determine if performance is real or noise.
    Warning: Destroys autocorrelation - use for significance only.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def bootstrap(
        self,
        returns: np.ndarray,
        n_samples: int,
        metric_fn: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """
        Generate bootstrap distribution of a metric.
        
        Args:
            returns: Return series (1D array)
            n_samples: Number of bootstrap samples
            metric_fn: Function to compute metric from returns
        
        Returns:
            Array of metric samples
        """
        rng = np.random.RandomState(self.seed)
        n = len(returns)
        
        samples = []
        for _ in range(n_samples):
            indices = rng.randint(0, n, size=n)
            sample_returns = returns[indices]
            metric = metric_fn(sample_returns)
            samples.append(metric)
        
        return np.array(samples)
    
    def compute_all_metrics(
        self,
        returns: np.ndarray,
        n_samples: int = 1000
    ) -> BootstrapResult:
        """Compute all performance metrics via IID bootstrap."""
        return BootstrapResult(
            n_samples=n_samples,
            sharpe_samples=self.bootstrap(returns, n_samples, self._sharpe),
            sortino_samples=self.bootstrap(returns, n_samples, self._sortino),
            max_drawdown_samples=self.bootstrap(returns, n_samples, self._max_drawdown),
            calmar_samples=self.bootstrap(returns, n_samples, self._calmar),
            total_return_samples=self.bootstrap(returns, n_samples, self._total_return),
            win_rate_samples=self.bootstrap(returns, n_samples, self._win_rate),
            profit_factor_samples=self.bootstrap(returns, n_samples, self._profit_factor)
        )
    
    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    @staticmethod
    def _sortino(returns: np.ndarray) -> float:
        neg_returns = returns[returns < 0]
        if len(neg_returns) == 0 or np.std(neg_returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(neg_returns) * np.sqrt(252)
    
    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        equity = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        return float(np.max(drawdowns))
    
    @staticmethod
    def _calmar(returns: np.ndarray) -> float:
        total_ret = np.prod(1 + returns) - 1
        max_dd = IIDBootstrap._max_drawdown(returns)
        if max_dd == 0:
            return 0.0
        return total_ret / max_dd
    
    @staticmethod
    def _total_return(returns: np.ndarray) -> float:
        return float(np.prod(1 + returns) - 1)
    
    @staticmethod
    def _win_rate(returns: np.ndarray) -> float:
        return float(np.mean(returns > 0))
    
    @staticmethod
    def _profit_factor(returns: np.ndarray) -> float:
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if len(losses) == 0 or np.sum(losses) == 0:
            return 0.0
        return abs(np.sum(wins) / np.sum(losses))


class BlockBootstrap:
    """
    Block bootstrap for real-world behavior estimation.
    
    Preserves autocorrelation structure by resampling contiguous blocks.
    Block length = sqrt(T) by default.
    """
    
    def __init__(
        self,
        seed: int = 42,
        block_length: Optional[int] = None
    ):
        self.seed = seed
        self.block_length = block_length
    
    def bootstrap(
        self,
        returns: np.ndarray,
        n_samples: int,
        metric_fn: Callable[[np.ndarray], float],
        block_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate block bootstrap distribution.
        
        Args:
            returns: Return series
            n_samples: Number of bootstrap samples
            metric_fn: Function to compute metric
            block_length: Override default sqrt(T)
        
        Returns:
            Array of metric samples
        """
        rng = np.random.RandomState(self.seed)
        T = len(returns)
        
        if block_length is None:
            block_length = max(1, int(np.sqrt(T)))
        
        if self.block_length:
            block_length = self.block_length
        
        n_blocks = (T + block_length - 1) // block_length
        
        samples = []
        for _ in range(n_samples):
            resampled = []
            for _ in range(n_blocks):
                start = rng.randint(0, T - block_length + 1)
                resampled.extend(returns[start:start + block_length])
            
            sample_returns = np.array(resampled[:T])
            metric = metric_fn(sample_returns)
            samples.append(metric)
        
        return np.array(samples)
    
    def compute_all_metrics(
        self,
        returns: np.ndarray,
        n_samples: int = 1000,
        block_length: Optional[int] = None
    ) -> BootstrapResult:
        """Compute all performance metrics via block bootstrap."""
        return BootstrapResult(
            n_samples=n_samples,
            sharpe_samples=self.bootstrap(returns, n_samples, IIDBootstrap._sharpe, block_length),
            sortino_samples=self.bootstrap(returns, n_samples, IIDBootstrap._sortino, block_length),
            max_drawdown_samples=self.bootstrap(returns, n_samples, IIDBootstrap._max_drawdown, block_length),
            calmar_samples=self.bootstrap(returns, n_samples, IIDBootstrap._calmar, block_length),
            total_return_samples=self.bootstrap(returns, n_samples, IIDBootstrap._total_return, block_length),
            win_rate_samples=self.bootstrap(returns, n_samples, IIDBootstrap._win_rate, block_length),
            profit_factor_samples=self.bootstrap(returns, n_samples, IIDBootstrap._profit_factor, block_length)
        )


class BootstrapEnsemble:
    """
    Section 14.9: Full bootstrap ensemble with sister models.
    
    Runs multiple bootstrap iterations with independent policy updates
    to estimate distribution of outcomes.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        agent_factory: Callable,
        n_samples: int = 1000,
        use_block: bool = True,
        seed: int = 42
    ):
        self.config = config
        self.agent_factory = agent_factory
        self.n_samples = n_samples
        self.use_block = use_block
        self.seed = seed
    
    def run(
        self,
        checkpoint: Dict[str, Any],
        test_returns: np.ndarray
    ) -> BootstrapResult:
        """
        Run bootstrap ensemble.
        
        Args:
            checkpoint: Saved policy weights
            test_returns: Test period returns
        
        Returns:
            BootstrapResult with metric distributions
        """
        if self.use_block:
            boot = BlockBootstrap(seed=self.seed)
        else:
            boot = IIDBootstrap(seed=self.seed)
        
        results = boot.compute_all_metrics(test_returns, self.n_samples)
        
        return results
