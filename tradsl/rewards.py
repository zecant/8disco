"""
Reward Functions for TradSL

Section 11: BaseRewardFunction interface and implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class RewardContext:
    """
    Context for reward computation.
    
    Contains all information available at fill time
    (Section 11.2).
    """
    fill_price: float
    fill_quantity: float
    fill_side: str
    commission_paid: float
    
    pre_fill_portfolio_value: float
    post_fill_portfolio_value: float
    
    unrealized_pnl: float
    realized_pnl: float
    
    bars_in_trade: int
    high_water_mark: float
    baseline_capital: float
    
    current_drawdown: float


@dataclass
class RewardSpec:
    """Specification for a registered reward function."""
    cls: type
    description: str


class BaseRewardFunction(ABC):
    """
    Section 11.3: Reward function interface.
    
    Computes scalar reward from fill context.
    """
    
    @abstractmethod
    def compute(self, ctx: RewardContext) -> float:
        """
        Compute scalar reward.
        
        Args:
            ctx: RewardContext with fill and portfolio info
        
        Returns:
            Float reward (positive = good, negative = bad)
        """
        pass
    
    @abstractmethod
    def reset(self, baseline_capital: float) -> None:
        """
        Reset internal state at block boundaries.
        
        Args:
            baseline_capital: Starting capital for new block
        """
        pass
    
    def validate(self) -> None:
        """Validate configuration. Override if needed."""
        pass


class SimplePnLReward(BaseRewardFunction):
    """
    Simple P&L reward: percentage return on trade.
    
    reward = (post_value - pre_value) / baseline
    """
    
    def __init__(self):
        self.baseline_capital = 100000.0
    
    def compute(self, ctx: RewardContext) -> float:
        if self.baseline_capital <= 0:
            return 0.0
        
        pnl = ctx.post_fill_portfolio_value - ctx.pre_fill_portfolio_value
        return pnl / self.baseline_capital
    
    def reset(self, baseline_capital: float) -> None:
        self.baseline_capital = baseline_capital


class AsymmetricHighWaterMarkReward(BaseRewardFunction):
    """
    Section 11.4: Asymmetric high water mark reward.
    
    Exponential reward for new highs, linear for gains,
    exponential punishment for losses.
    
    Args:
        upper_multiplier: Exponential base for gains (default 2.0)
        lower_multiplier: Exponential base for losses (default 3.0)
        time_penalty: Per-bar holding cost (default 0.01)
    """
    
    def __init__(
        self,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 3.0,
        time_penalty: float = 0.01,
        high_water_mark_reset: str = "above_baseline"
    ):
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
        self.time_penalty = time_penalty
        self.high_water_mark_reset = high_water_mark_reset
        
        self.high_water_mark = 0.0
        self.loss_accumulator = 0.0
        self.baseline_capital = 100000.0
    
    def compute(self, ctx: RewardContext) -> float:
        E = ctx.post_fill_portfolio_value
        B = ctx.baseline_capital
        H = self.high_water_mark
        
        reward = 0.0
        
        if E > H:
            self.high_water_mark = E
            gain_pct = (E - H) / B
            reward = (self.upper_multiplier ** gain_pct) - 1.0
        
        elif E > B:
            self.loss_accumulator = 0.0
            pnl_pct = (E - ctx.pre_fill_portfolio_value) / B
            reward = pnl_pct
        
        else:
            loss_pct = (B - E) / B
            self.loss_accumulator = max(self.loss_accumulator, loss_pct)
            reward = -(self.lower_multiplier ** self.loss_accumulator)
        
        reward -= self.time_penalty * ctx.bars_in_trade
        
        return reward
    
    def reset(self, baseline_capital: float) -> None:
        self.baseline_capital = baseline_capital
        self.high_water_mark = baseline_capital
        self.loss_accumulator = 0.0


class DifferentialSharpeReward(BaseRewardFunction):
    """
    Section 11.5: Differential Sharpe ratio reward.
    
    Measures marginal contribution to running Sharpe ratio.
    
    Args:
        adaptation_rate: EMA rate for statistics (default 0.01)
    """
    
    def __init__(self, adaptation_rate: float = 0.01):
        self.eta = adaptation_rate
        self.A = 0.0
        self.B = 0.0
        self.baseline_capital = 100000.0
    
    def compute(self, ctx: RewardContext) -> float:
        if ctx.pre_fill_portfolio_value <= 0:
            return 0.0
        
        r_t = (ctx.post_fill_portfolio_value - ctx.pre_fill_portfolio_value) / ctx.pre_fill_portfolio_value
        
        delta_A = r_t - self.A
        self.A = self.A + self.eta * delta_A
        self.B = self.B + self.eta * (r_t ** 2 - self.B)
        
        denominator = (self.B - self.A ** 2) ** 1.5
        if denominator < 1e-10:
            return 0.0
        
        reward = (self.B * delta_A - 0.5 * self.A * (delta_A ** 2)) / denominator
        
        return reward
    
    def reset(self, baseline_capital: float) -> None:
        self.baseline_capital = baseline_capital
        self.A = 0.0
        self.B = 0.0


# Reward registry
REWARD_REGISTRY: Dict[str, RewardSpec] = {
    'simple_pnl': RewardSpec(
        cls=SimplePnLReward,
        description="Simple P&L percentage reward"
    ),
    'asymmetric_high_water_mark': RewardSpec(
        cls=AsymmetricHighWaterMarkReward,
        description="Asymmetric reward with high water mark"
    ),
    'differential_sharpe': RewardSpec(
        cls=DifferentialSharpeReward,
        description="Differential Sharpe ratio reward"
    ),
}


def get_reward(name: str) -> Optional[type]:
    """Get reward class by name."""
    spec = REWARD_REGISTRY.get(name)
    return spec.cls if spec else None
