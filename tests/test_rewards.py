"""Tests for reward functions."""
import pytest
from tradsl.rewards import (
    SimplePnLReward, AsymmetricHighWaterMarkReward, DifferentialSharpeReward,
    RewardContext, REWARD_REGISTRY, get_reward
)


class TestSimplePnLReward:
    def test_positive_pnl(self):
        reward_fn = SimplePnLReward()
        reward_fn.reset(100000.0)
        
        ctx = RewardContext(
            fill_price=100.0,
            fill_quantity=10.0,
            fill_side="buy",
            commission_paid=1.0,
            pre_fill_portfolio_value=100000.0,
            post_fill_portfolio_value=100100.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            bars_in_trade=0,
            high_water_mark=100000.0,
            baseline_capital=100000.0,
            current_drawdown=0.0
        )
        
        result = reward_fn.compute(ctx)
        assert result > 0
    
    def test_negative_pnl(self):
        reward_fn = SimplePnLReward()
        reward_fn.reset(100000.0)
        
        ctx = RewardContext(
            fill_price=100.0,
            fill_quantity=10.0,
            fill_side="buy",
            commission_paid=1.0,
            pre_fill_portfolio_value=100000.0,
            post_fill_portfolio_value=99900.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            bars_in_trade=0,
            high_water_mark=100000.0,
            baseline_capital=100000.0,
            current_drawdown=0.0
        )
        
        result = reward_fn.compute(ctx)
        assert result < 0
    
    def test_reset(self):
        reward_fn = SimplePnLReward()
        reward_fn.reset(50000.0)
        
        assert reward_fn.baseline_capital == 50000.0


class TestAsymmetricHighWaterMarkReward:
    def test_new_high_reward(self):
        reward_fn = AsymmetricHighWaterMarkReward(
            upper_multiplier=2.0,
            lower_multiplier=3.0,
            time_penalty=0.0
        )
        reward_fn.reset(100000.0)
        
        ctx = RewardContext(
            fill_price=100.0,
            fill_quantity=10.0,
            fill_side="buy",
            commission_paid=0.0,
            pre_fill_portfolio_value=100000.0,
            post_fill_portfolio_value=105000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            bars_in_trade=0,
            high_water_mark=100000.0,
            baseline_capital=100000.0,
            current_drawdown=0.0
        )
        
        result = reward_fn.compute(ctx)
        
        gain_pct = 5000.0 / 100000.0
        expected = (2.0 ** gain_pct) - 1.0
        assert abs(result - expected) < 1e-6
    
    def test_loss_punishment(self):
        reward_fn = AsymmetricHighWaterMarkReward(
            upper_multiplier=2.0,
            lower_multiplier=3.0,
            time_penalty=0.0
        )
        reward_fn.reset(100000.0)
        reward_fn.high_water_mark = 100000.0
        
        ctx = RewardContext(
            fill_price=100.0,
            fill_quantity=10.0,
            fill_side="buy",
            commission_paid=0.0,
            pre_fill_portfolio_value=100000.0,
            post_fill_portfolio_value=95000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            bars_in_trade=5,
            high_water_mark=100000.0,
            baseline_capital=100000.0,
            current_drawdown=0.0
        )
        
        result = reward_fn.compute(ctx)
        
        loss_pct = 5000.0 / 100000.0
        expected = -(3.0 ** loss_pct)
        assert result <= expected
    
    def test_time_penalty(self):
        reward_fn = AsymmetricHighWaterMarkReward(
            upper_multiplier=2.0,
            lower_multiplier=3.0,
            time_penalty=0.01
        )
        reward_fn.reset(100000.0)
        
        ctx = RewardContext(
            fill_price=100.0,
            fill_quantity=10.0,
            fill_side="buy",
            commission_paid=0.0,
            pre_fill_portfolio_value=100000.0,
            post_fill_portfolio_value=105000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            bars_in_trade=10,
            high_water_mark=100000.0,
            baseline_capital=100000.0,
            current_drawdown=0.0
        )
        
        result = reward_fn.compute(ctx)
        
        gain_pct = 5000.0 / 100000.0
        base = (2.0 ** gain_pct) - 1.0
        expected = base - (0.01 * 10)
        
        assert result < base
    
    def test_reset_clears_state(self):
        reward_fn = AsymmetricHighWaterMarkReward()
        reward_fn.reset(100000.0)
        reward_fn.high_water_mark = 150000.0
        reward_fn.loss_accumulator = 0.5
        
        reward_fn.reset(100000.0)
        
        assert reward_fn.high_water_mark == 100000.0
        assert reward_fn.loss_accumulator == 0.0


class TestDifferentialSharpeReward:
    def test_basic_computation(self):
        reward_fn = DifferentialSharpeReward(adaptation_rate=0.01)
        reward_fn.reset(100000.0)
        
        ctx1 = RewardContext(
            fill_price=100.0,
            fill_quantity=10.0,
            fill_side="buy",
            commission_paid=0.0,
            pre_fill_portfolio_value=100000.0,
            post_fill_portfolio_value=101000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            bars_in_trade=0,
            high_water_mark=100000.0,
            baseline_capital=100000.0,
            current_drawdown=0.0
        )
        
        result1 = reward_fn.compute(ctx1)
        
        ctx2 = RewardContext(
            fill_price=100.0,
            fill_quantity=10.0,
            fill_side="buy",
            commission_paid=0.0,
            pre_fill_portfolio_value=101000.0,
            post_fill_portfolio_value=102000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            bars_in_trade=0,
            high_water_mark=100000.0,
            baseline_capital=100000.0,
            current_drawdown=0.0
        )
        
        result2 = reward_fn.compute(ctx2)
    
    def test_reset_clears_statistics(self):
        reward_fn = DifferentialSharpeReward(adaptation_rate=0.01)
        reward_fn.reset(100000.0)
        reward_fn.A = 0.5
        reward_fn.B = 0.25
        
        reward_fn.reset(100000.0)
        
        assert reward_fn.A == 0.0
        assert reward_fn.B == 0.0


class TestRewardRegistry:
    def test_get_reward(self):
        cls = get_reward('simple_pnl')
        assert cls == SimplePnLReward
    
    def test_registry_has_expected_rewards(self):
        expected = ['simple_pnl', 'asymmetric_high_water_mark', 'differential_sharpe']
        for name in expected:
            assert name in REWARD_REGISTRY
