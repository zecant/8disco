"""
Tests for backtest module.
"""
import pytest
import numpy as np
from tradsl.backtest import (
    BacktestEngine,
    TransactionCosts,
    ExecutionMode,
    Trade,
    BacktestResult
)


class TestTransactionCosts:
    """Tests for TransactionCosts."""
    
    def test_default_values(self):
        """Test default commission and slippage."""
        tc = TransactionCosts()
        assert tc.commission_rate == 0.001
        assert tc.slippage == 0.0
        assert tc.market_impact_coeff == 0.0
    
    def test_commission_calculation(self):
        """Test commission calculation."""
        tc = TransactionCosts(commission_rate=0.001)
        comm = tc.compute_commission(100, 50.0)
        assert comm == 5.0  # 100 * 50 * 0.001
    
    def test_slippage_buy(self):
        """Test slippage for buy orders."""
        tc = TransactionCosts(slippage=0.001)
        slip = tc.compute_slippage(50.0, "buy")
        assert slip == 0.05  # 50 * 0.001
    
    def test_slippage_sell(self):
        """Test slippage for sell orders."""
        tc = TransactionCosts(slippage=0.001)
        slip = tc.compute_slippage(50.0, "sell")
        assert slip == -0.05  # negative for sell
    
    def test_market_impact(self):
        """Test market impact calculation."""
        tc = TransactionCosts(market_impact_coeff=0.1)
        impact = tc.compute_market_impact(1000, 50.0, 10000)
        # 0.1 * sqrt(1000/10000) * 50 = 0.1 * 0.316 * 50 = 1.58
        assert impact > 0
    
    def test_market_impact_zero_adv(self):
        """Test market impact with zero ADV."""
        tc = TransactionCosts(market_impact_coeff=0.1)
        impact = tc.compute_market_impact(1000, 50.0, 0)
        assert impact == 0.0
    
    def test_total_cost(self):
        """Test total cost calculation."""
        tc = TransactionCosts(commission_rate=0.001, slippage=0.001)
        total = tc.total_cost(100, 50.0, "buy", 10000)
        assert total > 0


class TestExecutionMode:
    """Tests for ExecutionMode enum."""
    
    def test_modes_exist(self):
        """Test all execution modes exist."""
        assert ExecutionMode.TRAINING is not None
        assert ExecutionMode.TEST is not None
        assert ExecutionMode.LIVE is not None
    
    def test_mode_values(self):
        """Test mode string values."""
        assert ExecutionMode.TRAINING.value == "training"
        assert ExecutionMode.TEST.value == "test"
        assert ExecutionMode.LIVE.value == "live"


class TestTrade:
    """Tests for Trade dataclass."""
    
    def test_create_trade(self):
        """Test creating a trade."""
        trade = Trade(
            timestamp=1000,
            symbol="SPY",
            side="buy",
            quantity=100,
            price=50.0,
            commission=5.0,
            slippage=0.5
        )
        
        assert trade.symbol == "SPY"
        assert trade.quantity == 100
        assert trade.side == "buy"


class TestBacktestEngine:
    """Tests for BacktestEngine."""
    
    def test_init(self):
        """Test engine initialization."""
        config = {'capital': 100000, 'commission': 0.001}
        
        class MockAgent:
            pass
        
        class MockSizer:
            pass
        
        class MockReward:
            pass
        
        engine = BacktestEngine(
            config=config,
            agent=MockAgent(),
            sizer=MockSizer(),
            reward_function=MockReward()
        )
        
        assert engine.portfolio_value == 100000
    
    def test_reset(self):
        """Test engine reset."""
        config = {'capital': 100000}
        
        class MockAgent:
            pass
        
        class MockSizer:
            pass
        
        class MockReward:
            pass
        
        engine = BacktestEngine(
            config=config,
            agent=MockAgent(),
            sizer=MockSizer(),
            reward_function=MockReward()
        )
        
        engine.portfolio_value = 50000
        engine._reset()
        
        assert engine.portfolio_value == 100000
    
    def test_set_mode(self):
        """Test setting execution mode."""
        config = {'capital': 100000}
        
        class MockAgent:
            pass
        
        class MockSizer:
            pass
        
        class MockReward:
            pass
        
        engine = BacktestEngine(
            config=config,
            agent=MockAgent(),
            sizer=MockSizer(),
            reward_function=MockReward()
        )
        
        engine.set_mode(ExecutionMode.TRAINING)
        assert engine.mode == ExecutionMode.TRAINING
    
    def test_execute_trade(self):
        """Test executing a trade."""
        config = {'capital': 100000, 'commission': 0.001}
        
        class MockAgent:
            pass
        
        class MockSizer:
            pass
        
        class MockReward:
            pass
        
        engine = BacktestEngine(
            config=config,
            agent=MockAgent(),
            sizer=MockSizer(),
            reward_function=MockReward()
        )
        
        trade = engine.execute_trade(
            symbol="SPY",
            side="buy",
            quantity=100,
            price=50.0,
            timestamp=1000
        )
        
        assert trade.symbol == "SPY"
        assert trade.commission > 0


class TestBacktestResult:
    """Tests for BacktestResult."""
    
    def test_create_result(self):
        """Test creating a backtest result."""
        equity = np.array([100000, 110000])
        trades = []
        from tradsl.statistics import PerformanceMetrics
        
        metrics = PerformanceMetrics(
            total_return=0.1,
            cagr=0.08,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=0.5,
            max_drawdown=0.15,
            avg_drawdown=0.08,
            max_drawdown_duration=30,
            avg_drawdown_duration=15,
            win_rate=0.6,
            profit_factor=1.5,
            avg_win_loss=1.2,
            n_trades=10,
            avg_holding_period=5.0
        )
        
        result = BacktestResult(
            equity_curve=equity,
            trades=trades,
            metrics=metrics,
            mode=ExecutionMode.TEST
        )
        
        assert len(result.equity_curve) == 2
        assert result.mode == ExecutionMode.TEST
    
    def test_summary(self):
        """Test summary generation."""
        equity = np.array([100000, 110000])
        trades = []
        from tradsl.statistics import PerformanceMetrics
        
        metrics = PerformanceMetrics(
            total_return=0.1,
            cagr=0.08,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=0.5,
            max_drawdown=0.15,
            avg_drawdown=0.08,
            max_drawdown_duration=30,
            avg_drawdown_duration=15,
            win_rate=0.6,
            profit_factor=1.5,
            avg_win_loss=1.2,
            n_trades=10,
            avg_holding_period=5.0
        )
        
        result = BacktestResult(
            equity_curve=equity,
            trades=trades,
            metrics=metrics,
            mode=ExecutionMode.TEST
        )
        
        summary = result.summary()
        assert "Total Return" in summary
        assert "Sharpe" in summary
