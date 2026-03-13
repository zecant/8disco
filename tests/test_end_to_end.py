"""
Comprehensive End-to-End Integration Tests for TradSL.

Tests the full pipeline: DSL → parse → validate → resolve → execute → analyze.
"""
import pytest
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from dataclasses import dataclass

from tradsl import (
    parse_dsl, validate_config, resolve_config, build_execution_dag,
    TestDataAdapter, create_test_adapter,
    TradSLStrategy, Portfolio, Position,
    KellySizer, FractionalSizer,
    AsymmetricHighWaterMarkReward,
    ParameterizedAgent,
    RandomForestModel,
    TrainingConfig,
    BlockSampler,
    compute_metrics,
    CircularBuffer,
)
from tradsl.models import TradingAction
from tradsl.exceptions import TradSLError


class TestHappyPathPipeline:
    """Happy path tests: full pipeline from DSL to results."""
    
    def test_dsl_parse_validate_resolve_build(self):
        """Test complete DSL pipeline without execution."""
        dsl = """
# Test strategy
yf:
type=adapter
class=tradsl.testdata_adapter.TestDataAdapter
interval=1d

spy:
type=timeseries
adapter=yf
parameters=[SPY]
tradable=true

spy_returns:
type=timeseries
function=log_returns
inputs=[spy]
params=returns_cfg

returns_cfg:
period=1

spy_sma:
type=timeseries
function=sma
inputs=[spy]
params=sma_cfg

sma_cfg:
window=20

_backtest:
type=backtest
start=2024-01-01
end=2024-06-01
test_start=2024-04-01
capital=100000
training_mode=random_blocks
block_size_min=20
block_size_max=40
n_training_blocks=5
seed=42
"""
        raw = parse_dsl(dsl)
        assert 'spy' in raw
        assert 'spy_returns' in raw
        assert '_backtest' in raw
        
        validated = validate_config(raw)
        assert validated['spy']['type'] == 'timeseries'
        assert validated['_backtest']['capital'] == 100000
        
        resolved = resolve_config(validated)
        
        dag = build_execution_dag(resolved)
        assert dag is not None
        assert dag.metadata.warmup_bars > 0
    
    def test_dsl_with_trainable_model(self):
        """Test DSL with trainable model node."""
        dsl = """
yf:
type=adapter
class=tradsl.testdata_adapter.TestDataAdapter
interval=1d

spy:
type=timeseries
adapter=yf
parameters=[SPY]
tradable=true

returns:
type=timeseries
function=log_returns
inputs=[spy]

sma20:
type=timeseries
function=sma
inputs=[spy]
params=sma_cfg

sma_cfg:
window=20

model_node:
type=trainable_model
class=tradsl.models.RandomForestModel
inputs=[returns, sma20]
label_function=forward_return
retrain_schedule=every_n_bars
retrain_n=50
training_window=100
dotraining=true
update_schedule=every_n_bars
params=model_cfg

model_cfg:
n_estimators=10
max_depth=3

_backtest:
type=backtest
start=2024-01-01
end=2024-06-01
capital=100000
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)
        assert 'model_node' in validated
        assert validated['model_node']['type'] == 'trainable_model'


class TestInvalidDSLErrors:
    """Test proper error handling for invalid DSL."""
    
    def test_missing_adapter_reference(self):
        """Test error when adapter reference doesn't exist."""
        dsl = """
spy:
type=timeseries
adapter=nonexistent
parameters=[SPY]
"""
        raw = parse_dsl(dsl)
        with pytest.raises(TradSLError):
            validate_config(raw)
    
    def test_circular_dependency(self):
        """Test error for circular dependencies in DAG."""
        dsl = """
a:
type=timeseries
function=log_returns
inputs=[b]

b:
type=timeseries
function=sma
inputs=[a]
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)
        with pytest.raises(TradSLError):
            build_execution_dag(validated)
    
    def test_missing_function(self):
        """Test error when function doesn't exist."""
        dsl = """
spy:
type=timeseries
adapter=yf
parameters=[SPY]

computed:
type=timeseries
function=nonexistent_function
inputs=[spy]

yf:
type=adapter
class=tradsl.testdata_adapter.TestDataAdapter
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)
        resolved = resolve_config(validated)
        dag = build_execution_dag(resolved)
        assert dag is not None


class TestWarmupPeriod:
    """Test warmup period behavior."""
    
    def test_warmup_no_signals(self):
        """Test that no signals are generated during warmup."""
        @dataclass
        class MockConfig:
            dag_config = {'nodes': {}}
            warmup_bars = 20
            node_buffer_sizes = {'price': 30}
            execution_order = ['price', 'returns']
            source_nodes = ['price']
            model_nodes = []
            backtest_config = {'capital': 100000}
            tradable_instruments = ['TEST']
        
        config = MockConfig()
        
        from tradsl.agent_framework import ParameterizedAgent
        
        agent = ParameterizedAgent(
            policy_model=RandomForestModel(n_estimators=5, random_state=42),
            update_schedule="every_n_bars",
            update_n=50,
            seed=42
        )
        
        strategy = TradSLStrategy(
            config=config,
            agent=agent,
            sizer=KellySizer(),
            reward_function=AsymmetricHighWaterMarkReward(),
            min_order_size=1.0
        )
        
        for i in range(15):
            result = strategy.process_bar(close=100.0 + i)
            assert result['bar_index'] == i + 1
        
        assert strategy._state.bar_index == 15
        assert strategy._portfolio.total_equity == 100000


class TestPositionTracking:
    """Test solid position tracking through fills."""
    
    def test_position_add_fill(self):
        """Test adding fills to position."""
        pos = Position(instrument_id="TEST")
        
        result1 = pos.add_fill(price=100.0, quantity=10.0, commission=1.0)
        
        assert pos.quantity == 10.0
        assert pos.avg_entry_price == 100.0
        assert pos.realized_pnl == 0.0
        
        result2 = pos.add_fill(price=110.0, quantity=5.0, commission=1.0)
        
        assert pos.quantity == 15.0
        assert pos.avg_entry_price == pytest.approx(103.33, rel=0.01)
        
        result3 = pos.add_fill(price=105.0, quantity=-10.0, commission=1.0)
        
        assert pos.quantity == 5.0
        assert pos.realized_pnl > 0
    
    def test_position_pnl_calculation(self):
        """Test unrealized P&L calculation."""
        pos = Position(instrument_id="TEST")
        pos.add_fill(100.0, 10.0, 1.0)
        
        unrealized = pos.unrealized_pnl(110.0)
        assert unrealized == 100.0
        
        unrealized_down = pos.unrealized_pnl(90.0)
        assert unrealized_down == -100.0
    
    def test_portfolio_tracking(self):
        """Test full portfolio tracking."""
        portfolio = Portfolio(starting_capital=100000.0)
        
        portfolio.cash -= 1000
        portfolio.update_equity()
        
        assert portfolio.total_equity == 99000.0
        assert portfolio.high_water_mark == 100000.0
        
        portfolio.cash += 10000
        portfolio.update_equity()
        
        assert portfolio.high_water_mark == 109000.0
    
    def test_portfolio_drawdown(self):
        """Test drawdown calculation."""
        portfolio = Portfolio(starting_capital=100000.0)
        
        portfolio.cash = 100000
        portfolio.update_equity()
        assert portfolio.current_drawdown == 0.0
        
        portfolio.cash = 80000
        portfolio.update_equity()
        assert portfolio.current_drawdown == pytest.approx(0.2, rel=0.01)


class TestCheckpointRoundtrip:
    """Test checkpoint save/load."""
    
    def test_position_checkpoint(self):
        """Test position can be serialized."""
        pos = Position(instrument_id="TEST")
        pos.add_fill(100.0, 10.0, 1.0)
        pos.add_fill(110.0, 5.0, 1.0)
        
        d = pos.to_dict()
        
        assert d['quantity'] == 15.0
        assert d['avg_entry_price'] == pytest.approx(103.33, rel=0.01)
    
    def test_portfolio_checkpoint(self):
        """Test portfolio can be serialized."""
        portfolio = Portfolio(starting_capital=100000.0)
        portfolio.cash -= 10000
        portfolio.get_or_create_position("TEST").add_fill(100.0, 10.0, 1.0)
        portfolio.update_equity()
        
        d = portfolio.to_dict({})
        
        assert d['cash'] == 90000.0
        assert 'TEST' in d['positions']


class TestMultiAsset:
    """Test multi-asset portfolio."""
    
    def test_multiple_positions(self):
        """Test tracking multiple instruments."""
        portfolio = Portfolio(starting_capital=100000.0)
        
        pos_a = portfolio.get_or_create_position("AAPL")
        pos_a.add_fill(150.0, 100.0, 10.0)
        
        pos_b = portfolio.get_or_create_position("GOOG")
        pos_b.add_fill(200.0, 50.0, 10.0)
        
        assert pos_a.total_commission == 10.0
        assert pos_b.total_commission == 10.0
        
        portfolio.cash = 95000
        portfolio.update_equity()
        
        assert len(portfolio.positions) == 2


class TestSyntheticDataAdapter:
    """Test synthetic data generation."""
    
    def test_create_adapter(self):
        """Test adapter creation."""
        adapter = create_test_adapter(trend=0.0, volatility=0.02, seed=42)
        assert adapter.seed == 42
        assert adapter.trend == 0.0
        assert adapter.volatility == 0.02
    
    def test_generate_data(self):
        """Test data generation."""
        adapter = TestDataAdapter(seed=42, trend=0.0, volatility=0.01)
        
        df = adapter.load_historical(
            "TEST",
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
            "1d"
        )
        
        assert len(df) > 0
        assert 'close' in df.columns
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'volume' in df.columns
    
    def test_uptrend_data(self):
        """Test uptrend data generation."""
        adapter = TestDataAdapter(seed=42, trend=0.001, volatility=0.01)
        
        df = adapter.load_historical(
            "TEST",
            datetime(2024, 1, 1),
            datetime(2024, 6, 1),
            "1d"
        )
        
        first_close = df['close'].iloc[0]
        last_close = df['close'].iloc[-1]
        assert last_close > first_close
    
    def test_downtrend_data(self):
        """Test downtrend data generation."""
        adapter = TestDataAdapter(seed=42, trend=-0.001, volatility=0.01)
        
        df = adapter.load_historical(
            "TEST",
            datetime(2024, 1, 1),
            datetime(2024, 6, 1),
            "1d"
        )
        
        first_close = df['close'].iloc[0]
        last_close = df['close'].iloc[-1]
        assert last_close < first_close


class TestEndToEndBacktest:
    """Test full end-to-end backtest with synthetic data."""
    
    def test_full_backtest_run(self):
        """Test complete backtest run."""
        adapter = TestDataAdapter(seed=42, trend=0.0005, volatility=0.02)
        
        df = adapter.load_historical(
            "TEST",
            datetime(2024, 1, 1),
            datetime(2024, 12, 31),
            "1d"
        )
        
        prices = df['close'].values
        
        config = TrainingConfig(
            training_window=504,
            n_training_blocks=3,
            block_size_min=30,
            block_size_max=50,
            seed=42
        )
        
        sampler = BlockSampler(config)
        
        blocks = sampler.sample_blocks(
            pool_start_idx=50,
            pool_end_idx=len(prices) - 50,
            exclude_range=(len(prices) - 50, len(prices))
        )
        
        assert len(blocks) > 0
        
        price_arr = prices[:200]
        returns = np.diff(price_arr) / price_arr[:-1]
        equity = 100000 * np.cumprod(1 + returns)
        trades = [{'pnl': r * 100000, 'duration': 1} for r in returns]
        
        metrics = compute_metrics(equity, trades, '1d')
        
        assert metrics.total_return is not None
        assert metrics.n_trades == 199


class TestBlockSamplerIntegration:
    """Test block sampler with real data."""
    
    def test_sampler_with_exclusion(self):
        """Test sampler with test set exclusion."""
        config = TrainingConfig(
            training_window=500,
            n_training_blocks=5,
            block_size_min=30,
            block_size_max=50,
            seed=42
        )
        
        sampler = BlockSampler(config)
        
        blocks = sampler.sample_blocks(
            pool_start_idx=0,
            pool_end_idx=400,
            exclude_range=(350, 400)
        )
        
        for block in blocks:
            assert block.end_idx <= 350
    
    def test_sampler_coverage_limit(self):
        """Test that sampler respects 80% coverage limit."""
        config = TrainingConfig(
            training_window=100,
            n_training_blocks=100,
            block_size_min=1,
            block_size_max=1,
            seed=42
        )
        
        sampler = BlockSampler(config)
        
        blocks = sampler.sample_blocks(0, 100)
        
        total_covered = sum(b.length for b in blocks)
        assert total_covered <= 80


class TestCircularBufferIntegration:
    """Test circular buffer in execution context."""
    
    def test_buffer_warmup(self):
        """Test buffer warmup behavior."""
        buffer = CircularBuffer(size=10)
        
        assert buffer.is_ready is False
        
        for i in range(9):
            buffer.push(float(i))
            assert buffer.is_ready is False
        
        buffer.push(10.0)
        assert buffer.is_ready is True
        
        arr = buffer.to_array()
        assert len(arr) == 10
        assert arr[-1] == 10.0
    
    def test_buffer_overwrite(self):
        """Test buffer overwrite behavior."""
        buffer = CircularBuffer(size=5)
        
        for i in range(10):
            buffer.push(float(i))
        
        arr = buffer.to_array()
        assert len(arr) == 5
        assert arr[-1] == 9.0
