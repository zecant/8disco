import pytest
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, '.')

from tradsl.signals import (
    TradingSignal, 
    SignalBatch, 
    SignalType, 
    TradingAction
)
from tradsl.sizing import (
    PositionSizer,
    Allocation,
    AllocationPlan,
    EqualWeightSizer,
    FixedFractionSizer,
    KellySizer,
    ConfidenceWeightedSizer,
    create_sizer
)
from tradsl.portfolio.tracker import PortfolioTracker, Position, PortfolioSnapshot
from tradsl.backtest.execution import (
    ExecutionBackend,
    ImmediateExecutionBackend,
    SimulationExecutionBackend,
    Order,
    OrderSide,
    OrderType,
    Fill,
    create_execution_backend
)
from tradsl.backtest.engine import BacktestEngine, BacktestConfig, run_backtest


class TestTradingSignals:
    """Tests for trading signal abstractions."""
    
    def test_create_action_signal(self):
        """Test creating action signals."""
        signal = TradingSignal.buy('AAPL', confidence=0.8)
        assert signal.symbol == 'AAPL'
        assert signal.action == TradingAction.BUY
        assert signal.confidence == 0.8
        
        signal = TradingSignal.sell('AAPL', confidence=0.7)
        assert signal.action == TradingAction.SELL
        
        signal = TradingSignal.hold('AAPL', confidence=0.3)
        assert signal.action == TradingAction.HOLD
    
    def test_signal_to_dict(self):
        """Test signal serialization."""
        signal = TradingSignal.buy('AAPL', confidence=0.8)
        d = signal.to_dict()
        assert d['symbol'] == 'AAPL'
        assert d['action'] == 'buy'
        assert d['confidence'] == 0.8
    
    def test_signal_from_dict(self):
        """Test signal deserialization."""
        d = {'symbol': 'AAPL', 'action': 'sell', 'confidence': 0.6}
        signal = TradingSignal.from_dict(d)
        assert signal.symbol == 'AAPL'
        assert signal.action == TradingAction.SELL
    
    def test_signal_from_model_output(self):
        """Test creating signal from model output dict."""
        model_output = {'action': 'buy', 'confidence': 0.75, 'proba_buy': 0.8}
        signal = TradingSignal.from_model_output('AAPL', model_output)
        assert signal.symbol == 'AAPL'
        assert signal.action == TradingAction.BUY
        assert signal.confidence == 0.75
    
    def test_signal_is_actionable(self):
        """Test actionable property."""
        buy_signal = TradingSignal.buy('AAPL', confidence=0.8)
        assert buy_signal.is_actionable is True
        
        hold_signal = TradingSignal.hold('AAPL', confidence=0.8)
        assert hold_signal.is_actionable is False


class TestSignalBatch:
    """Tests for signal batch."""
    
    def test_add_and_get(self):
        """Test adding and retrieving signals."""
        batch = SignalBatch()
        signal = TradingSignal.buy('AAPL', confidence=0.8)
        batch.add(signal)
        
        assert batch.get('AAPL') is not None
        assert batch.get('AAPL').action == TradingAction.BUY
    
    def test_get_actionable(self):
        """Test filtering actionable signals."""
        batch = SignalBatch()
        batch.add(TradingSignal.buy('AAPL', confidence=0.8))
        batch.add(TradingSignal.hold('GOOG', confidence=0.5))
        batch.add(TradingSignal.sell('MSFT', confidence=0.7))
        
        actionable = batch.get_actionable()
        assert len(actionable) == 2
        assert 'AAPL' in actionable
        assert 'MSFT' in actionable
        assert 'GOOG' not in actionable


class TestAllocations:
    """Tests for allocation structures."""
    
    def test_allocation_creation(self):
        """Test creating allocations."""
        alloc = Allocation(
            symbol='AAPL',
            weight=0.2,
            target_value=10000,
            target_quantity=50,
            action=TradingAction.BUY
        )
        assert alloc.symbol == 'AAPL'
        assert alloc.weight == 0.2
    
    def test_allocation_plan(self):
        """Test allocation plan."""
        plan = AllocationPlan()
        plan.add(Allocation('AAPL', 0.5, 5000, 25, TradingAction.BUY))
        plan.add(Allocation('GOOG', 0.5, 5000, 5, TradingAction.SELL))
        
        assert len(plan) == 2
        assert plan.total_weight == 1.0
        assert plan.get('AAPL').symbol == 'AAPL'
        
        weights = plan.get_weights()
        assert weights['AAPL'] == 0.5


class TestPositionSizers:
    """Tests for position sizers."""
    
    def test_equal_weight_sizer(self):
        """Test equal weight sizer."""
        sizer = EqualWeightSizer(max_positions=2)
        
        batch = SignalBatch()
        batch.add(TradingSignal.buy('AAPL', confidence=0.8))
        batch.add(TradingSignal.buy('GOOG', confidence=0.7))
        
        plan = sizer.calculate(batch, portfolio_value=100000)
        
        assert plan.total_weight == 1.0
        assert abs(plan.get('AAPL').weight - 0.5) < 0.01
    
    def test_fixed_fraction_sizer(self):
        """Test fixed fraction sizer."""
        sizer = FixedFractionSizer(fraction=0.1)
        
        batch = SignalBatch()
        batch.add(TradingSignal.buy('AAPL', confidence=0.8))
        
        plan = sizer.calculate(batch, portfolio_value=100000)
        
        assert plan.get('AAPL').weight == 0.1
    
    def test_kelly_sizer(self):
        """Test Kelly criterion sizer."""
        sizer = KellySizer(kelly_fraction=1.0, max_kelly=0.25)
        
        batch = SignalBatch()
        batch.add(TradingSignal.buy('AAPL', confidence=0.9))
        
        plan = sizer.calculate(batch, portfolio_value=100000)
        
        assert plan.get('AAPL').weight > 0
    
    def test_confidence_weighted_sizer(self):
        """Test confidence weighted sizer."""
        sizer = ConfidenceWeightedSizer()
        
        batch = SignalBatch()
        batch.add(TradingSignal.buy('AAPL', confidence=0.8))
        batch.add(TradingSignal.buy('GOOG', confidence=0.4))
        
        plan = sizer.calculate(batch, portfolio_value=100000)
        
        aapl_weight = plan.get('AAPL').weight
        googl_weight = plan.get('GOOG').weight
        assert aapl_weight > googl_weight
    
    def test_create_sizer(self):
        """Test sizer factory."""
        sizer = create_sizer('equal')
        assert isinstance(sizer, EqualWeightSizer)
        
        sizer = create_sizer('kelly')
        assert isinstance(sizer, KellySizer)


class TestPortfolioTracker:
    """Tests for portfolio tracker."""
    
    def test_initialization(self):
        """Test portfolio initialization."""
        tracker = PortfolioTracker(starting_cash=100000)
        
        assert tracker.cash == 100000
        assert tracker.equity == 100000
        assert tracker.total_pnl == 0
    
    def test_update_prices(self):
        """Test updating prices."""
        tracker = PortfolioTracker(starting_cash=100000)
        tracker.update_prices({'AAPL': 150.0})
        
        assert tracker.positions['AAPL'].current_price == 150.0
    
    def test_execute_buy_trade(self):
        """Test executing a buy trade."""
        tracker = PortfolioTracker(starting_cash=100000, commission_rate=0.0)
        tracker.update_prices({'AAPL': 100.0})
        
        trade = tracker.execute_trade('AAPL', 10, 100.0)
        
        assert trade['quantity'] == 10
        assert trade['side'] == 'buy'
        assert tracker.cash == 100000 - 1000
        assert tracker.positions['AAPL'].quantity == 10
    
    def test_execute_sell_trade(self):
        """Test executing a sell trade."""
        tracker = PortfolioTracker(starting_cash=100000, commission_rate=0.0)
        tracker.update_prices({'AAPL': 100.0})
        
        tracker.execute_trade('AAPL', 10, 100.0)
        trade = tracker.execute_trade('AAPL', -5, 100.0)
        
        assert trade['side'] == 'sell'
        assert tracker.positions['AAPL'].quantity == 5
    
    def test_portfolio_snapshot(self):
        """Test portfolio snapshot."""
        tracker = PortfolioTracker(starting_cash=100000)
        tracker.update_prices({'AAPL': 100.0})
        tracker.execute_trade('AAPL', 10, 100.0)
        
        snapshot = tracker.get_snapshot(datetime.now())
        
        assert snapshot.cash < 100000
        assert snapshot.position_value > 0
    
    def test_portfolio_to_features(self):
        """Test converting portfolio to features."""
        tracker = PortfolioTracker(starting_cash=100000)
        tracker.update_prices({'AAPL': 100.0})
        tracker.execute_trade('AAPL', 10, 100.0)
        
        features = tracker.get_snapshot().to_features()
        
        assert 'portfolio_cash' in features
        assert 'portfolio_equity' in features
        assert 'position_AAPL' in features
    
    def test_equity_curve(self):
        """Test equity curve tracking."""
        tracker = PortfolioTracker(starting_cash=100000)
        
        tracker.record_snapshot(100000)
        tracker.update_prices({'AAPL': 100.0})
        tracker.execute_trade('AAPL', 10, 100.0)
        tracker.record_snapshot(99000)
        
        curve = tracker.get_equity_curve()
        assert len(curve) == 2


class TestExecutionBackends:
    """Tests for execution backends."""
    
    def test_immediate_execution(self):
        """Test immediate execution backend."""
        backend = ImmediateExecutionBackend(commission_rate=0.001)
        
        order = Order(
            order_id='1',
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            metadata={'current_price': 100.0}
        )
        
        executed = backend.submit_order(order)
        
        assert executed.status.value == 'filled'
        assert executed.filled_quantity == 10
        
        fills = backend.get_fills()
        assert len(fills) == 1
        assert fills[0].price == 100.0
    
    def test_simulation_execution(self):
        """Test simulation execution backend."""
        backend = SimulationExecutionBackend(
            commission_rate=0.001,
            fill_probability=0.9
        )
        
        order = Order(
            order_id='1',
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            metadata={'current_price': 100.0}
        )
        
        executed = backend.submit_order(order)
        
        assert executed.status.value in ['filled', 'rejected']
    
    def test_create_execution_backend(self):
        """Test execution backend factory."""
        backend = create_execution_backend('immediate')
        assert isinstance(backend, ImmediateExecutionBackend)
        
        backend = create_execution_backend('simulation')
        assert isinstance(backend, SimulationExecutionBackend)


class TestBacktestEngine:
    """Tests for backtest engine."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'open': np.random.randn(50) + 100,
            'high': np.random.randn(50) + 102,
            'low': np.random.randn(50) + 98,
            'close': np.random.randn(50) + 100,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        data['symbol'] = 'AAPL'
        return data
    
    @pytest.fixture
    def dummy_model(self):
        """Create dummy model for testing."""
        class DummyModel:
            def __init__(self):
                self.is_trained = True
            
            def predict(self, X):
                return {'action': 'buy', 'confidence': 0.8}
        
        return DummyModel()
    
    def test_backtest_config(self, sample_data):
        """Test backtest configuration."""
        config = BacktestConfig(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 12, 31),
            starting_cash=100000,
            symbols=['AAPL']
        )
        
        assert config.starting_cash == 100000
        assert 'AAPL' in config.symbols
    
    def test_run_backtest_no_models(self, sample_data):
        """Test running backtest without models."""
        config = BacktestConfig(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 12, 31),
            starting_cash=100000,
            symbols=['AAPL']
        )
        
        engine = BacktestEngine(config=config, data=sample_data)
        result = engine.run()
        
        assert result.summary['num_trades'] == 0
        assert result.summary['ending_cash'] == 100000
    
    def test_run_backtest_with_model(self, sample_data, dummy_model):
        """Test running backtest with model."""
        config = BacktestConfig(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 12, 31),
            starting_cash=100000,
            symbols=['AAPL']
        )
        
        engine = BacktestEngine(
            config=config, 
            data=sample_data,
            models={'AAPL': dummy_model}
        )
        result = engine.run()
        
        assert result.summary['num_trades'] >= 0
    
    def test_run_backtest_convenience_function(self, sample_data, dummy_model):
        """Test convenience backtest function."""
        result = run_backtest(
            data=sample_data,
            models={'AAPL': dummy_model},
            start='2020-01-01',
            end='2020-12-31',
            symbols=['AAPL'],
            starting_cash=100000,
            sizer_type='equal'
        )
        
        assert result.summary['starting_cash'] == 100000
        assert 'sharpe_ratio' in result.summary
    
    def test_backtest_result_dataframe(self, sample_data):
        """Test converting results to dataframe."""
        config = BacktestConfig(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 12, 31),
            starting_cash=100000,
            symbols=['AAPL']
        )
        
        engine = BacktestEngine(config=config, data=sample_data)
        result = engine.run()
        
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_empty_data(self):
        """Test handling empty data."""
        empty_data = pd.DataFrame()
        
        config = BacktestConfig(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 12, 31),
            starting_cash=100000,
            symbols=['AAPL']
        )
        
        engine = BacktestEngine(config=config, data=empty_data)
        result = engine.run()
        
        assert len(result.equity_curve) == 0


class TestSizersWithNoSignals:
    """Tests for sizers with no actionable signals."""
    
    def test_equal_weight_no_signals(self):
        """Test equal weight sizer with no signals."""
        sizer = EqualWeightSizer()
        batch = SignalBatch()
        
        plan = sizer.calculate(batch, portfolio_value=100000)
        
        assert len(plan) == 0
    
    def test_kelly_no_signals(self):
        """Test Kelly sizer with no signals."""
        sizer = KellySizer()
        batch = SignalBatch()
        
        plan = sizer.calculate(batch, portfolio_value=100000)
        
        assert len(plan) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
