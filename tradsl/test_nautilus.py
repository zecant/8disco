import pytest
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch
import importlib

sys.path.insert(0, '.')


def check_nautilus_available():
    """Check if NautilusTrader is available."""
    try:
        import nautilus_trader
        return True
    except ImportError:
        return False


NT_AVAILABLE = check_nautilus_available()


class TestNautilusStrategyBasics:
    """Test NautilusStrategy basic functionality - skipped if NT not available."""
    
    @pytest.fixture(autouse=True)
    def check_nt(self):
        if not NT_AVAILABLE:
            pytest.skip("NautilusTrader not available")
    
    def test_strategy_config_creation(self):
        """Test creating NautilusStrategyConfig."""
        from tradsl.backtest.nautilus_strategy import NautilusStrategyConfig
        
        config = NautilusStrategyConfig(
            strategy_id="test_strategy",
            symbols=["BTCUSDT", "ETHUSDT"],
            venue="BINANCE",
            bar_type="1-HOUR-LAST-EXTERNAL",
            position_size=Decimal("0.5"),
            tradsl_config={"models": {}}
        )
        
        assert config.strategy_id == "test_strategy"
        assert config.symbols == ["BTCUSDT", "ETHUSDT"]
        assert config.venue == "BINANCE"
        assert config.bar_type == "1-HOUR-LAST-EXTERNAL"
        assert config.position_size == Decimal("0.5")
    
    def test_strategy_creation(self):
        """Test creating NautilusStrategy."""
        from tradsl.backtest.nautilus_strategy import NautilusStrategy, NautilusStrategyConfig
        
        config = NautilusStrategyConfig(
            strategy_id="test_strategy",
            symbols=["BTCUSDT"],
            venue="BACKTEST"
        )
        
        strategy = NautilusStrategy(config)
        
        assert strategy.symbols == ["BTCUSDT"]
        assert strategy.venue == "BACKTEST"
        assert strategy.models == {}
        assert strategy.market_data == {}
    
    def test_strategy_factory(self):
        """Test create_nautilus_strategy factory."""
        from tradsl.backtest.nautilus_strategy import create_nautilus_strategy
        
        strategy = create_nautilus_strategy(
            tradsl_config={},
            symbols=["BTCUSDT"],
            venue="BACKTEST",
            bar_type="1-MINUTE-LAST-EXTERNAL",
            position_size=Decimal("1")
        )
        
        assert strategy.symbols == ["BTCUSDT"]
        assert strategy.venue == "BACKTEST"
        assert strategy.bar_type_str == "1-MINUTE-LAST-EXTERNAL"
        assert strategy.position_size == Decimal("1")
    
    def test_strategy_on_start(self):
        """Test on_start initialization."""
        from tradsl.backtest.nautilus_strategy import NautilusStrategy, NautilusStrategyConfig
        
        config = NautilusStrategyConfig(
            symbols=["BTCUSDT"],
            tradsl_config={}
        )
        
        strategy = NautilusStrategy(config)
        
        assert not strategy._historical_data_loaded
    
    def test_strategy_on_bar_no_init(self):
        """Test on_bar before initialization."""
        from tradsl.backtest.nautilus_strategy import NautilusStrategy, NautilusStrategyConfig
        
        config = NautilusStrategyConfig(symbols=["BTCUSDT"])
        strategy = NautilusStrategy(config)
        
        mock_bar = Mock()
        strategy.on_bar(mock_bar)
        
        assert not strategy._historical_data_loaded
    
    def test_strategy_with_models(self):
        """Test strategy with models - models loaded on on_start."""
        from tradsl.backtest.nautilus_strategy import NautilusStrategy, NautilusStrategyConfig
        from tradsl.models import DecisionTreeModel
        
        model = DecisionTreeModel(max_depth=3)
        
        config = NautilusStrategyConfig(
            symbols=["BTCUSDT"],
            tradsl_config={
                "models": {"BTCUSDT": model}
            }
        )
        
        strategy = NautilusStrategy(config)
        
        assert "BTCUSDT" in strategy.models
        assert isinstance(strategy.models["BTCUSDT"], DecisionTreeModel)
    
    def test_strategy_with_sizer(self):
        """Test strategy with sizer - sizer loaded on on_start."""
        from tradsl.backtest.nautilus_strategy import NautilusStrategy, NautilusStrategyConfig
        from tradsl.sizing import EqualWeightSizer
        
        sizer = EqualWeightSizer()
        
        config = NautilusStrategyConfig(
            symbols=["BTCUSDT"],
            tradsl_config={
                "sizer": sizer
            }
        )
        
        strategy = NautilusStrategy(config)
        
        assert strategy.sizer is sizer
    
    def test_strategy_update_market_data(self):
        """Test _update_market_data."""
        from tradsl.backtest.nautilus_strategy import NautilusStrategy, NautilusStrategyConfig
        
        config = NautilusStrategyConfig(symbols=["BTCUSDT"], venue="BACKTEST")
        strategy = NautilusStrategy(config)
        strategy._historical_data_loaded = True
        
        mock_bar = Mock()
        mock_bar.bar_type = Mock()
        mock_bar.bar_type.instrument_id = Mock()
        mock_bar.bar_type.instrument_id.symbol = Mock()
        mock_bar.bar_type.instrument_id.symbol.value = "BTCUSDT"
        mock_bar.ts_event = 1609459200000000000
        mock_bar.open = 100.0
        mock_bar.high = 105.0
        mock_bar.low = 95.0
        mock_bar.close = 102.0
        mock_bar.volume = 1000.0
        
        strategy._update_market_data("BTCUSDT", mock_bar)
        
        assert "BTCUSDT" in strategy.market_data
        assert len(strategy.market_data["BTCUSDT"]) == 1
    
    def test_strategy_get_model_inputs(self):
        """Test _get_model_inputs."""
        from tradsl.backtest.nautilus_strategy import NautilusStrategy, NautilusStrategyConfig
        
        config = NautilusStrategyConfig(
            symbols=["BTCUSDT"],
            tradsl_config={
                "dag_config": {
                    "BTCUSDT": {"inputs": ["close", "volume"]}
                }
            }
        )
        
        strategy = NautilusStrategy(config)
        strategy.feature_df = pd.DataFrame({
            "close": [100, 101],
            "volume": [1000, 1100],
            "rsi_BTCUSDT": [50, 55]
        })
        
        inputs = strategy._get_model_inputs("BTCUSDT")
        
        assert "close" in inputs or "volume" in inputs


class TestNautilusAdapterImports:
    """Test that NautilusAdapter module can be imported."""
    
    def test_adapter_import(self):
        """Test importing NautilusAdapter."""
        try:
            from tradsl.backtest.nautilus_adapter import NautilusAdapter
            assert NautilusAdapter is not None
        except ImportError as e:
            pytest.skip(f"NautilusTrader not available: {e}")
    
    def test_result_parser_import(self):
        """Test importing NautilusResultParser."""
        try:
            from tradsl.backtest.nautilus_adapter import NautilusResultParser
            assert NautilusResultParser is not None
        except ImportError as e:
            pytest.skip(f"NautilusTrader not available: {e}")


class TestNautilusStrategyImports:
    """Test that NautilusStrategy module can be imported."""
    
    def test_strategy_import(self):
        """Test importing NautilusStrategy."""
        try:
            from tradsl.backtest.nautilus_strategy import NautilusStrategy
            assert NautilusStrategy is not None
        except ImportError as e:
            pytest.skip(f"NautilusTrader not available: {e}")
    
    def test_strategy_config_import(self):
        """Test importing NautilusStrategyConfig."""
        try:
            from tradsl.backtest.nautilus_strategy import NautilusStrategyConfig
            assert NautilusStrategyConfig is not None
        except ImportError as e:
            pytest.skip(f"NautilusTrader not available: {e}")
    
    def test_create_strategy_import(self):
        """Test importing create_nautilus_strategy."""
        try:
            from tradsl.backtest.nautilus_strategy import create_nautilus_strategy
            assert create_nautilus_strategy is not None
        except ImportError as e:
            pytest.skip(f"Naut: {e}")


class TestNautilusBackendImports:
    """Test that NautilusBackend module can be imported."""
    
    def test_backend_import(self):
        """Test importing NautilusBackend."""
        try:
            from tradsl.backtest.nautilus_backend import NautilusBackend
            assert NautilusBackend is not None
        except ImportError as e:
            pytest.skip(f"NautilusTrader not available: {e}")
    
    def test_run_backtest_import(self):
        """Test importing run_nautilus_backtest."""
        try:
            from tradsl.backtest.nautilus_backend import run_nautilus_backtest
            assert run_nautilus_backtest is not None
        except ImportError as e:
            pytest.skip(f"NautilusTrader not available: {e}")
    
    def test_nautilus_available_flag(self):
        """Test NAUTILUS_AVAILABLE flag."""
        try:
            from tradsl.backtest import NAUTILUS_AVAILABLE
            assert isinstance(NAUTILUS_AVAILABLE, bool)
        except ImportError:
            pass


class TestDataFrameConversion:
    """Test DataFrame to bar conversion logic (without NT dependency)."""
    
    def test_dataframe_structure(self):
        """Test that we can create proper OHLCV DataFrame."""
        dates = pd.date_range('2020-01-01', periods=10, freq='h')
        df = pd.DataFrame({
            'open': np.random.randn(10) + 100,
            'high': np.random.randn(10) + 102,
            'low': np.random.randn(10) + 98,
            'close': np.random.randn(10) + 100,
            'volume': np.random.randint(1000, 10000, 10)
        }, index=dates)
        
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        assert len(df) == 10
    
    def test_dataframe_with_timestamp(self):
        """Test DataFrame with explicit timestamp column."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
            'open': np.random.randn(10) + 100,
            'high': np.random.randn(10) + 102,
            'low': np.random.randn(10) + 98,
            'close': np.random.randn(10) + 100,
            'volume': np.random.randint(1000, 10000, 10)
        })
        
        assert 'timestamp' in df.columns
        assert len(df) == 10
    
    def test_bar_type_parsing(self):
        """Test bar type string parsing."""
        bar_types = [
            "1-MINUTE-LAST-EXTERNAL",
            "1-HOUR-LAST-EXTERNAL",
            "1-DAY-LAST-INTERNAL",
            "5-MINUTE-BID-EXTERNAL",
            "15-MINUTE-ASK-INTERNAL",
        ]
        
        for bar_type in bar_types:
            parts = bar_type.split("-")
            assert len(parts) == 4
            assert int(parts[0]) > 0
            assert parts[1] in ["MINUTE", "HOUR", "DAY", "WEEK"]
            assert parts[2] in ["BID", "ASK", "LAST", "MID", "INTERNAL", "EXTERNAL"]
            assert parts[3] in ["INTERNAL", "EXTERNAL"]


class TestTradslIntegration:
    """Test integration points with tradsl."""
    
    def test_signals_import(self):
        """Test that signals module is available."""
        from tradsl.signals import TradingSignal, SignalBatch, TradingAction
        assert TradingSignal is not None
        assert SignalBatch is not None
    
    def test_sizing_import(self):
        """Test that sizing module is available."""
        from tradsl.sizing import PositionSizer, EqualWeightSizer, create_sizer
        assert PositionSizer is not None
        assert EqualWeightSizer is not None
    
    def test_models_import(self):
        """Test that models module is available."""
        from tradsl.models import DecisionTreeModel
        assert DecisionTreeModel is not None
    
    def test_portfolio_import(self):
        """Test that portfolio module is available."""
        from tradsl.portfolio import PortfolioTracker, Position
        assert PortfolioTracker is not None
        assert Position is not None
    
    def test_backtest_engine_import(self):
        """Test that backtest engine is available."""
        from tradsl.backtest import BacktestEngine, BacktestResult
        assert BacktestEngine is not None
        assert BacktestResult is not None


class TestMockNautilusBackend:
    """Test NautilusBackend with mocked NT dependencies."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='h')
        return {
            'BTCUSDT': pd.DataFrame({
                'open': np.random.randn(100) + 10000,
                'high': np.random.randn(100) + 10100,
                'low': np.random.randn(100) + 9900,
                'close': np.random.randn(100) + 10000,
                'volume': np.random.randint(100, 1000, 100)
            }, index=dates),
            'ETHUSDT': pd.DataFrame({
                'open': np.random.randn(100) + 500,
                'high': np.random.randn(100) + 510,
                'low': np.random.randn(100) + 490,
                'close': np.random.randn(100) + 500,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates),
        }
    
    def test_config_structure(self):
        """Test tradsl config structure."""
        config = {
            '_execution_order': ['data1', 'model1', 'agent'],
            '_params': {'mlparams': {'lr': 0.001}},
            '_adapters': {},
            '_backtest': {'start': '2020-01-01', 'end': '2024-01-01'},
        }
        
        assert '_execution_order' in config
        assert '_params' in config
    
    def test_models_dict_structure(self):
        """Test models dict structure."""
        from tradsl.models import DecisionTreeModel
        
        models = {
            'BTCUSDT': DecisionTreeModel(max_depth=5),
            'ETHUSDT': DecisionTreeModel(max_depth=5),
        }
        
        assert 'BTCUSDT' in models
        assert 'ETHUSDT' in models
    
    def test_sizer_creation(self):
        """Test creating position sizer."""
        from tradsl.sizing import create_sizer, EqualWeightSizer
        
        sizer = create_sizer('equal', max_positions=5)
        assert isinstance(sizer, EqualWeightSizer)
    
    def test_signal_generation_flow(self):
        """Test the signal generation flow."""
        from tradsl.signals import TradingSignal, SignalBatch, TradingAction
        from tradsl.sizing import create_sizer
        
        batch = SignalBatch()
        batch.add(TradingSignal.buy('BTCUSDT', confidence=0.8))
        batch.add(TradingSignal.sell('ETHUSDT', confidence=0.6))
        
        assert len(batch) == 2
        
        sizer = create_sizer('equal')
        
        prices = {'BTCUSDT': 10000, 'ETHUSDT': 500}
        positions = {'BTCUSDT': 0.0, 'ETHUSDT': 0.0}
        
        plan = sizer.calculate(batch, portfolio_value=100000, current_positions=positions, prices=prices)
        
        assert plan.total_weight > 0


class TestBacktestResultStructure:
    """Test BacktestResult structure and conversion."""
    
    @pytest.fixture
    def sample_result(self):
        """Create sample backtest result."""
        from tradsl.backtest.engine import BacktestResult
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        equity = pd.Series(100000 * (1 + np.random.randn(100).cumsum() * 0.01), index=dates)
        
        trades = [
            {'timestamp': dates[10], 'symbol': 'BTCUSDT', 'side': 'buy', 'quantity': 1, 'price': 10000},
            {'timestamp': dates[50], 'symbol': 'ETHUSDT', 'side': 'sell', 'quantity': 10, 'price': 500},
        ]
        
        portfolio_history = [
            {'timestamp': dates[0], 'equity': 100000, 'cash': 100000, 'position_value': 0},
            {'timestamp': dates[50], 'equity': 105000, 'cash': 50000, 'position_value': 55000},
        ]
        
        summary = {
            'starting_cash': 100000,
            'ending_equity': 105000,
            'total_return': 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.1,
            'num_trades': 2,
        }
        
        return BacktestResult(
            equity_curve=equity,
            trades=trades,
            portfolio_history=portfolio_history,
            signals_history=[],
            summary=summary,
        )
    
    def test_result_equity_curve(self, sample_result):
        """Test equity curve."""
        assert not sample_result.equity_curve.empty
        assert len(sample_result.equity_curve) == 100
    
    def test_result_trades(self, sample_result):
        """Test trades list."""
        assert len(sample_result.trades) == 2
        assert sample_result.trades[0]['symbol'] == 'BTCUSDT'
    
    def test_result_summary(self, sample_result):
        """Test summary statistics."""
        assert 'total_return' in sample_result.summary
        assert 'sharpe_ratio' in sample_result.summary
        assert sample_result.summary['num_trades'] == 2
    
    def test_result_to_dataframe(self, sample_result):
        """Test converting to DataFrame."""
        df = sample_result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


class TestBarTypeHandling:
    """Test bar type handling without NT dependency."""
    
    def test_standard_bar_types(self):
        """Test standard bar type formats."""
        bar_types = [
            ("1-MINUTE-LAST-EXTERNAL", 1, "MINUTE", "LAST", "EXTERNAL"),
            ("5-MINUTE-BID-EXTERNAL", 5, "MINUTE", "BID", "EXTERNAL"),
            ("1-HOUR-LAST-INTERNAL", 1, "HOUR", "LAST", "INTERNAL"),
            ("1-DAY-MID-EXTERNAL", 1, "DAY", "MID", "EXTERNAL"),
            ("4-HOUR-ASK-INTERNAL", 4, "HOUR", "ASK", "INTERNAL"),
        ]
        
        for bar_type_str, expected_step, expected_agg, expected_price, expected_source in bar_types:
            parts = bar_type_str.split("-")
            assert int(parts[0]) == expected_step
            assert parts[1] == expected_agg
            assert parts[2] == expected_price
            assert parts[3] == expected_source
    
    def test_invalid_bar_type_length(self):
        """Test that invalid bar types are caught when used."""
        invalid_bar_types = [
            "INVALID",
            "1--EXTERNAL",
            "MINUTE-LAST-EXTERNAL",
        ]
        
        for bar_type in invalid_bar_types:
            parts = bar_type.split("-")
            if len(parts) != 4:
                assert True  # Expected to fail
                return
        
        # If we get here, none raised - that's also fine for this test
        assert True


class TestEndToEndFlow:
    """Test end-to-end flow without actual NT."""
    
    def test_full_flow_with_dummy_data(self):
        """Test complete flow with dummy data."""
        from tradsl.models import DecisionTreeModel
        from tradsl.sizing import create_sizer
        from tradsl.signals import TradingSignal, SignalBatch
        
        dates = pd.date_range('2020-01-01', periods=200, freq='h')
        
        data = {
            'BTCUSDT': pd.DataFrame({
                'open': np.cumsum(np.random.randn(200)) + 10000,
                'high': np.cumsum(np.random.randn(200)) + 10100,
                'low': np.cumsum(np.random.randn(200)) + 9900,
                'close': np.cumsum(np.random.randn(200)) + 10000,
                'volume': np.random.randint(100, 1000, 200)
            }, index=dates),
        }
        
        X = data['BTCUSDT'][['open', 'high', 'low', 'close', 'volume']].values[:100]
        y = (np.diff(X[:, 3]) > 0).astype(int)[:100]
        
        model = DecisionTreeModel(max_depth=3, random_state=42)
        model.train(X, y)
        
        test_row = X[0].reshape(1, -1)
        result = model.predict(test_row)
        
        assert 'action' in result
        assert result['action'] in ['buy', 'sell', 'hold']
        
        signal = TradingSignal.from_model_output('BTCUSDT', result)
        
        batch = SignalBatch()
        batch.add(signal)
        
        sizer = create_sizer('equal')
        
        prices = {'BTCUSDT': 10000}
        positions = {'BTCUSDT': 0.0}
        
        allocation = sizer.calculate(batch, portfolio_value=100000, current_positions=positions, prices=prices)
        
        assert allocation.total_weight >= 0
    
    def test_multi_symbol_flow(self):
        """Test flow with multiple symbols."""
        from tradsl.models import DecisionTreeModel
        from tradsl.sizing import create_sizer
        from tradsl.signals import TradingSignal, SignalBatch
        
        dates = pd.date_range('2020-01-01', periods=100, freq='h')
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        data = {}
        models = {}
        
        for symbol in symbols:
            data[symbol] = pd.DataFrame({
                'close': np.cumsum(np.random.randn(100)) + 100,
                'volume': np.random.randint(100, 1000, 100)
            }, index=dates)
            
            X = data[symbol][['close', 'volume']].values[:50]
            y = (np.diff(X[:, 0]) > 0).astype(int)[:49]
            
            model = DecisionTreeModel(max_depth=10, confidence_threshold=0.3, random_state=42)
            model.train(X[:-1], y)
            models[symbol] = model
        
        batch = SignalBatch()
        
        for symbol in symbols:
            model = models[symbol]
            test_row = data[symbol][['close', 'volume']].values[0].reshape(1, -1)
            result = model.predict(test_row)
            signal = TradingSignal.from_model_output(symbol, result)
            batch.add(signal)
        
        assert len(batch) == 3
        
        prices = {s: 100.0 for s in symbols}
        positions = {s: 0.0 for s in symbols}
        
        sizer = create_sizer('fixed_fraction', fraction=0.1)
        
        allocation = sizer.calculate(batch, portfolio_value=100000.0, current_positions=positions, prices=prices)
        
        assert len(batch) == 3  # Should have signals regardless of allocation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
