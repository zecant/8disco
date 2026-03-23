"""
Tests for Function and Adapter base classes and DAG pipeline execution.
"""
import pytest
import pandas as pd
from tradsl.dag import DAG, Node
from tradsl.functions import Function
from tradsl.pricetransforms import EMA, PairwiseCorrelation
from tradsl.adapters import Adapter, YFinanceAdapter


class TestFunctionBaseClass:
    """Tests for Function abstract base class."""

    def test_function_subclass(self):
        class MyFunc(Function):
            def apply(self, data):
                return data
        
        assert issubclass(MyFunc, Function)

    def test_function_apply_called(self):
        class Identity(Function):
            def apply(self, data):
                return data
        
        fn = Identity()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = fn.apply(df)
        assert isinstance(result, pd.DataFrame)


class TestAdapterBaseClass:
    """Tests for Adapter abstract base class."""

    def test_adapter_subclass(self):
        class MyAdapter(Adapter):
            def set_start(self, start_time):
                pass
            
            def tick(self):
                import pandas as pd
                return pd.DataFrame({"value": [42]})
        
        assert issubclass(MyAdapter, Adapter)

    def test_adapter_tick_called(self):
        class CounterAdapter(Adapter):
            def __init__(self):
                super().__init__()
                self.count = 0
            
            def set_start(self, start_time):
                pass
            
            def tick(self):
                import pandas as pd
                self.count += 1
                return pd.DataFrame({"count": [self.count]})
        
        adapter = CounterAdapter()
        assert adapter.tick()["count"].iloc[0] == 1
        assert adapter.tick()["count"].iloc[0] == 2


class TestResolveWithFunctions:
    """Tests for resolve with Function classes."""

    def test_resolve_function_class_auto_initializes(self):
        class SMA(Function):
            def __init__(self, window=5):
                self.window = window
            
            def apply(self, data):
                return data.rolling(self.window).mean().to_frame()
        
        config = {
            "fn": {"type": "function", "function": "sma", "inputs": ["ts"], "window": 10},
            "ts": {"type": "timeseries", "adapter": "a"},
        }
        dag = DAG.from_config(config).build()
        dag.resolve({"sma": SMA, "a": None})
        
        assert isinstance(dag.nodes["fn"].attrs["function"], SMA)
        assert dag.nodes["fn"].attrs["function"].window == 10

    def test_resolve_function_instance(self):
        class Mul(Function):
            def apply(self, data):
                return data * 2
        
        config = {
            "fn": {"type": "function", "function": "mul", "inputs": ["ts"]},
            "ts": {"type": "timeseries", "adapter": "a"},
        }
        dag = DAG.from_config(config).build()
        mul_instance = Mul()
        dag.resolve({"mul": mul_instance, "a": None})
        
        assert dag.nodes["fn"].attrs["function"] is mul_instance


class TestResolveWithAdapters:
    """Tests for resolve with Adapter classes."""

    def test_resolve_adapter_class_auto_initializes(self):
        class PriceAdapter(Adapter):
            def __init__(self, dag=None):
                super().__init__(dag=dag)
                self.step = 0
            
            def set_start(self, start_time):
                pass
            
            def tick(self):
                import pandas as pd
                self.step += 1
                return pd.DataFrame({"price": [self.step * 10]})
        
        config = {
            "prices": {"type": "timeseries", "adapter": "yfinance"},
        }
        dag = DAG.from_config(config).build()
        dag.resolve({"yfinance": PriceAdapter})
        
        assert isinstance(dag.nodes["prices"].attrs["adapter"], PriceAdapter)
        result = dag.nodes["prices"].attrs["adapter"].tick()
        assert result["price"].iloc[0] == 10


class TestYFinanceAdapter:
    """Tests for YFinanceAdapter."""

    def test_yfinance_adapter_returns_dataframe(self):
        from datetime import datetime, timedelta
        import pandas as pd
        adapter = YFinanceAdapter(symbol="AAPL", interval="1m")
        recent_date = datetime.now() - timedelta(days=2)
        adapter.set_start(recent_date)
        result = adapter.tick()
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 5  # OHLCV
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_yfinance_adapter_streams(self):
        from datetime import datetime, timedelta
        adapter = YFinanceAdapter(symbol="AAPL", interval="1m")
        recent_date = datetime.now() - timedelta(days=2)
        adapter.set_start(recent_date)
        first = adapter.tick()
        second = adapter.tick()
        assert (first is None and second is None) or not first.equals(second)


class TestDAGStep:
    """Tests for DAG step execution."""

    def test_step_timeseries_only(self):
        class SimpleAdapter(Adapter):
            def __init__(self, dag=None):
                super().__init__(dag=dag)
                self.step = 0
            
            def set_start(self, start_time):
                pass
            
            def tick(self):
                import pandas as pd
                self.step += 1
                return pd.DataFrame({"value": [self.step]})
        
        config = {
            "prices": {"type": "timeseries", "adapter": "adapter"},
        }
        dag = DAG.from_config(config).build()
        dag.resolve({"adapter": SimpleAdapter})
        
        dag.step()
        dag.step()
        
        values = dag.values()
        assert values[0][0] == "prices"
        result = values[0][1]
        assert isinstance(result, pd.DataFrame)
        assert result["value"].iloc[-1] == 2

    def test_step_with_function(self):
        class CountAdapter(Adapter):
            def __init__(self, dag=None):
                super().__init__(dag=dag)
                self.count = 0
            
            def set_start(self, start_time):
                pass
            
            def tick(self):
                import pandas as pd
                self.count += 1
                return pd.DataFrame({"count": [self.count]})
        
        class Double(Function):
            def apply(self, data):
                import pandas as pd
                result = data.copy()
                for col in result.columns:
                    result[col] = result[col] * 2
                return result
        
        config = {
            "source": {"type": "timeseries", "adapter": "adapter"},
            "doubled": {"type": "function", "function": "double", "inputs": ["source"]},
        }
        dag = DAG.from_config(config).build()
        dag.resolve({"adapter": CountAdapter, "double": Double})
        
        dag.step()
        dag.step()
        
        values = dag.values()
        assert values[0][0] == "source"
        assert values[1][0] == "doubled"

    def test_values_returns_ordered_results(self):
        class SimpleAdapter(Adapter):
            def __init__(self, dag=None):
                super().__init__(dag=dag)
            
            def set_start(self, start_time):
                pass
            
            def tick(self):
                import pandas as pd
                return pd.DataFrame({"value": [42]})
        
        class Identity(Function):
            def apply(self, data):
                return data
        
        config = {
            "a": {"type": "timeseries", "adapter": "adapter"},
            "b": {"type": "function", "function": "id", "inputs": ["a"]},
            "c": {"type": "function", "function": "id", "inputs": ["b"]},
        }
        dag = DAG.from_config(config).build()
        dag.resolve({"adapter": SimpleAdapter, "id": Identity})
        
        dag.step()
        
        values = dag.values()
        assert len(values) == 3
        names = [v[0] for v in values]
        assert names == ["a", "b", "c"]


class TestEMA:
    """Tests for EMA function."""

    def test_ema_initialization(self):
        ema = EMA(window=20)
        assert ema.window == 20
        assert ema.alpha == 2 / 21

    def test_ema_applies_correctly(self):
        ema = EMA(window=3)
        data = pd.DataFrame({"price": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = ema.apply(data)
        assert result is not None
        assert "price" in result.columns
        assert len(result) == 1

    def test_ema_returns_none_when_insufficient_data(self):
        ema = EMA(window=10)
        data = pd.DataFrame({"price": [1.0, 2.0, 3.0]})
        result = ema.apply(data)
        assert result is None

    def test_ema_stateful_across_calls(self):
        ema = EMA(window=3)
        # First call with full history (like DAG provides)
        data1 = pd.DataFrame({"price": [None, None, 1.0, 2.0, 3.0]})
        result1 = ema.apply(data1)
        assert result1 is not None
        
        # Subsequent calls receive full buffer (including new values)
        data2 = pd.DataFrame({"price": [None, 1.0, 2.0, 3.0, 4.0]})
        result2 = ema.apply(data2)
        assert result2 is not None
        
        # Verify EMA is actually updating (value should change)
        val1 = result1.iloc[0, 0]
        val2 = result2.iloc[0, 0]
        assert val1 != val2


class TestPairwiseCorrelation:
    """Tests for PairwiseCorrelation function."""

    def test_correlation_initialization(self):
        corr = PairwiseCorrelation(window=20)
        assert corr.window == 20

    def test_correlation_perfect_positive(self):
        corr = PairwiseCorrelation(window=5)
        data = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        result = corr.apply(data)
        assert result is not None
        assert abs(result.iloc[0, 0] - 1.0) < 0.001

    def test_correlation_perfect_negative(self):
        corr = PairwiseCorrelation(window=5)
        data = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0]
        })
        result = corr.apply(data)
        assert result is not None
        assert abs(result.iloc[0, 0] + 1.0) < 0.001

    def test_correlation_returns_none_when_insufficient_data(self):
        corr = PairwiseCorrelation(window=10)
        data = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})
        result = corr.apply(data)
        assert result is None


class TestMLFunctions:
    """Tests for ML functions."""

    def test_ml_function_warmup(self):
        from tradsl.mlfunctions import Regressor
        regressor = Regressor(warmup=10)
        data = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        
        # Before warmup - should return None
        for _ in range(9):
            result = regressor.apply(data)
            assert result is None
        
        # After warmup - should be ready (no model so returns None)
        result = regressor.apply(data)
        assert result is None

    def test_ml_function_with_model(self):
        import sklearn.ensemble
        from tradsl.mlfunctions import Regressor
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
        regressor = Regressor(model=model, warmup=1)
        
        # Create simple training data
        import numpy as np
        X_train = pd.DataFrame({"feature": np.arange(100).astype(float)})
        y_train = np.arange(100).astype(float)
        model.fit(X_train, y_train)
        
        # After warmup - should predict
        data = pd.DataFrame({"feature": [50.0, 51.0, 52.0]})
        result = regressor.apply(data)
        assert result is not None

    def test_classifier(self):
        import sklearn.ensemble
        from tradsl.mlfunctions import Classifier
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
        classifier = Classifier(model=model, warmup=1)
        
        import numpy as np
        X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [0.0, 1.0, 0.0, 1.0]})
        y_train = np.array([0, 0, 1, 1])
        model.fit(X_train, y_train)
        
        data = pd.DataFrame({"f1": [2.5, 3.5], "f2": [0.5, 0.5]})
        result = classifier.apply(data)
        assert result is not None

    def test_agent_no_model(self):
        from tradsl.mlfunctions import Agent
        agent = Agent(warmup=1, n_actions=3)
        
        data = pd.DataFrame({"state": [1.0, 2.0, 3.0]})
        # Agent without model returns None after warmup
        agent.apply(data)  # warmup tick
        result = agent.apply(data)
        assert result is None  # No model, returns None

    def test_dummy_agent(self):
        from tradsl.ml.agents import DummyAgent
        agent = DummyAgent(warmup=1, n_actions=3)
        
        data = pd.DataFrame({"state": [1.0, 2.0, 3.0]})
        result = agent.apply(data)
        assert result is not None
        assert "action" in result.columns
        assert "confidence" in result.columns
        assert "asset" in result.columns

    def test_random_forest_regressor_no_model(self):
        from tradsl.ml.regressors import RandomForestRegressor
        model = RandomForestRegressor(warmup=1, n_estimators=10)
        
        data = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        model.apply(data)  # warmup
        result = model.apply(data)
        assert result is None  # No model trained yet

    def test_random_forest_classifier_no_model(self):
        from tradsl.ml.classifiers import RandomForestClassifier
        model = RandomForestClassifier(warmup=1, n_estimators=10)
        
        data = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        model.apply(data)  # warmup
        result = model.apply(data)
        assert result is None  # No model trained yet


class TestExecutionModel:
    """Tests for ExecutionModel and OhlcAvgExecution."""

    def test_ohlc_avg_calculates_midpoint(self):
        from tradsl.execution import OhlcAvgExecution
        execution = OhlcAvgExecution()
        
        sizing_output = pd.DataFrame({
            "quantity": [10],
            "action": [0],
            "asset": ["AAPL"],
            "confidence": [0.8],
        })
        
        price_data = pd.DataFrame({
            "price.open": [100.0],
            "price.high": [105.0],
            "price.low": [95.0],
            "price.close": [102.0],
            "price.volume": [1000000],
        })
        price_data.index = pd.DatetimeIndex(["2024-01-01"])
        
        result = execution.calculate(sizing_output, price_data)
        
        assert "execution_price" in result.columns
        assert "execution_cost" in result.columns
        assert result["execution_price"].iloc[-1] == 101.0  # (100 + 102) / 2
        assert result["execution_cost"].iloc[-1] == 1010.0  # 10 * 101

    def test_ohlc_avg_handles_negative_quantity(self):
        from tradsl.execution import OhlcAvgExecution
        execution = OhlcAvgExecution()
        
        sizing_output = pd.DataFrame({
            "quantity": [-5],
            "action": [2],
            "asset": ["AAPL"],
            "confidence": [0.7],
        })
        
        price_data = pd.DataFrame({
            "price.open": [50.0],
            "price.close": [52.0],
        })
        price_data.index = pd.DatetimeIndex(["2024-01-01"])
        
        result = execution.calculate(sizing_output, price_data)
        
        assert result["execution_price"].iloc[-1] == 51.0  # (50 + 52) / 2
        assert result["execution_cost"].iloc[-1] == 255.0  # 5 * 51 (absolute)


class TestPortfolioFunction:
    """Tests for PortfolioFunction with execution model."""

    def test_executor_updates_holdings_on_buy(self):
        from tradsl.portfolio_function import PortfolioFunction
        from tradsl.portfolio_state import PortfolioState
        from tradsl.execution import OhlcAvgExecution
        
        state = PortfolioState(cash=10000, holdings={"AAPL": 0})
        executor = PortfolioFunction(
            state=state,
            execution_model=OhlcAvgExecution
        )
        
        agent_output = pd.DataFrame({
            "action": [0],  # buy
            "confidence": [0.8],
            "asset": ["AAPL"],
        })
        
        price_data = pd.DataFrame({
            "price.open": [100.0],
            "price.high": [105.0],
            "price.low": [95.0],
            "price.close": [102.0],
            "price.volume": [1000000],
        })
        price_data.index = pd.DatetimeIndex(["2024-01-01"])
        
        data = pd.concat([agent_output, price_data], axis=1)
        executor.apply(data)
        
        assert state.get_holding("AAPL") > 0
        assert state.cash < 10000

    def test_executor_updates_holdings_on_sell(self):
        from tradsl.portfolio_function import PortfolioFunction
        from tradsl.portfolio_state import PortfolioState
        from tradsl.execution import OhlcAvgExecution
        
        state = PortfolioState(cash=9000, holdings={"AAPL": 10})
        executor = PortfolioFunction(
            state=state,
            execution_model=OhlcAvgExecution
        )
        
        agent_output = pd.DataFrame({
            "action": [2],  # sell
            "confidence": [0.8],
            "asset": ["AAPL"],
        })
        
        price_data = pd.DataFrame({
            "price.open": [100.0],
            "price.close": [102.0],
        })
        price_data.index = pd.DatetimeIndex(["2024-01-01"])
        
        data = pd.concat([agent_output, price_data], axis=1)
        executor.apply(data)
        
        assert state.get_holding("AAPL") < 10
        assert state.cash > 9000

    def test_executor_does_nothing_on_hold(self):
        from tradsl.portfolio_function import PortfolioFunction
        from tradsl.portfolio_state import PortfolioState
        from tradsl.execution import OhlcAvgExecution
        
        state = PortfolioState(cash=10000, holdings={"AAPL": 5})
        initial_cash = state.cash
        initial_holding = state.get_holding("AAPL")
        
        executor = PortfolioFunction(
            state=state,
            execution_model=OhlcAvgExecution
        )
        
        agent_output = pd.DataFrame({
            "action": [1],  # hold
            "confidence": [0.5],
            "asset": ["AAPL"],
        })
        
        price_data = pd.DataFrame({
            "price.open": [100.0],
            "price.close": [102.0],
        })
        price_data.index = pd.DatetimeIndex(["2024-01-01"])
        
        data = pd.concat([agent_output, price_data], axis=1)
        result = executor.apply(data)
        
        assert state.cash == initial_cash
        assert state.get_holding("AAPL") == initial_holding
        assert result["quantity"].iloc[-1] == 0

    def test_executor_uses_sizing_for_quantity(self):
        from tradsl.portfolio_function import PortfolioFunction
        from tradsl.portfolio_state import PortfolioState
        from tradsl.sizing import FractionalSizing
        from tradsl.execution import OhlcAvgExecution
        
        state = PortfolioState(cash=10000, holdings={"AAPL": 0})
        executor = PortfolioFunction(
            state=state,
            sizing_fn=FractionalSizing,
            sizing_params={"fraction": 0.1},
            execution_model=OhlcAvgExecution
        )
        
        agent_output = pd.DataFrame({
            "action": [0],  # buy
            "confidence": [0.8],
            "asset": ["AAPL"],
        })
        
        price_data = pd.DataFrame({
            "price.open": [100.0],
            "price.close": [100.0],
        })
        price_data.index = pd.DatetimeIndex(["2024-01-01"])
        
        data = pd.concat([agent_output, price_data], axis=1)
        result = executor.apply(data)
        
        expected_quantity = int(10000 * 0.1 / 100)  # 10% of 10000 / 100
        assert result["quantity"].iloc[-1] == expected_quantity

    def test_executor_returns_execution_details(self):
        from tradsl.portfolio_function import PortfolioFunction
        from tradsl.portfolio_state import PortfolioState
        from tradsl.execution import OhlcAvgExecution
        
        state = PortfolioState(cash=10000, holdings={"AAPL": 0})
        executor = PortfolioFunction(
            state=state,
            execution_model=OhlcAvgExecution
        )
        
        agent_output = pd.DataFrame({
            "action": [0],
            "confidence": [0.8],
            "asset": ["AAPL"],
        })
        
        price_data = pd.DataFrame({
            "price.open": [100.0],
            "price.close": [102.0],
        })
        price_data.index = pd.DatetimeIndex(["2024-01-01"])
        
        data = pd.concat([agent_output, price_data], axis=1)
        result = executor.apply(data)
        
        assert "execution_price" in result.columns
        assert "execution_cost" in result.columns
        assert result["execution_price"].iloc[-1] == 101.0

    def test_executor_default_execution_model(self):
        from tradsl.portfolio_function import PortfolioFunction
        from tradsl.portfolio_state import PortfolioState
        from tradsl.execution import OhlcAvgExecution
        
        state = PortfolioState(cash=10000, holdings={"AAPL": 0})
        executor = PortfolioFunction(state=state)
        
        assert isinstance(executor.execution, OhlcAvgExecution)


class TestPortfolioState:
    """Tests for PortfolioState."""

    def test_state_initializes_correctly(self):
        from tradsl.portfolio_state import PortfolioState
        
        state = PortfolioState(cash=5000, holdings={"AAPL": 10, "TSLA": 5})
        
        assert state.cash == 5000
        assert state.get_holding("AAPL") == 10
        assert state.get_holding("TSLA") == 5
        assert state.get_holding("UNKNOWN") == 0

    def test_state_update_holding(self):
        from tradsl.portfolio_state import PortfolioState
        
        state = PortfolioState(cash=1000, holdings={"AAPL": 10})
        state.update_holding("AAPL", 5)
        
        assert state.get_holding("AAPL") == 15

    def test_state_update_holding_creates_new_symbol(self):
        from tradsl.portfolio_state import PortfolioState
        
        state = PortfolioState(cash=1000, holdings={})
        state.update_holding("AAPL", 10)
        
        assert state.get_holding("AAPL") == 10

    def test_state_reset(self):
        from tradsl.portfolio_state import PortfolioState
        
        state = PortfolioState(cash=1000, holdings={"AAPL": 50})
        state.reset(initial_cash=5000, symbols=["AAPL", "TSLA"])
        
        assert state.cash == 5000
        assert state.get_holding("AAPL") == 0
        assert state.get_holding("TSLA") == 0


class TestFractionalSizing:
    """Tests for FractionalSizing."""

    def test_fractional_buy_quantity(self):
        from tradsl.sizing import FractionalSizing
        from tradsl.portfolio_state import PortfolioState
        
        sizing = FractionalSizing(fraction=0.1)
        state = PortfolioState(cash=10000, holdings={})
        
        agent_output = pd.DataFrame({
            "action": [0],  # buy
            "confidence": [0.8],
            "asset": ["AAPL"],
        })
        
        quantity = sizing.compute(agent_output, state, price=100)
        expected = int(10000 * 0.1 / 100)  # 10 shares
        
        assert quantity == expected

    def test_fractional_returns_zero_on_hold(self):
        from tradsl.sizing import FractionalSizing
        from tradsl.portfolio_state import PortfolioState
        
        sizing = FractionalSizing(fraction=0.1)
        state = PortfolioState(cash=10000, holdings={})
        
        agent_output = pd.DataFrame({
            "action": [1],  # hold
            "confidence": [0.5],
            "asset": ["AAPL"],
        })
        
        quantity = sizing.compute(agent_output, state, price=100)
        
        assert quantity == 0

    def test_fractional_sell_quantity(self):
        from tradsl.sizing import FractionalSizing
        from tradsl.portfolio_state import PortfolioState
        
        sizing = FractionalSizing(fraction=0.1)
        state = PortfolioState(cash=5000, holdings={"AAPL": 20})
        
        agent_output = pd.DataFrame({
            "action": [2],  # sell
            "confidence": [0.9],
            "asset": ["AAPL"],
        })
        
        quantity = sizing.compute(agent_output, state, price=100)
        
        assert quantity < 0  # negative for sell
        # NAV = cash + 0.5 * holdings * price = 5000 + 0.5 * 20 * 100 = 6000
        # Target = 10% of 6000 = 600, at price 100 = 6 shares
        assert quantity == -6

    def test_fractional_invalid_fraction_raises(self):
        from tradsl.sizing import FractionalSizing
        
        with pytest.raises(ValueError):
            FractionalSizing(fraction=1.5)  # > 1 is invalid
        
        with pytest.raises(ValueError):
            FractionalSizing(fraction=0)  # must be > 0

    def test_fractional_zero_price_returns_zero(self):
        from tradsl.sizing import FractionalSizing
        from tradsl.portfolio_state import PortfolioState
        
        sizing = FractionalSizing(fraction=0.1)
        state = PortfolioState(cash=10000, holdings={})
        
        agent_output = pd.DataFrame({
            "action": [0],
            "confidence": [0.8],
            "asset": ["AAPL"],
        })
        
        quantity = sizing.compute(agent_output, state, price=0)
        
        assert quantity == 0


class TestPortfolioAdapter:
    """Tests for PortfolioAdapter."""

    def test_portfolio_adapter_returns_dataframe(self):
        from tradsl.portfolio_adapter import PortfolioAdapter
        from datetime import datetime
        
        adapter = PortfolioAdapter(symbols=["AAPL", "TSLA"], initial_cash=10000)
        adapter.set_start(datetime.now())
        
        result = adapter.tick()
        
        assert isinstance(result, pd.DataFrame)
        assert "cash" in result.columns
        assert "AAPL_holding" in result.columns
        assert "TSLA_holding" in result.columns

    def test_portfolio_adapter_updates_holdings(self):
        from tradsl.portfolio_adapter import PortfolioAdapter
        from datetime import datetime
        
        adapter = PortfolioAdapter(symbols=["AAPL"], initial_cash=10000)
        adapter.set_start(datetime.now())
        
        adapter.state.update_holding("AAPL", 10)
        result = adapter.tick()
        
        assert result["AAPL_holding"].iloc[-1] == 10

    def test_portfolio_adapter_uses_dag_timestamp(self):
        from tradsl.portfolio_adapter import PortfolioAdapter
        from datetime import datetime
        
        class MockDAG:
            _current_timestamp = datetime(2024, 1, 15)
        
        adapter = PortfolioAdapter(symbols=["AAPL"], initial_cash=10000, dag=MockDAG())
        adapter.set_start(datetime.now())
        
        result = adapter.tick()
        
        assert result.index[0] == datetime(2024, 1, 15)


class TestPortfolioFunctionEdgeCases:
    """Edge case tests for PortfolioFunction."""

    def test_executor_handles_insufficient_data(self):
        from tradsl.portfolio_function import PortfolioFunction
        from tradsl.portfolio_state import PortfolioState
        from tradsl.execution import OhlcAvgExecution
        
        state = PortfolioState(cash=10000, holdings={"AAPL": 0})
        executor = PortfolioFunction(state=state, execution_model=OhlcAvgExecution)
        
        # Only one row of data - insufficient
        agent_output = pd.DataFrame({
            "action": [0],
            "confidence": [0.8],
            "asset": ["AAPL"],
        })
        
        result = executor.apply(agent_output)
        
        assert result is None

    def test_executor_prevents_shorting(self):
        from tradsl.portfolio_function import PortfolioFunction
        from tradsl.portfolio_state import PortfolioState
        from tradsl.execution import OhlcAvgExecution
        
        # Start with no position
        state = PortfolioState(cash=10000, holdings={"AAPL": 0})
        executor = PortfolioFunction(state=state, execution_model=OhlcAvgExecution)
        
        agent_output = pd.DataFrame({
            "action": [2],  # sell
            "confidence": [0.8],
            "asset": ["AAPL"],
        })
        
        price_data = pd.DataFrame({
            "price.open": [100.0],
            "price.close": [100.0],
        })
        price_data.index = pd.DatetimeIndex(["2024-01-01"])
        
        data = pd.concat([agent_output, price_data], axis=1)
        result = executor.apply(data)
        
        # Cannot sell what you don't have - quantity should be 0
        assert result["quantity"].iloc[-1] == 0
        assert state.get_holding("AAPL") == 0
        assert state.cash == 10000  # no change

    def test_executor_cash_goes_negative_on_large_buy(self):
        from tradsl.portfolio_function import PortfolioFunction
        from tradsl.portfolio_state import PortfolioState
        from tradsl.sizing import FractionalSizing
        from tradsl.execution import OhlcAvgExecution
        
        state = PortfolioState(cash=100, holdings={"AAPL": 0})
        executor = PortfolioFunction(
            state=state,
            sizing_fn=FractionalSizing,
            sizing_params={"fraction": 1.0},  # buy with 100% of cash
            execution_model=OhlcAvgExecution
        )
        
        agent_output = pd.DataFrame({
            "action": [0],  # buy
            "confidence": [0.8],
            "asset": ["AAPL"],
        })
        
        price_data = pd.DataFrame({
            "price.open": [10.0],  # low price
            "price.close": [10.0],
        })
        price_data.index = pd.DatetimeIndex(["2024-01-01"])
        
        data = pd.concat([agent_output, price_data], axis=1)
        executor.apply(data)
        
        # Cash can go negative if we buy more than we can afford
        # (in real trading you'd have margin or order sizing limits)
        assert state.get_holding("AAPL") > 0

    def test_executor_output_preserves_agent_columns(self):
        from tradsl.portfolio_function import PortfolioFunction
        from tradsl.portfolio_state import PortfolioState
        from tradsl.execution import OhlcAvgExecution
        
        state = PortfolioState(cash=10000, holdings={"AAPL": 0})
        executor = PortfolioFunction(state=state, execution_model=OhlcAvgExecution)
        
        agent_output = pd.DataFrame({
            "action": [0],
            "confidence": [0.75],
            "asset": ["TSLA"],
        })
        
        price_data = pd.DataFrame({
            "price.open": [100.0],
            "price.close": [100.0],
        })
        price_data.index = pd.DatetimeIndex(["2024-01-01"])
        
        data = pd.concat([agent_output, price_data], axis=1)
        result = executor.apply(data)
        
        assert "action" in result.columns
        assert "confidence" in result.columns
        assert "asset" in result.columns
        assert result["asset"].iloc[-1] == "TSLA"
        assert result["confidence"].iloc[-1] == 0.75
