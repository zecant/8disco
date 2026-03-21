"""
Tests for Function and Adapter base classes and DAG pipeline execution.
"""
import pytest
import pandas as pd
from tradsl.dag import DAG, Node
from tradsl.functions import Function
from tradsl.adapters import Adapter


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
            def tick(self):
                return [42]
        
        assert issubclass(MyAdapter, Adapter)

    def test_adapter_tick_called(self):
        class CounterAdapter(Adapter):
            def __init__(self):
                self.count = 0
            
            def tick(self):
                self.count += 1
                return [self.count]
        
        adapter = CounterAdapter()
        assert adapter.tick() == [1]
        assert adapter.tick() == [2]


class TestResolveWithFunctions:
    """Tests for resolve with Function classes."""

    def test_resolve_function_class_auto_initializes(self):
        class SMA(Function):
            def __init__(self, window=5):
                self.window = window
            
            def apply(self, data):
                return data.rolling(self.window).mean()
        
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
            def __init__(self):
                self.step = 0
            
            def tick(self):
                self.step += 1
                return [self.step * 10]
        
        config = {
            "prices": {"type": "timeseries", "adapter": "yfinance"},
        }
        dag = DAG.from_config(config).build()
        dag.resolve({"yfinance": PriceAdapter})
        
        assert isinstance(dag.nodes["prices"].attrs["adapter"], PriceAdapter)
        assert dag.nodes["prices"].attrs["adapter"].tick() == [10]


class TestDAGStep:
    """Tests for DAG step execution."""

    def test_step_timeseries_only(self):
        class SimpleAdapter(Adapter):
            def __init__(self):
                self.step = 0
            
            def tick(self):
                self.step += 1
                return [self.step]
        
        config = {
            "prices": {"type": "timeseries", "adapter": "adapter"},
        }
        dag = DAG.from_config(config).build()
        dag.resolve({"adapter": SimpleAdapter})
        
        dag.step()
        dag.step()
        
        values = dag.values()
        assert values[0][0] == "prices"
        assert values[0][1] == [2]  # Latest value is [2]

    def test_step_with_function(self):
        class CountAdapter(Adapter):
            def __init__(self):
                self.count = 0
            
            def tick(self):
                self.count += 1
                return [self.count]
        
        class Double(Function):
            def apply(self, data):
                return data * 2
        
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
            def tick(self):
                return [42]
        
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
