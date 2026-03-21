"""
Tests for the DAG construction and validation.
"""
import pytest
from tradsl.dag import DAG, Node
from tradsl.exceptions import CycleError, ConfigError, ResolutionError


class TestNodeConstruction:
    """Tests for Node dataclass creation."""

    def test_timeseries_node(self):
        node = Node(name="ts", type="timeseries", attrs={"adapter": "a"}, adapter="a")
        assert node.name == "ts"
        assert node.type == "timeseries"
        assert node.adapter == "a"
        assert node.window == 1

    def test_function_node(self):
        node = Node(name="fn", type="function", function="avg", inputs=["a", "b"])
        assert node.name == "fn"
        assert node.type == "function"
        assert node.function == "avg"
        assert node.inputs == ["a", "b"]

    def test_custom_window(self):
        node = Node(name="n", type="function", window=30)
        assert node.window == 30

    def test_invalid_type(self):
        with pytest.raises(ConfigError):
            Node(name="bad", type="invalid")


class TestDAGFromConfig:
    """Tests for DAG.from_config()."""

    def test_empty_config(self):
        dag = DAG.from_config({})
        assert dag.nodes == {}

    def test_ignores_non_nodes(self):
        config = {
            "n1": {"type": "timeseries", "adapter": "a"},
            "other": {"key": "value"},
        }
        dag = DAG.from_config(config)
        assert "n1" in dag.nodes
        assert "other" not in dag.nodes


class TestValidation:
    """Tests for validation constraints."""

    def test_valid_timeseries(self):
        config = {"ts": {"type": "timeseries", "adapter": "y"}}
        dag = DAG.from_config(config)
        dag.validate()

    def test_valid_function(self):
        config = {
            "ts": {"type": "timeseries", "adapter": "y"},
            "fn": {"type": "function", "function": "f", "inputs": ["ts"]},
        }
        dag = DAG.from_config(config)
        dag.validate()

    def test_timeseries_no_function(self):
        config = {"bad": {"type": "timeseries", "adapter": "y", "function": "f"}}
        dag = DAG.from_config(config)
        with pytest.raises(ConfigError):
            dag.validate()

    def test_timeseries_no_inputs(self):
        config = {"bad": {"type": "timeseries", "adapter": "y", "inputs": ["x"]}}
        dag = DAG.from_config(config)
        with pytest.raises(ConfigError):
            dag.validate()

    def test_function_requires_function(self):
        config = {"bad": {"type": "function", "inputs": ["ts"]}}
        dag = DAG.from_config(config)
        with pytest.raises(ConfigError):
            dag.validate()

    def test_function_requires_inputs(self):
        config = {"bad": {"type": "function", "function": "f"}}
        dag = DAG.from_config(config)
        with pytest.raises(ConfigError):
            dag.validate()

    def test_invalid_input_ref(self):
        config = {"fn": {"type": "function", "function": "f", "inputs": ["missing"]}}
        dag = DAG.from_config(config)
        with pytest.raises(ConfigError):
            dag.validate()


class TestCycleDetection:
    """Tests for cycle detection."""

    def test_no_cycle_chain(self):
        config = {
            "ts": {"type": "timeseries", "adapter": "a"},
            "fn": {"type": "function", "function": "f", "inputs": ["ts"]},
        }
        dag = DAG.from_config(config)
        dag.detect_cycles()

    def test_detects_direct_cycle(self):
        config = {"fn": {"type": "function", "function": "f", "inputs": ["fn"]}}
        dag = DAG.from_config(config)
        with pytest.raises(CycleError):
            dag.detect_cycles()

    def test_detects_longer_cycle(self):
        config = {
            "a": {"type": "function", "function": "f", "inputs": ["b"]},
            "b": {"type": "function", "function": "g", "inputs": ["c"]},
            "c": {"type": "function", "function": "h", "inputs": ["a"]},
        }
        dag = DAG.from_config(config)
        with pytest.raises(CycleError):
            dag.detect_cycles()


class TestTopologicalSort:
    """Tests for topological sort."""

    def test_empty(self):
        dag = DAG.from_config({})
        assert dag.topological_sort() == []

    def test_single_node(self):
        config = {"ts": {"type": "timeseries", "adapter": "a"}}
        dag = DAG.from_config(config)
        assert dag.topological_sort() == ["ts"]

    def test_respects_order(self):
        config = {
            "ts": {"type": "timeseries", "adapter": "a"},
            "fn": {"type": "function", "function": "f", "inputs": ["ts"]},
        }
        dag = DAG.from_config(config)
        order = dag.topological_sort()
        assert order.index("ts") < order.index("fn")

    def test_stable_alphabetical(self):
        config = {
            "z": {"type": "timeseries", "adapter": "a"},
            "a": {"type": "timeseries", "adapter": "b"},
        }
        dag = DAG.from_config(config)
        assert dag.topological_sort() == ["a", "z"]


class TestBufferSizeComputation:
    """Tests for buffer size computation (BFS from sinks)."""

    def test_single_node(self):
        config = {"ts": {"type": "timeseries", "adapter": "a"}}
        dag = DAG.from_config(config).build()
        assert dag.nodes["ts"].buffer_size == 1

    def test_simple_chain(self):
        config = {
            "ts": {"type": "timeseries", "adapter": "a"},
            "fn": {"type": "function", "function": "f", "inputs": ["ts"], "window": 5},
        }
        dag = DAG.from_config(config).build()
        assert dag.nodes["fn"].buffer_size == 1
        assert dag.nodes["ts"].buffer_size == 5

    def test_chain_with_windows(self):
        config = {
            "ts": {"type": "timeseries", "adapter": "a"},
            "fn1": {"type": "function", "function": "f", "inputs": ["ts"], "window": 5},
            "fn2": {"type": "function", "function": "g", "inputs": ["fn1"], "window": 20},
            "fn3": {"type": "function", "function": "h", "inputs": ["fn2"]},
        }
        dag = DAG.from_config(config).build()
        assert dag.nodes["fn3"].buffer_size == 1
        assert dag.nodes["fn2"].buffer_size == 1
        assert dag.nodes["fn1"].buffer_size == 20
        assert dag.nodes["ts"].buffer_size == 5

    def test_parallel_branches(self):
        config = {
            "data": {"type": "timeseries", "adapter": "a"},
            "fast": {"type": "function", "function": "f", "inputs": ["data"], "window": 5},
            "slow": {"type": "function", "function": "g", "inputs": ["data"], "window": 20},
            "signal": {"type": "function", "function": "h", "inputs": ["fast", "slow"]},
        }
        dag = DAG.from_config(config).build()
        assert dag.nodes["signal"].buffer_size == 1
        assert dag.nodes["fast"].buffer_size == 1
        assert dag.nodes["slow"].buffer_size == 1
        assert dag.nodes["data"].buffer_size == 20

    def test_diamond(self):
        config = {
            "a": {"type": "timeseries", "adapter": "a"},
            "b": {"type": "function", "function": "f", "inputs": ["a"], "window": 10},
            "c": {"type": "function", "function": "g", "inputs": ["a"], "window": 5},
            "d": {"type": "function", "function": "h", "inputs": ["b", "c"]},
        }
        dag = DAG.from_config(config).build()
        assert dag.nodes["d"].buffer_size == 1
        assert dag.nodes["b"].buffer_size == 1
        assert dag.nodes["c"].buffer_size == 1
        assert dag.nodes["a"].buffer_size == 10


class TestResolution:
    """Tests for resolution."""

    def test_resolve_function(self):
        config = {
            "fn": {"type": "function", "function": "my_func", "inputs": ["ts"]},
            "ts": {"type": "timeseries", "adapter": "a"},
        }
        dag = DAG.from_config(config).build()

        registry = {"my_func": lambda x: x}
        dag.resolve(registry)
        assert callable(dag.nodes["fn"].attrs["function"])

    def test_resolve_dot_notation(self):
        import operator
        config = {
            "fn": {"type": "function", "function": "operator.add", "inputs": ["ts"]},
            "ts": {"type": "timeseries", "adapter": "a"},
        }
        dag = DAG.from_config(config).build()

        registry = {"operator": operator}
        dag.resolve(registry)
        assert dag.nodes["fn"].attrs["function"] is operator.add

    def test_resolve_nonexistent_raises(self):
        config = {
            "fn": {"type": "function", "function": "missing", "inputs": ["ts"]},
            "ts": {"type": "timeseries", "adapter": "a"},
        }
        dag = DAG.from_config(config).build()

        registry = {}
        with pytest.raises(ResolutionError):
            dag.resolve(registry)

    def test_preserves_buffers(self):
        config = {
            "ts": {"type": "timeseries", "adapter": "a", "window": 10},
            "fn": {"type": "function", "function": "f", "inputs": ["ts"]},
        }
        dag = DAG.from_config(config).build()
        registry = {"f": lambda x: x}
        dag.resolve(registry)
        assert dag.nodes["fn"].buffer_size == 1
        assert dag.nodes["ts"].buffer_size == 1


class TestBuildPipeline:
    """Tests for full build pipeline."""

    def test_build_validates(self):
        config = {"bad": {"type": "function", "function": "f"}}
        dag = DAG.from_config(config)
        with pytest.raises(ConfigError):
            dag.build()

    def test_build_detects_cycles(self):
        config = {"a": {"type": "function", "function": "f", "inputs": ["a"]}}
        dag = DAG.from_config(config)
        with pytest.raises(CycleError):
            dag.build()


class TestIntegration:
    """Integration tests with parser."""

    def test_crossover_strategy(self):
        from tradsl.parser import parse

        dsl = """
price:
type=timeseries
adapter=yfinance

sma_30:
type=function
function=sma
inputs=[price]
window=30

sma_5:
type=function
function=sma
inputs=[price]
window=5

signal:
type=function
function=crossover
inputs=[sma_30, sma_5]
"""
        config = parse(dsl)
        dag = DAG.from_config(config).build()

        assert dag.execution_order[-1] == "signal"
        assert dag.nodes["signal"].buffer_size == 1
        assert dag.nodes["sma_30"].buffer_size == 1
        assert dag.nodes["sma_5"].buffer_size == 1
        assert dag.nodes["price"].buffer_size == 30

    def test_complex_strategy(self):
        from tradsl.parser import parse

        dsl = """
ohlcv:
type=timeseries
adapter=yfinance

close:
type=function
function=close
inputs=[ohlcv]

ma_fast:
type=function
function=ma
inputs=[close]
window=10

ma_slow:
type=function
function=ma
inputs=[close]
window=50

agent:
type=function
function=agent
inputs=[ma_fast, ma_slow]
"""
        config = parse(dsl)
        dag = DAG.from_config(config).build()

        assert dag.nodes["agent"].buffer_size == 1
        assert dag.nodes["ma_fast"].buffer_size == 1
        assert dag.nodes["ma_slow"].buffer_size == 1
        assert dag.nodes["close"].buffer_size == 50
        assert dag.nodes["ohlcv"].buffer_size == 1


class TestFuzz:
    """Randomized tests."""

    def test_random_graph(self):
        import random
        random.seed(42)

        config = {}
        for i in range(20):
            if i < 5:
                config[f"n{i}"] = {
                    "type": "timeseries",
                    "adapter": f"a{i}",
                    "window": random.randint(1, 10),
                }
            else:
                num_inputs = random.randint(1, min(3, i))
                inputs = random.sample([f"n{j}" for j in range(i)], num_inputs)
                config[f"n{i}"] = {
                    "type": "function",
                    "function": f"f{i}",
                    "inputs": inputs,
                    "window": random.randint(1, 20),
                }

        dag = DAG.from_config(config)
        dag.build()
        assert len(dag.execution_order) == 20

    def test_deep_chain(self):
        config = {}
        for i in range(100):
            if i == 0:
                config[f"n{i}"] = {"type": "timeseries", "adapter": "a"}
            else:
                config[f"n{i}"] = {
                    "type": "function",
                    "function": f"f{i}",
                    "inputs": [f"n{i-1}"],
                    "window": 1,
                }
        config["n99"]["window"] = 100

        dag = DAG.from_config(config).build()
        assert dag.nodes["n0"].buffer_size == 1
        assert dag.nodes["n99"].buffer_size == 1

    def test_wide_graph(self):
        config = {f"leaf{i}": {"type": "timeseries", "adapter": "a", "window": i + 1} for i in range(10)}
        config["root"] = {
            "type": "function",
            "function": "combine",
            "inputs": [f"leaf{i}" for i in range(10)],
        }

        dag = DAG.from_config(config).build()
        assert dag.nodes["root"].buffer_size == 1
        assert dag.nodes["leaf9"].buffer_size == 1
        assert dag.nodes["leaf0"].buffer_size == 1
