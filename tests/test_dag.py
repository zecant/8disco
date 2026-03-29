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

    def test_function_node(self):
        node = Node(name="fn", type="function", function="avg", inputs=["a", "b"])
        assert node.name == "fn"
        assert node.type == "function"
        assert node.function == "avg"
        assert node.inputs == ["a", "b"]

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
                }
            else:
                num_inputs = random.randint(1, min(3, i))
                inputs = random.sample([f"n{j}" for j in range(i)], num_inputs)
                config[f"n{i}"] = {
                    "type": "function",
                    "function": f"f{i}",
                    "inputs": inputs,
                }

        dag = DAG.from_config(config)
        dag.build()
        assert len(dag.execution_order) == 20
