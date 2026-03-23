"""
DAG Construction and Validation.

Builds a directed acyclic graph from parsed TradSL config,
computes topological sort, calculates buffer sizes, and resolves references.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union
import pandas as pd

from tradsl.adapters import Adapter
from tradsl.circular_buffer import CircularBuffer
from tradsl.exceptions import CycleError, ConfigError, InvariantError, ResolutionError
from tradsl.functions import Function


@dataclass
class Node:
    """
    Represents a node in the DAG.

    Attributes:
        name: Unique identifier for this node.
        type: Node type - "timeseries" or "function".
        attrs: All parsed key-value pairs (unresolved).
        inputs: List of node names this node depends on.
        function: Name of the function to call (for type=function).
        adapter: Name of the data adapter (for type=timeseries).
        window: Required lookback window size (default 1).
        buffer_size: Computed buffer size after propagation (set during build).
    """

    name: str
    type: str
    attrs: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    function: str | None = None
    adapter: str | None = None
    window: int = 1
    buffer_size: int | None = None

    def __post_init__(self):
        if self.type not in ("timeseries", "function"):
            raise ConfigError(
                f"Unknown node type: '{self.type}'. Must be 'timeseries' or 'function'.",
                node=self.name,
                key="type"
            )


class DAG:
    """
    Directed Acyclic Graph for TradSL execution.

    Build a DAG from parsed config, validate constraints, compute execution
    order and buffer sizes, then resolve references to Python objects.
    """

    def __init__(self, nodes: dict[str, Node] | None = None):
        self.nodes: dict[str, Node] = nodes or {}
        self.execution_order: list[str] = []
        self._buffers: dict[str, CircularBuffer] = {}
        self._function_registry: dict[str, Function] = {}
        self._adapter_registry: dict[str, Adapter] = {}
        self._current_timestamp: datetime | None = None

    @classmethod
    def from_config(cls, config: dict[str, dict[str, Any]]) -> "DAG":
        """
        Build a DAG from parsed TradSL configuration.

        Args:
            config: Dict mapping block names to their parsed attributes.

        Returns:
            DAG instance with nodes built but not yet validated/sorted.
        """
        dag = cls()
        for name, attrs in config.items():
            node_type = attrs.get("type", "")
            if node_type not in ("timeseries", "function"):
                continue

            node = Node(
                name=name,
                type=node_type,
                attrs=attrs.copy(),
                inputs=attrs.get("inputs", []) if isinstance(attrs.get("inputs"), list) else [],
                function=attrs.get("function"),
                adapter=attrs.get("adapter"),
                window=attrs.get("window", 1),
            )
            dag.nodes[name] = node

        return dag

    def validate(self) -> None:
        """
        Validate DAG configuration constraints.

        Raises:
            ConfigError: If any constraint is violated.

        Constraints:
            - type=timeseries must have adapter (no inputs)
            - type=function must have function name and inputs
            - All input references must exist
        """
        for name, node in self.nodes.items():
            if node.type == "timeseries":
                if node.function is not None:
                    raise ConfigError(
                        "timeseries node cannot have 'function' attribute",
                        node=name,
                        key="function"
                    )
                if node.inputs:
                    raise ConfigError(
                        "timeseries node cannot have 'inputs' attribute",
                        node=name,
                        key="inputs"
                    )

            elif node.type == "function":
                if node.function is None:
                    raise ConfigError(
                        "function node requires 'function' attribute",
                        node=name,
                        key="function"
                    )
                if not node.inputs:
                    raise ConfigError(
                        "function node requires non-empty 'inputs' attribute",
                        node=name,
                        key="inputs"
                    )

            for input_name in node.inputs:
                if input_name not in self.nodes:
                    raise ConfigError(
                        f"Input node '{input_name}' does not exist",
                        node=name,
                        key="inputs"
                    )
    
    def validate_agent_invariant(self) -> None:
        """
        Validate that the DAG ends with agent + portfolio executor (last 2 nodes).
        
        Raises:
            InvariantError: If the last two nodes don't match the pattern.
        """
        if len(self.execution_order) < 2:
            raise InvariantError(
                "DAG must have at least 2 nodes (agent + executor)",
            )
        
        agent_name = self.execution_order[-2]
        executor_name = self.execution_order[-1]
        agent_node = self.nodes[agent_name]
        executor_node = self.nodes[executor_name]
        
        if agent_node.type != "function" or agent_node.function is None:
            raise InvariantError(
                "Second-to-last node must be an agent (ml.agents.*)",
                node=agent_name
            )
        
        if not agent_node.function.startswith("ml.agents."):
            raise InvariantError(
                "Second-to-last node must be an agent (ml.agents.*)",
                node=agent_name
            )
        
        if executor_node.type != "function" or executor_node.function is None:
            raise InvariantError(
                "Last node must be a portfolio executor (portfolio.*)",
                node=executor_name
            )
        
        if not executor_node.function.startswith("portfolio."):
            raise InvariantError(
                "Last node must be a portfolio executor (portfolio.*)",
                node=executor_name
            )

    def detect_cycles(self) -> None:
        """
        Detect cycles in the graph using DFS.

        Raises:
            CycleError: If a cycle is found, includes the cycle path.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in self.nodes}
        parent: dict[str, str | None] = {name: None for name in self.nodes}

        def dfs(node: str) -> list[str] | None:
            color[node] = GRAY
            for neighbor in self.nodes[node].inputs:
                if neighbor not in self.nodes:
                    continue
                if color[neighbor] == GRAY:
                    path = [neighbor, node]
                    current: str | None = node
                    while current is not None and parent.get(current) and parent[current] != neighbor:
                        current = parent[current]
                        if current:
                            path.append(current)
                    path.append(neighbor)
                    path.reverse()
                    raise CycleError(path)
                if color[neighbor] == WHITE:
                    parent[neighbor] = node
                    result = dfs(neighbor)
                    if result:
                        return result
            color[node] = BLACK
            return None

        for node in self.nodes:
            if color[node] == WHITE:
                result = dfs(node)
                if result:
                    raise CycleError(result)

    def topological_sort(self) -> list[str]:
        """
        Compute topological sort using Kahn's algorithm with deque.

        Returns:
            List of node names in execution order (dependencies before dependents).

        Raises:
            CycleError: If the graph contains a cycle.

        Note:
            Uses collections.deque for O(1) popleft, ensuring O(V+E) complexity.
            Stable sort: nodes with same in-degree are processed alphabetically.
        """
        in_degree: dict[str, int] = {name: len(node.inputs) for name, node in self.nodes.items()}

        forward_edges: dict[str, list[str]] = {name: [] for name in self.nodes}
        for name, node in self.nodes.items():
            for input_name in node.inputs:
                if input_name in forward_edges:
                    forward_edges[input_name].append(name)

        queue = deque(sorted([name for name, deg in in_degree.items() if deg == 0]))
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for dependent in forward_edges.get(node, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self.nodes):
            remaining = [n for n in self.nodes if n not in result]
            raise CycleError(remaining)

        self.execution_order = result
        return result

    def compute_buffer_sizes(self) -> None:
        """
        Compute buffer sizes using reverse topological propagation.

        Algorithm:
            1. Initialize buffer_map: all 0
            2. Sink node gets buffer = 1
            3. For each dependency: buffer_map[dep] = max(buffer_map[dep], node.window)
            4. Set each node's buffer_size from buffer_map
        """
        if not self.execution_order:
            self.topological_sort()

        buffer_map: dict[str, int] = {name: 0 for name in self.nodes}
        if self.execution_order:
            buffer_map[self.execution_order[-1]] = 1

        for node_name in reversed(self.execution_order):
            node = self.nodes[node_name]
            for dep in node.inputs:
                if dep in buffer_map:
                    buffer_map[dep] = max(buffer_map[dep], node.window)

        for name in self.nodes:
            self.nodes[name].buffer_size = buffer_map[name]

    def _resolve_sizing_fn(self, name: str, registry: dict) -> Any:
        if name is None:
            return None
        if "." in name:
            parts = name.split(".")
            obj = registry.get(parts[0])
            if obj is None:
                obj = __import__(parts[0], fromlist=[""])
            for part in parts[1:]:
                obj = getattr(obj, part)
            return obj
        return registry.get(name)

    def resolve(self, registry: dict[str, Any]) -> None:
        """
        Resolve all references using a registry dictionary.

        Args:
            registry: Dictionary mapping function/adapter names to class or
                     instance. For function nodes, if a class is provided,
                     it is instantiated with parameters from the DSL (e.g., window).
                     Supports dot notation (e.g., "numpy.mean" resolves
                     to numpy.mean).

        Raises:
            ResolutionError: If a name cannot be resolved.

        Note:
            Resolves:
            - 'function' attribute to instantiated Function objects
            - 'adapter' attribute to instantiated Adapter objects
            - All other string values are passed through unchanged
        """
        def can_resolve(name: str) -> bool:
            if "." in name:
                parts = name.split(".")
                obj = registry.get(parts[0])
                if obj is None:
                    return False
                for part in parts[1:]:
                    if not hasattr(obj, part):
                        return False
                    obj = getattr(obj, part)
                return True
            return name in registry

        def instantiate(cls_or_instance, attrs: dict, auto_state=None, is_adapter=False) -> Any:
            if isinstance(cls_or_instance, type):
                kwargs = {}
                sizing_fn_cls = None
                sizing_params = {}
                execution_model_cls = None
                
                for key, value in attrs.items():
                    if key in ("type", "function", "adapter", "inputs"):
                        continue
                    if key == "sizing_fn":
                        sizing_fn_cls = self._resolve_sizing_fn(value, registry)
                    elif key == "sizing_params":
                        sizing_params = value
                    elif key == "execution_model":
                        execution_model_cls = self._resolve_sizing_fn(value, registry)
                    else:
                        kwargs[key] = value
                
                if auto_state is not None:
                    import inspect
                    sig = inspect.signature(cls_or_instance.__init__)
                    if "state" in sig.parameters:
                        kwargs["state"] = auto_state
                
                if sizing_fn_cls is not None:
                    kwargs["sizing_fn"] = sizing_fn_cls
                    kwargs["sizing_params"] = sizing_params
                
                if execution_model_cls is not None:
                    kwargs["execution_model"] = execution_model_cls
                
                if is_adapter:
                    kwargs["dag"] = self
                
                return cls_or_instance(**kwargs)
            return cls_or_instance
        
        def resolve_dotted(name: str, reg: dict) -> Any:
            if "." in name:
                parts = name.split(".")
                obj = reg[parts[0]]
                for part in parts[1:]:
                    obj = getattr(obj, part)
                return obj
            return reg.get(name)

        for node in self.nodes.values():
            if node.type == "function" and node.function is not None:
                if not can_resolve(node.function):
                    raise ResolutionError(node.function, node=node.name, key="function")

        auto_states = []
        for name in self.execution_order:
            if name not in self.nodes:
                continue

            node = self.nodes[name]

            if node.type == "timeseries" and node.adapter is not None:
                adapter_name = node.adapter
                if adapter_name in registry:
                    if "." in adapter_name:
                        parts = adapter_name.split(".")
                        obj = registry[parts[0]]
                        for part in parts[1:]:
                            obj = getattr(obj, part)
                        resolved_adapter = instantiate(obj, node.attrs, is_adapter=True)
                    else:
                        resolved_adapter = instantiate(registry[adapter_name], node.attrs, is_adapter=True)
                    node.attrs["adapter"] = resolved_adapter
                    self._adapter_registry[name] = resolved_adapter
                    if hasattr(resolved_adapter, "state"):
                        auto_states.append(resolved_adapter.state)

            if node.buffer_size and node.buffer_size > 0:
                self._buffers[name] = CircularBuffer(size=node.buffer_size)

        default_state = auto_states[0] if auto_states else None

        for name in self.execution_order:
            if name not in self.nodes:
                continue

            node = self.nodes[name]

            if node.type == "function" and node.function is not None:
                func_name = node.function
                if "." in func_name:
                    parts = func_name.split(".")
                    obj = registry[parts[0]]
                    for part in parts[1:]:
                        obj = getattr(obj, part)
                    resolved_fn = instantiate(obj, node.attrs, auto_state=default_state)
                else:
                    resolved_fn = instantiate(registry[func_name], node.attrs, auto_state=default_state)
                node.attrs["function"] = resolved_fn
                self._function_registry[name] = resolved_fn

    def step(self) -> None:
        """Advance the DAG by one tick."""
        for name in self.execution_order:
            node = self.nodes[name]
            if node.type == "timeseries":
                adapter = self._adapter_registry.get(name)
                if adapter:
                    value = adapter.tick()
                else:
                    value = None
                self._push_to_buffer(name, value)

            elif node.type == "function":
                fn = self._function_registry.get(name)
                if fn:
                    df = self._glue_inputs(node.inputs)
                    result = fn.apply(df)
                    if result is not None:
                        value = result
                    else:
                        value = None
                else:
                    value = None
                self._push_to_buffer(name, value)

    def _glue_inputs(self, input_names: list[str]) -> pd.DataFrame:
        """Join all input buffers into a single DataFrame with prefixed columns and timestamp alignment."""
        dfs = []
        for dep in input_names:
            contents = self._get_buffer_contents(dep)
            if contents is None:
                continue
            
            if isinstance(contents, pd.DataFrame):
                contents = contents.copy()
                contents.columns = [f"{dep}.{col}" for col in contents.columns]
                dfs.append(contents)
            else:
                df = pd.DataFrame({dep: contents})
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        result = dfs[0]
        for df in dfs[1:]:
            if isinstance(result.index, pd.DatetimeIndex) and isinstance(df.index, pd.DatetimeIndex):
                result = pd.merge_asof(
                    result.sort_index(),
                    df.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction='nearest'
                )
            else:
                result = pd.concat([result, df], axis=1)
        
        return result

    def _get_buffer_contents(self, name: str) -> Optional[Union[list, pd.DataFrame]]:
        """Get buffer contents in oldest-to-newest order."""
        if name in self._buffers:
            return self._buffers[name].contents()
        return None

    def _push_to_buffer(self, name: str, value: Any) -> None:
        if name in self._buffers:
            self._buffers[name].push(value)

    def values(self) -> list[tuple[str, Any]]:
        """Return ordered list of (node_name, latest_value) tuples."""
        result = []
        for name in self.execution_order:
            if name in self._buffers and self._buffers[name].is_ready:
                result.append((name, self._buffers[name].latest()))
            else:
                result.append((name, None))
        return result

    def build(self, validate_agent: bool = False) -> "DAG":
        """
        Full DAG build pipeline: validate, detect cycles, sort, compute buffers.

        Args:
            validate_agent: If True, enforce that sink node is ml.agents.*.
                           Set to False for non-trading DAGs (analysis, etc).

        Returns:
            Self for method chaining.
        """
        self.validate()
        self.detect_cycles()
        self.topological_sort()
        if validate_agent:
            self.validate_agent_invariant()
        self.compute_buffer_sizes()
        return self
