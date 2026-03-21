"""
DAG Construction and Validation.

Builds a directed acyclic graph from parsed TradSL config,
computes topological sort, calculates buffer sizes, and resolves references.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd

from tradsl.adapters import Adapter
from tradsl.circular_buffer import CircularBuffer
from tradsl.exceptions import CycleError, ConfigError, ResolutionError
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
            2. Last node (sink) gets buffer = 1 (no dependents)
            3. Process nodes in reverse topological order
            4. For each dependency: buffer_map[dep] = max(buffer_map[dep], node.window)
            5. Set each node's buffer_size to max(buffer_map[name], node.window)
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

        def instantiate(cls_or_instance, attrs: dict) -> Any:
            if isinstance(cls_or_instance, type):
                kwargs = {}
                for key, value in attrs.items():
                    if key in ("type", "function", "adapter", "inputs"):
                        continue
                    kwargs[key] = value
                return cls_or_instance(**kwargs)
            return cls_or_instance

        for node in self.nodes.values():
            if node.type == "function" and node.function is not None:
                if not can_resolve(node.function):
                    raise ResolutionError(node.function, node=node.name, key="function")

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
                    resolved_fn = instantiate(obj, node.attrs)
                else:
                    resolved_fn = instantiate(registry[func_name], node.attrs)
                node.attrs["function"] = resolved_fn
                self._function_registry[name] = resolved_fn

            if node.type == "timeseries" and node.adapter is not None:
                adapter_name = node.adapter
                if adapter_name in registry:
                    if "." in adapter_name:
                        parts = adapter_name.split(".")
                        obj = registry[parts[0]]
                        for part in parts[1:]:
                            obj = getattr(obj, part)
                        resolved_adapter = instantiate(obj, node.attrs)
                    else:
                        resolved_adapter = instantiate(registry[adapter_name], node.attrs)
                    node.attrs["adapter"] = resolved_adapter
                    self._adapter_registry[name] = resolved_adapter

            if node.buffer_size and node.buffer_size > 0:
                self._buffers[name] = CircularBuffer(size=node.buffer_size)

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
        """Join all input buffers into a single DataFrame."""
        columns = []
        for dep in input_names:
            contents = self._get_buffer_contents(dep)
            columns.append(contents if contents else [None] * self._buffers[dep].size)
        
        max_len = max(len(c) for c in columns) if columns else 0
        padded = []
        for c in columns:
            if len(c) < max_len:
                padded.append([None] * (max_len - len(c)) + c)
            else:
                padded.append(c)
        
        return pd.DataFrame(dict(zip(input_names, padded)))

    def _get_buffer_contents(self, name: str) -> Optional[list]:
        """Get buffer contents in oldest-to-newest order, padded with None if partial."""
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

    def build(self) -> "DAG":
        """
        Full DAG build pipeline: validate, detect cycles, sort, compute buffers.

        Returns:
            Self for method chaining.
        """
        self.validate()
        self.detect_cycles()
        self.topological_sort()
        self.compute_buffer_sizes()
        return self
