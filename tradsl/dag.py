"""
DAG Construction and Validation.

Builds a directed acyclic graph from parsed TradSL config,
computes topological sort, and resolves references.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
import uuid

from tradsl.adapters import Adapter
from tradsl.exceptions import CycleError, ConfigError, ResolutionError

if TYPE_CHECKING:
    from tradsl.storage.connection import ClickHouseConnection


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
    """

    name: str
    type: str
    attrs: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    function: str | None = None
    adapter: str | None = None

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
    order, then resolve references to Python objects.
    """

    def __init__(self, nodes: dict[str, Node] | None = None):
        self.nodes: dict[str, Node] = nodes or {}
        self.execution_order: list[str] = []
        self._function_registry: dict[str, Any] = {}
        self._adapter_registry: dict[str, Adapter] = {}
        self._table_names: dict[str, str] = {}
        self._config_blocks: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_config(cls, config: dict[str, dict[str, Any]]) -> "DAG":
        """
        Build a DAG from parsed TradSL configuration.

        Args:
            config: Dict mapping block names to their parsed attributes.

        Returns:
            DAG instance with nodes built but not yet validated/sorted.
        Also stores "config" type blocks in dag._config_blocks (not processed as nodes).
        """
        dag = cls()
        for name, attrs in config.items():
            node_type = attrs.get("type", "")
            
            # Store config blocks separately (not processed as DAG nodes)
            if node_type == "config":
                dag._config_blocks[name] = attrs.copy()
                continue
            
            if node_type not in ("timeseries", "function"):
                continue

            node = Node(
                name=name,
                type=node_type,
                attrs=attrs.copy(),
                inputs=attrs.get("inputs", []) if isinstance(attrs.get("inputs"), list) else [],
                function=attrs.get("function"),
                adapter=attrs.get("adapter"),
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

    def resolve(self, registry: dict[str, Any]) -> None:
        """
        Resolve all references using a registry dictionary.

        Args:
            registry: Dictionary mapping function/adapter names to class or
                     instance. For function nodes, if a class is provided,
                     it is instantiated with parameters from the DSL.
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
            if name in registry:
                return True
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
            return False

        def instantiate(cls_or_instance, attrs: dict, is_adapter: bool = False) -> Any:
            if isinstance(cls_or_instance, type):
                kwargs = {}

                for key, value in attrs.items():
                    if key in ("type", "function", "adapter", "inputs"):
                        continue
                    kwargs[key] = value

                if is_adapter:
                    kwargs["dag"] = self

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

        for name in self.execution_order:
            if name not in self.nodes:
                continue

            node = self.nodes[name]

            if node.type == "function" and node.function is not None:
                func_name = node.function
                if func_name in registry:
                    resolved_fn = instantiate(registry[func_name], node.attrs)
                elif "." in func_name:
                    parts = func_name.split(".")
                    obj = registry[parts[0]]
                    for part in parts[1:]:
                        obj = getattr(obj, part)
                    resolved_fn = instantiate(obj, node.attrs)
                else:
                    resolved_fn = instantiate(registry[func_name], node.attrs)
                node.attrs["function"] = resolved_fn
                self._function_registry[name] = resolved_fn

    def execute(self, conn: "ClickHouseConnection") -> dict[str, str]:
        """
        Execute the DAG against ClickHouse.

        All data stays in ClickHouse. Each node produces a table that can be
        queried or used as input for downstream nodes.

        Args:
            conn: ClickHouseConnection instance

        Returns:
            Dictionary mapping node names to their table names in ClickHouse
        """
        table_names = {}

        for node_name in self.execution_order:
            node = self.nodes[node_name]

            if node.type == "timeseries":
                adapter = node.attrs.get("adapter")
                if adapter is not None:
                    table_name = adapter.load(conn)
                    table_names[node_name] = table_name

            elif node.type == "function":
                fn = node.attrs.get("function")
                if fn is not None:
                    # Collect all input tables for this function
                    input_tables = {}
                    for input_name in node.inputs:
                        if input_name in table_names:
                            input_tables[input_name] = table_names[input_name]

                    if input_tables:
                        # If multiple inputs, join them using _join_tables helper
                        if len(input_tables) > 1:
                            joined_table = fn._join_tables(conn, input_tables)
                            output_table = fn.apply(conn, joined_table)
                        else:
                            # Single input - pass directly
                            input_table = list(input_tables.values())[0]
                            output_table = fn.apply(conn, input_table)
                        table_names[node_name] = output_table

        self._table_names = table_names
        return table_names

    def get_table(self, node_name: str) -> Optional[str]:
        """
        Get the ClickHouse table name for a node after execution.
        
        Args:
            node_name: Name of the node
            
        Returns:
            Table name in ClickHouse, or None if not executed/found
        """
        return self._table_names.get(node_name)

    def query(self, node_name: str, sql: str = "SELECT * FROM", conn: "ClickHouseConnection" = None):
        """
        Query data from a node's table in ClickHouse.
        
        Args:
            node_name: Name of the node to query
            sql: SQL query template (table name will be appended)
            conn: ClickHouseConnection instance (required)
            
        Returns:
            DataFrame with query results
        """
        table_name = self._table_names.get(node_name)
        if table_name is None:
            raise ValueError(f"No table found for node: {node_name}")
        if conn is None:
            raise ValueError("conn is required to query")
        
        full_sql = f"{sql} {table_name}"
        return conn.query(full_sql)

    def build(self) -> "DAG":
        """
        Full DAG build pipeline: validate, detect cycles, sort.

        Returns:
            Self for method chaining.
        """
        self.validate()
        self.detect_cycles()
        self.topological_sort()
        return self
