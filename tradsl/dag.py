"""
DAG Construction and Execution

Builds directed acyclic graph from validated config,
computes topological sort, and extracts execution metadata.
See Section 6 of the specification.
"""
from typing import Any, Dict, List, Set, Optional
from collections import deque
from dataclasses import dataclass, field
from tradsl.exceptions import CycleError, ConfigError


@dataclass
class DAGMetadata:
    """Metadata computed from DAG structure."""
    warmup_bars: int
    node_buffer_sizes: Dict[str, int]
    execution_order: List[str]
    source_nodes: Set[str]
    model_nodes: Set[str]
    trainable_model_nodes: Set[str]


@dataclass
class DAG:
    """Directed acyclic graph for strategy execution."""
    config: Dict[str, Any]
    edges: Dict[str, List[str]] = field(default_factory=dict)
    reverse_edges: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Optional[DAGMetadata] = None


def build_dag(config: Dict[str, Any]) -> DAG:
    """
    Build DAG from validated config.

    Args:
        config: Output from schema.validate()

    Returns:
        DAG with computed metadata

    Raises:
        CycleError: If circular dependency detected
    """
    nodes: Dict[str, Dict] = {}

    for name, block in config.items():
        if name.startswith('_'):
            continue
        block_type = block.get('type')
        if block_type in ('timeseries', 'trainable_model', 'model', 'agent'):
            nodes[name] = block

    edges: Dict[str, List[str]] = {name: [] for name in nodes}
    reverse_edges: Dict[str, List[str]] = {name: [] for name in nodes}

    for name, block in nodes.items():
        inputs = block.get('inputs', [])
        for inp in inputs:
            if inp in nodes:
                edges[inp].append(name)
                reverse_edges[name].append(inp)

    _check_cycles(nodes, edges)

    execution_order = _topological_sort(nodes, reverse_edges)

    source_nodes = {
        name for name, block in nodes.items()
        if block.get('type') == 'timeseries' and not block.get('inputs')
    }

    model_nodes = {
        name for name, block in nodes.items()
        if block.get('type') == 'model'
    }

    trainable_model_nodes = {
        name for name, block in nodes.items()
        if block.get('type') == 'trainable_model'
    }

    node_buffer_sizes = _compute_buffer_sizes(nodes)

    warmup_bars = _compute_warmup_bars(nodes, node_buffer_sizes)

    metadata = DAGMetadata(
        warmup_bars=warmup_bars,
        node_buffer_sizes=node_buffer_sizes,
        execution_order=execution_order,
        source_nodes=source_nodes,
        model_nodes=model_nodes,
        trainable_model_nodes=trainable_model_nodes,
    )

    return DAG(
        config=config,
        edges=edges,
        reverse_edges=reverse_edges,
        metadata=metadata
    )


def _check_cycles(nodes: Dict[str, Dict], edges: Dict[str, List[str]]) -> None:
    """Check for cycles using DFS with explicit stack."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {name: WHITE for name in nodes}
    parent: Dict[str, Optional[str]] = {name: None for name in nodes}

    def dfs(node: str) -> Optional[List[str]]:
        color[node] = GRAY
        for neighbor in edges.get(node, []):
            if neighbor not in nodes:
                continue
            if color[neighbor] == GRAY:
                path = [neighbor, node]
                current: Optional[str] = node
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

    for node in nodes:
        if color[node] == WHITE:
            try:
                dfs(node)
            except CycleError:
                raise


def _topological_sort(nodes: Dict[str, Dict], reverse_edges: Dict[str, List[str]]) -> List[str]:
    """
    Topological sort using Kahn's algorithm with deque.

    Stable sort: nodes with same in-degree are processed alphabetically.
    
    Args:
        nodes: Dict of node name -> node config
        reverse_edges: Dict mapping node -> its dependencies (nodes it depends on)
    """
    in_degree = {name: len(reverse_edges.get(name, [])) for name in nodes}

    queue = deque([name for name, deg in in_degree.items() if deg == 0])
    queue = deque(sorted(queue))

    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for dependent in nodes:
            if node in reverse_edges.get(dependent, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        if queue:
            queue = deque(sorted(queue))

    if len(result) != len(nodes):
        remaining = [n for n in nodes if n not in result]
        raise CycleError(remaining)

    return result


def _compute_buffer_sizes(nodes: Dict[str, Dict]) -> Dict[str, int]:
    """Compute required buffer size for each node."""
    buffer_sizes = {}

    for name, block in nodes.items():
        block_type = block.get('type')
        if block_type == 'timeseries':
            window = block.get('window', 20)
            buffer_sizes[name] = window
        elif block_type == 'trainable_model':
            window = block.get('training_window', 504)
            buffer_sizes[name] = window
        elif block_type == 'model':
            window = block.get('training_window', 504)
            buffer_sizes[name] = window
        else:
            buffer_sizes[name] = 1

    return buffer_sizes


def _compute_warmup_bars(nodes: Dict[str, Dict], buffer_sizes: Dict[str, int]) -> int:
    """Compute maximum warmup bars required across all nodes."""
    max_warmup = 0

    for name, block in nodes.items():
        block_type = block.get('type')
        if block_type == 'timeseries':
            own_lookback = buffer_sizes.get(name, 20)
        elif block_type == 'trainable_model':
            own_lookback = buffer_sizes.get(name, 504)
        elif block_type == 'model':
            own_lookback = 0
        else:
            own_lookback = 0

        inputs = block.get('inputs', [])
        max_input_lookback = 0
        for inp in inputs:
            if inp in buffer_sizes:
                max_input_lookback = max(max_input_lookback, buffer_sizes.get(inp, 0))

        node_warmup = max_input_lookback + own_lookback
        max_warmup = max(max_warmup, node_warmup)

    return max_warmup


def resolve(config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Resolve string names to Python objects.

    Args:
        config: Validated config from schema.validate()
        context: Optional context with registered functions, adapters, etc.

    Returns:
        Config with resolved Python objects
    """
    from tradsl import _registry

    ctx = context or {}
    registry = {**_registry, **ctx}

    resolved = {}

    for key, value in config.items():
        if key.startswith('_'):
            resolved[key] = value
            continue

        block = dict(value)
        block_type = block.get('type')

        if block_type == 'timeseries':
            func_name = block.get('function')
            if func_name and func_name in registry.get('functions', {}):
                block['_function'] = registry['functions'][func_name]

            adapter_name = block.get('adapter')
            if adapter_name and adapter_name in registry.get('adapters', {}):
                block['_adapter'] = registry['adapters'][adapter_name]

        elif block_type == 'trainable_model':
            class_name = block.get('class')
            if class_name and class_name in registry.get('trainable_models', {}):
                block['_class'] = registry['trainable_models'][class_name]

            label_func = block.get('label_function')
            if label_func and label_func in registry.get('label_functions', {}):
                block['_label_function'] = registry['label_functions'][label_func]

        elif block_type == 'model':
            class_name = block.get('class')
            if class_name and class_name in registry.get('agents', {}):
                block['_class'] = registry['agents'][class_name]

            reward_func = block.get('reward_function')
            if reward_func and reward_func in registry.get('rewards', {}):
                block['_reward_function'] = registry['rewards'][reward_func]

        elif block_type == 'agent':
            sizer = block.get('sizer')
            if sizer and sizer in registry.get('sizers', {}):
                block['_sizer'] = registry['sizers'][sizer]

        resolved[key] = block

    return resolved
