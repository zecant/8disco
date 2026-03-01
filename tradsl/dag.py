from collections import defaultdict


class CycleError(Exception):
    pass


def build_dag(config: dict[str, dict]) -> dict[str, dict]:
    """
    Build DAG from config and verify acyclicity.
    Returns config with execution order and dependency info.
    
    Skips metadata blocks (_params, _adapters, _backtest).
    """
    skip_keys = {'_params', '_adapters', '_backtest'}
    names = {k for k in config.keys() if not k.startswith('_')}
    deps = {name: fields.get('_deps', []) for name, fields in config.items() if name not in skip_keys}
    
    for name, dep_list in deps.items():
        for dep in dep_list:
            if dep not in names:
                raise ValueError(f"'{name}': dependency '{dep}' does not exist in config")
    
    order = topological_sort(names, deps)
    
    result = dict(config)
    result['_execution_order'] = order
    result['_graph'] = {
        'nodes': list(names),
        'deps': deps,
        'reverse_deps': _reverse_deps(deps)
    }
    
    return result


def topological_sort(nodes: set[str], deps: dict[str, list[str]]) -> list[str]:
    in_degree = defaultdict(int)
    graph = defaultdict(list)
    
    for node in nodes:
        in_degree[node] = 0
    
    for node, dep_list in deps.items():
        for dep in dep_list:
            graph[dep].append(node)
            in_degree[node] += 1
    
    queue = [n for n in nodes if in_degree[n] == 0]
    result = []
    
    while queue:
        node = queue.pop(0)
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(result) != len(nodes):
        cycle = _find_cycle(nodes, deps)
        raise CycleError(f"Cycle detected in dependency graph: {' -> '.join(cycle)}")
    
    return result


def _reverse_deps(deps: dict[str, list[str]]) -> dict[str, list[str]]:
    reverse = defaultdict(list)
    for node, dep_list in deps.items():
        for dep in dep_list:
            reverse[dep].append(node)
    return dict(reverse)


def _find_cycle(nodes: set[str], deps: dict[str, list[str]]) -> list[str]:
    visited = set()
    rec_stack = set()
    
    def dfs(node: str, path: list[str]) -> list[str] | None:
        visited.add(node)
        rec_stack.add(node)
        
        for dep in deps.get(node, []):
            if dep not in visited:
                cycle = dfs(dep, path + [dep])
                if cycle:
                    return cycle
            elif dep in rec_stack:
                cycle_start = path.index(dep)
                return path[cycle_start:] + [dep]
        
        rec_stack.remove(node)
        return None
    
    for node in nodes:
        if node not in visited:
            cycle = dfs(node, [node])
            if cycle:
                return cycle
    
    return []
