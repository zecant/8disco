import importlib
from typing import Any, Callable


class ResolutionError(Exception):
    pass


def resolve(config: dict[str, dict], context: dict = None) -> dict[str, dict]:
    """
    Resolve string references to Python callables in the given context.
    
    Handles:
    - Param refs: 'params=mlparams' -> resolved to actual dict from _params
    - Adapter class: 'adapter=yf' -> instantiated adapter
    - Callable refs: 'function', 'class', 'sizer' fields
    
    Also extracts 'inputs' as '_deps' for DAG building.
    """
    if context is None:
        context = {}
    
    params = config.get('_params', {})
    adapters = config.get('_adapters', {})
    
    resolved_adapters = _resolve_adapters(adapters, context)
    
    resolved = {}
    
    for name, fields in config.items():
        if name.startswith('_'):
            continue
        
        resolved_fields = dict(fields)
        
        resolved_fields['_deps'] = _extract_deps(fields)
        
        if 'params' in fields and isinstance(fields['params'], str):
            param_name = fields['params']
            if param_name in params:
                resolved_fields['params'] = dict(params[param_name])
            else:
                raise ResolutionError(f"'{name}': Param ref '{param_name}' not found in _params")
        
        if 'adapter' in fields and isinstance(fields['adapter'], str):
            adapter_name = fields['adapter']
            if adapter_name in resolved_adapters:
                resolved_fields['adapter'] = resolved_adapters[adapter_name]
            else:
                raise ResolutionError(f"'{name}': Adapter '{adapter_name}' not found")
        
        if 'function' in fields:
            resolved_fields['function'] = _resolve_callable(fields['function'], context, f'{name}.function')
        
        if 'class' in fields:
            resolved_fields['class'] = _resolve_callable(fields['class'], context, f'{name}.class')
        
        if 'sizer' in fields:
            resolved_fields['sizer'] = _resolve_callable(fields['sizer'], context, f'{name}.sizer')
        
        resolved[name] = resolved_fields
    
    resolved['_params'] = params
    resolved['_adapters'] = resolved_adapters
    resolved['_backtest'] = config.get('_backtest', {})
    
    return resolved


def _resolve_adapters(adapters: dict, context: dict) -> dict[str, Any]:
    resolved = {}
    
    for name, fields in adapters.items():
        adapter_class_path = fields.get('class')
        if not adapter_class_path:
            continue
        
        adapter_class = _resolve_class_path(adapter_class_path, context, f'adapter.{name}.class')
        
        adapter_params = {}
        for key, value in fields.items():
            if key in ('type', 'class'):
                continue
            adapter_params[key] = value
        
        try:
            resolved[name] = adapter_class(**adapter_params)
        except TypeError as e:
            raise ResolutionError(f"Failed to instantiate adapter '{name}': {e}")
    
    return resolved


def _resolve_class_path(class_path: str, context: dict, field_path: str) -> Any:
    if class_path in context:
        obj = context[class_path]
        if not callable(obj):
            raise ResolutionError(f"'{field_path}': '{class_path}' is not callable, got {type(obj).__name__}")
        return obj
    
    if '.' not in class_path:
        raise ResolutionError(f"'{field_path}': '{class_path}' not found in context")
    
    module_path, class_name = class_path.rsplit('.', 1)
    
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ResolutionError(f"'{field_path}': Could not import '{class_path}': {e}")


def _extract_deps(fields: dict) -> list[str]:
    deps = []
    if 'inputs' in fields and isinstance(fields['inputs'], list):
        for inp in fields['inputs']:
            if isinstance(inp, str):
                deps.append(inp)
    return deps


def _resolve_callable(name: str, context: dict, field_path: str) -> Any:
    if name in context:
        obj = context[name]
        if not callable(obj):
            raise ResolutionError(f"'{field_path}': '{name}' is not callable, got {type(obj).__name__}")
        return obj
    
    if '.' in name:
        return _resolve_class_path(name, context, field_path)
    
    raise ResolutionError(f"'{field_path}': '{name}' not found in context")
