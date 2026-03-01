import re
from typing import Any


def parse_config(source: str) -> dict[str, dict]:
    """
    Parse DSL config string into raw dict.
    
    Format:
        :name
        key=value
        key=[val1, val2, val3]
        
    Or param blocks (no colon prefix):
        paramname:
        key=value
        
    Returns dict mapping name -> {key: value, ...}
    with special keys:
        _params: {name: {key: value}} for param blocks
        _adapters: {name: {key: value}} for adapter definitions
        _backtest: {key: value} for backtest config
    """
    config = {
        '_params': {},
        '_adapters': {},
        '_backtest': {}
    }
    
    lines = source.strip().split('\n')
    
    current_block = None
    current_type = None
    current_fields = {}
    
    param_pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*):$')
    named_block_pattern = re.compile(r'^:([a-zA-Z_][a-zA-Z0-9_]*)$')
    key_value_pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)=(.*)$')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        param_match = param_pattern.match(line)
        named_match = named_block_pattern.match(line)
        kv_match = key_value_pattern.match(line)
        
        if param_match:
            if current_block is not None:
                _save_block(config, current_block, current_type, current_fields)
            
            current_block = param_match.group(1)
            current_type = 'param'
            current_fields = {}
        
        elif named_match:
            if current_block is not None:
                _save_block(config, current_block, current_type, current_fields)
            
            current_block = named_match.group(1)
            current_type = 'named'
            current_fields = {}
        
        elif kv_match:
            key = kv_match.group(1)
            value = _parse_value(kv_match.group(2))
            current_fields[key] = value
        
        else:
            pass
    
    if current_block is not None:
        _save_block(config, current_block, current_type, current_fields)
    
    return config


def _save_block(config: dict, name: str, block_type: str, fields: dict):
    if block_type == 'param':
        config['_params'][name] = fields
    elif name == 'backtest' and 'type' in fields and fields['type'] == 'backtest':
        config['_backtest'] = fields
    elif 'type' in fields and fields['type'] == 'adapter':
        config['_adapters'][name] = fields
    else:
        config[name] = fields


def _parse_value(value: str) -> Any:
    value = value.strip()
    
    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(v.strip()) for v in inner.split(',')]
    
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    if value.lower() == 'none':
        return None
    
    try:
        return int(value)
    except ValueError:
        pass
    
    try:
        return float(value)
    except ValueError:
        pass
    
    return value
