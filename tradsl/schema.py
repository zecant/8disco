from typing import Any


class ConfigError(Exception):
    pass


FIELD_SCHEMA = {
    'timeseries': {
        'required': ['type'],
        'optional': ['adapter', 'parameters', 'function', 'inputs', 'params', 'tradable'],
        'defaults': {'tradable': False, 'inputs': []}
    },
    'model': {
        'required': ['type', 'class', 'inputs'],
        'optional': [
            'params',
            'dotraining',
            'load_from',
            'retrain_schedule',
            'training_window',
            'training_window_size',
        ],
    },
    'agent': {
        'required': ['type', 'inputs', 'tradable'],
        'optional': ['sizer', 'params'],
    },
    'adapter': {
        'required': ['type', 'class'],
        'optional': [],
    },
    'backtest': {
        'required': ['type', 'start', 'end', 'capital'],
        'optional': ['commission', 'leverage'],
    }
}


def validate(config: dict[str, dict]) -> dict[str, dict]:
    """
    Validate config against schema.
    Returns validated config with defaults filled in.
    """
    validated = {}
    
    for name, fields in config.items():
        if name.startswith('_'):
            continue
        
        if 'type' not in fields:
            raise ConfigError(f"'{name}': missing required 'type' field")
        
        type_name = fields['type']
        
        if type_name not in FIELD_SCHEMA:
            raise ConfigError(f"'{name}': unknown type '{type_name}'. Must be one of: {list(FIELD_SCHEMA.keys())}")
        
        schema = FIELD_SCHEMA[type_name]
        
        for req_field in schema['required']:
            if req_field not in fields:
                raise ConfigError(f"'{name}': missing required field '{req_field}' for type '{type_name}'")
        
        validated[name] = dict(fields)
        
        if 'defaults' in schema:
            for key, default in schema['defaults'].items():
                if key not in validated[name]:
                    validated[name][key] = default
    
    for key in ['_params', '_adapters', '_backtest']:
        if key in config:
            validated[key] = config[key]
    
    return validated
