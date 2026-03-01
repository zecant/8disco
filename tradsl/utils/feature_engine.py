from typing import Any
import pandas as pd


class FeatureEngineError(Exception):
    pass


def compute_features(config: dict, data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features by traversing DAG in topological order.
    
    Args:
        config: Parsed DSL config with resolved callables
        data: DataFrame with base timeseries columns (e.g., nvda_close, vix_close)
    
    Returns:
        DataFrame with all computed features as columns
    
    Raises:
        FeatureEngineError: If feature computation fails
    """
    execution_order = config.get('_execution_order', [])
    graph = config.get('_graph', {}).get('deps', {})
    
    result_df = data.copy()
    computed = set(data.columns)
    
    for node_name in execution_order:
        if node_name not in config:
            continue
        
        node = config[node_name]
        node_type = node.get('type')
        
        if node_type == 'timeseries':
            result_df = _compute_timeseries(node, node_name, result_df, computed)
        
        elif node_type == 'model':
            result_df = _compute_model(node, node_name, result_df, computed)
    
    return result_df


def compute_features_incremental(
    config: dict,
    data: pd.DataFrame,
    new_row: pd.Series
) -> pd.DataFrame:
    """
    Incrementally compute features for a new data point.
    
    Args:
        config: Parsed DSL config
        data: DataFrame with all computed features so far
        new_row: Series with new base timeseries values
    
    Returns:
        DataFrame with new feature row appended
    """
    execution_order = config.get('_execution_order', [])
    graph = config.get('_graph', {}).get('deps', {})
    
    result_df = data.copy()
    result_df = pd.concat([result_df, new_row.to_frame().T], ignore_index=True)
    
    computed = set(data.columns)
    
    for node_name in execution_order:
        if node_name not in config:
            continue
        
        node = config[node_name]
        node_type = node.get('type')
        
        if node_type == 'timeseries':
            result_df = _compute_timeseries(node, node_name, result_df, computed)
        
        elif node_type == 'model':
            result_df = _compute_model(node, node_name, result_df, computed)
    
    return result_df


def _compute_timeseries(
    node: dict,
    node_name: str,
    df: pd.DataFrame,
    computed: set
) -> pd.DataFrame:
    """
    Compute a timeseries node with a function.
    
    If the node has a 'function' field, apply it to input columns.
    """
    function = node.get('function')
    inputs = node.get('inputs', [])
    params = node.get('params', {})
    
    if not function:
        return df
    
    input_cols = _get_input_columns(inputs, df.columns.tolist())
    
    if not input_cols:
        return df
    
    try:
        if len(input_cols) == 1:
            input_series = df[input_cols[0]]
            result = function(input_series, **params)
        else:
            input_df = df[input_cols]
            result = function(input_df, **params)
        
        if isinstance(result, pd.Series):
            df[node_name] = result
        elif isinstance(result, pd.DataFrame):
            for col in result.columns:
                df[f"{node_name}_{col}"] = result[col]
        
    except Exception as e:
        raise FeatureEngineError(
            f"Failed to compute timeseries '{node_name}': {e}"
        )
    
    return df


def _compute_model(
    node: dict,
    node_name: str,
    df: pd.DataFrame,
    computed: set
) -> pd.DataFrame:
    """
    Compute a model node by running predict().
    
    Model should return a dict mapping output_name -> Series.
    """
    model = node.get('class')
    inputs = node.get('inputs', [])
    params = node.get('params', {})
    
    if not model:
        return df
    
    if isinstance(model, type):
        model = model()
    
    input_cols = _get_input_columns(inputs, df.columns.tolist())
    
    if not input_cols:
        return df
    
    try:
        input_df = df[input_cols]
        
        if hasattr(model, 'predict'):
            result = model.predict(input_df, **params)
        else:
            raise FeatureEngineError(
                f"Model '{node_name}' has no 'predict' method"
            )
        
        if isinstance(result, dict):
            for output_name, output_val in result.items():
                if isinstance(output_val, list):
                    if len(output_val) == len(df):
                        df[f"{node_name}_{output_name}"] = output_val
                    elif len(output_val) == 1:
                        df[f"{node_name}_{output_name}"] = output_val[0]
                elif isinstance(output_val, (int, float)):
                    df[f"{node_name}_{output_name}"] = output_val
                elif isinstance(output_val, pd.Series):
                    output_val = output_val.reset_index(drop=True)
                    output_val.index = df.index
                    df[f"{node_name}_{output_name}"] = output_val
        elif isinstance(result, pd.Series):
            result = result.reset_index(drop=True)
            result.index = df.index
            df[node_name] = result
        elif isinstance(result, pd.DataFrame):
            for col in result.columns:
                df[f"{node_name}_{col}"] = result[col]
            for col in result.columns:
                df[f"{node_name}_{col}"] = result[col]
        
    except Exception as e:
        raise FeatureEngineError(
            f"Failed to compute model '{node_name}': {e}"
        )
    
    return df


def _get_input_columns(inputs: list, available_cols: list) -> list:
    """
    Get the actual column names for given input references.
    
    Handles:
    - Direct column names (e.g., 'nvda_close')
    - Model output columns (e.g., 'signal_model_allocation')
    """
    result = []
    
    for inp in inputs:
        matched = [c for c in available_cols if c == inp or c.endswith(f"_{inp}")]
        result.extend(matched)
    
    return result
