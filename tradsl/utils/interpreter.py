from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from ..utils.data_loader import load_timeseries
from ..utils.feature_engine import compute_features, compute_features_incremental, FeatureEngineError
from ..utils.training import TrainingScheduler, create_scheduler_from_config


class InterpreterError(Exception):
    pass


class TradslInterpreter:
    """
    DSL Interpreter for trading systems.
    
    This class interprets the parsed DSL config and orchestrates:
    - Data loading from adapters
    - Feature computation via DAG
    - Model training and inference
    - Position sizing
    
    Can be used standalone or integrated with NautilusTrader.
    """
    
    def __init__(self, config: dict):
        """
        Initialize interpreter with parsed DSL config.
        
        Args:
            config: Parsed and resolved DSL config from tradsl.parse()
        """
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.feature_df: Optional[pd.DataFrame] = None
        
        self.models: Dict[str, Any] = {}
        self.schedulers: Dict[str, TrainingScheduler] = {}
        self.position_sizer: Optional[callable] = None
        
        self.agent_config: Optional[dict] = None
        self.tradable_symbols: List[str] = []
        
        self._initialize()
    
    def _initialize(self):
        """Initialize models, schedulers, and sizer from config."""
        for name, node in self.config.items():
            if name.startswith('_'):
                continue
            
            node_type = node.get('type')
            
            if node_type == 'model':
                self._init_model(name, node)
            
            elif node_type == 'agent':
                self.agent_config = node
                self.position_sizer = node.get('sizer')
                self.tradable_symbols = node.get('tradable', [])
        
        if self.agent_config is None:
            raise InterpreterError("No agent defined in config")
    
    def _init_model(self, name: str, node: dict):
        """Initialize a model instance."""
        model_class = node.get('class')
        
        if not model_class:
            raise InterpreterError(f"Model '{name}' has no class defined")
        
        model_params = node.get('params', {})
        
        try:
            model = model_class(**model_params)
        except Exception as e:
            raise InterpreterError(f"Failed to instantiate model '{name}': {e}")
        
        if node.get('dotraining'):
            scheduler = create_scheduler_from_config(self.config, name)
            if scheduler:
                self.schedulers[name] = scheduler
        
        if node.get('load_from'):
            scheduler = self.schedulers.get(name)
            if scheduler:
                loaded = scheduler.load_model(model_class, **model_params)
                if loaded is not None:
                    model = loaded
        
        self.models[name] = model
    
    def load_data(
        self,
        start: datetime,
        end: datetime,
        frequency: str = '1min'
    ):
        """
        Load historical data from configured adapters.
        
        Args:
            start: Start datetime
            end: End datetime
            frequency: Bar frequency
        """
        self.data = load_timeseries(self.config, start, end, frequency)
        
        if self.data is None or self.data.empty:
            raise InterpreterError("No data loaded from adapters")
    
    def compute_initial_features(self):
        """Compute all features from loaded data."""
        if self.data is None:
            raise InterpreterError("No data loaded. Call load_data() first.")
        
        self.feature_df = compute_features(self.config, self.data)
    
    def train_models(self):
        """Train all models marked with dotraining=true."""
        for model_name, scheduler in self.schedulers.items():
            model = self.models.get(model_name)
            if not model or not hasattr(model, 'train'):
                continue
            
            if self.feature_df is None:
                continue
            
            target_col = model_name
            if target_col not in self.feature_df.columns:
                target_col = f"{model_name}_target"
            
            if target_col not in self.feature_df.columns:
                continue
            
            for idx in range(len(self.feature_df)):
                try:
                    X, y = scheduler.get_training_window(
                        self.feature_df, 
                        idx, 
                        target_col
                    )
                    
                    if len(X) < 10:
                        continue
                    
                    model.train(X, y)
                    scheduler.update_last_train(idx, self.feature_df.index[idx])
                    break
                except Exception as e:
                    raise InterpreterError(
                        f"Training failed for model '{model_name}' at idx {idx}: {e}"
                    )
    
    def predict(self, new_data: pd.Series) -> Dict[str, Any]:
        """
        Make prediction for new data point.
        
        Args:
            new_data: Series with new timeseries values
        
        Returns:
            Dict mapping output_name -> prediction value
        """
        if self.feature_df is None:
            self.feature_df = pd.DataFrame([new_data])
        else:
            self.feature_df = compute_features_incremental(
                self.config, 
                self.feature_df, 
                new_data
            )
        
        latest_row = self.feature_df.iloc[-1]
        
        predictions = {}
        
        for model_name, model in self.models.items():
            if not hasattr(model, 'predict'):
                continue
            
            model_config = self.config.get(model_name, {})
            model_inputs = model_config.get('inputs', [])
            
            input_cols = self._get_model_inputs(model_name)
            
            if not input_cols and model_inputs:
                continue
            
            try:
                if input_cols:
                    input_df = self.feature_df[input_cols].iloc[-1:]
                else:
                    input_df = pd.DataFrame([self.feature_df.iloc[-1]])
                
                result = model.predict(input_df)
                
                if not isinstance(result, dict):
                    raise ValueError(
                        f"Model predict() must return a dict, got {type(result).__name__}"
                    )
                
                for key, value in result.items():
                    predictions[f"{model_name}_{key}"] = value
            
            except FeatureEngineError:
                raise
            except Exception as e:
                raise FeatureEngineError(
                    f"Model '{model_name}' prediction failed: {e}"
                )
        
        return predictions
    
    def get_allocation(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """
        Get portfolio allocation from predictions.
        
        Args:
            predictions: Dict of predictions from models
        
        Returns:
            Dict mapping symbol -> weight (0 to 1)
        
        Raises:
            ValueError: If sizer fails or returns invalid allocation
        """
        if not self.position_sizer:
            if not self.tradable_symbols:
                return {}
            return {sym: 1.0 / len(self.tradable_symbols) 
                   for sym in self.tradable_symbols}
        
        allocation = self.position_sizer(
            predictions, 
            self.tradable_symbols
        )
        
        if not isinstance(allocation, dict):
            raise ValueError(
                f"Sizer must return a dict, got {type(allocation).__name__}"
            )
        
        valid_allocation = {}
        for sym in self.tradable_symbols:
            if sym not in allocation:
                raise ValueError(
                    f"Sizer must return all tradable symbols. "
                    f"Missing: {sym}. Got keys: {list(allocation.keys())}"
                )
            
            val = allocation[sym]
            
            if val is None:
                raise ValueError(f"Sizer returned None for symbol {sym}")
            
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                raise ValueError(f"Sizer returned NaN/Inf for symbol {sym}")
            
            try:
                v = float(val)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Sizer returned non-numeric value for {sym}: {val}"
                )
            
            if v < 0:
                raise ValueError(
                    f"Sizer returned negative allocation for {sym}: {v}"
                )
            
            valid_allocation[sym] = v
        
        total = sum(valid_allocation.values())
        if total <= 0 or np.isnan(total) or np.isinf(total):
            raise ValueError(
                f"Invalid total allocation: {total}"
            )
        
        return valid_allocation
    
    def should_retrain(self, bar_count: int, current_time: datetime) -> List[str]:
        """
        Check which models need retraining.
        
        Args:
            bar_count: Current bar number
            current_time: Current timestamp
        
        Returns:
            List of model names that need retraining
        """
        retrain_list = []
        
        for model_name, scheduler in self.schedulers.items():
            if scheduler.should_retrain(bar_count, current_time):
                retrain_list.append(model_name)
        
        return retrain_list
    
    def retrain_model(self, model_name: str):
        """
        Retrain a specific model.
        
        Args:
            model_name: Name of model to retrain
        """
        if model_name not in self.schedulers:
            return
        
        scheduler = self.schedulers[model_name]
        model = self.models.get(model_name)
        
        if not model or not hasattr(model, 'train'):
            return
        
        if self.feature_df is None:
            return
        
        target_col = model_name
        if target_col not in self.feature_df.columns:
            target_col = f"{model_name}_target"
        
        if target_col not in self.feature_df.columns:
            return
        
        current_idx = len(self.feature_df) - 1
        
        try:
            X, y = scheduler.get_training_window(
                self.feature_df,
                current_idx,
                target_col
            )
            
            if len(X) >= 10:
                model.train(X, y)
                scheduler.update_last_train(
                    current_idx,
                    self.feature_df.index[current_idx]
                )
        except Exception as e:
            raise InterpreterError(
                f"Retrain failed for model '{model_name}': {e}"
            )
    
    def _get_model_inputs(self, model_name: str) -> List[str]:
        """Get input columns for a model."""
        if model_name not in self.config:
            return []
        
        inputs = self.config[model_name].get('inputs', [])
        
        if self.feature_df is None:
            return []
        
        cols = self.feature_df.columns.tolist()
        result = []
        
        for inp in inputs:
            matched = [c for c in cols if c == inp or c.endswith(f"_{inp}")]
            result.extend(matched)
        
        return result
    
    def run_backtest(
        self,
        start: datetime,
        end: datetime,
        frequency: str = '1min',
        on_trade: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        Run full backtest.
        
        Args:
            start: Start datetime
            end: End datetime
            frequency: Bar frequency
            on_trade: Optional callback(trade_dict) for each trade
        
        Returns:
            DataFrame with backtest results
        """
        self.load_data(start, end, frequency)
        self.compute_initial_features()
        self.train_models()
        
        results = []
        
        for idx in range(len(self.feature_df)):
            row = self.feature_df.iloc[idx]
            current_time = self.feature_df.index[idx]
            
            predictions = {}
            for model_name, model in self.models.items():
                if not hasattr(model, 'predict'):
                    continue
                
                input_cols = self._get_model_inputs(model_name)
                if not input_cols:
                    continue
                
                try:
                    X = self.feature_df[input_cols].iloc[:idx+1].iloc[-1:]
                    result = model.predict(X)
                    
                    if isinstance(result, dict):
                        predictions.update(result)
                    else:
                        predictions[model_name] = result
                except Exception as e:
                    raise InterpreterError(
                        f"Prediction failed for model '{model_name}' at idx {idx}: {e}"
                    )
            
            allocation = self.get_allocation(predictions)
            
            result_row = {
                'timestamp': current_time,
                'bar_count': idx,
            }
            result_row.update(allocation)
            results.append(result_row)
            
            retrain_models = self.should_retrain(idx, current_time)
            for model_name in retrain_models:
                self.retrain_model(model_name)
            
            if on_trade:
                on_trade(result_row)
        
        return pd.DataFrame(results)


def create_interpreter(config: dict) -> TradslInterpreter:
    """
    Factory function to create interpreter from parsed config.
    
    Args:
        config: Parsed DSL config
    
    Returns:
        Configured TradslInterpreter
    """
    return TradslInterpreter(config)
