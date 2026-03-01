from typing import Any, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pickle


class TrainingScheduler:
    """
    Manages model retraining during backtesting.
    
    Supports:
    - Schedule types: every_n_bars, daily, weekly, monthly
    - Window types: rolling (fixed size), expanding (all history)
    - Model persistence: save/load to disk
    """
    
    def __init__(
        self,
        schedule: str = 'every_n_bars',
        n_bars: int = 252,
        window_type: str = 'rolling',
        window_size: int = 500,
        save_path: Optional[Path] = None
    ):
        """
        Args:
            schedule: 'every_n_bars', 'daily', 'weekly', 'monthly'
            n_bars: Number of bars between retrains (for 'every_n_bars' schedule)
            window_type: 'rolling' or 'expanding'
            window_size: Number of bars to use for training (for rolling window)
            save_path: Path to save/load trained models
        """
        self.schedule = schedule
        self.n_bars = n_bars
        self.window_type = window_type
        self.window_size = window_size
        self.save_path = save_path
        
        self.last_train_bar = -1
        self.last_train_time: Optional[datetime] = None
    
    def should_retrain(
        self,
        bar_count: int,
        current_time: datetime
    ) -> bool:
        """
        Determine if model should be retrained.
        
        Args:
            bar_count: Current bar number
            current_time: Current timestamp
        
        Returns:
            True if model should be retrained
        """
        if self.last_train_bar < 0:
            return True
        
        if self.schedule == 'every_n_bars':
            return (bar_count - self.last_train_bar) >= self.n_bars
        
        elif self.schedule == 'daily':
            if self.last_train_time is None:
                return True
            return current_time.date() > self.last_train_time.date()
        
        elif self.schedule == 'weekly':
            if self.last_train_time is None:
                return True
            current_week = current_time.isocalendar()[1]
            last_week = self.last_train_time.isocalendar()[1]
            return current_week != last_week
        
        elif self.schedule == 'monthly':
            if self.last_train_time is None:
                return True
            return current_time.month != self.last_train_time.month
        
        return False
    
    def get_training_window(
        self,
        feature_df: pd.DataFrame,
        current_idx: int,
        target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get training data window up to current index.
        
        Args:
            feature_df: Full feature DataFrame
            current_idx: Current row index
            target_col: Name of target column
        
        Returns:
            (X, y) tuple for training
        """
        if self.window_type == 'rolling':
            start_idx = max(0, current_idx - self.window_size)
            window_df = feature_df.iloc[start_idx:current_idx]
        else:
            window_df = feature_df.iloc[:current_idx]
        
        if target_col not in window_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in feature DataFrame")
        
        X = window_df.drop(columns=[target_col], errors='ignore')
        y = window_df[target_col]
        
        X = X.dropna()
        y = y.loc[X.index]
        
        return X, y
    
    def train(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        **params
    ) -> Any:
        """
        Train model on given data.
        
        Args:
            model: Model instance with 'train(X, y, **params)' method
            X: Feature DataFrame
            y: Target Series
        
        Returns:
            Trained model
        """
        if not hasattr(model, 'train'):
            raise ValueError(f"Model {type(model)} has no 'train' method")
        
        model.train(X, y, **params)
        
        if self.save_path:
            self._save_model(model)
        
        return model
    
    def load_model(self, model_class: type, **init_params) -> Optional[Any]:
        """
        Load pre-trained model from disk.
        
        Args:
            model_class: Model class to instantiate
            **init_params: Additional parameters for model initialization
        
        Returns:
            Loaded model or None if no saved model exists
        """
        if not self.save_path or not self.save_path.exists():
            return None
        
        model_path = self.save_path
        if model_path.is_dir():
            model_path = model_path / 'model.pkl'
        
        if not model_path.exists():
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_class(**init_params)
            
            if hasattr(model, 'load_state'):
                model.load_state(model_data)
            else:
                for key, value in model_data.items():
                    setattr(model, key, value)
            
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    def _save_model(self, model: Any):
        """Save model to disk."""
        if not self.save_path:
            return
        
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        model_path = self.save_path
        if model_path.is_dir():
            model_path = model_path / 'model.pkl'
        
        model_data = {}
        if hasattr(model, 'state'):
            model_data = model.state
        elif hasattr(model, '__dict__'):
            model_data = {k: v for k, v in model.__dict__.items() 
                         if not k.startswith('_')}
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def update_last_train(self, bar_count: int, current_time: datetime):
        """Update training state after retraining."""
        self.last_train_bar = bar_count
        self.last_train_time = current_time


def create_scheduler_from_config(config: dict, node_name: str) -> Optional[TrainingScheduler]:
    """
    Create TrainingScheduler from DSL config node.
    
    Args:
        config: Parsed DSL config
        node_name: Name of model node
    
    Returns:
        TrainingScheduler or None if not configured for training
    """
    if node_name not in config:
        return None
    
    node = config[node_name]
    
    has_training_config = (
        node.get('dotraining') or
        node.get('load_from') or
        node.get('retrain_schedule') or
        node.get('training_window') or
        node.get('training_window_size')
    )
    
    if not has_training_config:
        return None
    
    schedule = node.get('retrain_schedule', 'every_n_bars')
    n_bars = 252
    
    if 'n_bars' in node:
        n_bars = node['n_bars']
    
    window_type = node.get('training_window', 'rolling')
    window_size = node.get('training_window_size', 500)
    
    load_from = node.get('load_from')
    save_path = Path(load_from) if load_from else None
    
    scheduler = TrainingScheduler(
        schedule=schedule,
        n_bars=n_bars,
        window_type=window_type,
        window_size=window_size,
        save_path=save_path
    )
    
    return scheduler
