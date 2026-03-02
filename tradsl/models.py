"""
ML Models for trading signal generation.

Provides DecisionTreeModel for classification-based trading strategies.
"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Union


class DecisionTreeModel:
    """
    Decision Tree classifier for generating trading signals.
    
    Supports both 3-class (sell/hold/buy) and binary (sell/buy) classification.
    Weak signals below confidence_threshold are treated as HOLD.
    
    DSL Usage:
        :signal_model
        type=model
        class=tradsl.models.DecisionTreeModel
        inputs=[nvda, vix]
        params=max_depth=10
        dotraining=true
    
    Parameters:
        max_depth: Maximum tree depth (default 10)
        min_samples_split: Min samples to split node (default 2)
        min_samples_leaf: Min samples at leaf (default 1)
        criterion: Split criterion 'gini' or 'entropy' (default 'gini')
        class_weight: 'balanced', dict, or None (default None)
        confidence_threshold: Min confidence to signal (default 0.4)
        random_state: Random seed for reproducibility (default 42)
        n_classes: Number of classes - 2 for binary, 3 for 3-class (default 3)
    """
    
    ACTION_MAP_3CLASS = {
        0: 'sell',
        1: 'hold',
        2: 'buy'
    }
    
    ACTION_MAP_BINARY = {
        0: 'sell',
        1: 'buy'
    }
    
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini',
        class_weight: Optional[Union[str, Dict]] = None,
        confidence_threshold: float = 0.4,
        random_state: int = 42,
        n_classes: int = 3,
        **kwargs
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.class_weight = class_weight
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        self.n_classes = n_classes
        
        if criterion not in ('gini', 'entropy'):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got '{criterion}'")
        
        if n_classes not in (2, 3):
            raise ValueError(f"n_classes must be 2 or 3, got {n_classes}")
        
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            class_weight=class_weight,
            random_state=random_state
        )
        
        self.is_trained = False
    
    def train(self, X: Union[np.ndarray, pd.DataFrame, pd.Series], 
              y: Union[np.ndarray, pd.Series, pd.DataFrame] = None,
              **kwargs) -> 'DecisionTreeModel':
        """
        Train the decision tree on features and labels.
        
        Args:
            X: Feature matrix (n_samples, n_features) or DataFrame
            y: Target labels (n_samples,) - must be encoded as:
               - 3-class: 0=sell, 1=hold, 2=buy
               - Binary: 0=sell, 1=buy (hold signals come from low confidence)
            **kwargs: Additional training parameters (ignored)
        
        Returns:
            self for chaining
        """
        if y is None:
            raise ValueError(
                "y (target labels) is required. "
                "Provide pre-computed target derived timeseries in DSL."
            )
        
        X_arr = self._to_numpy(X)
        y_arr = self._to_numpy(y)
        
        if len(X_arr) != len(y_arr):
            min_len = min(len(X_arr), len(y_arr))
            X_arr = X_arr[:min_len]
            y_arr = y_arr[:min_len]
        
        if len(X_arr) == 0:
            raise ValueError("No training samples provided")
        
        unique_classes = np.unique(y_arr)
        if len(unique_classes) > self.n_classes:
            raise ValueError(
                f"Found {len(unique_classes)} unique classes in y, "
                f"but model configured for {self.n_classes} classes"
            )
        
        self.model.fit(X_arr, y_arr)
        self.is_trained = True
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series], 
                **kwargs) -> Dict[str, Any]:
        """
        Predict trading action from features.
        
        Args:
            X: Feature vector (1, n_features) or DataFrame
            **kwargs: Additional parameters (ignored)
        
        Returns:
            dict with keys:
                action: 'buy', 'sell', or 'hold'
                confidence: probability of predicted class
                proba_sell: probability of sell class
                proba_hold: probability of hold class (0 for binary)
                proba_buy: probability of buy class
        """
        if not self.is_trained:
            return self._hold_output(confidence=0.0)
        
        X_arr = self._to_numpy(X)
        
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        
        if X_arr.shape[0] == 0:
            return self._hold_output(confidence=0.0)
        
        proba = self.model.predict_proba(X_arr)[0]
        
        return self._build_output(proba)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get class probabilities without thresholding.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        if not self.is_trained:
            n_classes = self.n_classes
            return np.ones((1, n_classes)) / n_classes
        
        X_arr = self._to_numpy(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        
        return self.model.predict_proba(X_arr)
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Get feature importances from the tree.
        
        Returns:
            Array of feature importances or None if not trained
        """
        if not self.is_trained:
            return None
        return self.model.feature_importances_
    
    def get_tree_depth(self) -> Optional[int]:
        """
        Get the depth of the trained tree.
        
        Returns:
            Tree depth or None if not trained
        """
        if not self.is_trained:
            return None
        return self.model.get_depth()
    
    def get_n_leaves(self) -> Optional[int]:
        """
        Get the number of leaf nodes in the tree.
        
        Returns:
            Number of leaves or None if not trained
        """
        if not self.is_trained:
            return None
        return self.model.get_n_leaves()
    
    def save_state(self, path: Union[str, Path]):
        """
        Save model state to disk:
            path:.
        
        Args File path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model': self.model,
            'is_trained': self.is_trained,
            'params': {
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'criterion': self.criterion,
                'class_weight': self.class_weight,
                'confidence_threshold': self.confidence_threshold,
                'random_state': self.random_state,
                'n_classes': self.n_classes
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, path: Union[str, Path]):
        """
        Load model state from disk.
        
        Args:
            path: File path to load model from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.model = state['model']
        self.is_trained = state['is_trained']
        
        for key, value in state['params'].items():
            setattr(self, key, value)
    
    def _to_numpy(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, pd.Series):
            return X.values
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.array(X)
    
    def _build_output(self, proba: np.ndarray) -> Dict[str, Any]:
        """Build output dict from probabilities."""
        n_proba = len(proba)
        
        if n_proba == 1:
            proba_sell = 1.0 if self.n_classes == 2 else 0.0
            proba_hold = 0.0
            proba_buy = 0.0 if self.n_classes == 2 else 1.0
            pred_class = 0 if self.n_classes == 2 else 1
        else:
            proba_sell = float(proba[0])
            proba_buy = float(proba[-1])
            proba_hold = float(proba[1]) if n_proba > 2 and self.n_classes == 3 else 0.0
            pred_class = np.argmax(proba)
        
        confidence = float(proba[pred_class])
        
        action_map = self.ACTION_MAP_BINARY if self.n_classes == 2 else self.ACTION_MAP_3CLASS
        
        if confidence < self.confidence_threshold:
            action = 'hold'
        else:
            action = action_map.get(pred_class, 'hold')
        
        return {
            'action': action,
            'confidence': confidence,
            'proba_sell': proba_sell,
            'proba_hold': proba_hold,
            'proba_buy': proba_buy
        }
    
    def _hold_output(self, confidence: float) -> Dict[str, Any]:
        """Return hold output (for untrained model or empty input)."""
        return {
            'action': 'hold',
            'confidence': confidence,
            'proba_sell': 0.0,
            'proba_hold': 1.0,
            'proba_buy': 0.0
        }
    
    def __repr__(self) -> str:
        return (
            f"DecisionTreeModel("
            f"max_depth={self.max_depth}, "
            f"n_classes={self.n_classes}, "
            f"trained={self.is_trained})"
        )
