"""
Model Implementations for TradSL

Section 10.4.4: Required trainable model implementations.
"""
from typing import Optional, Dict, Any, Type
import numpy as np
import joblib
import os

from tradsl.models import BaseTrainableModel


class SklearnModel(BaseTrainableModel):
    """
    Thin wrapper around any scikit-learn estimator.
    
    Handles fit/predict delegation and checkpoint serialization via joblib.
    Model class is configured via params.
    """
    
    def __init__(
        self,
        model_class: Optional[Type] = None,
        params: Optional[Dict[str, Any]] = None,
        model_type: str = "regressor"
    ):
        """
        Initialize sklearn model.
        
        Args:
            model_class: sklearn estimator class (e.g., RandomForestRegressor)
            params: Dict of model hyperparameters
            model_type: 'regressor' or 'classifier'
        """
        self.model_class = model_class
        self.params = params or {}
        self.model_type = model_type
        
        self._model = None
        self._is_fitted = False
        
        if model_class is not None:
            self._model = model_class(**self.params)
    
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit the model to historical features and labels."""
        if self._model is None:
            raise ValueError("Model not initialized: model_class is None")
        
        valid_indices = ~np.isnan(labels)
        if not np.any(valid_indices):
            return
        
        X = features[valid_indices]
        y = labels[valid_indices]
        
        if len(X) == 0:
            return
        
        self._model.fit(X, y)
        self._is_fitted = True
    
    def predict(self, features: np.ndarray) -> float:
        """Produce output value for the current bar."""
        if not self._is_fitted or self._model is None:
            return 0.0
        
        try:
            X = features.reshape(1, -1)
            pred = self._model.predict(X)
            
            if self.model_type == "classifier":
                if hasattr(self._model, 'predict_proba'):
                    return float(pred[0, 1] if pred.shape[1] > 1 else pred[0])
                return float(pred[0])
            
            return float(pred[0])
        except Exception:
            return 0.0
    
    def save_checkpoint(self, path: str) -> None:
        """Save model state to file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            'model': self._model,
            'params': self.params,
            'model_type': self.model_type,
            'is_fitted': self._is_fitted
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model state from file."""
        data = joblib.load(path)
        self._model = data['model']
        self.params = data['params']
        self.model_type = data['model_type']
        self._is_fitted = data['is_fitted']
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class RandomForestModel(BaseTrainableModel):
    """
    Random forest classifier/regressor.
    
    Suitable for triple barrier label prediction, regime classification,
    volatility forecasting.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        model_type: str = "classifier",
        random_state: Optional[int] = None
    ):
        """
        Initialize random forest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Min samples to split
            min_samples_leaf: Min samples in leaf
            model_type: 'classifier' or 'regressor'
            random_state: RNG seed
        """
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            self._Classifier = RandomForestClassifier
            self._Regressor = RandomForestRegressor
        except ImportError:
            raise ImportError("scikit-learn required for RandomForestModel")
        
        self.model_type = model_type
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        }
        
        self._model = self._create_model()
        self._is_fitted = False
    
    def _create_model(self):
        """Create the underlying model instance."""
        if self.model_type == "classifier":
            return self._Classifier(**self.params)
        return self._Regressor(**self.params)
    
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit random forest to historical data."""
        valid_mask = ~np.isnan(labels)
        if not np.any(valid_mask):
            return
        
        X = features[valid_mask]
        y = labels[valid_mask]
        
        if len(X) < 10:
            return
        
        nan_cols = np.all(np.isnan(X), axis=0)
        if np.all(nan_cols):
            return
        
        X = X[:, ~nan_cols]
        
        if self.model_type == "classifier":
            y = np.round(y).astype(int)
            y = np.clip(y, 0, 2)
        
        self._model.fit(X, y)
        self._is_fitted = True
    
    def predict(self, features: np.ndarray) -> float:
        """Predict output for current bar."""
        if not self._is_fitted:
            return 0.0
        
        try:
            X = features.reshape(1, -1)
            
            nan_cols = np.all(np.isnan(X), axis=0)
            if np.any(nan_cols):
                X = X[:, ~nan_cols]
            
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            if self.model_type == "classifier":
                proba = self._model.predict_proba(X)
                if proba.shape[1] >= 2:
                    return float(proba[0, 1])
                return float(proba[0, 0])
            else:
                return float(self._model.predict(X)[0])
        except Exception:
            return 0.0
    
    def save_checkpoint(self, path: str) -> None:
        """Save model state."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            'model': self._model,
            'params': self.params,
            'model_type': self.model_type,
            'is_fitted': self._is_fitted
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model state."""
        data = joblib.load(path)
        self._model = data['model']
        self.params = data['params']
        self.model_type = data['model_type']
        self._is_fitted = data['is_fitted']
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class LinearModel(BaseTrainableModel):
    """
    Linear or logistic regression model.
    
    Suitable as a baseline to establish predictive value before
    using more complex models.
    """
    
    def __init__(
        self,
        model_type: str = "regressor",
        regularization: float = 1.0,
        fit_intercept: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize linear model.
        
        Args:
            model_type: 'regressor' (linear) or 'classifier' (logistic)
            regularization: L2 regularization strength (alpha)
            fit_intercept: Whether to fit intercept
            random_state: RNG seed (for SGD solver)
        """
        try:
            from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
            self._LinearRegression = LinearRegression
            self._LogisticRegression = LogisticRegression
            self._Ridge = Ridge
        except ImportError:
            raise ImportError("scikit-learn required for LinearModel")
        
        self.model_type = model_type
        self.regularization = regularization
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        
        self._model = self._create_model()
        self._is_fitted = False
    
    def _create_model(self):
        """Create the underlying model."""
        if self.model_type == "classifier":
            return self._LogisticRegression(
                C=self.regularization,
                fit_intercept=self.fit_intercept,
                solver='lbfgs',
                max_iter=1000,
                random_state=self.random_state
            )
        else:
            if self.regularization != 1.0:
                return self._Ridge(
                    alpha=self.regularization,
                    fit_intercept=self.fit_intercept
                )
            return self._LinearRegression(fit_intercept=self.fit_intercept)
    
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit linear model to historical data."""
        valid_mask = ~np.isnan(labels)
        if not np.any(valid_mask):
            return
        
        X = features[valid_mask]
        y = labels[valid_mask]
        
        if len(X) < 10:
            return
        
        nan_cols = np.all(np.isnan(X), axis=0)
        if np.all(nan_cols):
            return
        
        X = X[:, ~nan_cols]
        
        if self.model_type == "classifier":
            y = np.round(y).astype(int)
            y = np.clip(y, 0, 1)
        
        self._model.fit(X, y)
        self._is_fitted = True
    
    def predict(self, features: np.ndarray) -> float:
        """Predict output for current bar."""
        if not self._is_fitted:
            return 0.0
        
        try:
            X = features.reshape(1, -1)
            
            nan_cols = np.all(np.isnan(X), axis=0)
            if np.any(nan_cols):
                X = X[:, ~nan_cols]
            
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            if self.model_type == "classifier":
                proba = self._model.predict_proba(X)
                return float(proba[0, 1])
            else:
                return float(self._model.predict(X)[0])
        except Exception:
            return 0.0
    
    def save_checkpoint(self, path: str) -> None:
        """Save model state."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            'model': self._model,
            'model_type': self.model_type,
            'regularization': self.regularization,
            'fit_intercept': self.fit_intercept,
            'is_fitted': self._is_fitted
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model state."""
        data = joblib.load(path)
        self._model = data['model']
        self.model_type = data['model_type']
        self.regularization = data['regularization']
        self.fit_intercept = data['fit_intercept']
        self._is_fitted = data['is_fitted']
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class DecisionTreeModel(BaseTrainableModel):
    """
    Single decision tree model.
    
    Fast training, interpretable. Suitable for rapid hypothesis testing.
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        model_type: str = "classifier",
        random_state: Optional[int] = None
    ):
        """Initialize decision tree model."""
        try:
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            self._Classifier = DecisionTreeClassifier
            self._Regressor = DecisionTreeRegressor
        except ImportError:
            raise ImportError("scikit-learn required for DecisionTreeModel")
        
        self.model_type = model_type
        self.params = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        }
        
        self._model = self._create_model()
        self._is_fitted = False
    
    def _create_model(self):
        if self.model_type == "classifier":
            return self._Classifier(**self.params)
        return self._Regressor(**self.params)
    
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        valid_mask = ~np.isnan(labels)
        if not np.any(valid_mask):
            return
        
        X = features[valid_mask]
        y = labels[valid_mask]
        
        if len(X) < 5:
            return
        
        nan_cols = np.all(np.isnan(X), axis=0)
        if np.all(nan_cols):
            return
        
        X = X[:, ~nan_cols]
        
        if self.model_type == "classifier":
            y = np.round(y).astype(int)
            y = np.clip(y, 0, 2)
        
        self._model.fit(X, y)
        self._is_fitted = True
    
    def predict(self, features: np.ndarray) -> float:
        if not self._is_fitted:
            return 0.0
        
        try:
            X = features.reshape(1, -1)
            nan_cols = np.all(np.isnan(X), axis=0)
            if np.any(nan_cols):
                X = X[:, ~nan_cols]
            
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            if self.model_type == "classifier":
                proba = self._model.predict_proba(X)
                if proba.shape[1] >= 2:
                    return float(proba[0, 1])
                return float(proba[0, 0])
            return float(self._model.predict(X)[0])
        except Exception:
            return 0.0
    
    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            'model': self._model,
            'params': self.params,
            'model_type': self.model_type,
            'is_fitted': self._is_fitted
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        data = joblib.load(path)
        self._model = data['model']
        self.params = data['params']
        self.model_type = data['model_type']
        self._is_fitted = data['is_fitted']
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class GradientBoostingModel(BaseTrainableModel):
    """
    Gradient boosted trees model.
    
    Higher capacity than random forest.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        model_type: str = "regressor",
        random_state: Optional[int] = None
    ):
        """Initialize gradient boosting model."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            self._Classifier = GradientBoostingClassifier
            self._Regressor = GradientBoostingRegressor
        except ImportError:
            raise ImportError("scikit-learn required for GradientBoostingModel")
        
        self.model_type = model_type
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        }
        
        self._model = self._create_model()
        self._is_fitted = False
    
    def _create_model(self):
        if self.model_type == "classifier":
            return self._Classifier(**self.params)
        return self._Regressor(**self.params)
    
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        valid_mask = ~np.isnan(labels)
        if not np.any(valid_mask):
            return
        
        X = features[valid_mask]
        y = labels[valid_mask]
        
        if len(X) < 20:
            return
        
        nan_cols = np.all(np.isnan(X), axis=0)
        if np.all(nan_cols):
            return
        
        X = X[:, ~nan_cols]
        
        if self.model_type == "classifier":
            y = np.round(y).astype(int)
            y = np.clip(y, 0, 2)
        
        self._model.fit(X, y)
        self._is_fitted = True
    
    def predict(self, features: np.ndarray) -> float:
        if not self._is_fitted:
            return 0.0
        
        try:
            X = features.reshape(1, -1)
            nan_cols = np.all(np.isnan(X), axis=0)
            if np.any(nan_cols):
                X = X[:, ~nan_cols]
            
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            if self.model_type == "classifier":
                proba = self._model.predict_proba(X)
                if proba.shape[1] >= 2:
                    return float(proba[0, 1])
                return float(proba[0, 0])
            return float(self._model.predict(X)[0])
        except Exception:
            return 0.0
    
    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            'model': self._model,
            'params': self.params,
            'model_type': self.model_type,
            'is_fitted': self._is_fitted
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        data = joblib.load(path)
        self._model = data['model']
        self.params = data['params']
        self.model_type = data['model_type']
        self._is_fitted = data['is_fitted']
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
