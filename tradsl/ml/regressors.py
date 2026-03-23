"""
Regressors for TradSL.

Example usage in DSL:
    predictor:
      type=function
      function=ml.regressors.random_forest
      inputs=[features]
      warmup=100
      model_params:
        n_estimators=100
        max_depth=10
"""
from tradsl.mlfunctions import Regressor


class RandomForestRegressor(Regressor):
    """Random Forest regressor wrapper."""
    
    def __init__(self, warmup: int = 50, model=None, **model_params):
        import sklearn.ensemble
        if model is None:
            model = sklearn.ensemble.RandomForestRegressor(**model_params)
        super().__init__(model=model, warmup=warmup)


class GradientBoostingRegressor(Regressor):
    """Gradient Boosting regressor wrapper."""
    
    def __init__(self, warmup: int = 50, model=None, **model_params):
        import sklearn.ensemble
        if model is None:
            model = sklearn.ensemble.GradientBoostingRegressor(**model_params)
        super().__init__(model=model, warmup=warmup)


class LinearRegressor(Regressor):
    """Linear regression wrapper."""
    
    def __init__(self, warmup: int = 50, model=None, **model_params):
        import sklearn.linear_model
        if model is None:
            model = sklearn.linear_model.LinearRegression(**model_params)
        super().__init__(model=model, warmup=warmup)


class SVRRegressor(Regressor):
    """Support Vector Regression wrapper."""
    
    def __init__(self, warmup: int = 50, model=None, **model_params):
        import sklearn.svm
        if model is None:
            model = sklearn.svm.SVR(**model_params)
        super().__init__(model=model, warmup=warmup)


class KNNRegressor(Regressor):
    """K-Nearest Neighbors regressor wrapper."""
    
    def __init__(self, warmup: int = 50, model=None, **model_params):
        import sklearn.neighbors
        if model is None:
            model = sklearn.neighbors.KNeighborsRegressor(**model_params)
        super().__init__(model=model, warmup=warmup)
