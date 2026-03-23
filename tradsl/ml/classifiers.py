"""
Classifiers for TradSL.

Example usage in DSL:
    predictor:
      type=function
      function=ml.classifiers.random_forest
      inputs=[features]
      warmup=100
      model_params:
        n_estimators=100
        max_depth=10
"""
from tradsl.mlfunctions import Classifier


class RandomForestClassifier(Classifier):
    """Random Forest classifier wrapper."""
    
    def __init__(self, warmup: int = 50, model=None, **model_params):
        import sklearn.ensemble
        if model is None:
            model = sklearn.ensemble.RandomForestClassifier(**model_params)
        super().__init__(model=model, warmup=warmup)


class GradientBoostingClassifier(Classifier):
    """Gradient Boosting classifier wrapper."""
    
    def __init__(self, warmup: int = 50, model=None, **model_params):
        import sklearn.ensemble
        if model is None:
            model = sklearn.ensemble.GradientBoostingClassifier(**model_params)
        super().__init__(model=model, warmup=warmup)


class SVMClassifier(Classifier):
    """Support Vector Machine classifier wrapper."""
    
    def __init__(self, warmup: int = 50, model=None, **model_params):
        import sklearn.svm
        if model is None:
            model = sklearn.svm.SVC(**model_params)
        super().__init__(model=model, warmup=warmup)


class KNNClassifier(Classifier):
    """K-Nearest Neighbors classifier wrapper."""
    
    def __init__(self, warmup: int = 50, model=None, **model_params):
        import sklearn.neighbors
        if model is None:
            model = sklearn.neighbors.KNeighborsClassifier(**model_params)
        super().__init__(model=model, warmup=warmup)
