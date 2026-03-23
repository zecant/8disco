"""
ML package for TradSL.

Contains ML-powered functions organized by task type:
- classifiers: Classification models
- regressors: Regression models
- agents: Reinforcement learning agents

Example usage in DSL:
    predictor:
      type=function
      function=ml.regressors.random_forest
      inputs=[features]
      warmup=100
      model_params:
        n_estimators=100
        max_depth=10

Registry setup:
    import tradsl.ml as ml
    dag.resolve({"ml": ml, ...})
"""
from tradsl.mlfunctions import MLFunction, Regressor, Classifier, Agent
from tradsl.ml import agents
from tradsl.ml import regressors
from tradsl.ml import classifiers

__all__ = ["MLFunction", "Regressor", "Classifier", "Agent", "agents", "regressors", "classifiers"]
