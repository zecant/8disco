"""Tests for model implementations."""
import pytest
import numpy as np
import os
import tempfile

from tradsl.models_impl import (
    SklearnModel, RandomForestModel, LinearModel,
    DecisionTreeModel, GradientBoostingModel
)


class TestRandomForestModel:
    def test_init_default(self):
        model = RandomForestModel()
        assert model.model_type == "classifier"
        assert model.params['n_estimators'] == 100
        assert model.params['max_depth'] == 5
    
    def test_init_regressor(self):
        model = RandomForestModel(model_type="regressor")
        assert model.model_type == "regressor"
    
    def test_init_custom_params(self):
        model = RandomForestModel(
            n_estimators=50,
            max_depth=3,
            model_type="classifier"
        )
        assert model.params['n_estimators'] == 50
        assert model.params['max_depth'] == 3
    
    def test_is_fitted_initially_false(self):
        model = RandomForestModel()
        assert model.is_fitted is False
    
    def test_predict_not_fitted_returns_zero(self):
        model = RandomForestModel()
        features = np.array([1.0, 2.0, 3.0])
        result = model.predict(features)
        assert result == 0.0
    
    def test_fit_and_predict_classifier(self):
        model = RandomForestModel(n_estimators=10, max_depth=3, random_state=42)
        
        np.random.seed(42)
        features = np.random.randn(100, 5)
        labels = np.random.choice([0, 1, 2], size=100)
        
        model.fit(features, labels)
        
        assert model.is_fitted is True
        
        test_features = np.random.randn(1, 5)
        result = model.predict(test_features)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_fit_and_predict_regressor(self):
        model = RandomForestModel(
            n_estimators=10,
            max_depth=3,
            model_type="regressor",
            random_state=42
        )
        
        np.random.seed(42)
        features = np.random.randn(100, 5)
        labels = np.random.randn(100)
        
        model.fit(features, labels)
        
        assert model.is_fitted is True
        
        test_features = np.random.randn(1, 5)
        result = model.predict(test_features)
        
        assert isinstance(result, float)
    
    def test_fit_with_nan_labels(self):
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        features = np.random.randn(100, 5)
        labels = np.full(100, np.nan)
        labels[50] = 1.0
        
        model.fit(features, labels)
        
        assert model.is_fitted is False
    
    def test_save_and_load_checkpoint(self):
        model = RandomForestModel(n_estimators=10, max_depth=3, random_state=42)
        
        features = np.random.randn(50, 5)
        labels = np.random.choice([0, 1], size=50)
        model.fit(features, labels)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.joblib')
            model.save_checkpoint(path)
            
            new_model = RandomForestModel(n_estimators=10, max_depth=3)
            new_model.load_checkpoint(path)
            
            assert new_model.is_fitted is True


class TestLinearModel:
    def test_init_default(self):
        model = LinearModel()
        assert model.model_type == "regressor"
        assert model.regularization == 1.0
    
    def test_init_classifier(self):
        model = LinearModel(model_type="classifier")
        assert model.model_type == "classifier"
    
    def test_is_fitted_initially_false(self):
        model = LinearModel()
        assert model.is_fitted is False
    
    def test_predict_not_fitted_returns_zero(self):
        model = LinearModel()
        features = np.array([1.0, 2.0, 3.0])
        result = model.predict(features)
        assert result == 0.0
    
    def test_fit_and_predict_regressor(self):
        model = LinearModel(model_type="regressor", random_state=42)
        
        np.random.seed(42)
        features = np.random.randn(100, 5)
        labels = np.sum(features * np.array([1, 2, 3, 4, 5]), axis=1) + np.random.randn(100) * 0.1
        
        model.fit(features, labels)
        
        assert model.is_fitted is True
        
        test_features = np.random.randn(1, 5)
        result = model.predict(test_features)
        
        assert isinstance(result, float)
    
    def test_fit_and_predict_classifier(self):
        model = LinearModel(model_type="classifier", random_state=42)
        
        np.random.seed(42)
        features = np.random.randn(100, 5)
        labels = (np.sum(features, axis=1) > 0).astype(int)
        
        model.fit(features, labels)
        
        assert model.is_fitted is True
        
        test_features = np.random.randn(1, 5)
        result = model.predict(test_features)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_save_and_load_checkpoint(self):
        model = LinearModel(model_type="regressor", random_state=42)
        
        features = np.random.randn(50, 5)
        labels = np.random.randn(50)
        model.fit(features, labels)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.joblib')
            model.save_checkpoint(path)
            
            new_model = LinearModel(model_type="regressor")
            new_model.load_checkpoint(path)
            
            assert new_model.is_fitted is True


class TestDecisionTreeModel:
    def test_init(self):
        model = DecisionTreeModel(max_depth=5, random_state=42)
        assert model.model_type == "classifier"
        assert model.params['max_depth'] == 5
    
    def test_fit_and_predict(self):
        model = DecisionTreeModel(max_depth=3, random_state=42)
        
        np.random.seed(42)
        features = np.random.randn(50, 3)
        labels = np.random.choice([0, 1], size=50)
        
        model.fit(features, labels)
        
        assert model.is_fitted is True
        
        test_features = np.random.randn(1, 3)
        result = model.predict(test_features)
        
        assert isinstance(result, float)


class TestGradientBoostingModel:
    def test_init(self):
        model = GradientBoostingModel(n_estimators=10, max_depth=3, random_state=42)
        assert model.model_type == "regressor"
        assert model.params['n_estimators'] == 10
    
    def test_fit_and_predict(self):
        model = GradientBoostingModel(n_estimators=10, max_depth=3, random_state=42)
        
        np.random.seed(42)
        features = np.random.randn(50, 3)
        labels = np.random.randn(50)
        
        model.fit(features, labels)
        
        assert model.is_fitted is True
        
        test_features = np.random.randn(1, 3)
        result = model.predict(test_features)
        
        assert isinstance(result, float)


class TestSklearnModel:
    def test_init_with_class(self):
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            model = SklearnModel(
                model_class=RandomForestRegressor,
                params={'n_estimators': 10, 'max_depth': 3},
                model_type="regressor"
            )
            
            assert model.model_type == "regressor"
            assert model._model is not None
        except ImportError:
            pytest.skip("scikit-learn not installed")
    
    def test_predict_not_initialized(self):
        model = SklearnModel()
        features = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError):
            model.fit(np.array([[1, 2]]), np.array([1]))
    
    def test_fit_and_predict_with_sklearn(self):
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            model = SklearnModel(
                model_class=RandomForestRegressor,
                params={'n_estimators': 10, 'max_depth': 3, 'random_state': 42},
                model_type="regressor"
            )
            
            np.random.seed(42)
            features = np.random.randn(50, 5)
            labels = np.random.randn(50)
            
            model.fit(features, labels)
            
            assert model.is_fitted is True
            
            test_features = np.random.randn(1, 5)
            result = model.predict(test_features)
            
            assert isinstance(result, float)
        except ImportError:
            pytest.skip("scikit-learn not installed")


class TestModelEdgeCases:
    def test_predict_with_nan_features(self):
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        features = np.random.randn(50, 5)
        labels = np.random.choice([0, 1], size=50)
        model.fit(features, labels)
        
        nan_features = np.full(5, np.nan)
        result = model.predict(nan_features)
        
        assert isinstance(result, float)
    
    def test_fit_insufficient_samples(self):
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        features = np.random.randn(3, 5)
        labels = np.array([0, 1, 0])
        
        model.fit(features, labels)
        
        assert model.is_fitted is False
    
    def test_all_nan_features(self):
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        features = np.full((50, 5), np.nan)
        labels = np.random.choice([0, 1], size=50)
        
        model.fit(features, labels)
        
        assert model.is_fitted is False
