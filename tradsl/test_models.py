import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

sys.path.insert(0, '.')

from tradsl.models import DecisionTreeModel


class TestDecisionTreeModelBasics:
    """Basic functionality tests."""
    
    def test_default_initialization(self):
        """Test default parameter initialization."""
        model = DecisionTreeModel()
        
        assert model.max_depth == 10
        assert model.min_samples_split == 2
        assert model.min_samples_leaf == 1
        assert model.criterion == 'gini'
        assert model.class_weight is None
        assert model.confidence_threshold == 0.4
        assert model.random_state == 42
        assert model.n_classes == 3
        assert model.is_trained is False
    
    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        model = DecisionTreeModel(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            criterion='entropy',
            class_weight='balanced',
            confidence_threshold=0.6,
            random_state=123,
            n_classes=2
        )
        
        assert model.max_depth == 5
        assert model.min_samples_split == 10
        assert model.min_samples_leaf == 5
        assert model.criterion == 'entropy'
        assert model.class_weight == 'balanced'
        assert model.confidence_threshold == 0.6
        assert model.random_state == 123
        assert model.n_classes == 2
    
    def test_invalid_criterion(self):
        """Test invalid criterion raises error."""
        with pytest.raises(ValueError) as exc:
            DecisionTreeModel(criterion='invalid')
        assert "criterion must be 'gini' or 'entropy'" in str(exc.value)
    
    def test_invalid_n_classes(self):
        """Test invalid n_classes raises error."""
        with pytest.raises(ValueError) as exc:
            DecisionTreeModel(n_classes=4)
        assert "n_classes must be 2 or 3" in str(exc.value)
    
    def test_repr(self):
        """Test string representation."""
        model = DecisionTreeModel(max_depth=5, n_classes=2)
        assert "max_depth=5" in repr(model)
        assert "n_classes=2" in repr(model)
        assert "trained=False" in repr(model)


class TestDecisionTreeModel3Class:
    """Tests for 3-class classification (sell/hold/buy)."""
    
    @pytest.fixture
    def model_3class(self):
        """Create a trained 3-class model."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.random.choice([0, 1, 2], size=100)
        
        model = DecisionTreeModel(max_depth=5, random_state=42)
        model.train(X, y)
        return model
    
    def test_train_3class(self, model_3class):
        """Test 3-class training."""
        assert model_3class.is_trained is True
    
    def test_predict_returns_dict(self, model_3class):
        """Test predict returns proper dict structure."""
        X = np.random.randn(1, 4)
        result = model_3class.predict(X)
        
        assert isinstance(result, dict)
        assert 'action' in result
        assert 'confidence' in result
        assert 'proba_sell' in result
        assert 'proba_hold' in result
        assert 'proba_buy' in result
    
    def test_predict_action_values(self, model_3class):
        """Test predict returns valid action values."""
        X = np.random.randn(1, 4)
        result = model_3class.predict(X)
        
        assert result['action'] in ['buy', 'sell', 'hold']
        assert 0 <= result['confidence'] <= 1
        assert 0 <= result['proba_sell'] <= 1
        assert 0 <= result['proba_hold'] <= 1
        assert 0 <= result['proba_buy'] <= 1
    
    def test_predict_probabilities_sum_to_one(self, model_3class):
        """Test probabilities sum to approximately 1."""
        X = np.random.randn(1, 4)
        result = model_3class.predict(X)
        
        total = result['proba_sell'] + result['proba_hold'] + result['proba_buy']
        assert abs(total - 1.0) < 1e-6
    
    def test_confidence_threshold_hold(self):
        """Test that low confidence returns hold."""
        model = DecisionTreeModel(
            max_depth=1, 
            confidence_threshold=0.9,
            random_state=42
        )
        
        X = np.random.randn(20, 2)
        y = np.array([0] * 7 + [1] * 6 + [2] * 7)
        model.train(X, y)
        
        result = model.predict(X[:1])
        assert result['action'] == 'hold'
    
    def test_high_confidence_returns_signal(self):
        """Test that high confidence returns buy/sell."""
        model = DecisionTreeModel(
            max_depth=10, 
            confidence_threshold=0.3,
            random_state=42
        )
        
        X = np.array([
            [1, 2], [1, 2], [1, 2],
            [5, 6], [5, 6], [5, 6],
            [10, 11], [10, 11], [10, 11]
        ])
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        model.train(X, y)
        
        result = model.predict([[1, 2]])
        assert result['action'] in ['buy', 'sell']


class TestDecisionTreeModelBinary:
    """Tests for binary classification (sell/buy)."""
    
    @pytest.fixture
    def model_binary(self):
        """Create a trained binary model."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.random.choice([0, 1], size=100)
        
        model = DecisionTreeModel(max_depth=5, n_classes=2, random_state=42)
        model.train(X, y)
        return model
    
    def test_train_binary(self, model_binary):
        """Test binary training."""
        assert model_binary.is_trained is True
        assert model_binary.n_classes == 2
    
    def test_predict_binary(self, model_binary):
        """Test binary predict returns valid actions."""
        X = np.random.randn(1, 4)
        result = model_binary.predict(X)
        
        assert result['action'] in ['buy', 'sell', 'hold']
        assert result['proba_hold'] == 0.0
    
    def test_binary_probabilities_sum(self, model_binary):
        """Test binary probabilities sum to 1."""
        X = np.random.randn(1, 4)
        result = model_binary.predict(X)
        
        total = result['proba_sell'] + result['proba_buy']
        assert abs(total - 1.0) < 1e-6


class TestDecisionTreeModelUntrained:
    """Tests for untrained model behavior."""
    
    def test_untrained_predict_returns_hold(self):
        """Test untrained model returns hold."""
        model = DecisionTreeModel()
        result = model.predict(np.random.randn(1, 4))
        
        assert result['action'] == 'hold'
        assert result['confidence'] == 0.0
    
    def test_untrained_predict_proba(self):
        """Test untrained model predict_proba returns uniform."""
        model = DecisionTreeModel(n_classes=3)
        proba = model.predict_proba(np.random.randn(1, 4))
        
        expected = np.array([[1/3, 1/3, 1/3]])
        np.testing.assert_array_almost_equal(proba, expected)
    
    def test_untrained_binary_predict_proba(self):
        """Test untrained binary model predict_proba returns uniform."""
        model = DecisionTreeModel(n_classes=2)
        proba = model.predict_proba(np.random.randn(1, 4))
        
        expected = np.array([[0.5, 0.5]])
        np.testing.assert_array_almost_equal(proba, expected)
    
    def test_untrained_feature_importances(self):
        """Test untrained model returns None for importances."""
        model = DecisionTreeModel()
        assert model.get_feature_importances() is None
    
    def test_untrained_tree_depth(self):
        """Test untrained model returns None for depth."""
        model = DecisionTreeModel()
        assert model.get_tree_depth() is None


class TestDecisionTreeModelInputTypes:
    """Tests for different input types."""
    
    def test_numpy_array_input(self):
        """Test numpy array input."""
        model = DecisionTreeModel(max_depth=3, random_state=42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], size=50)
        model.train(X, y)
        
        result = model.predict(X[:1])
        assert 'action' in result
    
    def test_pandas_dataframe_input(self):
        """Test pandas DataFrame input."""
        model = DecisionTreeModel(max_depth=3, random_state=42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])
        y = pd.Series(np.random.choice([0, 1, 2], size=50))
        model.train(X, y)
        
        result = model.predict(X.iloc[:1])
        assert 'action' in result
    
    def test_pandas_series_input(self):
        """Test pandas Series input (single sample)."""
        model = DecisionTreeModel(max_depth=3, random_state=42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])
        y = pd.Series(np.random.choice([0, 1, 2], size=50))
        model.train(X, y)
        
        result = model.predict(X.iloc[0])
        assert 'action' in result
    
    def test_1d_numpy_array(self):
        """Test 1D numpy array input."""
        model = DecisionTreeModel(max_depth=3, random_state=42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], size=50)
        model.train(X, y)
        
        result = model.predict(X[0])
        assert 'action' in result


class TestDecisionTreeModelEdgeCases:
    """Edge case tests."""
    
    def test_empty_input_returns_hold(self):
        """Test empty input returns hold."""
        model = DecisionTreeModel()
        result = model.predict(np.array([]).reshape(0, 4))
        
        assert result['action'] == 'hold'
    
    def test_mismatched_xy_lengths(self):
        """Test X/y length mismatch is handled."""
        model = DecisionTreeModel(max_depth=3, random_state=42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], size=30)
        
        model.train(X, y)
        assert model.is_trained
    
    def test_missing_y_raises_error(self):
        """Test missing y raises appropriate error."""
        model = DecisionTreeModel()
        
        with pytest.raises(ValueError) as exc:
            model.train(np.random.randn(10, 3), y=None)
        assert "y (target labels) is required" in str(exc.value)
    
    def test_too_many_classes_raises_error(self):
        """Test too many unique classes raises error."""
        model = DecisionTreeModel(n_classes=3)
        
        with pytest.raises(ValueError) as exc:
            model.train(np.random.randn(10, 3), y=np.array([0, 1, 2, 3, 4]))
        assert "Found 5 unique classes" in str(exc.value)
    
    def test_empty_training_data_raises_error(self):
        """Test empty training data raises error."""
        model = DecisionTreeModel()
        
        with pytest.raises(ValueError) as exc:
            model.train(np.array([]).reshape(0, 3), y=np.array([]))
        assert "No training samples" in str(exc.value)


class TestDecisionTreeModelSaveLoad:
    """Tests for save/load functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_save_state(self, temp_dir):
        """Test save state to disk."""
        model = DecisionTreeModel(max_depth=5, random_state=42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], size=50)
        model.train(X, y)
        
        path = temp_dir / "model.pkl"
        model.save_state(path)
        
        assert path.exists()
    
    def test_load_state(self, temp_dir):
        """Test load state from disk."""
        model = DecisionTreeModel(max_depth=5, random_state=42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], size=50)
        model.train(X, y)
        
        path = temp_dir / "model.pkl"
        model.save_state(path)
        
        new_model = DecisionTreeModel(max_depth=999)
        new_model.load_state(path)
        
        assert new_model.is_trained is True
        assert new_model.max_depth == 5
    
    def test_load_nonexistent_raises_error(self):
        """Test loading nonexistent file raises error."""
        model = DecisionTreeModel()
        
        with pytest.raises(FileNotFoundError):
            model.load_state("/nonexistent/path/model.pkl")
    
    def test_predict_after_load(self, temp_dir):
        """Test prediction works after loading."""
        model = DecisionTreeModel(
            max_depth=3, 
            confidence_threshold=0.4,
            random_state=42
        )
        X = np.random.randn(50, 3)
        y = np.array([0] * 15 + [1] * 20 + [2] * 15)
        model.train(X, y)
        
        path = temp_dir / "model.pkl"
        model.save_state(path)
        
        new_model = DecisionTreeModel()
        new_model.load_state(path)
        
        result = new_model.predict(X[:1])
        assert result['action'] in ['buy', 'sell', 'hold']


class TestDecisionTreeModelFeatureImportances:
    """Tests for feature importances."""
    
    def test_feature_importances_after_training(self):
        """Test feature importances are available after training."""
        model = DecisionTreeModel(max_depth=5, random_state=42)
        X = np.random.randn(50, 4)
        y = np.random.choice([0, 1, 2], size=50)
        model.train(X, y)
        
        importances = model.get_feature_importances()
        
        assert importances is not None
        assert len(importances) == 4
        assert np.all(importances >= 0)
        assert np.all(importances <= 1)
        assert abs(np.sum(importances) - 1.0) < 1e-6
    
    def test_feature_importances_untrained(self):
        """Test feature importances return None when untrained."""
        model = DecisionTreeModel()
        assert model.get_feature_importances() is None


class TestDecisionTreeModelTreeProperties:
    """Tests for tree properties."""
    
    def test_tree_depth(self):
        """Test tree depth is computed correctly."""
        model = DecisionTreeModel(max_depth=7, random_state=42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], size=50)
        model.train(X, y)
        
        depth = model.get_tree_depth()
        assert depth == 7
    
    def test_n_leaves(self):
        """Test number of leaves is computed."""
        model = DecisionTreeModel(max_depth=3, random_state=42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], size=50)
        model.train(X, y)
        
        n_leaves = model.get_n_leaves()
        assert n_leaves is not None
        assert n_leaves >= 1
    
    def test_tree_properties_untrained(self):
        """Test tree properties return None when untrained."""
        model = DecisionTreeModel()
        assert model.get_tree_depth() is None
        assert model.get_n_leaves() is None


class TestDecisionTreeModelDeterminism:
    """Tests for deterministic behavior."""
    
    def test_same_seed_same_results(self):
        """Test same random seed produces same results."""
        X = np.random.randn(100, 4)
        y = np.random.choice([0, 1, 2], size=100)
        
        model1 = DecisionTreeModel(max_depth=5, random_state=42)
        model1.train(X[:50], y[:50])
        
        model2 = DecisionTreeModel(max_depth=5, random_state=42)
        model2.train(X[:50], y[:50])
        
        result1 = model1.predict(X[50:51])
        result2 = model2.predict(X[50:51])
        
        assert result1['action'] == result2['action']
        assert result1['confidence'] == result2['confidence']


class TestDecisionTreeModelTrainingReturnsSelf:
    """Test that train returns self for chaining."""
    
    def test_train_returns_self(self):
        """Test train method returns self."""
        model = DecisionTreeModel(max_depth=3, random_state=42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], size=50)
        
        result = model.train(X, y)
        
        assert result is model


class TestDecisionTreeModelWithPandasIndex:
    """Test using DataFrame with pandas index."""
    
    def test_pandas_dataframe_with_index(self):
        """Test DataFrame with DatetimeIndex."""
        model = DecisionTreeModel(max_depth=3, random_state=42)
        
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        X = pd.DataFrame(
            np.random.randn(50, 3),
            index=dates,
            columns=['a', 'b', 'c']
        )
        y = pd.Series(np.random.choice([0, 1, 2], size=50), index=dates)
        
        model.train(X, y)
        
        result = model.predict(X.iloc[:1])
        assert 'action' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
