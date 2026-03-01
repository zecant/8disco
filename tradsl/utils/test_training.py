import pytest
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from tradsl.utils.training import TrainingScheduler, create_scheduler_from_config


class MockModel:
    def __init__(self):
        self.trained = False
        self.X_train = None
        self.y_train = None
    
    def train(self, X, y, **params):
        self.trained = True
        self.X_train = X
        self.y_train = y
    
    def predict(self, X, **params):
        return {'output': [0.5] * len(X)}


class TestTrainingScheduler:
    def test_should_retrain_every_n_bars(self):
        scheduler = TrainingScheduler(schedule='every_n_bars', n_bars=10)
        
        assert scheduler.should_retrain(0, datetime.now()) is True
        scheduler.update_last_train(0, datetime.now())
        assert scheduler.should_retrain(5, datetime.now()) is False
        assert scheduler.should_retrain(9, datetime.now()) is False
        assert scheduler.should_retrain(10, datetime.now()) is True
        scheduler.update_last_train(10, datetime.now())
        assert scheduler.should_retrain(15, datetime.now()) is False
        assert scheduler.should_retrain(20, datetime.now()) is True
    
    def test_should_retrain_daily(self):
        scheduler = TrainingScheduler(schedule='daily')
        
        day1 = datetime(2024, 1, 1, 10, 0)
        day2 = datetime(2024, 1, 2, 10, 0)
        day3 = datetime(2024, 1, 2, 11, 0)
        
        assert scheduler.should_retrain(0, day1) is True
        scheduler.last_train_time = day1
        scheduler.last_train_bar = 10
        
        assert scheduler.should_retrain(10, day2) is True
        scheduler.last_train_time = day2
        assert scheduler.should_retrain(20, day3) is False
    
    def test_should_retrain_weekly(self):
        scheduler = TrainingScheduler(schedule='weekly')
        
        week1 = datetime(2024, 1, 1)
        week2 = datetime(2024, 1, 8)
        week3 = datetime(2024, 1, 9)
        
        assert scheduler.should_retrain(0, week1) is True
        scheduler.last_train_time = week1
        scheduler.last_train_bar = 10
        
        assert scheduler.should_retrain(10, week2) is True
        scheduler.last_train_time = week2
        assert scheduler.should_retrain(20, week3) is False
    
    def test_should_retrain_monthly(self):
        scheduler = TrainingScheduler(schedule='monthly')
        
        jan = datetime(2024, 1, 15)
        feb = datetime(2024, 2, 1)
        feb_late = datetime(2024, 2, 15)
        
        assert scheduler.should_retrain(0, jan) is True
        scheduler.last_train_time = jan
        scheduler.last_train_bar = 10
        
        assert scheduler.should_retrain(10, feb) is True
        scheduler.last_train_time = feb
        assert scheduler.should_retrain(20, feb_late) is False
    
    def test_get_training_window_rolling(self):
        scheduler = TrainingScheduler(schedule='every_n_bars', window_type='rolling', window_size=20)
        
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'feature1': np.arange(100),
            'target': np.arange(100) * 2
        }, index=dates)
        
        X, y = scheduler.get_training_window(df, 50, 'target')
        
        assert len(X) == 20
        assert len(y) == 20
        assert X.index[0] == dates[30]
        assert X.index[-1] == dates[49]
    
    def test_get_training_window_expanding(self):
        scheduler = TrainingScheduler(schedule='every_n_bars', window_type='expanding')
        
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'feature1': np.arange(100),
            'target': np.arange(100) * 2
        }, index=dates)
        
        X, y = scheduler.get_training_window(df, 50, 'target')
        
        assert len(X) == 50
        assert len(y) == 50
        assert X.index[0] == dates[0]
        assert X.index[-1] == dates[49]
    
    def test_get_training_window_at_start(self):
        scheduler = TrainingScheduler(schedule='every_n_bars', window_type='rolling', window_size=20)
        
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'feature1': np.arange(100),
            'target': np.arange(100) * 2
        }, index=dates)
        
        X, y = scheduler.get_training_window(df, 5, 'target')
        
        assert len(X) == 5
    
    def test_train_model(self):
        scheduler = TrainingScheduler()
        model = MockModel()
        
        X = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6]})
        y = pd.Series([1, 2, 3])
        
        result = scheduler.train(model, X, y)
        
        assert model.trained is True
        assert result is model
    
    def test_update_last_train(self):
        scheduler = TrainingScheduler()
        
        bar_count = 100
        current_time = datetime(2024, 1, 1)
        
        scheduler.update_last_train(bar_count, current_time)
        
        assert scheduler.last_train_bar == bar_count
        assert scheduler.last_train_time == current_time
    
    def test_save_load_model(self):
        temp_dir = tempfile.mkdtemp()
        try:
            scheduler = TrainingScheduler(save_path=Path(temp_dir))
            model = MockModel()
            
            X = pd.DataFrame({'f1': [1, 2, 3]})
            y = pd.Series([1, 2, 3])
            
            scheduler.train(model, X, y)
            
            loaded_model = scheduler.load_model(MockModel)
            
            assert loaded_model is not None
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_model_not_exists(self):
        scheduler = TrainingScheduler(save_path=Path('/nonexistent'))
        
        result = scheduler.load_model(MockModel)
        
        assert result is None
    
    def test_create_scheduler_from_config(self):
        config = {
            'signal_model': {
                'type': 'model',
                'retrain_schedule': 'weekly',
                'training_window': 'rolling',
                'training_window_size': 100,
                'load_from': './models'
            }
        }
        
        scheduler = create_scheduler_from_config(config, 'signal_model')
        
        assert scheduler is not None
        assert scheduler.schedule == 'weekly'
        assert scheduler.window_type == 'rolling'
        assert scheduler.window_size == 100
    
    def test_create_scheduler_from_config_no_training(self):
        config = {
            'signal_model': {
                'type': 'model',
                'class': MockModel,
            }
        }
        
        scheduler = create_scheduler_from_config(config, 'signal_model')
        
        assert scheduler is None
    
    def test_create_scheduler_from_config_missing_node(self):
        config = {}
        
        scheduler = create_scheduler_from_config(config, 'nonexistent')
        
        assert scheduler is None
    
    def test_unknown_schedule_type(self):
        scheduler = TrainingScheduler(schedule='unknown')
        
        result = scheduler.should_retrain(0, datetime.now())
        
        assert result is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
