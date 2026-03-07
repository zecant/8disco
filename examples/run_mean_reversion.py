"""
Mean Reversion Strategy on SPY - Execution Script

This script demonstrates end-to-end usage of tradsl for building
and backtesting a mean reversion strategy on SPY.

The strategy:
- Uses Z-Score to identify mean reversion opportunities
- Z < -2.0 -> BUY (oversold)
- Z > 2.0 -> SELL (overbought)
- Otherwise -> HOLD

Usage:
    python examples/run_mean_reversion.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# tradsl components
import tradsl
from tradsl.adapters.yfinance import YFAdapter
from tradsl.models import DecisionTreeModel
from tradsl.sizing import EqualWeightSizer
from tradsl.signals import TradingSignal, TradingAction, SignalType
from tradsl.functions import z_score, sma, rolling_std
from tradsl.backtest.engine import BacktestEngine, BacktestConfig


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'symbol': 'SPY',
    'start_date': '2018-01-01',
    'end_date': '2024-01-01',
    'window': 20,              # Z-score lookback window
    'buy_threshold': -2.0,     # Z-score threshold for buy signal
    'sell_threshold': 2.0,     # Z-score threshold for sell signal
    'train_size': 252 * 2,     # ~2 years for training (504 days)
    'max_depth': 3,            # Decision tree max depth
    'starting_cash': 100000,
    'commission': 0.001,       # 0.1% commission
}


# ============================================================================
# Load Data
# ============================================================================

def load_spy_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Load SPY data from Yahoo Finance."""
    print(f"Loading SPY data from {start_date} to {end_date}...")
    
    adapter = YFAdapter(interval='1d')
    
    df = adapter.load_historical(
        symbol='SPY',
        start=datetime.strptime(start_date, '%Y-%m-%d'),
        end=datetime.strptime(end_date, '%Y-%m-%d'),
    )
    
    print(f"  Loaded {len(df)} bars")
    return df


# ============================================================================
# Feature Engineering
# ============================================================================

def compute_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute z-score and other features."""
    print(f"Computing features (window={window})...")
    
    # Use close price
    close = df['close']
    
    # Compute components for z-score
    df['sma'] = sma(close, window)
    df['std'] = rolling_std(close, window)
    
    # Compute z-score
    df['zscore'] = z_score(close, window)
    
    # Add some additional features for the model
    df['returns'] = close.pct_change()
    df['volume'] = df['volume']
    
    # Drop NaN rows (from window calculations)
    df = df.dropna()
    
    print(f"  Features computed, {len(df)} usable bars after dropping NaN")
    return df


# ============================================================================
# Label Generation
# ============================================================================

def generate_labels(df: pd.DataFrame, buy_threshold: float, sell_threshold: float) -> pd.Series:
    """Generate trading labels from z-score."""
    print(f"Generating labels (buy<{buy_threshold}, sell>{sell_threshold})...")
    
    z = df['zscore']
    
    # Labels: 0=sell, 1=hold, 2=buy
    labels = pd.Series(np.ones(len(df), dtype=int), index=df.index)
    labels[z < buy_threshold] = 2   # buy
    labels[z > sell_threshold] = 0  # sell
    
    # Count signals
    buy_count = (labels == 2).sum()
    sell_count = (labels == 0).sum()
    hold_count = (labels == 1).sum()
    
    print(f"  Labels: {buy_count} buy, {hold_count} hold, {sell_count} sell")
    
    return labels


# ============================================================================
# Model Training
# ============================================================================

def train_model(df: pd.DataFrame, train_size: int, max_depth: int):
    """Train the decision tree model."""
    print(f"\nTraining DecisionTreeModel (max_depth={max_depth})...")
    
    # Split data
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"  Training set: {len(train_df)} bars")
    print(f"  Test set: {len(test_df)} bars")
    
    # Prepare features - use only zscore for cleaner signal
    feature_cols = ['zscore']
    X_train = train_df[feature_cols].values
    y_train = train_df['labels'].values
    
    # Train model
    model = DecisionTreeModel(
        max_depth=max_depth,
        n_classes=3,
        random_state=42
    )
    
    model.train(X_train, y_train)
    
    # Check training accuracy using predict_proba
    train_proba = model.predict_proba(X_train)
    train_preds_idx = np.argmax(train_proba, axis=1)
    train_acc = (train_preds_idx == y_train).mean()
    print(f"  Training accuracy: {train_acc:.2%}")
    
    return model, train_df, test_df


# ============================================================================
# Backtest
# ============================================================================

def run_backtest(
    model: DecisionTreeModel,
    test_df: pd.DataFrame,
    starting_cash: float,
    commission: float
):
    """Run backtest on test data."""
    print(f"\nRunning backtest...")
    
    # Initialize
    cash = starting_cash
    position = 0  # shares held
    entry_price = 0
    
    trades = []
    equity_curve = []
    
    feature_cols = ['zscore']
    
    for i, (idx, row) in enumerate(test_df.iterrows()):
        # Get features
        X = test_df[feature_cols].iloc[i:i+1].values
        
        # Get model prediction
        pred = model.predict(X)
        action_str = pred['action']  # 'sell', 'hold', or 'buy'
        action_map = {'sell': 0, 'hold': 1, 'buy': 2}
        action = action_map.get(action_str, 1)  # default to hold
        
        current_price = row['close']
        
        # Execute based on signal
        if action == 2 and position == 0:  # BUY signal, no position
            shares = cash // current_price
            if shares > 0:
                cost = shares * current_price * (1 + commission)
                if cost <= cash:
                    cash -= cost
                    position = shares
                    entry_price = current_price
                    trades.append({
                        'date': idx,
                        'action': 'BUY',
                        'shares': shares,
                        'price': current_price
                    })
        
        elif action == 0 and position > 0:  # SELL signal, have position
            proceeds = position * current_price * (1 - commission)
            cash += proceeds
            pnl = proceeds - (position * entry_price)
            trades.append({
                'date': idx,
                'action': 'SELL',
                'shares': position,
                'price': current_price,
                'pnl': pnl
            })
            position = 0
            entry_price = 0
        
        # Track equity
        equity = cash + (position * current_price)
        equity_curve.append({'date': idx, 'equity': equity})
    
    # Close any remaining position at end
    if position > 0:
        final_price = test_df.iloc[-1]['close']
        proceeds = position * final_price * (1 - commission)
        cash += proceeds
        trades.append({
            'date': test_df.index[-1],
            'action': 'SELL',
            'shares': position,
            'price': final_price,
            'pnl': proceeds - (position * entry_price)
        })
        position = 0
    
    # Calculate metrics
    final_equity = cash
    total_return = (final_equity - starting_cash) / starting_cash
    
    # Calculate max drawdown
    equity_df = pd.DataFrame(equity_curve)
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
    max_drawdown = equity_df['drawdown'].min()
    
    # Count trades
    num_trades = len([t for t in trades if t['action'] == 'BUY'])
    
    # Winning trades
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
    
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Symbol:              {CONFIG['symbol']}")
    print(f"Period:              {test_df.index[0].date()} to {test_df.index[-1].date()}")
    print(f"Starting Capital:    ${starting_cash:,.2f}")
    print(f"Final Equity:        ${final_equity:,.2f}")
    print(f"Total Return:        {total_return:.2%}")
    print(f"Max Drawdown:        {max_drawdown:.2%}")
    print(f"Number of Trades:    {num_trades}")
    print(f"Winning Trades:      {len(winning_trades)}")
    print(f"Losing Trades:       {len(losing_trades)}")
    
    if num_trades > 0:
        win_rate = len(winning_trades) / num_trades
        print(f"Win Rate:            {win_rate:.2%}")
    
    print(f"{'='*60}")
    
    return {
        'trades': trades,
        'equity_curve': equity_df,
        'final_equity': final_equity,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
    }


# ============================================================================
# DSL Config Demo (Alternative Approach)
# ============================================================================

def show_dsl_example():
    """Show how this would look using the DSL."""
    print("\n" + "="*60)
    print("DSL CONFIG EXAMPLE")
    print("="*60)
    
    # Read the DSL file
    dsl_path = Path(__file__).parent / 'mean_reversion_spy.tradsl'
    if dsl_path.exists():
        with open(dsl_path) as f:
            print(f.read())
    
    print("\nTo use this DSL config with tradsl.parse():")
    print("""
from tradsl import parse
from tradsl.models import DecisionTreeModel

config = parse(dsl_content, context={
    'DecisionTreeModel': DecisionTreeModel,
    'equal_weight': lambda signals, tradable: ...,
})

# Then use TradslInterpreter to load data, compute features,
# train models, and run backtest
""")


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*60)
    print("MEAN REVERSION STRATEGY ON SPY")
    print("Using Z-Score with DecisionTreeModel")
    print("="*60)
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    # Load data
    df = load_spy_data(CONFIG['start_date'], CONFIG['end_date'])
    
    # Compute features
    df = compute_features(df, CONFIG['window'])
    
    # Generate labels
    labels = generate_labels(
        df, 
        CONFIG['buy_threshold'], 
        CONFIG['sell_threshold']
    )
    df['labels'] = labels
    
    # Train model
    model, train_df, test_df = train_model(
        df, 
        CONFIG['train_size'], 
        CONFIG['max_depth']
    )
    
    # Run backtest
    results = run_backtest(
        model,
        test_df,
        CONFIG['starting_cash'],
        CONFIG['commission']
    )
    
    # Show DSL example
    show_dsl_example()
    
    return results


if __name__ == '__main__':
    main()
