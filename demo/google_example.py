"""
GOOGL Trading Strategy Demo using TradSL

This demo creates a strategy that trades GOOGL using:
- SPY, VIX, GOOGL data from YFinance adapter
- Raw OHLCV data from each symbol
- Returns computed for each symbol
- Rolling correlations between returns of 3 pairs (SPY-GOOGL, SPY-VIX, GOOGL-VIX)
- RandomForest model trained on all features
- Fractional position sizing (5% per trade)

The DSL is fully self-contained - no Python data generation required.
"""
from tradsl import (
    run,
    parse_dsl,
    validate_config,
    resolve_config,
    build_execution_dag,
    load,
)


DSL = """
# ===== ADAPTER =====
yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter
interval=1d

# ===== DATA SOURCES (OHLCV) =====
spy:
type=timeseries
adapter=yf
parameters=[SPY]
tradable=true

vix:
type=timeseries
adapter=yf
parameters=[^VIX]

googl:
type=timeseries
adapter=yf
parameters=[GOOGL]
tradable=true

# ===== RETURNS =====
spy_returns:
type=timeseries
function=log_returns
inputs=[spy]

vix_returns:
type=timeseries
function=log_returns
inputs=[vix]

googl_returns:
type=timeseries
function=log_returns
inputs=[googl]

# ===== CORRELATIONS (between returns of 3 pairs) =====
spy_googl_corr:
type=timeseries
function=rolling_correlation
inputs=[spy_returns, googl_returns]
params=corr_20

spy_vix_corr:
type=timeseries
function=rolling_correlation
inputs=[spy_returns, vix_returns]
params=corr_20

googl_vix_corr:
type=timeseries
function=rolling_correlation
inputs=[googl_returns, vix_returns]
params=corr_20

corr_20:
period=20

# ===== MODEL =====
# Inputs: all OHLCV + all returns + all correlations
model:
type=trainable_model
class=tradsl.models.RandomForestModel
inputs=[spy, vix, googl, spy_returns, vix_returns, googl_returns, spy_googl_corr, spy_vix_corr, googl_vix_corr]
label_function=forward_return
training_window=50
retrain_n=10
update_schedule=every_n_bars
n_estimators=50
max_depth=5
dotraining=true

# ===== AGENT =====
agent:
type=agent
inputs=[spy, vix, googl]
tradable=[googl]
sizer=fractional
sizer_params=fractional_cfg
update_schedule=every_n_bars
update_n=10

fractional_cfg:
fraction=0.05

# ===== BACKTEST =====
_backtest:
type=backtest
start=2024-01-01
end=2025-12-31
test_start=2024-07-01
capital=100000
commission=0.001
training_window=150
n_training_blocks=3
block_size_min=20
block_size_max=30
retrain_n=10
"""


def run_demo():
    """Run the GOOGL trading strategy demo using TradSL."""
    print("=" * 60)
    print("GOOGL Trading Strategy Demo - TradSL")
    print("=" * 60)
    print()
    
    print("Strategy: GOOGL based on SPY, VIX, OHLCV + correlations")
    print("Period: 2024-01-01 to 2025-12-31")
    print("Training: 2024-01-01 to 2024-06-30")
    print("Testing: 2024-07-01 to 2025-12-31")
    print()
    
    print("Step 1: Parsing DSL...")
    raw = parse_dsl(DSL)
    print(f"   Parsed {len(raw)} configuration blocks")
    
    print("\nStep 2: Validating configuration...")
    validated = validate_config(raw)
    print("   Configuration is valid")
    
    print("\nStep 3: Resolving references...")
    resolved = resolve_config(validated)
    print("   References resolved")
    
    print("\nStep 4: Building execution DAG...")
    dag = build_execution_dag(resolved)
    print(f"   DAG built with {dag.metadata.warmup_bars} warmup bars")
    print(f"   Execution order: {dag.metadata.execution_order}")
    
    print("\nStep 5: Loading data and running backtest...")
    print("   (This will fetch real data from Yahoo Finance)")
    print("   (This may take a few minutes)")
    print()
    
    # Run the full pipeline - this will load data from YFinance
    result = run(DSL)
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    metrics = result.test_result.metrics
    print(f"\nTotal Return: {metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    print(f"Total Trades: {metrics.n_trades}")
    print(f"\nFinal Portfolio Value: ${result.test_result.equity_curve[-1]:,.2f}")
    
    print("\n" + "=" * 60)
    print("TRADE LOG (last 10 trades)")
    print("=" * 60)
    trades = result.test_result.trades
    if trades:
        for trade in trades[-10:]:
            print(f"  Bar {trade.get('timestamp', 'N/A')}: {trade['side'].upper()} {trade['quantity']:.2f} @ ${trade['price']:.2f} (pnl: ${trade.get('pnl', 0):.2f})")
    else:
        print("  No trades executed")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
