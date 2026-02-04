#!/usr/bin/env python3
"""
Run daily backtest comparison with 3 sizing variants.

Compares:
A) all_in
B) risk_per_trade=1%
C) risk_per_trade=2%
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest, print_backtest_summary
from src.data.providers.vectorbt_provider import VectorBTProvider

# Configuration
SYMBOL = "BTC/USDT"
TIMEFRAME = "1d"
START_DATE = "2019-01-01"
END_DATE = "2025-12-31"
INITIAL_CAPITAL = 10000.0
DATA_DIR = PROJECT_ROOT / "data"


def fetch_and_cache_data():
    """Fetch daily data from VectorBT and cache it."""
    print("=" * 80)
    print("FETCHING DAILY DATA VIA VECTORBT")
    print("=" * 80)
    
    provider = VectorBTProvider()
    df = provider.fetch_range(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    
    if df.empty:
        print("ERROR: No data fetched!")
        sys.exit(1)
    
    # Save to cache
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / f"{SYMBOL.replace('/', '_')}_{TIMEFRAME}.parquet"
    df.to_parquet(cache_path, engine="pyarrow")
    print(f"Cached {len(df)} candles to {cache_path}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print()
    return df


def run_variant(name, sizing_mode, risk_per_trade=0.01):
    """Run a single backtest variant."""
    print(f"\n{'='*80}")
    print(f"RUNNING VARIANT: {name}")
    print(f"{'='*80}")
    
    result = run_backtest(
        strategy_name="sma_atr",
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        cost_profile="baseline",
        sizing_mode=sizing_mode,
        risk_per_trade=risk_per_trade,
        max_leverage=1.0,
        save_reports=True,
    )
    
    print_backtest_summary(result)
    return result


def print_comparison_table(results):
    """Print comparison table of all variants."""
    print("\n")
    print("=" * 120)
    print("DAILY BACKTEST COMPARISON: sma_atr | BTC/USDT 1d | 2019-01-01 to 2025-12-31")
    print("=" * 120)
    
    # Header
    print(f"{'Variant':<25} | {'Return %':>10} | {'MaxDD %':>10} | {'Sharpe':>8} | {'PF':>8} | {'Trades':>8} | {'Total Costs':>12} | {'Cost/Gross%':>12}")
    print("-" * 120)
    
    for name, r in results.items():
        total_return = (r.final_equity / r.initial_capital - 1) * 100
        total_costs = r.total_fees_paid + r.total_slippage_cost
        
        print(f"{name:<25} | {total_return:>10.2f} | {r.max_drawdown_pct:>10.2f} | "
              f"{r.sharpe_ratio:>8.3f} | {r.profit_factor:>8.2f} | {r.total_trades:>8} | "
              f"${total_costs:>11.2f} | {r.cost_as_pct_of_gross_pnl:>11.2f}%")
    
    print("=" * 120)
    
    # Print run IDs for reference
    print("\nReport locations:")
    for name, r in results.items():
        print(f"  {name}: reports/backtest/{r.run_id}/")


def main():
    # Step 1: Fetch and cache data
    fetch_and_cache_data()
    
    # Step 2: Run all variants
    results = {}
    
    # A) all_in
    results["A) all_in"] = run_variant(
        name="A) all_in",
        sizing_mode="all_in",
    )
    
    # B) risk_per_trade=1%
    results["B) risk_per_trade=1%"] = run_variant(
        name="B) risk_per_trade=1%",
        sizing_mode="risk_per_trade",
        risk_per_trade=0.01,
    )
    
    # C) risk_per_trade=2%
    results["C) risk_per_trade=2%"] = run_variant(
        name="C) risk_per_trade=2%",
        sizing_mode="risk_per_trade",
        risk_per_trade=0.02,
    )
    
    # Step 3: Print comparison table
    print_comparison_table(results)


if __name__ == "__main__":
    main()
