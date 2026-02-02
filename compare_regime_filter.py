#!/usr/bin/env python3
"""
Compare backtests with and without daily regime filter.

Tests:
A) regime_filter_enabled=false
B) regime_filter_enabled=true (default)

Period: 2024-02-01 → 2025-12-31
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest, print_backtest_summary
from src.strategies import STRATEGIES, SMACrossATRStrategy


def main():
    print("=" * 80)
    print("REGIME FILTER COMPARISON")
    print("=" * 80)
    print("Period: 2024-02-01 to 2025-12-31")
    print("=" * 80)
    
    results = {}
    
    # A) Without regime filter
    print("\n[A] Running WITHOUT regime filter...")
    
    class NoRegimeStrategy(SMACrossATRStrategy):
        def __init__(self, **kwargs):
            kwargs['regime_filter_enabled'] = False
            kwargs['filter_enabled'] = False  # Also disable ADX for fair comparison
            kwargs['slope_filter_enabled'] = False
            super().__init__(**kwargs)
    
    STRATEGIES["sma_atr"] = NoRegimeStrategy
    
    result_a = run_backtest(
        strategy_name="sma_atr",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-02-01",
        end_date="2025-12-31",
        save_reports=False,
    )
    results["No Filters"] = result_a
    
    # B) With regime filter only
    print("\n[B] Running WITH regime filter (no ADX/slope)...")
    
    class RegimeOnlyStrategy(SMACrossATRStrategy):
        def __init__(self, **kwargs):
            kwargs['regime_filter_enabled'] = True
            kwargs['filter_enabled'] = False  # ADX off
            kwargs['slope_filter_enabled'] = False  # Slope off
            super().__init__(**kwargs)
    
    STRATEGIES["sma_atr"] = RegimeOnlyStrategy
    
    result_b = run_backtest(
        strategy_name="sma_atr",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-02-01",
        end_date="2025-12-31",
        save_reports=False,
    )
    results["Regime Only"] = result_b
    
    # C) With all filters (regime + ADX + slope)
    print("\n[C] Running WITH ALL filters (regime + ADX + slope)...")
    
    class AllFiltersStrategy(SMACrossATRStrategy):
        def __init__(self, **kwargs):
            kwargs['regime_filter_enabled'] = True
            kwargs['filter_enabled'] = True
            kwargs['slope_filter_enabled'] = True
            super().__init__(**kwargs)
    
    STRATEGIES["sma_atr"] = AllFiltersStrategy
    
    result_c = run_backtest(
        strategy_name="sma_atr",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-02-01",
        end_date="2025-12-31",
        save_reports=False,
    )
    results["All Filters"] = result_c
    
    # Restore original
    STRATEGIES["sma_atr"] = SMACrossATRStrategy
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Config':<20} | {'Return%':>10} | {'MaxDD':>10} | {'PF':>8} | {'Trades':>7} | {'Costs':>12}")
    print("-" * 80)
    
    for name, r in results.items():
        total_costs = r.total_fees_paid + r.total_slippage_cost
        total_return = (r.final_equity / 10000 - 1) * 100  # Assuming initial capital 10000
        print(f"{name:<20} | {total_return:>9.2f}% | {r.max_drawdown_pct:>9.2f}% | {r.profit_factor:>8.2f} | {r.total_trades:>7} | ${total_costs:>10.2f}")
    
    print("=" * 80)
    
    # Print detailed summary for best config
    print("\n[Best Config: All Filters]")
    print_backtest_summary(result_c)


if __name__ == "__main__":
    main()
