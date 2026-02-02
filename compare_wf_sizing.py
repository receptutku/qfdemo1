#!/usr/bin/env python3
"""
Compare Walk-Forward Analysis with different sizing modes.

Modes:
A) all_in: Use 100% of equity per trade
B) risk_per_trade: Size based on 1% risk and ATR stop distance

Period: 2024-02-01 → 2025-12-31 (approx, via splits)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.walkforward import run_walk_forward

def print_wf_summary(name, r):
    print(f"\n{name} RESULTS:")
    print("-" * 60)
    print(f"Splits: {r.mean_return_pct:.2f}% Mean Ret | {r.mean_pf:.2f} Mean PF")
    print(f"Aggregated: {r.weighted_mean_return_pct:.2f}% W.Ret | {r.weighted_mean_profit_factor:.2f} W.PF")
    print("-" * 60)

def main():
    print("=" * 90)
    print("WALK-FORWARD SIZING COMPARISON")
    print("=" * 90)
    print("Strategy: sma_atr")
    print("Period:   2024-02-01 to 2025-12-31")
    print("Splits:   Train 240d, Test 90d, Step 30d")
    print("=" * 90)
    
    # A) All-in sizing
    print("\n[A] Running Walk-Forward with ALL_IN (100% equity)...")
    wf_all_in = run_walk_forward(
        strategy_name="sma_atr",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-02-01",
        end_date="2025-12-31",
        sizing_mode="all_in",
        save_reports=False
    )
    
    # B) Risk-per-trade sizing (1%)
    print("\n[B] Running Walk-Forward with RISK_PER_TRADE (1% risk)...")
    wf_rpt = run_walk_forward(
        strategy_name="sma_atr",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-02-01",
        end_date="2025-12-31",
        sizing_mode="risk_per_trade",
        risk_per_trade=0.01,
        max_leverage=1.0,
        save_reports=False
    )
    
    # B2) Risk-per-trade sizing (2%)
    print("\n[C] Running Walk-Forward with RISK_PER_TRADE (2% risk)...")
    wf_rpt2 = run_walk_forward(
        strategy_name="sma_atr",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-02-01",
        end_date="2025-12-31",
        sizing_mode="risk_per_trade",
        risk_per_trade=0.02,
        max_leverage=1.0,
        save_reports=False
    )

    # Comparison Table
    print("\n" + "=" * 120)
    print("WALK-FORWARD COMPARISON RESULTS (Weighted Means)")
    print("=" * 120)
    print(f"{'Mode':<22} | {'W.Return%':>10} | {'W.PF':>7} | {'Splits':>7} | {'Profitable%':>12}")
    print("-" * 120)
    
    modes = [("all_in (100%)", wf_all_in), 
             ("risk_per_trade (1%)", wf_rpt),
             ("risk_per_trade (2%)", wf_rpt2)]
             
    for name, r in modes:
        # Access metrics from all_splits summary
        summary = r.all_splits
        print(f"{name:<22} | {summary.weighted_mean_return:>9.2f}% | {summary.weighted_mean_pf:>7.2f} | "
              f"{r.total_splits:>7} | {summary.profitable_splits_pct:>11.1f}%")
    print("=" * 120)
    
    # Detailed check of first split to verify sizing metrics
    if r.split_metrics:
        s0 = r.split_metrics[0]
        print(f"\n[Verification] Last Mode First Split (Risk 2%):")
        print(f"  Avg Notional:  ${s0.avg_position_notional:,.2f}")
        print(f"  Avg Risk Used: {s0.avg_risk_used_per_trade*100:.2f}%")
        print(f"  Return:        {s0.total_return_pct:.2f}%")
        print(f"  MaxDD:         {s0.max_drawdown_pct:.2f}%")
        print("-" * 60)

if __name__ == "__main__":
    main()
