#!/usr/bin/env python3
"""
Compare backtests with different sizing modes.

Modes:
A) all_in: Use 100% of equity per trade
B) risk_per_trade: Size based on 1% risk and ATR stop distance

Period: 2024-02-01 → 2025-12-31
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest, print_backtest_summary


def main():
    print("=" * 90)
    print("SIZING MODE COMPARISON")
    print("=" * 90)
    print("Period: 2024-02-01 to 2025-12-31")
    print("=" * 90)
    
    results = {}
    
    # A) All-in sizing (100% equity)
    print("\n[A] Running with ALL_IN sizing (100% equity)...")
    result_all_in = run_backtest(
        strategy_name="sma_atr",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-02-01",
        end_date="2025-12-31",
        sizing_mode="all_in",
        position_size_pct=1.0,
        save_reports=False,
    )
    results["all_in (100%)"] = result_all_in
    
    # B) Risk-per-trade sizing (1% risk)
    print("\n[B] Running with RISK_PER_TRADE sizing (1% risk)...")
    result_rpt = run_backtest(
        strategy_name="sma_atr",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-02-01",
        end_date="2025-12-31",
        sizing_mode="risk_per_trade",
        risk_per_trade=0.01,
        max_leverage=1.0,
        save_reports=False,
    )
    results["risk_per_trade (1%)"] = result_rpt
    
    # C) Risk-per-trade with 2% risk
    print("\n[C] Running with RISK_PER_TRADE sizing (2% risk)...")
    result_rpt2 = run_backtest(
        strategy_name="sma_atr",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-02-01",
        end_date="2025-12-31",
        sizing_mode="risk_per_trade",
        risk_per_trade=0.02,
        max_leverage=1.0,
        save_reports=False,
    )
    results["risk_per_trade (2%)"] = result_rpt2
    
    # Print comparison table
    print("\n" + "=" * 120)
    print("COMPARISON RESULTS")
    print("=" * 120)
    print(f"{'Mode':<22} | {'Return%':>10} | {'MaxDD':>10} | {'PF':>7} | {'Trades':>7} | {'AvgNotional':>12} | {'AvgRisk':>8} | {'Costs':>10} | {'Cost%':>7}")
    print("-" * 120)
    
    for name, r in results.items():
        total_return = (r.final_equity / 10000 - 1) * 100
        total_costs = r.total_fees_paid + r.total_slippage_cost
        print(f"{name:<22} | {total_return:>9.2f}% | {r.max_drawdown_pct:>9.2f}% | {r.profit_factor:>7.2f} | "
              f"{r.total_trades:>7} | ${r.avg_position_notional:>10,.0f} | {r.avg_risk_used_per_trade*100:>7.2f}% | "
              f"${total_costs:>8.2f} | {r.cost_as_pct_of_gross_pnl:>6.1f}%")
    
    print("=" * 120)
    
    # Print detailed summary for risk_per_trade
    print("\n[Detailed: risk_per_trade (1%)]")
    print_backtest_summary(result_rpt)


if __name__ == "__main__":
    main()
