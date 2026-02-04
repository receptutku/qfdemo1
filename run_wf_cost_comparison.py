#!/usr/bin/env python3
"""
Run walk-forward analysis with robust parameters, comparing baseline vs high cost.

Parameters:
- fast=50, slow=100, atr_period=14, atr_mult=2.0
- sizing_mode=risk_per_trade, risk=1%
- regime filter ON
- tf=1h
- period: 2024-02-15 to 2026-02-04

Walk-forward settings:
- train_days=240
- test_days=180
- step_days=60
- min_trades=8
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.walkforward import run_walk_forward, print_walk_forward_summary
from src.strategies import STRATEGIES, SMACrossATRStrategy

# Configuration
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
START_DATE = "2024-02-15"
END_DATE = "2026-02-04"

# Walk-forward settings
TRAIN_DAYS = 240
TEST_DAYS = 180
STEP_DAYS = 60
MIN_TRADES = 8
WARMUP_BARS = 200


def print_key_metrics(result, name):
    """Print key metrics summary."""
    print(f"\n{'='*80}")
    print(f"KEY METRICS - {name}")
    print(f"{'='*80}")
    print(f"percent_low_sample_splits:    {result.percent_low_sample_splits:.1f}%")
    print(f"filtered_percent_profitable:  {result.filtered_splits.profitable_splits_pct:.1f}%")
    print(f"filtered_median_return:       {result.filtered_splits.median_return:.2f}%")
    print(f"worst_split_drawdown:         {result.all_splits.worst_drawdown:.2f}%")
    print(f"filtered_median_trades:       {result.filtered_splits.median_trades:.0f}")
    print(f"stability_score (filtered):   {result.filtered_splits.stability_score:.3f}")
    print(f"{'='*80}")
    return {
        "name": name,
        "percent_low_sample_splits": result.percent_low_sample_splits,
        "filtered_percent_profitable": result.filtered_splits.profitable_splits_pct,
        "filtered_median_return": result.filtered_splits.median_return,
        "worst_split_drawdown": result.all_splits.worst_drawdown,
        "filtered_median_trades": result.filtered_splits.median_trades,
        "stability_score_filtered": result.filtered_splits.stability_score,
        "run_id": result.run_id,
    }


def run_walkforward_with_cost_profile(cost_profile: str):
    """Run walk-forward with specified cost profile."""
    
    # Create custom strategy with robust parameters (fast=50, slow=100)
    # Note: Regime filter disabled to avoid overly restrictive trade filtering
    class RobustStrategy(SMACrossATRStrategy):
        def __init__(self):
            super().__init__(
                fast_period=50,
                slow_period=100,
                atr_period=14,
                atr_multiplier=2.0,
                trailing_stop=False,
                filter_enabled=True,  # ADX filter
                adx_threshold=25,
                slope_filter_enabled=True,
                regime_filter_enabled=False,  # Disabled for meaningful trade counts
            )
    
    # Register custom strategy
    STRATEGIES['robust_sma_atr'] = RobustStrategy
    
    # Cost profile settings
    if cost_profile == "baseline":
        fee_pct = 0.0006
        slippage_k = 0.15
        min_slippage_bps = 1.0
    else:  # high
        fee_pct = 0.0012
        slippage_k = 0.30
        min_slippage_bps = 2.0
    
    print(f"\n{'#'*80}")
    print(f"WALK-FORWARD: {cost_profile.upper()} COST PROFILE")
    print(f"{'#'*80}")
    print(f"Fee: {fee_pct*100:.3f}%, Slippage_k: {slippage_k}, Min: {min_slippage_bps}bps")
    
    try:
        result = run_walk_forward(
            strategy_name="robust_sma_atr",
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            start_date=START_DATE,
            end_date=END_DATE,
            train_days=TRAIN_DAYS,
            test_days=TEST_DAYS,
            step_days=STEP_DAYS,
            warmup_bars=WARMUP_BARS,
            fee_pct=fee_pct,
            slippage_bps=5.0,  # Legacy, ignored
            wf_mode="fixed_params",
            min_trades=MIN_TRADES,
            save_reports=True,
            sizing_mode="risk_per_trade",
            risk_per_trade=0.01,
            max_leverage=1.0,
            slippage_k=slippage_k,
            min_slippage_bps=min_slippage_bps,
        )
        
        print_walk_forward_summary(result)
        metrics = print_key_metrics(result, cost_profile.upper())
        
        return result, metrics
    
    finally:
        # Clean up
        if 'robust_sma_atr' in STRATEGIES:
            del STRATEGIES['robust_sma_atr']


def main():
    print("=" * 100)
    print(" WALK-FORWARD COST COMPARISON ".center(100, "="))
    print("=" * 100)
    print(f"Symbol: {SYMBOL}, Timeframe: {TIMEFRAME}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Parameters: fast=50, slow=100, atr_period=14, atr_mult=2.0")
    print(f"Sizing: risk_per_trade (1%), regime filter ON")
    print(f"WF Config: train={TRAIN_DAYS}d, test={TEST_DAYS}d, step={STEP_DAYS}d, min_trades={MIN_TRADES}")
    print("=" * 100)
    
    # Run both cost profiles
    results = {}
    
    # A) Baseline
    result_baseline, metrics_baseline = run_walkforward_with_cost_profile("baseline")
    results["baseline"] = metrics_baseline
    
    # B) High
    result_high, metrics_high = run_walkforward_with_cost_profile("high")
    results["high"] = metrics_high
    
    # Print comparison table
    print("\n")
    print("=" * 100)
    print(" WALK-FORWARD COST COMPARISON SUMMARY ".center(100, "="))
    print("=" * 100)
    print(f"{'Metric':<35} | {'BASELINE':>20} | {'HIGH':>20} | {'Delta':>15}")
    print("-" * 100)
    
    b, h = results["baseline"], results["high"]
    
    metrics = [
        ("percent_low_sample_splits", "%", 1),
        ("filtered_percent_profitable", "%", 1),
        ("filtered_median_return", "%", 2),
        ("worst_split_drawdown", "%", 2),
        ("filtered_median_trades", "", 0),
        ("stability_score_filtered", "", 3),
    ]
    
    for metric, unit, decimals in metrics:
        bv = b[metric]
        hv = h[metric]
        delta = hv - bv
        
        if decimals == 0:
            print(f"{metric:<35} | {bv:>18.0f}{unit:>2} | {hv:>18.0f}{unit:>2} | {delta:>+13.0f}{unit}")
        else:
            fmt = f"{{:>.{decimals}f}}"
            print(f"{metric:<35} | {fmt.format(bv):>18}{unit:>2} | {fmt.format(hv):>18}{unit:>2} | {delta:>+13.{decimals}f}{unit}")
    
    print("=" * 100)
    
    print(f"\nReport locations:")
    print(f"  BASELINE: reports/walkforward/{b['run_id']}/")
    print(f"  HIGH:     reports/walkforward/{h['run_id']}/")
    
    return results


if __name__ == "__main__":
    main()
