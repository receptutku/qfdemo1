#!/usr/bin/env python3
"""
Run walk-forward analysis with stability score.

Configuration:
- sizing_mode: risk_per_trade
- ADX filter: enabled
- Slope filter: enabled  
- train_days: 180, test_days: 30, step_days: 30

Prints the new stability summary including:
- percent_profitable_splits
- median_net_return
- worst_split_drawdown
- median_trades
- stability_score
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.walkforward import run_walk_forward, print_walk_forward_summary
from src.strategies import SMACrossATRStrategy, STRATEGIES


def main():
    """Run walk-forward with stability score configuration."""
    
    print("=" * 70)
    print("WALK-FORWARD STABILITY SCORE ANALYSIS")
    print("=" * 70)
    print("\nConfiguration:")
    print("  - Sizing Mode:    risk_per_trade")
    print("  - ADX Filter:     ENABLED (threshold=25)")
    print("  - Slope Filter:   ENABLED (lookback=10)")
    print("  - Train Days:     180")
    print("  - Test Days:      30")
    print("  - Step Days:      30")
    print("-" * 70)
    
    # Configure strategy with filters enabled
    # Override the default strategy in the registry temporarily
    original_strategy = STRATEGIES["sma_atr"]
    
    # Create a custom strategy class with filters enabled
    class FilteredSMAATRStrategy(SMACrossATRStrategy):
        def __init__(self, **kwargs):
            # Force filters on
            kwargs.setdefault('filter_enabled', True)  # ADX filter
            kwargs.setdefault('adx_threshold', 25.0)
            kwargs.setdefault('slope_filter_enabled', True)  # Slope filter
            kwargs.setdefault('slope_lookback', 10)
            super().__init__(**kwargs)
    
    # Register the filtered strategy
    STRATEGIES["sma_atr"] = FilteredSMAATRStrategy
    
    try:
        # Run walk-forward analysis
        result = run_walk_forward(
            strategy_name="sma_atr",
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-02-02",
            end_date="2026-01-31",
            train_days=180,
            test_days=30,
            step_days=30,
            warmup_bars=200,
            fee_pct=0.0005,
            slippage_bps=5.0,
            wf_mode="fixed_params",
            save_reports=True,
        )
        
        # Print full summary (includes new stability metrics)
        print_walk_forward_summary(result)
        
        # Print additional stability details
        print("\n" + "=" * 70)
        print("STABILITY SCORE COMPONENTS")
        print("=" * 70)
        print(f"  percent_profitable_splits: {result.profitable_splits_pct:.1f}%")
        print(f"  median_net_return:         {result.median_return:.2f}%")
        print(f"  worst_split_drawdown:      {result.worst_drawdown:.2f}%")
        print(f"  median_trades:             {result.median_trades:.0f}")
        print("-" * 70)
        print(f"  STABILITY_SCORE = {result.profitable_splits_pct:.1f} * ({result.median_return:.2f} / {abs(result.worst_drawdown):.2f})")
        print(f"                  = {result.stability_score:.3f}")
        print("=" * 70)
        
        print(f"\n[Done] Reports saved: reports/walkforward/{result.run_id}")
        
    finally:
        # Restore original strategy
        STRATEGIES["sma_atr"] = original_strategy


if __name__ == "__main__":
    main()
