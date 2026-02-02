"""
Robustness testing utilities for strategy validation.

Includes:
1. Parameter perturbation test - tests sensitivity to param changes
2. Cost shock test - tests sensitivity to execution cost increases
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from src.data_loader import load_ohlcv
from src.strategies import SMACrossATRStrategy
from src.backtest import run_backtest


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass
class PerturbationResult:
    """Result for a single parameter combination."""
    fast_period: int
    slow_period: int
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    total_trades: int


@dataclass
class CostShockResult:
    """Result for cost shock comparison."""
    scenario: str
    fee_pct: float
    slippage_bps: float
    total_return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    sharpe_ratio: float


def run_param_perturbation(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = "2024-02-01",
    end_date: str = "2025-12-31",
    fast_values: List[int] = None,
    slow_values: List[int] = None,
    save_reports: bool = True,
) -> Dict[str, Any]:
    """
    Run parameter perturbation test around baseline.
    
    Tests fast_period x slow_period grid and outputs Return/MaxDD table.
    
    Args:
        fast_values: List of fast_period values to test (default: [30, 40, 50])
        slow_values: List of slow_period values to test (default: [80, 100, 120])
    
    Returns:
        Dict with results grid and metadata
    """
    if fast_values is None:
        fast_values = [30, 40, 50]
    if slow_values is None:
        slow_values = [80, 100, 120]
    
    print(f"\n{'='*70}")
    print("PARAMETER PERTURBATION TEST")
    print(f"{'='*70}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Fast: {fast_values}")
    print(f"Slow: {slow_values}")
    print(f"{'='*70}\n")
    
    results = []
    
    for fast in fast_values:
        for slow in slow_values:
            if fast >= slow:
                # Skip invalid combinations
                results.append(PerturbationResult(
                    fast_period=fast,
                    slow_period=slow,
                    total_return_pct=np.nan,
                    max_drawdown_pct=np.nan,
                    sharpe_ratio=np.nan,
                    profit_factor=np.nan,
                    total_trades=0,
                ))
                continue
            
            # Run backtest with custom params
            result = run_backtest(
                strategy_name="sma_atr",
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                save_reports=False,
                # These will be ignored - we need to modify strategy
            )
            
            # Actually we need to create the strategy with custom params
            # Run a simple backtest by creating custom strategy
            from src.strategies import STRATEGIES
            
            class CustomStrategy(SMACrossATRStrategy):
                def __init__(self):
                    super().__init__(fast_period=fast, slow_period=slow)
            
            # Temporarily register
            STRATEGIES['_temp_custom'] = CustomStrategy
            
            try:
                result = run_backtest(
                    strategy_name="_temp_custom",
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    save_reports=False,
                )
                
                results.append(PerturbationResult(
                    fast_period=fast,
                    slow_period=slow,
                    total_return_pct=round((result.final_equity / 10000 - 1) * 100, 2),
                    max_drawdown_pct=round(result.max_drawdown_pct, 2),
                    sharpe_ratio=round(result.sharpe_ratio, 3),
                    profit_factor=round(result.profit_factor, 2),
                    total_trades=result.total_trades,
                ))
            finally:
                del STRATEGIES['_temp_custom']
    
    # Print grid table
    print("\n--- RETURN % GRID ---")
    print(f"{'Fast\\Slow':>10}", end="")
    for slow in slow_values:
        print(f"{slow:>10}", end="")
    print()
    
    for fast in fast_values:
        print(f"{fast:>10}", end="")
        for slow in slow_values:
            r = next((x for x in results if x.fast_period == fast and x.slow_period == slow), None)
            if r and not np.isnan(r.total_return_pct):
                print(f"{r.total_return_pct:>9.1f}%", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()
    
    print("\n--- MAX DRAWDOWN % GRID ---")
    print(f"{'Fast\\Slow':>10}", end="")
    for slow in slow_values:
        print(f"{slow:>10}", end="")
    print()
    
    for fast in fast_values:
        print(f"{fast:>10}", end="")
        for slow in slow_values:
            r = next((x for x in results if x.fast_period == fast and x.slow_period == slow), None)
            if r and not np.isnan(r.max_drawdown_pct):
                print(f"{r.max_drawdown_pct:>9.1f}%", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()
    
    # Generate run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_perturbation"
    
    output = {
        "run_id": run_id,
        "test_type": "param_perturbation",
        "symbol": symbol,
        "timeframe": timeframe,
        "period": f"{start_date} to {end_date}",
        "fast_values": fast_values,
        "slow_values": slow_values,
        "results": [asdict(r) for r in results],
    }
    
    if save_reports:
        report_dir = REPORTS_DIR / "robustness" / run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        pd.DataFrame([asdict(r) for r in results]).to_csv(
            report_dir / "param_grid.csv", index=False
        )
        
        with open(report_dir / "summary.json", "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\n[Robustness] Reports saved: {report_dir}")
    
    return output


def run_cost_shock(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = "2024-02-01",
    end_date: str = "2025-12-31",
    baseline_fee_pct: float = 0.0005,
    baseline_slippage_bps: float = 5.0,
    shock_multiplier: float = 2.0,
    save_reports: bool = True,
) -> Dict[str, Any]:
    """
    Run cost shock test - compare baseline vs shocked execution costs.
    
    Args:
        shock_multiplier: Multiply fees and slippage by this factor
    
    Returns:
        Dict with baseline and shocked results
    """
    print(f"\n{'='*70}")
    print("COST SHOCK TEST")
    print(f"{'='*70}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Baseline: fee={baseline_fee_pct*100:.3f}%, slippage={baseline_slippage_bps}bps")
    print(f"Shocked:  fee={baseline_fee_pct*shock_multiplier*100:.3f}%, slippage={baseline_slippage_bps*shock_multiplier}bps")
    print(f"{'='*70}\n")
    
    results = []
    
    # Baseline
    print("[Cost Shock] Running baseline...")
    baseline = run_backtest(
        strategy_name="sma_atr",
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        fee_pct=baseline_fee_pct,
        slippage_bps=baseline_slippage_bps,
        save_reports=False,
    )
    
    results.append(CostShockResult(
        scenario="baseline",
        fee_pct=baseline_fee_pct,
        slippage_bps=baseline_slippage_bps,
        total_return_pct=round((baseline.final_equity / 10000 - 1) * 100, 2),
        max_drawdown_pct=round(baseline.max_drawdown_pct, 2),
        profit_factor=round(baseline.profit_factor, 2),
        sharpe_ratio=round(baseline.sharpe_ratio, 3),
    ))
    
    # Shocked
    print("[Cost Shock] Running shocked (2x costs)...")
    shocked = run_backtest(
        strategy_name="sma_atr",
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        fee_pct=baseline_fee_pct * shock_multiplier,
        slippage_bps=baseline_slippage_bps * shock_multiplier,
        save_reports=False,
    )
    
    results.append(CostShockResult(
        scenario=f"shocked_{int(shock_multiplier)}x",
        fee_pct=baseline_fee_pct * shock_multiplier,
        slippage_bps=baseline_slippage_bps * shock_multiplier,
        total_return_pct=round((shocked.final_equity / 10000 - 1) * 100, 2),
        max_drawdown_pct=round(shocked.max_drawdown_pct, 2),
        profit_factor=round(shocked.profit_factor, 2),
        sharpe_ratio=round(shocked.sharpe_ratio, 3),
    ))
    
    # Print comparison
    print(f"\n{'='*60}")
    print("COST SHOCK COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Baseline':>15} {'2x Costs':>15} {'Delta':>10}")
    print(f"{'-'*60}")
    
    b, s = results[0], results[1]
    print(f"{'Return %':<20} {b.total_return_pct:>14.2f}% {s.total_return_pct:>14.2f}% {s.total_return_pct - b.total_return_pct:>+9.2f}%")
    print(f"{'Max DD %':<20} {b.max_drawdown_pct:>14.2f}% {s.max_drawdown_pct:>14.2f}% {s.max_drawdown_pct - b.max_drawdown_pct:>+9.2f}%")
    print(f"{'Profit Factor':<20} {b.profit_factor:>15.2f} {s.profit_factor:>15.2f} {s.profit_factor - b.profit_factor:>+10.2f}")
    print(f"{'Sharpe':<20} {b.sharpe_ratio:>15.3f} {s.sharpe_ratio:>15.3f} {s.sharpe_ratio - b.sharpe_ratio:>+10.3f}")
    print(f"{'='*60}")
    
    # Generate run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_costshock"
    
    output = {
        "run_id": run_id,
        "test_type": "cost_shock",
        "symbol": symbol,
        "timeframe": timeframe,
        "period": f"{start_date} to {end_date}",
        "shock_multiplier": shock_multiplier,
        "results": [asdict(r) for r in results],
    }
    
    if save_reports:
        report_dir = REPORTS_DIR / "robustness" / run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        
        pd.DataFrame([asdict(r) for r in results]).to_csv(
            report_dir / "cost_shock.csv", index=False
        )
        
        with open(report_dir / "summary.json", "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\n[Robustness] Reports saved: {report_dir}")
    
    return output


def run_all_robustness_tests(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = "2024-02-01",
    end_date: str = "2025-12-31",
) -> Dict[str, Any]:
    """Run all robustness tests and save combined report."""
    
    print("\n" + "="*70)
    print(" ROBUSTNESS TEST SUITE ".center(70, "="))
    print("="*70)
    
    # Run tests
    perturbation = run_param_perturbation(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        save_reports=False,
    )
    
    cost_shock = run_cost_shock(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        save_reports=False,
    )
    
    # Save combined report
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_robustness"
    report_dir = REPORTS_DIR / "robustness" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    
    combined = {
        "run_id": run_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "period": f"{start_date} to {end_date}",
        "param_perturbation": perturbation,
        "cost_shock": cost_shock,
    }
    
    with open(report_dir / "robustness_report.json", "w") as f:
        json.dump(combined, f, indent=2)
    
    pd.DataFrame(perturbation["results"]).to_csv(
        report_dir / "param_grid.csv", index=False
    )
    
    pd.DataFrame(cost_shock["results"]).to_csv(
        report_dir / "cost_shock.csv", index=False
    )
    
    print(f"\n[Robustness] All reports saved: {report_dir}")
    
    return combined
