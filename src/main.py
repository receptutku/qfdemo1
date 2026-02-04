"""
CLI entry point for BTC Backtesting Framework.

Commands:
    update      - One-shot fetch/update OHLCV data
    live        - Continuous live data polling
    backtest    - Run single backtest
    walkforward - Run walk-forward analysis
    info        - Show cached data info
    diagnose    - Network diagnostics
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def ensure_directories():
    """Create required directories."""
    DATA_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    CONFIGS_DIR.mkdir(exist_ok=True)


def check_network_at_startup():
    """Check for proxy settings at startup."""
    from src.data.network_utils import print_proxy_warning
    print_proxy_warning()


def cmd_update(args):
    """One-shot OHLCV data update."""
    from src.data.live_fetcher import update_ohlcv
    
    fallbacks = args.fallback.split(",") if args.fallback else None
    
    filepath = update_ohlcv(
        symbol=args.symbol,
        timeframe=args.tf,
        start_date=args.start,
        exchange=args.exchange,
        fallback_exchanges=fallbacks,
        data_dir=DATA_DIR,
        allow_symbol_fallback=args.allow_symbol_fallback,
    )
    print(f"\n[Update] Done: {filepath}")


def cmd_live(args):
    """Continuous live data polling."""
    from src.data.live_fetcher import live_loop
    
    fallbacks = args.fallback.split(",") if args.fallback else None
    
    live_loop(
        symbol=args.symbol,
        timeframe=args.tf,
        exchange=args.exchange,
        fallback_exchanges=fallbacks,
        poll_seconds=args.poll_seconds,
        data_dir=DATA_DIR,
        allow_symbol_fallback=args.allow_symbol_fallback,
    )


def cmd_download(args):
    """Legacy download command - redirects to update."""
    from src.data.live_fetcher import update_ohlcv
    
    print("[Download] Note: Use 'update' command for better reliability")
    
    update_ohlcv(
        symbol=args.symbol,
        timeframe=args.tf,
        start_date=args.start,
        exchange=args.exchange if hasattr(args, 'exchange') else "okx",
        data_dir=DATA_DIR,
    )


def cmd_backtest(args):
    """Run single backtest."""
    from src.backtest import run_backtest, print_backtest_summary
    
    exchange = getattr(args, 'exchange', None)
    sizing_mode = getattr(args, 'sizing_mode', 'all_in')
    risk_per_trade = getattr(args, 'risk_per_trade', 0.01)
    max_leverage = getattr(args, 'max_leverage', 1.0)
    
    print(f"[Backtest] Strategy: {args.strategy}")
    print(f"[Backtest] Symbol: {args.symbol}, TF: {args.tf}")
    print(f"[Backtest] Period: {args.start} to {args.end}")
    print(f"[Backtest] Cost Profile: {args.cost_profile}")
    print(f"[Backtest] Sizing Mode: {sizing_mode}" + (f" (risk={risk_per_trade*100:.1f}%)" if sizing_mode == 'risk_per_trade' else ""))
    
    # If exchange is vectorbt, fetch data directly and save to cache
    if exchange == "vectorbt":
        from src.data.providers.vectorbt_provider import VectorBTProvider
        import pandas as pd
        
        provider = VectorBTProvider()
        print(f"[Backtest] Fetching data via VectorBT...")
        df = provider.fetch_range(
            symbol=args.symbol,
            timeframe=args.tf,
            start_date=args.start,
            end_date=args.end,
        )
        
        if not df.empty:
            # Save to cache for backtest engine to use
            cache_path = DATA_DIR / f"{args.symbol.replace('/', '_')}_{args.tf}.parquet"
            df.to_parquet(cache_path, engine="pyarrow")
            print(f"[Backtest] Cached {len(df)} candles to {cache_path}")
    
    result = run_backtest(
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.tf,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        cost_profile=args.cost_profile,
        sizing_mode=sizing_mode,
        risk_per_trade=risk_per_trade,
        max_leverage=max_leverage,
        save_reports=True,
    )
    
    print_backtest_summary(result)
    return result


def cmd_walkforward(args):
    """Run walk-forward analysis."""
    from src.walkforward import run_walk_forward, print_walk_forward_summary
    
    exchange = getattr(args, 'exchange', None)
    
    print(f"[WalkForward] Strategy: {args.strategy}")
    print(f"[WalkForward] Symbol: {args.symbol}, TF: {args.tf}")
    print(f"[WalkForward] Period: {args.start} to {args.end}")
    print(f"[WalkForward] Config: Train={args.train_days}d, Test={args.test_days}d, Step={args.step_days}d")
    print(f"[WalkForward] Mode: {args.wf_mode}, Min Trades: {args.min_trades}")
    
    # If exchange is vectorbt, fetch data directly and save to cache
    if exchange == "vectorbt":
        from src.data.providers.vectorbt_provider import VectorBTProvider
        
        provider = VectorBTProvider()
        print(f"[WalkForward] Fetching {args.tf} data via VectorBT...")
        df = provider.fetch_range(
            symbol=args.symbol,
            timeframe=args.tf,
            start_date=args.start,
            end_date=args.end,
        )
        
        if not df.empty:
            # Save to cache for walkforward engine to use
            cache_path = DATA_DIR / f"{args.symbol.replace('/', '_')}_{args.tf}.parquet"
            df.to_parquet(cache_path, engine="pyarrow")
            print(f"[WalkForward] Cached {len(df)} candles to {cache_path}")
        
        # Also fetch daily data for regime filter if not 1d
        if args.tf != "1d":
            print(f"[WalkForward] Fetching 1d data for regime filter...")
            daily_df = provider.fetch_range(
                symbol=args.symbol,
                timeframe="1d",
                start_date=args.start,
                end_date=args.end,
            )
            if not daily_df.empty:
                daily_cache_path = DATA_DIR / f"{args.symbol.replace('/', '_')}_1d.parquet"
                daily_df.to_parquet(daily_cache_path, engine="pyarrow")
                print(f"[WalkForward] Cached {len(daily_df)} daily candles")
    
    result = run_walk_forward(
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.tf,
        start_date=args.start,
        end_date=args.end,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        warmup_bars=args.warmup,
        fee_pct=args.fee,
        slippage_bps=args.slippage,
        wf_mode=args.wf_mode,
        min_trades=args.min_trades,
        save_reports=True,
    )
    
    print_walk_forward_summary(result)
    
    # Print condensed key metrics
    print("\n" + "=" * 80)
    print("KEY METRICS SUMMARY")
    print("=" * 80)
    print(f"percent_low_sample_splits:    {result.percent_low_sample_splits:.1f}%")
    print(f"filtered_percent_profitable:  {result.filtered_splits.profitable_splits_pct:.1f}%")
    print(f"filtered_median_return:       {result.filtered_splits.median_return:.2f}%")
    print(f"worst_split_drawdown:         {result.all_splits.worst_drawdown:.2f}%")
    print(f"filtered_median_trades:       {result.filtered_splits.median_trades:.0f}")
    print(f"stability_score (all):        {result.all_splits.stability_score:.3f}")
    print(f"stability_score (filtered):   {result.filtered_splits.stability_score:.3f}")
    print("=" * 80)
    
    print(f"\n[WalkForward] Reports: reports/walkforward/{result.run_id}/")
    return result


def cmd_info(args):
    """Show cached data info."""
    print(f"\n[Data Info] Looking for {args.symbol} {args.tf} data...")
    
    # Find all matching files
    pattern = f"*{args.symbol.replace('/', '_')}*{args.tf}*.parquet"
    files = list(DATA_DIR.glob(pattern))
    
    if not files:
        print(f"  No cached data found matching {pattern}")
        print(f"  Run: python -m src.main update --symbol {args.symbol} --tf {args.tf} --start 2019-01-01")
        return
    
    import pandas as pd
    for f in files:
        df = pd.read_parquet(f)
        print(f"\n  File: {f.name}")
        print(f"  Rows: {len(df)}")
        print(f"  Start: {df.index.min()}")
        print(f"  End: {df.index.max()}")
        print(f"  Size: {f.stat().st_size / 1024 / 1024:.2f} MB")


def cmd_audit(args):
    """Run data audit for correctness verification."""
    from src.audit import audit_data, save_audit_report, print_audit_summary
    
    exchange = getattr(args, 'exchange', 'vectorbt')
    print(f"[Audit] Symbol: {args.symbol}, TF: {args.tf}")
    print(f"[Audit] Period: {args.start} to {args.end}")
    print(f"[Audit] Exchange: {exchange}")
    
    # Determine canonical symbol and fetch data
    canonical = None
    data_source = exchange
    
    if exchange == "vectorbt":
        from src.data.providers.vectorbt_provider import VectorBTProvider
        provider = VectorBTProvider()
        canonical = provider.normalize_symbol(args.symbol)
        data_source = f"vectorbt/yfinance (via {canonical})"
        
        print(f"[Audit] Fetching data via VectorBT...")
        df = provider.fetch_range(
            symbol=args.symbol,
            timeframe=args.tf,
            start_date=args.start,
            end_date=args.end,
        )
    else:
        # Try to load from cache first
        from src.data_loader import load_ohlcv
        try:
            df = load_ohlcv(
                symbol=args.symbol,
                timeframe=args.tf,
                data_dir=DATA_DIR,
                start_date=args.start,
                end_date=args.end,
            )
        except FileNotFoundError:
            print(f"[Audit] No cached data, attempting to fetch via {exchange}...")
            from src.data.live_fetcher import update_ohlcv
            update_ohlcv(
                symbol=args.symbol,
                timeframe=args.tf,
                start_date=args.start,
                exchange=exchange,
                data_dir=DATA_DIR,
            )
            df = load_ohlcv(
                symbol=args.symbol,
                timeframe=args.tf,
                data_dir=DATA_DIR,
                start_date=args.start,
                end_date=args.end,
            )
    
    # Find data file path (for reference)
    data_file = DATA_DIR / f"{args.symbol.replace('/', '_')}_{args.tf}.parquet"
    
    # Run audit
    result = audit_data(
        symbol=args.symbol,
        timeframe=args.tf,
        df=df,
        canonical_symbol=canonical,
        data_source=data_source,
        data_file=str(data_file),
        stop_fill_mode=args.stop_fill_mode,
        fee_pct=args.fee,
        slippage_bps=args.slippage,
    )
    
    # Save report
    report_path = save_audit_report(result)
    
    # Print summary
    print_audit_summary(result)
    
    print(f"\n[Audit] Report saved: {report_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="btc-backtest",
        description="BTC-only backtesting research framework with multi-provider live data",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Update command
    up = subparsers.add_parser("update", help="One-shot fetch/update OHLCV data")
    up.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    up.add_argument("--tf", default="1h", help="Timeframe")
    up.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    up.add_argument("--exchange", default="okx", help="Primary exchange (okx, bybit, kucoin, kraken)")
    up.add_argument("--fallback", default=None, help="Fallback exchanges (comma-separated)")
    up.add_argument("--allow_symbol_fallback", action="store_true", help="Allow symbol variants")

    # Live command
    lv = subparsers.add_parser("live", help="Continuous live data polling")
    lv.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    lv.add_argument("--tf", default="1h", help="Timeframe")
    lv.add_argument("--exchange", default="okx", help="Primary exchange")
    lv.add_argument("--fallback", default=None, help="Fallback exchanges (comma-separated)")
    lv.add_argument("--poll_seconds", type=int, default=60, help="Poll interval")
    lv.add_argument("--allow_symbol_fallback", action="store_true", help="Allow symbol variants")

    # Download (legacy)
    dl = subparsers.add_parser("download", help="Download OHLCV (use 'update' instead)")
    dl.add_argument("--symbol", default="BTC/USDT")
    dl.add_argument("--tf", default="1h")
    dl.add_argument("--start", required=True)
    dl.add_argument("--end", default=None)
    dl.add_argument("--exchange", default="okx")

    # Backtest
    bt = subparsers.add_parser("backtest", help="Run backtest")
    bt.add_argument("--strategy", required=True)
    bt.add_argument("--symbol", default="BTC/USDT")
    bt.add_argument("--tf", default="1h")
    bt.add_argument("--start", required=True)
    bt.add_argument("--end", required=True)
    bt.add_argument("--capital", type=float, default=10000.0)
    bt.add_argument("--position-size", type=float, default=1.0)
    bt.add_argument("--cost_profile", default="baseline", choices=["baseline", "high"],
                    help="Cost profile: baseline (fee=0.06%%, slip_k=0.15) or high (fee=0.12%%, slip_k=0.30)")
    bt.add_argument("--exchange", default=None, help="Data provider (vectorbt, okx, etc.)")
    bt.add_argument("--config", default=None, help="Strategy config YAML file")
    bt.add_argument("--sizing_mode", default="all_in", choices=["all_in", "risk_per_trade"],
                    help="Position sizing mode")
    bt.add_argument("--risk_per_trade", type=float, default=0.01,
                    help="Risk per trade as fraction (e.g., 0.01 = 1%%)")
    bt.add_argument("--max_leverage", type=float, default=1.0, help="Max leverage (1.0 for spot)")

    # Walk-Forward
    # Presets:
    #   1h (2 years): train_days=240, test_days=180, step_days=60, min_trades=5
    #   1d (2019+):   train_days=720, test_days=365, step_days=180, min_trades=5
    wf = subparsers.add_parser("walkforward", help="Run walk-forward analysis")
    wf.add_argument("--strategy", required=True)
    wf.add_argument("--symbol", default="BTC/USDT")
    wf.add_argument("--tf", default="1h")
    wf.add_argument("--start", required=True)
    wf.add_argument("--end", required=True)
    wf.add_argument("--exchange", default=None, help="Data provider (vectorbt, okx, etc.)")
    wf.add_argument("--train_days", type=int, default=240)
    wf.add_argument("--test_days", type=int, default=90)
    wf.add_argument("--step_days", type=int, default=30)
    wf.add_argument("--warmup", type=int, default=200)
    wf.add_argument("--fee", type=float, default=0.0005)
    wf.add_argument("--slippage", type=float, default=5.0)
    wf.add_argument("--min_trades", type=int, default=10,
                    help="Minimum trades per split; below this is flagged as low_sample")
    wf.add_argument("--wf_mode", default="fixed_params", choices=["fixed_params", "train_then_test"],
                    help="Walk-forward mode: fixed_params (no tuning) or train_then_test (tune on train)")

    # Info
    info = subparsers.add_parser("info", help="Show cached data info")
    info.add_argument("--symbol", default="BTC/USDT")
    info.add_argument("--tf", default="1h")

    # Audit
    au = subparsers.add_parser("audit", help="Run data correctness audit")
    au.add_argument("--symbol", default="BTC/USDT")
    au.add_argument("--tf", default="1h")
    au.add_argument("--start", required=True)
    au.add_argument("--end", required=True)
    au.add_argument("--exchange", default="vectorbt", help="Data provider (vectorbt, okx, etc.)")
    au.add_argument("--stop_fill_mode", default="intrabar", choices=["intrabar", "next_open"])
    au.add_argument("--fee", type=float, default=0.0005)
    au.add_argument("--slippage", type=float, default=5.0)

    # Diagnose
    diag = subparsers.add_parser("diagnose", help="Network diagnostics")

    # Evaluate Holdout
    ho = subparsers.add_parser("evaluate_holdout", help="Evaluate strategy on DEV and HOLDOUT sets")
    ho.add_argument("--strategy", required=True)
    ho.add_argument("--symbol", default="BTC/USDT")
    ho.add_argument("--tf", default="1h")
    ho.add_argument("--start", default="2024-02-15", help="DEV start date")
    ho.add_argument("--dev_end", required=True, help="DEV end date")
    ho.add_argument("--holdout_start", required=True, help="HOLDOUT start date")
    ho.add_argument("--end", required=True, help="HOLDOUT end date")
    ho.add_argument("--exchange", default="vectorbt", help="Data provider")
    ho.add_argument("--cost_profile", default="baseline", choices=["baseline", "high"])

    args = parser.parse_args()
    ensure_directories()

    # Check network for data commands
    if args.command in ("update", "live", "download", "diagnose"):
        check_network_at_startup()

    commands = {
        "update": cmd_update,
        "live": cmd_live,
        "download": cmd_download,
        "backtest": cmd_backtest,
        "walkforward": cmd_walkforward,
        "info": cmd_info,
        "audit": cmd_audit,
        "diagnose": cmd_diagnose,
        "evaluate_holdout": cmd_evaluate_holdout,
    }
    
    commands[args.command](args)


def cmd_diagnose(args):
    """Run network diagnostics."""
    from src.data.network_utils import print_network_diagnostics
    print_network_diagnostics()


def cmd_evaluate_holdout(args):
    """
    Evaluate strategy on DEV and HOLDOUT sets.
    
    DEV: start to dev_end (for development/tuning - log only)
    HOLDOUT: holdout_start to end (final evaluation - run once)
    """
    import json
    import uuid
    from datetime import datetime
    from src.backtest import run_backtest, print_backtest_summary
    
    exchange = getattr(args, 'exchange', None)
    start_date = getattr(args, 'start', '2024-02-15')
    
    print("\n" + "="*100)
    print(" HOLDOUT EVALUATION PROTOCOL ".center(100, "="))
    print("="*100)
    print(f"Strategy: {args.strategy}")
    print(f"Symbol: {args.symbol}, TF: {args.tf}")
    print(f"Cost Profile: {args.cost_profile}")
    print(f"DEV Period:     {start_date} to {args.dev_end}")
    print(f"HOLDOUT Period: {args.holdout_start} to {args.end}")
    print("="*100)
    
    # Fetch data if using vectorbt
    if exchange == "vectorbt":
        from src.data.providers.vectorbt_provider import VectorBTProvider
        
        provider = VectorBTProvider()
        print(f"\n[Holdout] Fetching {args.tf} data via VectorBT...")
        df = provider.fetch_range(
            symbol=args.symbol,
            timeframe=args.tf,
            start_date=start_date,
            end_date=args.end,
        )
        
        if not df.empty:
            cache_path = DATA_DIR / f"{args.symbol.replace('/', '_')}_{args.tf}.parquet"
            df.to_parquet(cache_path, engine="pyarrow")
            print(f"[Holdout] Cached {len(df)} candles to {cache_path}")
        
        # Also fetch daily data for regime filter if not 1d
        if args.tf != "1d":
            print(f"[Holdout] Fetching 1d data for regime filter...")
            daily_df = provider.fetch_range(
                symbol=args.symbol,
                timeframe="1d",
                start_date=start_date,
                end_date=args.end,
            )
            if not daily_df.empty:
                daily_cache_path = DATA_DIR / f"{args.symbol.replace('/', '_')}_1d.parquet"
                daily_df.to_parquet(daily_cache_path, engine="pyarrow")
                print(f"[Holdout] Cached {len(daily_df)} daily candles")
    
    # Run DEV backtest
    print(f"\n{'#'*100}")
    print(f"DEV SET BACKTEST: {start_date} to {args.dev_end}")
    print(f"{'#'*100}")
    
    dev_result = run_backtest(
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.tf,
        start_date=start_date,
        end_date=args.dev_end,
        cost_profile=args.cost_profile,
        sizing_mode="risk_per_trade",
        risk_per_trade=0.01,
        save_reports=False,
    )
    
    # Run HOLDOUT backtest
    print(f"\n{'#'*100}")
    print(f"HOLDOUT SET BACKTEST: {args.holdout_start} to {args.end}")
    print(f"{'#'*100}")
    
    holdout_result = run_backtest(
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.tf,
        start_date=args.holdout_start,
        end_date=args.end,
        cost_profile=args.cost_profile,
        sizing_mode="risk_per_trade",
        risk_per_trade=0.01,
        save_reports=False,
    )
    
    # Calculate metrics
    def calc_metrics(result):
        total_return = (result.final_equity / result.initial_capital - 1) * 100
        total_costs = result.total_fees_paid + result.total_slippage_cost
        return {
            "return_pct": round(total_return, 2),
            "max_drawdown_pct": round(result.max_drawdown_pct, 2),
            "profit_factor": round(result.profit_factor, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 3),
            "trades": result.total_trades,
            "total_costs": round(total_costs, 2),
            "win_rate_pct": round(result.win_rate_pct, 1),
        }
    
    dev_metrics = calc_metrics(dev_result)
    holdout_metrics = calc_metrics(holdout_result)
    
    # Print side-by-side comparison
    print("\n")
    print("="*100)
    print(" HOLDOUT EVALUATION RESULTS ".center(100, "="))
    print("="*100)
    print(f"Cost Profile: {args.cost_profile.upper()}")
    print("-"*100)
    print(f"{'Metric':<25} | {'DEV':>20} | {'HOLDOUT':>20} | {'Delta':>15} | {'Check':>10}")
    print("-"*100)
    
    metrics_config = [
        ("return_pct", "Return", "%", 2, "higher_better"),
        ("max_drawdown_pct", "Max Drawdown", "%", 2, "lower_better"),
        ("profit_factor", "Profit Factor", "", 2, "higher_better"),
        ("sharpe_ratio", "Sharpe Ratio", "", 3, "higher_better"),
        ("trades", "Trades", "", 0, "similar"),
        ("total_costs", "Total Costs", "$", 2, "lower_better"),
        ("win_rate_pct", "Win Rate", "%", 1, "higher_better"),
    ]
    
    for key, name, unit, decimals, check_type in metrics_config:
        dv = dev_metrics[key]
        hv = holdout_metrics[key]
        delta = hv - dv
        
        # Determine check symbol
        if check_type == "higher_better":
            check = "✓" if hv >= dv * 0.7 else "⚠"  # Allow 30% degradation
        elif check_type == "lower_better":
            check = "✓" if hv <= dv * 1.3 else "⚠"  # Allow 30% worse
        else:
            check = "~"
        
        if unit == "$":
            print(f"{name:<25} | ${dv:>18.{decimals}f} | ${hv:>18.{decimals}f} | {delta:>+14.{decimals}f} | {check:>10}")
        elif unit == "%":
            print(f"{name:<25} | {dv:>18.{decimals}f}{unit} | {hv:>18.{decimals}f}{unit} | {delta:>+14.{decimals}f}{unit} | {check:>10}")
        else:
            print(f"{name:<25} | {dv:>19.{decimals}f} | {hv:>19.{decimals}f} | {delta:>+15.{decimals}f} | {check:>10}")
    
    print("="*100)
    
    # Assess holdout performance
    print("\nHOLDOUT ASSESSMENT:")
    
    issues = []
    if holdout_metrics["return_pct"] < dev_metrics["return_pct"] * 0.5:
        issues.append(f"Return degraded significantly ({holdout_metrics['return_pct']:.1f}% vs {dev_metrics['return_pct']:.1f}%)")
    if holdout_metrics["profit_factor"] < 1.0:
        issues.append(f"Profit factor below 1.0 ({holdout_metrics['profit_factor']:.2f})")
    if holdout_metrics["max_drawdown_pct"] < dev_metrics["max_drawdown_pct"] * 1.5:
        issues.append(f"Drawdown worse than expected ({holdout_metrics['max_drawdown_pct']:.1f}% vs {dev_metrics['max_drawdown_pct']:.1f}%)")
    
    if not issues:
        print("  ✓ Holdout performance is acceptable relative to DEV")
    else:
        for issue in issues:
            print(f"  ⚠ {issue}")
    
    # Save report
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    report_dir = REPORTS_DIR / "holdout" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        "run_id": run_id,
        "strategy": args.strategy,
        "symbol": args.symbol,
        "timeframe": args.tf,
        "cost_profile": args.cost_profile,
        "dev_period": f"{start_date} to {args.dev_end}",
        "holdout_period": f"{args.holdout_start} to {args.end}",
        "dev_metrics": dev_metrics,
        "holdout_metrics": holdout_metrics,
        "issues": issues,
    }
    
    with open(report_dir / "holdout_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n[Holdout] Report saved: {report_dir}/")
    
    return report_data


if __name__ == "__main__":
    main()

