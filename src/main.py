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
    
    print(f"[Backtest] Strategy: {args.strategy}")
    print(f"[Backtest] Symbol: {args.symbol}, TF: {args.tf}")
    print(f"[Backtest] Period: {args.start} to {args.end}")
    print(f"[Backtest] Cost Profile: {args.cost_profile}")
    
    result = run_backtest(
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.tf,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        cost_profile=args.cost_profile,
        save_reports=True,
    )
    
    print_backtest_summary(result)


def cmd_walkforward(args):
    """Run walk-forward analysis."""
    from src.walkforward import run_walk_forward, print_walk_forward_summary
    
    print(f"[WalkForward] Strategy: {args.strategy}")
    print(f"[WalkForward] Period: {args.start} to {args.end}")
    print(f"[WalkForward] Config: Train={args.train_days}d, Test={args.test_days}d, Step={args.step_days}d")
    print(f"[WalkForward] Mode: {args.wf_mode}, Min Trades: {args.min_trades}")
    
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
    print(f"\n[WalkForward] Reports: reports/walkforward/{result.run_id}")


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
    from src.data_loader import load_ohlcv
    
    print(f"[Audit] Symbol: {args.symbol}, TF: {args.tf}")
    print(f"[Audit] Period: {args.start} to {args.end}")
    
    # Load data
    df = load_ohlcv(
        symbol=args.symbol,
        timeframe=args.tf,
        data_dir=DATA_DIR,
        start_date=args.start,
        end_date=args.end,
    )
    
    # Determine canonical symbol (from VectorBT mapping if available)
    canonical = None
    try:
        from src.data.providers.vectorbt_provider import VectorBTProvider
        provider = VectorBTProvider()
        canonical = provider.normalize_symbol(args.symbol)
    except:
        pass
    
    # Find data file
    data_file = DATA_DIR / f"{args.symbol.replace('/', '_')}_{args.tf}.parquet"
    
    # Run audit
    result = audit_data(
        symbol=args.symbol,
        timeframe=args.tf,
        df=df,
        canonical_symbol=canonical,
        data_source="vectorbt/yfinance",
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

    # Walk-Forward
    wf = subparsers.add_parser("walkforward", help="Run walk-forward analysis")
    wf.add_argument("--strategy", required=True)
    wf.add_argument("--symbol", default="BTC/USDT")
    wf.add_argument("--tf", default="1h")
    wf.add_argument("--start", required=True)
    wf.add_argument("--end", required=True)
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
    au.add_argument("--stop_fill_mode", default="intrabar", choices=["intrabar", "next_open"])
    au.add_argument("--fee", type=float, default=0.0005)
    au.add_argument("--slippage", type=float, default=5.0)

    # Diagnose
    diag = subparsers.add_parser("diagnose", help="Network diagnostics")

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
    }
    
    commands[args.command](args)


def cmd_diagnose(args):
    """Run network diagnostics."""
    from src.data.network_utils import print_network_diagnostics
    print_network_diagnostics()


if __name__ == "__main__":
    main()

