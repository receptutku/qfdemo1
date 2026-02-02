"""
Live data fetcher with continuous update loop and robust fallback.

Provides one-shot update and continuous live polling modes.
Never crashes - gracefully handles failures and keeps retrying.
"""

import time
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

from src.data.providers.manager import ProviderManager
from src.data.providers.base import TIMEFRAME_MS, ProviderError
from src.data.network_utils import ExchangeHealth, run_full_diagnostics


def get_cache_filepath(
    symbol: str,
    timeframe: str,
    exchange: str,
    data_dir: Path,
) -> Path:
    """Generate deterministic cache filepath."""
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    filename = f"{safe_symbol}_{timeframe}_{exchange}.parquet"
    return data_dir / filename


def validate_dataframe(df: pd.DataFrame, timeframe: str) -> tuple:
    """
    Validate OHLCV dataframe integrity.
    
    Returns: (is_valid, issues_list, gaps_count)
    """
    issues = []
    gaps = 0
    
    if df.empty:
        return True, [], 0
    
    # Check sorted
    if not df.index.is_monotonic_increasing:
        issues.append("Not sorted")
    
    # Check duplicates
    dups = df.index.duplicated().sum()
    if dups > 0:
        issues.append(f"{dups} duplicates")
    
    # Check intervals
    if len(df) > 1:
        expected_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)
        expected_delta = pd.Timedelta(milliseconds=expected_ms)
        
        intervals = df.index.to_series().diff().dropna()
        gap_mask = intervals > expected_delta * 1.5
        gaps = gap_mask.sum()
        
        if gaps > 0:
            issues.append(f"{gaps} gaps")
    
    is_valid = not any("duplicate" in i.lower() or "sorted" in i.lower() for i in issues)
    return is_valid, issues, gaps


def safe_append_parquet(
    new_df: pd.DataFrame,
    filepath: Path,
) -> pd.DataFrame:
    """
    Safely append new data to existing parquet file.
    
    Handles duplicates and maintains sorted order.
    """
    if filepath.exists():
        existing = pd.read_parquet(filepath)
        combined = pd.concat([existing, new_df])
    else:
        combined = new_df
    
    # Remove duplicates, keep latest
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    
    # Ensure UTC
    if combined.index.tz is None:
        combined.index = combined.index.tz_localize('UTC')
    
    # Save
    combined.to_parquet(filepath, engine='pyarrow')
    
    return combined


class RobustLiveLoop:
    """
    Robust live data loop with per-exchange health tracking.
    
    Features:
    - Per-exchange health monitoring
    - TLS error cooldown (10 min per exchange)
    - Never crashes, keeps retrying
    - Status reporting
    """
    
    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        exchange: str = "okx",
        fallback_exchanges: Optional[List[str]] = None,
        poll_seconds: int = 60,
        data_dir: Path = Path("data"),
        allow_symbol_fallback: bool = False,
        cooldown_minutes: int = 10,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.primary_exchange = exchange
        self.fallback_exchanges = fallback_exchanges or ["okx", "bybit", "kucoin", "kraken_direct"]
        self.poll_seconds = poll_seconds
        self.data_dir = Path(data_dir)
        self.allow_symbol_fallback = allow_symbol_fallback
        self.cooldown_minutes = cooldown_minutes
        
        # Initialize paths
        self.data_dir.mkdir(exist_ok=True)
        self.filepath = get_cache_filepath(symbol, timeframe, exchange, self.data_dir)
        
        # Exchange health tracking
        all_exchanges = [exchange] + [e for e in self.fallback_exchanges if e != exchange]
        self.exchange_health: Dict[str, ExchangeHealth] = {
            ex: ExchangeHealth(name=ex) for ex in all_exchanges
        }
        
        # Provider manager (will be recreated on exchange switch)
        self.manager: Optional[ProviderManager] = None
        self._init_manager()
        
        # Stats
        self.iteration = 0
        self.total_failures = 0
        self.last_success_time: Optional[datetime] = None
        self.running = False
    
    def _init_manager(self):
        """Initialize provider manager with healthy exchanges only."""
        healthy_exchanges = [
            ex for ex, health in self.exchange_health.items()
            if health.is_healthy
        ]
        
        if not healthy_exchanges:
            # Reset all cooldowns if all unhealthy
            print("[Live] All exchanges unhealthy, resetting cooldowns...")
            for health in self.exchange_health.values():
                health.unhealthy_until = None
            healthy_exchanges = list(self.exchange_health.keys())
        
        self.manager = ProviderManager(
            primary_exchange=healthy_exchanges[0] if healthy_exchanges else self.primary_exchange,
            fallback_exchanges=healthy_exchanges[1:] if len(healthy_exchanges) > 1 else None,
            allow_symbol_fallback=self.allow_symbol_fallback,
        )
    
    def _record_success(self, exchange: str):
        """Record successful fetch."""
        if exchange in self.exchange_health:
            self.exchange_health[exchange].record_success()
        self.last_success_time = datetime.now(timezone.utc)
    
    def _record_failure(self, exchange: str, error: str):
        """Record failed fetch with cooldown for TLS errors."""
        self.total_failures += 1
        
        if exchange in self.exchange_health:
            self.exchange_health[exchange].record_failure(error, self.cooldown_minutes)
            
            # Reinit manager if exchange went unhealthy
            if not self.exchange_health[exchange].is_healthy:
                print(f"[Live] {exchange} marked unhealthy for {self.cooldown_minutes} min")
                self._init_manager()
    
    def print_status(self):
        """Print current exchange health status."""
        print("\n[Exchange Health]")
        for name, health in self.exchange_health.items():
            status = "✓" if health.is_healthy else "✗"
            cooldown = health.cooldown_remaining
            
            if cooldown:
                cooldown_str = f", cooldown: {cooldown//60}m{cooldown%60}s"
            else:
                cooldown_str = ""
            
            last = health.last_success.strftime("%H:%M:%S") if health.last_success else "never"
            print(f"  {status} {name}: failures={health.consecutive_failures}, last_ok={last}{cooldown_str}")
    
    def _do_fetch(self) -> Optional[pd.DataFrame]:
        """Attempt to fetch data, handling errors gracefully."""
        try:
            df = self.manager.fetch_latest(self.symbol, self.timeframe, lookback=100)
            
            if not df.empty and self.manager.current_provider:
                self._record_success(self.manager.current_provider.replace("ccxt_", ""))
            
            return df
        
        except ProviderError as e:
            error_str = str(e)
            
            # Try to identify which exchange failed
            if self.manager.current_provider:
                exchange = self.manager.current_provider.replace("ccxt_", "")
                self._record_failure(exchange, error_str)
            
            raise
    
    def run(self):
        """Run the main loop. Never crashes."""
        print(f"\n{'='*70}")
        print(f"LIVE DATA LOOP: {self.symbol} {self.timeframe}")
        print(f"{'='*70}")
        print(f"Primary: {self.primary_exchange}")
        print(f"Fallbacks: {self.fallback_exchanges}")
        print(f"Poll interval: {self.poll_seconds}s")
        print(f"Cooldown on TLS error: {self.cooldown_minutes} min")
        print(f"Cache: {self.filepath}")
        print(f"{'='*70}")
        print("\nPress Ctrl+C to stop\n")
        
        self.running = True
        
        while self.running:
            try:
                self.iteration += 1
                now = datetime.now(timezone.utc)
                
                # Fetch
                df = self._do_fetch()
                
                if df is None or df.empty:
                    print(f"[{now.strftime('%H:%M:%S')}] No data received")
                else:
                    # Safe append
                    combined = safe_append_parquet(df, self.filepath)
                    
                    # Validate
                    _, issues, gaps = validate_dataframe(combined, self.timeframe)
                    
                    # Get last candle info
                    last_candle = combined.index.max()
                    last_close = combined["close"].iloc[-1]
                    
                    # Build status line
                    provider = self.manager.current_provider or "unknown"
                    status_parts = [
                        f"Provider: {provider}",
                        f"Last: {last_candle.strftime('%Y-%m-%d %H:%M')} UTC",
                        f"Close: ${last_close:,.2f}",
                        f"Total: {len(combined)} rows",
                    ]
                    
                    if gaps > 0:
                        status_parts.append(f"Gaps: {gaps}")
                    
                    print(f"[{now.strftime('%H:%M:%S')}] {' | '.join(status_parts)}")
                
                # Wait for next poll
                self._wait_with_countdown(self.poll_seconds)
            
            except ProviderError as e:
                now = datetime.now(timezone.utc)
                print(f"[{now.strftime('%H:%M:%S')}] ⚠️ Provider error: {str(e)[:80]}...")
                self.print_status()
                print(f"\n  Retrying in {self.poll_seconds}s...")
                self._wait_with_countdown(self.poll_seconds)
            
            except KeyboardInterrupt:
                print("\n\n[Live] Stopped by user")
                self.print_status()
                self.running = False
                break
            
            except Exception as e:
                now = datetime.now(timezone.utc)
                print(f"[{now.strftime('%H:%M:%S')}] ❌ Unexpected error: {e}")
                print(f"  Continuing in {self.poll_seconds}s...")
                self._wait_with_countdown(self.poll_seconds)
    
    def _wait_with_countdown(self, seconds: int):
        """Wait with periodic status updates."""
        remaining = seconds
        while remaining > 0 and self.running:
            sleep_time = min(10, remaining)
            time.sleep(sleep_time)
            remaining -= sleep_time


def update_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = "2019-01-01",
    exchange: str = "okx",
    fallback_exchanges: Optional[List[str]] = None,
    data_dir: Path = Path("data"),
    allow_symbol_fallback: bool = False,
    save_network_profile: bool = True,
) -> Path:
    """
    One-shot update: fetch historical + recent data and cache.
    
    Returns path to cached parquet file.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    filepath = get_cache_filepath(symbol, timeframe, exchange, data_dir)
    
    print(f"[Update] Fetching {symbol} {timeframe} from {exchange}...")
    print(f"[Update] Start: {start_date}, Cache: {filepath}")
    
    # Save network profile
    if save_network_profile:
        reports_dir = data_dir.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        profile_path = reports_dir / "network_profile.json"
        try:
            profile = run_full_diagnostics(save_path=profile_path)
            if "HEALTHY" not in profile.classification:
                print(f"\n⚠️  Network issue detected: {profile.classification}")
                print(f"   {profile.recommendation[:100]}...\n")
        except Exception as e:
            print(f"[Update] Network diagnostics failed: {e}")
    
    # Initialize provider manager
    manager = ProviderManager(
        primary_exchange=exchange,
        fallback_exchanges=fallback_exchanges or ["okx", "bybit", "kucoin", "kraken_direct"],
        allow_symbol_fallback=allow_symbol_fallback,
    )
    
    # Check if we have existing data (incremental update)
    if filepath.exists():
        existing = pd.read_parquet(filepath)
        if not existing.empty:
            last_ts = existing.index.max()
            incremental_start = last_ts.strftime("%Y-%m-%d")
            print(f"[Update] Incremental from {incremental_start}")
            start_date = incremental_start
    
    # Fetch data
    df = manager.fetch_range(symbol, timeframe, start_date)
    
    if df.empty:
        print("[Update] No data fetched")
        return filepath
    
    # Validate
    is_valid, issues, gaps = validate_dataframe(df, timeframe)
    if issues:
        print(f"[Update] Integrity: {', '.join(issues)}")
    
    # Safe append
    combined = safe_append_parquet(df, filepath)
    
    print(f"[Update] Total: {len(combined)} candles")
    print(f"[Update] Range: {combined.index.min()} to {combined.index.max()}")
    print(f"[Update] Provider: {manager.current_provider}")
    print(f"[Update] Saved: {filepath}")
    
    return filepath


def live_loop(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    exchange: str = "okx",
    fallback_exchanges: Optional[List[str]] = None,
    poll_seconds: int = 60,
    data_dir: Path = Path("data"),
    allow_symbol_fallback: bool = False,
):
    """
    Continuous live polling loop.
    
    Uses RobustLiveLoop for graceful error handling.
    """
    loop = RobustLiveLoop(
        symbol=symbol,
        timeframe=timeframe,
        exchange=exchange,
        fallback_exchanges=fallback_exchanges,
        poll_seconds=poll_seconds,
        data_dir=data_dir,
        allow_symbol_fallback=allow_symbol_fallback,
    )
    loop.run()
