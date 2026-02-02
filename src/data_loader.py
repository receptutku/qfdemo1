"""
Data loading and downloading utilities.

Features:
- CCXT-based OHLCV download from Binance
- Parquet caching with deterministic filenames
- Incremental fetch (only missing candles)
- Integrity checks (monotonic, no duplicates, gap detection)
- All timestamps in UTC
- Retry with rate-limit compliant sleep
"""

import time
import ccxt
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List


# Timeframe to milliseconds mapping
TIMEFRAME_MS = {
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "8h": 8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "3d": 3 * 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
}


def get_cache_filepath(symbol: str, timeframe: str, data_dir: Path) -> Path:
    """Generate deterministic cache filepath."""
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    filename = f"{safe_symbol}_{timeframe}.parquet"
    return Path(data_dir) / filename


def load_cached_data(filepath: Path) -> Optional[pd.DataFrame]:
    """Load existing cached data if available."""
    if not filepath.exists():
        return None
    try:
        df = pd.read_parquet(filepath)
        if not df.empty:
            return df
    except Exception as e:
        print(f"[DataLoader] Warning: Could not read cache {filepath}: {e}")
    return None


def validate_dataframe(df: pd.DataFrame, timeframe: str) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV dataframe integrity.
    
    Checks:
    - Monotonic timestamps
    - No duplicates
    - Consistent interval
    - Gap detection
    
    Returns:
        (is_valid, list of issues)
    """
    issues = []
    
    if df.empty:
        return True, []
    
    # Check monotonic
    if not df.index.is_monotonic_increasing:
        issues.append("Timestamps are not monotonically increasing")
    
    # Check duplicates
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate timestamps")
    
    # Check interval consistency
    if len(df) > 1:
        expected_interval = pd.Timedelta(milliseconds=TIMEFRAME_MS.get(timeframe, 3600000))
        actual_intervals = df.index.to_series().diff().dropna()
        
        # Find gaps (intervals larger than expected)
        gaps = actual_intervals[actual_intervals > expected_interval * 1.5]
        if len(gaps) > 0:
            issues.append(f"Found {len(gaps)} gaps in data")
            # Report up to 5 gaps
            for i, (ts, gap) in enumerate(gaps.items()):
                if i >= 5:
                    issues.append(f"  ... and {len(gaps) - 5} more gaps")
                    break
                issues.append(f"  Gap at {ts}: {gap}")
    
    # Check for timezone awareness
    if df.index.tz is None:
        issues.append("Timestamps are not timezone-aware (should be UTC)")
    
    is_valid = len([i for i in issues if "Gap" not in i and "timezone" not in i.lower()]) == 0
    return is_valid, issues


def download_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = "2019-01-01",
    end_date: Optional[str] = None,
    output_dir: Path = Path("data"),
    exchange_id: str = "binance",
    max_retries: int = 3,
    rate_limit_sleep: float = 1.0,
) -> Path:
    """
    Download OHLCV data from exchange with incremental caching.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candlestick timeframe (e.g., '1h', '1d')
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD), defaults to now
        output_dir: Directory to save data
        exchange_id: Exchange to use (default: binance)
        max_retries: Maximum retry attempts per request
        rate_limit_sleep: Sleep between requests (seconds)
    
    Returns:
        Path to saved parquet file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    filepath = get_cache_filepath(symbol, timeframe, output_dir)
    
    # Initialize exchange
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        "enableRateLimit": True,
        "rateLimit": int(rate_limit_sleep * 1000),
    })
    
    # Parse dates
    start_ts = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ts = None
    if end_date:
        end_ts = exchange.parse8601(f"{end_date}T23:59:59Z")
    else:
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    # Load existing cache
    existing_df = load_cached_data(filepath)
    
    if existing_df is not None and not existing_df.empty:
        # Incremental fetch: start from last cached timestamp
        last_cached_ts = int(existing_df.index[-1].timestamp() * 1000)
        interval_ms = TIMEFRAME_MS.get(timeframe, 3600000)
        
        if last_cached_ts >= end_ts:
            print(f"[DataLoader] Cache is up to date: {filepath}")
            return filepath
        
        # Start from next candle after last cached
        fetch_start_ts = last_cached_ts + interval_ms
        print(f"[DataLoader] Incremental fetch from {datetime.fromtimestamp(fetch_start_ts/1000, tz=timezone.utc)}")
    else:
        existing_df = None
        fetch_start_ts = start_ts
    
    print(f"[DataLoader] Fetching {symbol} {timeframe} from {exchange_id}...")
    print(f"[DataLoader] Start: {datetime.fromtimestamp(fetch_start_ts/1000, tz=timezone.utc)}")
    print(f"[DataLoader] End: {datetime.fromtimestamp(end_ts/1000, tz=timezone.utc)}")
    
    # Fetch OHLCV in chunks with retry
    all_ohlcv = []
    current_since = fetch_start_ts
    
    while current_since < end_ts:
        ohlcv = None
        
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000,
                )
                break
            except ccxt.RateLimitExceeded as e:
                wait_time = rate_limit_sleep * (attempt + 2)
                print(f"[DataLoader] Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            except ccxt.NetworkError as e:
                wait_time = rate_limit_sleep * (attempt + 1)
                print(f"[DataLoader] Network error, retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"[DataLoader] Error: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(rate_limit_sleep)
        
        if not ohlcv:
            break
        
        all_ohlcv.extend(ohlcv)
        
        # Update cursor
        last_timestamp = ohlcv[-1][0]
        
        # Check if we've reached the end
        if last_timestamp >= end_ts:
            break
        
        # Break if no progress
        if last_timestamp <= current_since:
            break
        
        interval_ms = TIMEFRAME_MS.get(timeframe, 3600000)
        current_since = last_timestamp + interval_ms
        
        # Progress update every 5000 candles
        if len(all_ohlcv) % 5000 == 0:
            print(f"[DataLoader] Fetched {len(all_ohlcv)} candles...")
        
        # Rate limit compliance
        time.sleep(rate_limit_sleep)
    
    if all_ohlcv:
        # Convert to DataFrame
        new_df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        
        # Convert timestamp to UTC datetime
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
        new_df = new_df.set_index("timestamp")
        
        # Filter by end date
        if end_date:
            new_df = new_df[new_df.index <= f"{end_date}T23:59:59+00:00"]
        
        # Merge with existing data
        if existing_df is not None:
            combined_df = pd.concat([existing_df, new_df])
        else:
            combined_df = new_df
        
        # Remove duplicates and sort
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
        combined_df = combined_df.sort_index()
        
        print(f"[DataLoader] New candles fetched: {len(new_df)}")
    else:
        combined_df = existing_df if existing_df is not None else pd.DataFrame()
    
    if combined_df is not None and not combined_df.empty:
        print(f"[DataLoader] Total candles: {len(combined_df)}")
        print(f"[DataLoader] Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        
        # Validate data
        is_valid, issues = validate_dataframe(combined_df, timeframe)
        if issues:
            print("[DataLoader] Integrity check:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Save to parquet
        combined_df.to_parquet(filepath, engine="pyarrow")
        print(f"[DataLoader] Saved to: {filepath}")
    
    return filepath


def load_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    data_dir: Path = Path("data"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data from local parquet file.
    
    Args:
        symbol: Trading pair
        timeframe: Candlestick timeframe
        data_dir: Directory containing data files
        start_date: Optional filter start date
        end_date: Optional filter end date
    
    Returns:
        DataFrame with OHLCV data (UTC timestamps)
    """
    filepath = get_cache_filepath(symbol, timeframe, Path(data_dir))
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Run: python -m src.main download --symbol {symbol} --tf {timeframe} --start <date>"
        )
    
    df = pd.read_parquet(filepath)
    
    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    
    # Filter by date range
    if start_date:
        df = df[df.index >= f"{start_date}T00:00:00+00:00"]
    if end_date:
        df = df[df.index <= f"{end_date}T23:59:59+00:00"]
    
    return df


def get_data_info(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    data_dir: Path = Path("data"),
) -> dict:
    """Get information about cached data file."""
    filepath = get_cache_filepath(symbol, timeframe, Path(data_dir))
    
    if not filepath.exists():
        return {"exists": False, "filepath": str(filepath)}
    
    df = pd.read_parquet(filepath)
    is_valid, issues = validate_dataframe(df, timeframe)
    
    return {
        "exists": True,
        "filepath": str(filepath),
        "rows": len(df),
        "start": str(df.index.min()) if not df.empty else None,
        "end": str(df.index.max()) if not df.empty else None,
        "is_valid": is_valid,
        "issues": issues,
        "file_size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
    }
