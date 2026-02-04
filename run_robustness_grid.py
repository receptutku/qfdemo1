#!/usr/bin/env python3
"""
Run extended grid search for robustness testing.

Grid: fast x slow (5x5 = 25 combinations)
- fast: [20, 30, 40, 50, 60]
- slow: [60, 80, 100, 120, 140]

Runs with both baseline and high cost profiles.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.providers.vectorbt_provider import VectorBTProvider
from src.robustness import run_extended_grid_search

# Configuration
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
# Yahoo 1h limit is 730 days, so use 2024-02-15 to 2026-02-04
START_DATE = "2024-02-15"
END_DATE = "2026-02-04"
DATA_DIR = PROJECT_ROOT / "data"


def fetch_and_cache_data():
    """Fetch 1h data from VectorBT and cache it."""
    print("=" * 80)
    print("FETCHING 1H DATA VIA VECTORBT")
    print("=" * 80)
    
    provider = VectorBTProvider()
    df = provider.fetch_range(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    
    if df.empty:
        print("ERROR: No data fetched!")
        sys.exit(1)
    
    # Save to cache
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / f"{SYMBOL.replace('/', '_')}_{TIMEFRAME}.parquet"
    df.to_parquet(cache_path, engine="pyarrow")
    print(f"Cached {len(df)} candles to {cache_path}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Also fetch daily data for regime filter
    print("\nFetching daily data for regime filter...")
    daily_df = provider.fetch_range(
        symbol=SYMBOL,
        timeframe="1d",
        start_date=START_DATE,
        end_date=END_DATE,
    )
    if not daily_df.empty:
        daily_cache_path = DATA_DIR / f"{SYMBOL.replace('/', '_')}_1d.parquet"
        daily_df.to_parquet(daily_cache_path, engine="pyarrow")
        print(f"Cached {len(daily_df)} daily candles")
    
    print()
    return df


def main():
    # Step 1: Fetch and cache data
    fetch_and_cache_data()
    
    # Step 2: Run extended grid search
    result = run_extended_grid_search(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE,
        fast_values=[20, 30, 40, 50, 60],
        slow_values=[60, 80, 100, 120, 140],
        sizing_mode="risk_per_trade",
        risk_per_trade=0.01,
    )
    
    return result


if __name__ == "__main__":
    main()
