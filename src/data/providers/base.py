"""
Base provider interface for OHLCV data fetching.

All providers implement this interface to allow seamless exchange switching.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import pandas as pd


# Timeframe to milliseconds
TIMEFRAME_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
}


@dataclass
class ProviderHealth:
    """Health status for a provider."""
    name: str
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    last_error: Optional[str] = None
    
    def record_success(self):
        self.last_success = datetime.now(timezone.utc)
        self.consecutive_failures = 0
        self.total_requests += 1
    
    def record_failure(self, error: str):
        self.last_failure = datetime.now(timezone.utc)
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_requests += 1
        self.last_error = error
    
    @property
    def is_healthy(self) -> bool:
        """Provider is healthy if < 3 consecutive failures."""
        return self.consecutive_failures < 3
    
    def __repr__(self) -> str:
        status = "✓" if self.is_healthy else "✗"
        return f"{status} {self.name}: {self.consecutive_failures} failures"


class OHLCVProvider(ABC):
    """Abstract base class for OHLCV data providers."""
    
    name: str = "base"
    
    def __init__(self):
        self.health = ProviderHealth(name=self.name)
    
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h')
            since_ms: Start timestamp in milliseconds (UTC)
            limit: Maximum candles to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            Index should be DatetimeIndex (UTC)
        
        Raises:
            ProviderError: On any fetch failure
        """
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols."""
        pass
    
    @abstractmethod
    def normalize_symbol(self, symbol: str) -> Optional[str]:
        """
        Normalize symbol to exchange format.
        
        Returns None if symbol not supported.
        """
        pass
    
    def fetch_latest(self, symbol: str, timeframe: str, lookback: int = 100) -> pd.DataFrame:
        """Fetch the most recent candles."""
        return self.fetch_ohlcv(symbol, timeframe, limit=lookback)
    
    def fetch_range(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for a date range with pagination.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to now
        """
        from datetime import datetime
        
        # Parse dates to milliseconds
        since_ms = int(datetime.fromisoformat(f"{start_date}T00:00:00+00:00").timestamp() * 1000)
        
        if end_date:
            until_ms = int(datetime.fromisoformat(f"{end_date}T23:59:59+00:00").timestamp() * 1000)
        else:
            until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        interval_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)
        all_data = []
        current_since = since_ms
        
        while current_since < until_ms:
            df = self.fetch_ohlcv(symbol, timeframe, since_ms=current_since, limit=1000)
            
            if df.empty:
                break
            
            all_data.append(df)
            
            # Get last timestamp and move forward
            last_ts = int(df.index[-1].timestamp() * 1000)
            
            if last_ts >= until_ms:
                break
            if last_ts <= current_since:
                break
            
            current_since = last_ts + interval_ms
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()
        
        # Filter to requested range
        if end_date:
            combined = combined[combined.index <= f"{end_date}T23:59:59+00:00"]
        
        return combined


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded."""
    pass


class SymbolNotFoundError(ProviderError):
    """Symbol not available on exchange."""
    pass


class NetworkError(ProviderError):
    """Network/connection error."""
    pass
