"""
Direct Kraken REST API provider as fallback.

Uses requests directly (no ccxt) for maximum reliability.
"""

import time
import requests
from datetime import datetime, timezone
from typing import Optional, List

import pandas as pd

from src.data.providers.base import (
    OHLCVProvider, ProviderError, RateLimitError,
    SymbolNotFoundError, NetworkError, TIMEFRAME_MS
)


# Kraken timeframe mapping (minutes)
KRAKEN_TF_MAP = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440, "1w": 10080,
}

# Kraken symbol mapping
KRAKEN_SYMBOL_MAP = {
    "BTC/USDT": "XBTUSDT",
    "BTC/USD": "XXBTZUSD",
    "ETH/USDT": "ETHUSDT",
    "ETH/USD": "XETHZUSD",
}


class KrakenDirectProvider(OHLCVProvider):
    """
    Direct Kraken REST API provider.
    
    No ccxt dependency - uses requests directly for maximum reliability.
    """
    
    name = "kraken_direct"
    BASE_URL = "https://api.kraken.com/0/public"
    
    def __init__(self, max_retries: int = 3, timeout: int = 30):
        super().__init__()
        self.max_retries = max_retries
        self.timeout = timeout
        self._asset_pairs = None
    
    def _load_asset_pairs(self):
        """Load available asset pairs from Kraken."""
        if self._asset_pairs is None:
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/AssetPairs",
                    timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                
                if data.get("error"):
                    raise ProviderError(f"Kraken API error: {data['error']}")
                
                self._asset_pairs = list(data.get("result", {}).keys())
            except requests.RequestException as e:
                raise NetworkError(f"Failed to load Kraken pairs: {e}")
    
    def get_supported_symbols(self) -> List[str]:
        """Get supported symbols in standard format."""
        self._load_asset_pairs()
        return list(KRAKEN_SYMBOL_MAP.keys())
    
    def normalize_symbol(self, symbol: str) -> Optional[str]:
        """Convert standard symbol to Kraken format."""
        # Check direct mapping
        if symbol in KRAKEN_SYMBOL_MAP:
            return KRAKEN_SYMBOL_MAP[symbol]
        
        # Try loading asset pairs and direct match
        self._load_asset_pairs()
        if symbol.replace("/", "") in self._asset_pairs:
            return symbol.replace("/", "")
        
        return None
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since_ms: Optional[int] = None,
        limit: int = 720,  # Kraken max is 720
    ) -> pd.DataFrame:
        """
        Fetch OHLCV from Kraken REST API.
        """
        # Normalize symbol
        kraken_symbol = self.normalize_symbol(symbol)
        if not kraken_symbol:
            self.health.record_failure(f"Symbol not found: {symbol}")
            raise SymbolNotFoundError(f"Symbol {symbol} not supported on Kraken")
        
        # Get timeframe
        interval = KRAKEN_TF_MAP.get(timeframe)
        if not interval:
            raise ProviderError(f"Timeframe {timeframe} not supported on Kraken")
        
        # Build request
        params = {
            "pair": kraken_symbol,
            "interval": interval,
        }
        if since_ms:
            params["since"] = since_ms // 1000  # Kraken uses seconds
        
        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/OHLC",
                    params=params,
                    timeout=self.timeout,
                )
                
                if resp.status_code == 429:
                    wait_time = (attempt + 1) * 5
                    self.health.record_failure("Rate limited")
                    print(f"[kraken] Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    last_error = RateLimitError("Kraken rate limit")
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                
                if data.get("error") and data["error"]:
                    self.health.record_failure(str(data["error"]))
                    raise ProviderError(f"Kraken error: {data['error']}")
                
                result = data.get("result", {})
                
                # Find the OHLC data (Kraken returns with dynamic key)
                ohlc_data = None
                for key, value in result.items():
                    if isinstance(value, list):
                        ohlc_data = value
                        break
                
                if not ohlc_data:
                    self.health.record_success()
                    return pd.DataFrame()
                
                # Parse OHLC: [time, open, high, low, close, vwap, volume, count]
                rows = []
                for candle in ohlc_data:
                    rows.append({
                        "timestamp": int(candle[0]) * 1000,  # Convert to ms
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[6]),
                    })
                
                df = pd.DataFrame(rows)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp")
                
                # Limit rows
                if limit and len(df) > limit:
                    df = df.tail(limit)
                
                self.health.record_success()
                return df
            
            except requests.Timeout:
                wait_time = (attempt + 1) * 2
                self.health.record_failure("Timeout")
                print(f"[kraken] Timeout, retry {attempt+1}/{self.max_retries}...")
                time.sleep(wait_time)
                last_error = NetworkError("Request timeout")
            
            except requests.RequestException as e:
                wait_time = (attempt + 1) * 2
                self.health.record_failure(str(e))
                print(f"[kraken] Error: {e}, retry {attempt+1}/{self.max_retries}...")
                time.sleep(wait_time)
                last_error = NetworkError(str(e))
        
        raise last_error or ProviderError(f"Failed after {self.max_retries} retries")
    
    def __repr__(self) -> str:
        return "KrakenDirectProvider()"
