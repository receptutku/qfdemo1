"""
VectorBT provider for OHLCV data.

Uses vectorbt library which provides flexible data fetching.
Not blocked by most ISPs and doesn't require API keys.
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List

import pandas as pd

try:
    import vectorbt as vbt
except ImportError:
    vbt = None

from .base import OHLCVProvider, ProviderError, NetworkError


class VectorBTProvider(OHLCVProvider):
    """VectorBT provider for crypto OHLCV data."""
    
    name: str = "vectorbt"
    
    # VectorBT/Yahoo Finance timeframe mapping
    TIMEFRAME_MAP = {
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "1h",  # Will aggregate from 1h
        "1d": "1d",
        "1w": "1wk",
        "1M": "1mo",
    }
    
    # Symbol mapping to Yahoo Finance format
    SYMBOL_MAP = {
        "BTC/USDT": "BTC-USD",
        "BTC/USD": "BTC-USD",
        "ETH/USDT": "ETH-USD",
        "ETH/USD": "ETH-USD",
        "SOL/USDT": "SOL-USD",
        "SOL/USD": "SOL-USD",
        "XRP/USDT": "XRP-USD",
        "DOGE/USDT": "DOGE-USD",
        "ADA/USDT": "ADA-USD",
        "AVAX/USDT": "AVAX-USD",
        "DOT/USDT": "DOT-USD",
        "LINK/USDT": "LINK-USD",
        "MATIC/USDT": "MATIC-USD",
        "LTC/USDT": "LTC-USD",
        "UNI/USDT": "UNI-USD",
        "ATOM/USDT": "ATOM-USD",
    }
    
    def __init__(self):
        super().__init__()
        if vbt is None:
            raise ImportError("vectorbt is required: pip install vectorbt")
        self.last_canonical_symbol: Optional[str] = None
    
    def normalize_symbol(self, symbol: str) -> Optional[str]:
        """Convert trading pair to Yahoo Finance format."""
        symbol = symbol.upper().replace("_", "/")
        
        if symbol in self.SYMBOL_MAP:
            return self.SYMBOL_MAP[symbol]
        
        # Try direct mapping for *-USD format
        if "-USD" in symbol:
            return symbol
        
        # Try converting X/USDT -> X-USD
        if "/USDT" in symbol:
            base = symbol.replace("/USDT", "")
            return f"{base}-USD"
        
        if "/USD" in symbol:
            base = symbol.replace("/USD", "")
            return f"{base}-USD"
        
        # Warn about unknown mapping
        print(f"[vectorbt] WARNING: Unknown symbol '{symbol}', cannot map to Yahoo format")
        return None
    
    def get_supported_symbols(self) -> List[str]:
        """Return list of commonly available crypto symbols."""
        return list(self.SYMBOL_MAP.keys())
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data using VectorBT.
        
        Note: Data availability limits:
        - 1m: Last 7 days only
        - 1h: Last 730 days (~2 years)
        - 1d: Full history
        """
        vbt_symbol = self.normalize_symbol(symbol)
        if not vbt_symbol:
            raise ProviderError(f"Symbol not supported: {symbol}")
        
        vbt_tf = self.TIMEFRAME_MAP.get(timeframe)
        if not vbt_tf:
            raise ProviderError(f"Timeframe not supported: {timeframe}")
        
        try:
            # Determine date range
            if since_ms:
                start_dt = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)
                start_str = start_dt.strftime("%Y-%m-%d")
            else:
                # Default to last 2 years for hourly data
                start_str = (datetime.now(timezone.utc) - timedelta(days=729)).strftime("%Y-%m-%d")
            
            end_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
            # Fetch data using VectorBT
            data = vbt.YFData.download(
                vbt_symbol,
                start=start_str,
                end=end_str,
                interval=vbt_tf,
            )
            
            df = data.data[vbt_symbol]
            
            # Track canonical symbol
            self.last_canonical_symbol = vbt_symbol
            print(f"[vectorbt] {symbol} → {vbt_symbol}")
            
            if df.empty:
                self.health.record_failure(f"No data returned for {vbt_symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })
            
            # Keep only OHLCV columns
            df = df[["open", "high", "low", "close", "volume"]]
            
            # Ensure UTC timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            
            # Add timestamp column (milliseconds)
            df["timestamp"] = (df.index.astype("int64") // 10**6).astype("int64")
            
            # Apply limit
            if limit and len(df) > limit:
                df = df.tail(limit)
            
            self.health.record_success()
            return df
            
        except Exception as e:
            self.health.record_failure(str(e))
            if "connection" in str(e).lower() or "network" in str(e).lower():
                raise NetworkError(f"VectorBT connection error: {e}")
            raise ProviderError(f"VectorBT error: {e}")
    
    def fetch_range(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for a date range.
        
        VectorBT handles long ranges well for daily data.
        For hourly data, we fetch in chunks.
        """
        vbt_symbol = self.normalize_symbol(symbol)
        if not vbt_symbol:
            raise ProviderError(f"Symbol not supported: {symbol}")
        
        vbt_tf = self.TIMEFRAME_MAP.get(timeframe)
        if not vbt_tf:
            raise ProviderError(f"Timeframe not supported: {timeframe}")
        
        try:
            # For hourly data, fetch in chunks
            if timeframe in ["1h", "4h"]:
                return self._fetch_hourly_range(vbt_symbol, start_date, end_date)
            
            # For daily data, fetch directly
            end = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
            print(f"[vectorbt] Downloading {vbt_symbol} {timeframe} from {start_date} to {end}...")
            
            data = vbt.YFData.download(
                vbt_symbol,
                start=start_date,
                end=end,
                interval=vbt_tf,
            )
            
            df = data.data[vbt_symbol]
            
            if df.empty:
                return pd.DataFrame()
            
            # Standardize
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]]
            
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            
            df["timestamp"] = (df.index.astype("int64") // 10**6).astype("int64")
            
            self.health.record_success()
            print(f"[vectorbt] Downloaded {len(df)} candles")
            return df
            
        except Exception as e:
            self.health.record_failure(str(e))
            raise ProviderError(f"VectorBT error: {e}")
    
    def _fetch_hourly_range(
        self,
        vbt_symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch hourly data in chunks (Yahoo limits to ~730 days per request)."""
        all_data = []
        
        start_dt = datetime.fromisoformat(f"{start_date}T00:00:00+00:00")
        end_dt = datetime.fromisoformat(f"{end_date}T23:59:59+00:00") if end_date else datetime.now(timezone.utc)
        
        chunk_days = 700  # Stay under 730 day limit
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_days), end_dt)
            
            print(f"[vectorbt] Fetching {vbt_symbol} from {current_start.date()} to {current_end.date()}...")
            
            try:
                data = vbt.YFData.download(
                    vbt_symbol,
                    start=current_start.strftime("%Y-%m-%d"),
                    end=current_end.strftime("%Y-%m-%d"),
                    interval="1h",
                )
                
                df = data.data[vbt_symbol]
                
                if not df.empty:
                    all_data.append(df)
                    print(f"[vectorbt] Got {len(df)} candles")
                
            except Exception as e:
                print(f"[vectorbt] Warning: chunk fetch failed: {e}")
            
            current_start = current_end
            time.sleep(0.5)  # Rate limiting
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()
        
        # Standardize
        combined = combined.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        combined = combined[["open", "high", "low", "close", "volume"]]
        
        if combined.index.tz is None:
            combined.index = combined.index.tz_localize("UTC")
        else:
            combined.index = combined.index.tz_convert("UTC")
        
        combined["timestamp"] = (combined.index.astype("int64") // 10**6).astype("int64")
        
        self.health.record_success()
        print(f"[vectorbt] Total: {len(combined)} candles")
        return combined
