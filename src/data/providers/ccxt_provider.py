"""
CCXT-based OHLCV provider supporting multiple exchanges.

Uses ccxt library for exchange-agnostic access to public OHLCV endpoints.
"""

import time
from datetime import datetime, timezone
from typing import Optional, List

import ccxt
import pandas as pd

from src.data.providers.base import (
    OHLCVProvider, ProviderError, RateLimitError,
    SymbolNotFoundError, NetworkError, TIMEFRAME_MS
)


class CCXTProvider(OHLCVProvider):
    """
    CCXT-based provider for fetching OHLCV data.
    
    Supports: okx, bybit, kucoin, kraken, coinbase, binance, etc.
    """
    
    def __init__(
        self,
        exchange_id: str = "okx",
        rate_limit_ms: int = 1000,
        max_retries: int = 3,
    ):
        self.exchange_id = exchange_id.lower()
        self.rate_limit_ms = rate_limit_ms
        self.max_retries = max_retries
        self.name = f"ccxt_{self.exchange_id}"
        
        super().__init__()
        
        # Initialize exchange
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                "enableRateLimit": True,
                "rateLimit": rate_limit_ms,
                "timeout": 30000,
            })
        except AttributeError:
            raise ProviderError(f"Unknown exchange: {exchange_id}")
        
        self._markets_loaded = False
        self._supported_symbols = []
    
    def _load_markets(self):
        """Lazy load markets on first use."""
        if not self._markets_loaded:
            try:
                self.exchange.load_markets()
                self._supported_symbols = list(self.exchange.symbols)
                self._markets_loaded = True
            except Exception as e:
                raise NetworkError(f"Failed to load markets: {e}")
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        self._load_markets()
        return self._supported_symbols
    
    def normalize_symbol(self, symbol: str) -> Optional[str]:
        """
        Normalize symbol to exchange format.
        
        Handles cases like BTC/USDT vs BTCUSDT vs BTC-USDT.
        """
        self._load_markets()
        
        # Direct match
        if symbol in self._supported_symbols:
            return symbol
        
        # Try common variations
        variations = [
            symbol,
            symbol.replace("/", ""),
            symbol.replace("/", "-"),
            symbol.replace("-", "/"),
        ]
        
        for var in variations:
            if var in self._supported_symbols:
                return var
        
        return None
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with retry and error handling.
        """
        self._load_markets()
        
        # Normalize symbol
        exchange_symbol = self.normalize_symbol(symbol)
        if not exchange_symbol:
            self.health.record_failure(f"Symbol not found: {symbol}")
            raise SymbolNotFoundError(
                f"Symbol {symbol} not found on {self.exchange_id}. "
                f"Try: {self._find_similar_symbols(symbol)}"
            )
        
        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Fetch OHLCV
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=exchange_symbol,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=limit,
                )
                
                if not ohlcv:
                    self.health.record_success()
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp")
                
                # Record success
                self.health.record_success()
                
                return df
            
            except ccxt.RateLimitExceeded as e:
                wait_time = self.rate_limit_ms * (attempt + 2) / 1000
                self.health.record_failure(f"Rate limit: {e}")
                print(f"[{self.name}] Rate limited, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                last_error = RateLimitError(str(e))
            
            except ccxt.NetworkError as e:
                wait_time = (attempt + 1) * 2
                self.health.record_failure(f"Network error: {e}")
                print(f"[{self.name}] Network error, retry {attempt+1}/{self.max_retries}...")
                time.sleep(wait_time)
                last_error = NetworkError(str(e))
            
            except ccxt.ExchangeNotAvailable as e:
                wait_time = (attempt + 1) * 3
                self.health.record_failure(f"Exchange unavailable: {e}")
                print(f"[{self.name}] Exchange unavailable, retry {attempt+1}/{self.max_retries}...")
                time.sleep(wait_time)
                last_error = NetworkError(str(e))
            
            except ccxt.BadSymbol as e:
                self.health.record_failure(f"Bad symbol: {e}")
                raise SymbolNotFoundError(str(e))
            
            except Exception as e:
                self.health.record_failure(str(e))
                last_error = ProviderError(f"Unexpected error: {e}")
                time.sleep(1)
        
        raise last_error or ProviderError(f"Failed after {self.max_retries} retries")
    
    def _find_similar_symbols(self, symbol: str, top_n: int = 5) -> List[str]:
        """Find similar symbols on the exchange."""
        base = symbol.split("/")[0] if "/" in symbol else symbol[:3]
        matches = [s for s in self._supported_symbols if base in s]
        return matches[:top_n]
    
    def __repr__(self) -> str:
        return f"CCXTProvider({self.exchange_id})"
