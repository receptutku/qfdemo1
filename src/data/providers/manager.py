"""
Provider manager with automatic fallback support.

Manages multiple OHLCV providers and handles failover between them.
"""

import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import pandas as pd

from src.data.providers.base import (
    OHLCVProvider, ProviderHealth, ProviderError,
    NetworkError, SymbolNotFoundError, TIMEFRAME_MS
)
from src.data.providers.ccxt_provider import CCXTProvider
from src.data.providers.kraken_provider import KrakenDirectProvider
from src.data.providers.vectorbt_provider import VectorBTProvider


# Default provider priority
DEFAULT_EXCHANGES = ["okx", "bybit", "kucoin", "kraken", "coinbase"]


class ProviderManager:
    """
    Manages multiple OHLCV providers with automatic fallback.
    
    When one provider fails, automatically tries the next one in the list.
    Tracks health status of each provider.
    """
    
    def __init__(
        self,
        primary_exchange: str = "okx",
        fallback_exchanges: Optional[List[str]] = None,
        allow_symbol_fallback: bool = False,
    ):
        self.primary_exchange = primary_exchange
        self.fallback_exchanges = fallback_exchanges or DEFAULT_EXCHANGES
        self.allow_symbol_fallback = allow_symbol_fallback
        
        # Ensure primary is first
        all_exchanges = [primary_exchange] + [
            e for e in self.fallback_exchanges if e != primary_exchange
        ]
        
        # Initialize providers
        self.providers: Dict[str, OHLCVProvider] = {}
        for exchange in all_exchanges:
            try:
                if exchange == "kraken_direct":
                    self.providers[exchange] = KrakenDirectProvider()
                elif exchange == "vectorbt":
                    self.providers[exchange] = VectorBTProvider()
                else:
                    self.providers[exchange] = CCXTProvider(exchange)
            except Exception as e:
                print(f"[ProviderManager] Failed to init {exchange}: {e}")
        
        # Add VectorBT as ultimate fallback (rarely blocked)
        if "vectorbt" not in self.providers:
            try:
                self.providers["vectorbt"] = VectorBTProvider()
            except Exception:
                pass
        
        # Add Kraken direct as additional fallback
        if "kraken_direct" not in self.providers:
            try:
                self.providers["kraken_direct"] = KrakenDirectProvider()
            except Exception:
                pass
        
        self._current_provider: Optional[str] = None
        self._symbol_cache: Dict[str, Dict[str, str]] = {}  # exchange -> {symbol: normalized}
    
    @property
    def current_provider(self) -> Optional[str]:
        """Name of currently active provider."""
        return self._current_provider
    
    def get_health_status(self) -> Dict[str, ProviderHealth]:
        """Get health status of all providers."""
        return {name: p.health for name, p in self.providers.items()}
    
    def print_health_status(self):
        """Print health status to console."""
        print("\n[Provider Health Status]")
        for name, provider in self.providers.items():
            h = provider.health
            status = "✓" if h.is_healthy else "✗"
            last = h.last_success.strftime("%H:%M:%S") if h.last_success else "never"
            print(f"  {status} {name}: failures={h.consecutive_failures}, last_success={last}")
    
    def _get_symbol_mapping(self, symbol: str, provider: OHLCVProvider) -> Optional[str]:
        """Get normalized symbol for provider, with optional fallback."""
        # Check cache
        cache_key = f"{provider.name}"
        if cache_key not in self._symbol_cache:
            self._symbol_cache[cache_key] = {}
        
        if symbol in self._symbol_cache[cache_key]:
            return self._symbol_cache[cache_key][symbol]
        
        # Try direct normalization
        normalized = provider.normalize_symbol(symbol)
        if normalized:
            self._symbol_cache[cache_key][symbol] = normalized
            return normalized
        
        # Try fallback variants if enabled
        if self.allow_symbol_fallback:
            variants = self._get_symbol_variants(symbol)
            for variant in variants:
                normalized = provider.normalize_symbol(variant)
                if normalized:
                    print(f"[ProviderManager] Using {variant} instead of {symbol} on {provider.name}")
                    self._symbol_cache[cache_key][symbol] = normalized
                    return normalized
        
        return None
    
    def _get_symbol_variants(self, symbol: str) -> List[str]:
        """Generate symbol variants for fallback matching."""
        variants = []
        
        # BTC/USDT <-> BTC/USD
        if "/USDT" in symbol:
            variants.append(symbol.replace("/USDT", "/USD"))
        elif "/USD" in symbol and "/USDT" not in symbol:
            variants.append(symbol.replace("/USD", "/USDT"))
        
        # ETH/USDT <-> ETH/USD
        # (same pattern)
        
        return variants
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV with automatic provider fallback.
        
        Tries each provider in order until one succeeds.
        """
        errors = []
        
        for name, provider in self.providers.items():
            # Skip unhealthy providers
            if not provider.health.is_healthy:
                continue
            
            try:
                # Check symbol availability
                normalized = self._get_symbol_mapping(symbol, provider)
                if not normalized:
                    errors.append((name, f"Symbol {symbol} not available"))
                    continue
                
                # Fetch data
                df = provider.fetch_ohlcv(symbol, timeframe, since_ms, limit)
                
                self._current_provider = name
                return df
            
            except SymbolNotFoundError as e:
                errors.append((name, str(e)))
                continue
            
            except (NetworkError, ProviderError) as e:
                errors.append((name, str(e)))
                print(f"[ProviderManager] {name} failed: {e}")
                continue
        
        # All providers failed
        error_summary = "; ".join([f"{n}: {e}" for n, e in errors])
        raise ProviderError(f"All providers failed: {error_summary}")
    
    def fetch_range(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for a date range with pagination and fallback.
        """
        from datetime import datetime
        
        since_ms = int(datetime.fromisoformat(f"{start_date}T00:00:00+00:00").timestamp() * 1000)
        
        if end_date:
            until_ms = int(datetime.fromisoformat(f"{end_date}T23:59:59+00:00").timestamp() * 1000)
        else:
            until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        interval_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)
        all_data = []
        current_since = since_ms
        
        print(f"[ProviderManager] Fetching {symbol} {timeframe} from {start_date}...")
        
        while current_since < until_ms:
            df = self.fetch_ohlcv(symbol, timeframe, since_ms=current_since, limit=1000)
            
            if df.empty:
                break
            
            all_data.append(df)
            
            last_ts = int(df.index[-1].timestamp() * 1000)
            
            if last_ts >= until_ms or last_ts <= current_since:
                break
            
            current_since = last_ts + interval_ms
            
            if len(all_data) % 5 == 0:
                total = sum(len(d) for d in all_data)
                print(f"  {total} candles fetched (via {self._current_provider})...")
            
            time.sleep(0.5)  # Rate limit between requests
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()
        
        if end_date:
            combined = combined[combined.index <= f"{end_date}T23:59:59+00:00"]
        
        return combined
    
    def fetch_latest(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback: int = 100,
    ) -> pd.DataFrame:
        """Fetch the most recent candles."""
        return self.fetch_ohlcv(symbol, timeframe, limit=lookback)
