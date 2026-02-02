"""
Strategy definitions.

All strategies are defined in this single file.
Currently implements only sma_atr for BTC TRAIN phase.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple


@dataclass
class StrategySignal:
    """
    Strategy signal with stop-loss levels.
    
    Attributes:
        position: 1 = long, 0 = flat
        stop_price: Stop-loss price (None if no position)
        atr_at_entry: ATR value at entry (for trailing stop calculation)
    """
    position: int
    stop_price: Optional[float] = None
    atr_at_entry: Optional[float] = None


class BaseStrategy(ABC):
    """Abstract base class for all strategies."""
    
    name: str = "base"
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from OHLCV data.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
        
        Returns:
            DataFrame with columns:
                - signal: 1 (long), 0 (flat)
                - stop_price: Stop-loss price level
                - atr: Current ATR value
        """
        pass


class SMACrossATRStrategy(BaseStrategy):
    """
    SMA Crossover with ATR-based Stop-Loss.
    
    Entry: SMA_fast crosses above SMA_slow -> enter long next bar open
    Exit: SMA_fast crosses below SMA_slow -> exit next bar open
          OR stop-loss hit (intrabar)
    
    Stop-loss: entry_price - k * ATR_at_entry
    Trailing (optional): highest_close_since_entry - k * ATR_current
    
    All indicators use shifted data to prevent look-ahead.
    """
    
    name = "sma_atr"
    
    def __init__(
        self,
        fast_period: int = 40,
        slow_period: int = 100,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        trailing_stop: bool = False,
        filter_enabled: bool = True,  # ADX filter param
        adx_threshold: float = 25.0,
        adx_period: int = 14,
        slope_filter_enabled: bool = True,  # Slope filter param
        slope_lookback: int = 10,
        # Daily regime filter params
        regime_filter_enabled: bool = True,
        regime_sma_period: int = 200,
        regime_slope_enabled: bool = False,
        regime_slope_lookback_days: int = 5,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.trailing_stop = trailing_stop
        self.filter_enabled = filter_enabled
        self.adx_threshold = adx_threshold
        self.adx_period = adx_period
        self.slope_filter_enabled = slope_filter_enabled
        self.slope_lookback = slope_lookback
        # Regime filter
        self.regime_filter_enabled = regime_filter_enabled
        self.regime_sma_period = regime_sma_period
        self.regime_slope_enabled = regime_slope_enabled
        self.regime_slope_lookback_days = regime_slope_lookback_days
        # Cached daily regime series (set by prepare_regime)
        self._regime_ok: Optional[pd.Series] = None
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range.
        
        LEAKAGE AUDIT (verified):
        - tr1 = high[i] - low[i]  → uses only bar[i] data ✓
        - tr2 = abs(high[i] - close[i-1])  → uses high[i] vs previous close ✓
        - tr3 = abs(low[i] - close[i-1])  → uses low[i] vs previous close ✓
        - ATR[i] = rolling mean over [i-N+1, i]  → past data only ✓
        
        Result: ATR[i] is fully known at bar[i] close. No look-ahead bias.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # True Range components
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(self.atr_period).mean()
        
        return atr
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        ADX measures trend strength (not direction):
        - ADX >= 25: Strong trend (trending market)
        - ADX < 20: Weak trend (ranging/choppy market)
        
        LEAKAGE AUDIT:
        - +DM, -DM use current and previous bar ✓
        - Smoothed values use rolling mean over past data ✓
        - ADX[i] uses only data up to bar[i] ✓
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Calculate +DM and -DM
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        # Use pandas where to preserve index
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0),
            index=df.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0),
            index=df.index
        )
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
        period = self.adx_period
        atr_smooth = true_range.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_smooth
        minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_smooth
        
        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx
    
    def prepare_regime(self, df_1h: pd.DataFrame, daily_df: pd.DataFrame) -> None:
        """
        Prepare daily regime indicator and align to 1h bars.
        
        Regime logic:
        - daily_sma200 = SMA(daily_close, regime_sma_period)
        - regime_ok = daily_close > daily_sma200
        - Optional: require daily_sma200_slope >= 0
        
        Alignment (no look-ahead):
        - For each 1h bar, use the PREVIOUS day's regime value
        - This ensures no same-day leakage (daily close not available until end of day)
        
        Args:
            df_1h: 1h OHLCV DataFrame
            daily_df: Daily OHLCV DataFrame
        """
        if not self.regime_filter_enabled:
            self._regime_ok = pd.Series(True, index=df_1h.index)
            return
        
        # Calculate daily SMA200
        daily_close = daily_df["close"]
        daily_sma = daily_close.rolling(self.regime_sma_period).mean()
        
        # Base regime: close > sma
        regime_ok_daily = daily_close > daily_sma
        
        # Optional slope filter
        if self.regime_slope_enabled:
            slope = daily_sma - daily_sma.shift(self.regime_slope_lookback_days)
            slope_ok = slope >= 0
            regime_ok_daily = regime_ok_daily & slope_ok
        
        # Convert to DataFrame with date as index
        regime_df = regime_ok_daily.to_frame(name="regime_ok")
        regime_df["date"] = regime_df.index.date
        
        # Create mapping: for each 1h bar, use PREVIOUS day's regime (no look-ahead)
        # Get the date of each 1h bar
        df_1h_dates = pd.Series(df_1h.index.date, index=df_1h.index)
        
        # Shift regime by 1 day to avoid same-day leakage
        regime_shifted = regime_ok_daily.shift(1)  # Use yesterday's regime
        
        # Create date-to-regime mapping
        date_to_regime = regime_shifted.to_dict()
        
        # Map to 1h bars
        regime_1h = df_1h_dates.map(lambda d: date_to_regime.get(pd.Timestamp(d, tz='UTC'), False))
        
        # Forward-fill any NaN from warmup period
        regime_1h = regime_1h.fillna(False)
        
        self._regime_ok = regime_1h
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals with stop-loss levels.
        
        IMPORTANT: All indicators are calculated on current bar data,
        but signals are meant to be EXECUTED on the NEXT bar.
        The backtest engine handles this execution delay.
        
        Filters:
        1. ADX Filter (if filter_enabled): Requires ADX >= adx_threshold
        2. Slope Filter (if slope_filter_enabled): Requires SMA_slow slope > 0
           Slope = SMA_slow[t] - SMA_slow[t - slope_lookback]
        
        Returns DataFrame with:
            - signal: 1 (want long), 0 (want flat)
            - stop_price: Initial stop-loss price (set at entry signal)
            - atr: Current ATR value (for trailing stop calculation)
            - adx: Current ADX value (for regime detection)
        """
        close = df["close"]
        
        # Calculate indicators (no shift here - backtest engine handles execution delay)
        sma_fast = close.rolling(self.fast_period).mean()
        sma_slow = close.rolling(self.slow_period).mean()
        atr = self._calculate_atr(df)
        adx = self._calculate_adx(df)
        
        # Calculate slope if needed
        slope_is_positive = pd.Series(True, index=df.index)
        if self.slope_filter_enabled:
            # slope = current - lookback
            slope_val = sma_slow - sma_slow.shift(self.slope_lookback)
            slope_is_positive = slope_val > 0
        
        # Initialize output
        signals = pd.DataFrame(index=df.index)
        signals["signal"] = 0
        signals["stop_price"] = np.nan
        signals["atr"] = atr
        signals["adx"] = adx
        
        # Detect crossovers
        # Cross above: fast was below slow, now fast >= slow
        cross_above = (sma_fast.shift(1) < sma_slow.shift(1)) & (sma_fast >= sma_slow)
        # Cross below: fast was above slow, now fast < slow
        cross_below = (sma_fast.shift(1) > sma_slow.shift(1)) & (sma_fast < sma_slow)
        
        # Generate raw signal (1 = in uptrend, 0 = not)
        # When cross_above: start long (if filter allows)
        # When cross_below: end long (always allowed)
        in_long = False
        signal_values = []
        stop_prices = []
        
        for i in range(len(df)):
            if pd.isna(sma_fast.iloc[i]) or pd.isna(sma_slow.iloc[i]):
                signal_values.append(0)
                stop_prices.append(np.nan)
                continue
            
            if cross_above.iloc[i]:
                # Entry signal - check filters
                allow_entry = True
                
                # 1. ADX Filter
                if self.filter_enabled:
                    adx_val = adx.iloc[i]
                    if pd.isna(adx_val) or adx_val < self.adx_threshold:
                        allow_entry = False
                
                # 2. Slope Filter
                if self.slope_filter_enabled and allow_entry:
                    if not slope_is_positive.iloc[i]:
                        allow_entry = False
                
                # 3. Daily Regime Filter
                if self.regime_filter_enabled and allow_entry:
                    if self._regime_ok is not None:
                        regime_val = self._regime_ok.iloc[i] if i < len(self._regime_ok) else False
                        if not regime_val:
                            allow_entry = False
                
                if allow_entry:
                    in_long = True
                    
            elif cross_below.iloc[i]:
                # Exit signal - always allowed
                in_long = False
            
            signal_values.append(1 if in_long else 0)
            
            # Set initial stop price at entry signal
            # This condition means current signal is 1 and previous was 0 (an entry)
            if in_long and (len(signal_values) > 1 and signal_values[-2] == 0):
                current_atr = atr.iloc[i]
                if not pd.isna(current_atr):
                    # We store the desired stop distance or price here
                    # For simplicity, we'll store the exact price level derived from close
                    # But the backtest engine re-calculates entry price + ATR
                    # So here we just conceptually mark it.
                    # Actually, the backtest engine uses 'stop_price' column from signals?
                    # Let's check backtest.py... backtest uses run_backtest logic:
                    # "stop_signal = signals_df['stop_price'].iloc[i]"
                    # So we should calculate it here based on CLOSE (approximation)
                    # Real execution uses entry_price, but we don't have entry price here.
                    # We'll calculate a proxy stop based on close.
                    stop_p = close.iloc[i] - self.atr_multiplier * current_atr
                    stop_prices.append(stop_p)
                else:
                    stop_prices.append(np.nan)
            else:
                stop_prices.append(np.nan)

        signals["signal"] = signal_values
        signals["stop_price"] = stop_prices
        
        return signals


# Strategy registry - only sma_atr for now
STRATEGIES: Dict[str, type] = {
    "sma_atr": SMACrossATRStrategy,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """
    Get strategy instance by name.
    
    Args:
        name: Strategy name (currently only 'sma_atr')
        **kwargs: Strategy parameters
    
    Returns:
        Strategy instance
    """
    if name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")
    
    return STRATEGIES[name](**kwargs)


def list_strategies() -> list:
    """List all available strategy names."""
    return list(STRATEGIES.keys())
