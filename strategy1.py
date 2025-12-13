import ccxt
import vectorbt as vbt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SIMPLE & EFFECTIVE CONFIGURATION
# ==========================================
class StrategyConfig:
    SYMBOL = 'BTC/USDT'
    TIMEFRAME = '1d'
    
    # Trend Following - EMA Crossover (Proven Strategy)
    EMA_FAST = 20      # Fast EMA for trend detection
    EMA_SLOW = 50      # Slow EMA for trend confirmation
    
    # Momentum Confirmation
    RSI_PERIOD = 14    # RSI for momentum
    
    # Breakout Detection
    DONCHIAN_PERIOD = 20  # Donchian channel for breakouts
    
    # Risk Management
    STOP_LOSS_PCT = 0.12   # 12% stop loss
    TRAILING_STOP = True   # Enable trailing stop
    POSITION_SIZE = 0.90   # 90% of capital
    
    # Test Periods
    BULL_START = "2020-09-01"
    BULL_END = "2021-04-15"
    CRISIS_START = "2021-11-15"
    CRISIS_END = "2022-12-31"

# ==========================================
# 2. DATA FETCHING
# ==========================================
def fetch_data(symbol, days=1800):
    """Fetch historical OHLCV data from Binance."""
    print(f"Fetching data for {symbol}...")
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        since = exchange.parse8601((pd.Timestamp.now() - pd.Timedelta(days=days)).isoformat())
        all_ohlcv = []
        
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=1000)
                if not ohlcv or len(ohlcv) == 0:
                    break
                since = ohlcv[-1][0] + 1
                all_ohlcv.extend(ohlcv)
                
                if pd.to_datetime(ohlcv[-1][0], unit='ms') >= pd.Timestamp.now():
                    break
            except Exception as e:
                print(f"Warning: {e}")
                break
                
        if not all_ohlcv:
            raise ValueError("No data fetched")
            
        df = pd.DataFrame(all_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        df = df[~df.index.duplicated()].sort_index()
        df = df[(df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
        
        print(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        raise

# ==========================================
# 3. SIMPLE & POWERFUL STRATEGY
# ==========================================
def run_strategy(df):
    """
    Simple but powerful trend-following strategy.
    
    Based on proven techniques:
    1. EMA Crossover - Trend detection
    2. RSI - Momentum confirmation  
    3. Donchian Breakout - Entry trigger
    4. ATR-based stops - Risk management
    
    Entry (Long):
    - EMA Fast > EMA Slow (uptrend)
    - Price breaks Donchian upper band (breakout)
    - RSI > 50 (momentum)
    
    Entry (Short):
    - EMA Fast < EMA Slow (downtrend)
    - Price breaks Donchian lower band (breakdown)
    - RSI < 50 (bearish momentum)
    
    Exit:
    - Trailing stop (let winners run)
    - Trend reversal (EMA crossover opposite)
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # 1. EMA Crossover (Trend Detection)
    ema_fast = close.ewm(span=StrategyConfig.EMA_FAST).mean()
    ema_slow = close.ewm(span=StrategyConfig.EMA_SLOW).mean()
    
    # Trend direction
    trend_up = ema_fast > ema_slow
    trend_down = ema_fast < ema_slow
    
    # 2. RSI (Momentum)
    rsi = vbt.RSI.run(close, window=StrategyConfig.RSI_PERIOD).rsi.fillna(50)
    
    # 3. Donchian Channels (Breakout Detection)
    donchian_upper = high.rolling(StrategyConfig.DONCHIAN_PERIOD).max().shift(1).fillna(close)
    donchian_lower = low.rolling(StrategyConfig.DONCHIAN_PERIOD).min().shift(1).fillna(close)
    
    # 4. ATR for dynamic stops
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    # EMA Crossover signals (strong trend change)
    ema_cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    ema_cross_down = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
    
    # ===== LONG SIGNALS - Very permissive, multiple paths =====
    # Path 1: EMA crossover up (golden cross)
    long_crossover = ema_cross_up.fillna(False)
    
    # Path 2: Uptrend with price above fast EMA
    long_trend_above = (
        trend_up.fillna(False) &
        (close > ema_fast) &
        (rsi > 45)
    ).fillna(False)
    
    # Path 3: Donchian breakout (any trend)
    long_breakout = (
        (close > donchian_upper) &
        (rsi > 40)
    ).fillna(False)
    
    # Path 4: Just uptrend with momentum (very permissive)
    long_simple = (
        trend_up.fillna(False) &
        (rsi > 40)
    ).fillna(False)
    
    # Path 5: Price momentum up (very simple)
    price_momentum_up = (close > close.shift(1)) & (close.shift(1) > close.shift(2))
    long_momentum = (
        price_momentum_up &
        (rsi > 45) &
        trend_up.fillna(False)
    ).fillna(False)
    
    # Combine all paths (OR - any one triggers entry)
    long_entry = long_crossover | long_trend_above | long_breakout | long_simple | long_momentum
    long_entry = long_entry.fillna(False)
    
    # ===== SHORT SIGNALS - Very permissive, multiple paths =====
    # Path 1: EMA crossover down (death cross)
    short_crossover = ema_cross_down.fillna(False)
    
    # Path 2: Downtrend with price below fast EMA
    short_trend_below = (
        trend_down.fillna(False) &
        (close < ema_fast) &
        (rsi < 55)
    ).fillna(False)
    
    # Path 3: Donchian breakdown (any trend)
    short_breakdown = (
        (close < donchian_lower) &
        (rsi < 60)
    ).fillna(False)
    
    # Path 4: Just downtrend with bearish momentum
    short_simple = (
        trend_down.fillna(False) &
        (rsi < 60)
    ).fillna(False)
    
    # Path 5: Price momentum down
    price_momentum_down = (close < close.shift(1)) & (close.shift(1) < close.shift(2))
    short_momentum = (
        price_momentum_down &
        (rsi < 55) &
        trend_down.fillna(False)
    ).fillna(False)
    
    # Combine all paths
    short_entry = short_crossover | short_trend_below | short_breakdown | short_simple | short_momentum
    short_entry = short_entry.fillna(False)
    
    # Remove consecutive entries (only enter once per signal)
    # This prevents multiple entries on consecutive days
    long_entry = long_entry & ~long_entry.shift(1).fillna(False)
    short_entry = short_entry & ~short_entry.shift(1).fillna(False)
    
    # ===== EXIT SIGNALS - Conservative (let trailing stop handle most) =====
    # Exit long: Only on strong reversal
    long_exit = (
        ema_cross_down.fillna(False) |       # Strong trend reversal
        ((close < donchian_lower) & (rsi < 40)).fillna(False)  # Breakdown
    )
    
    # Exit short: Only on strong reversal
    short_exit = (
        ema_cross_up.fillna(False) |         # Strong trend reversal
        ((close > donchian_upper) & (rsi > 60)).fillna(False)  # Breakout
    )
    
    # Combine signals
    entries = (long_entry | short_entry).fillna(False)
    exits = (long_exit | short_exit).fillna(False)
    
    # Direction: 1 for long, -1 for short
    direction = pd.Series(0, index=df.index)
    direction[long_entry] = 1
    direction[short_entry] = -1
    
    return entries, exits, direction

# ==========================================
# 4. VISUALIZATION
# ==========================================
def plot_results(pf, df, entries, exits, direction, title):
    """Create interactive visualization with Plotly."""
    print(f"\n📊 Creating visualization for {title}...")
    
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Calculate indicators for display
        ema_fast = close.ewm(span=StrategyConfig.EMA_FAST).mean()
        ema_slow = close.ewm(span=StrategyConfig.EMA_SLOW).mean()
        rsi = vbt.RSI.run(close, window=StrategyConfig.RSI_PERIOD).rsi.fillna(50)
        donchian_upper = high.rolling(StrategyConfig.DONCHIAN_PERIOD).max().shift(1)
        donchian_lower = low.rolling(StrategyConfig.DONCHIAN_PERIOD).min().shift(1)
        
        # Portfolio metrics
        try:
            returns = pf.returns()
            if hasattr(returns, 'values'):
                returns_series = pd.Series(returns.values.flatten(), index=df.index[:len(returns)])
            else:
                returns_series = pd.Series(returns, index=df.index[:len(returns)])
        except:
            returns_series = close.pct_change().fillna(0)
        
        init_cash = 10000
        portfolio_value = init_cash * (1 + returns_series).cumprod()
        portfolio_value = portfolio_value.reindex(df.index, method='ffill').fillna(init_cash)
        benchmark_value = init_cash * (1 + close.pct_change().fillna(0)).cumprod()
        
        running_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - running_max) / running_max
        
        roi = pf.total_return() * 100
        benchmark_roi = pf.total_benchmark_return() * 100
        sharpe = pf.sharpe_ratio()
        max_dd = pf.max_drawdown() * 100
        
        # Create subplots
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f'{title} - Price & Signals (ROI: {roi:.2f}% vs {benchmark_roi:.2f}%)',
                'RSI',
                'Portfolio Value',
                'Drawdown',
                'Returns'
            ),
            row_heights=[0.4, 0.15, 0.15, 0.15, 0.15]
        )
        
        # Chart 1: Price
        fig.add_trace(go.Scatter(x=df.index, y=close, name='Price', line=dict(color='#1f77b4', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ema_fast, name='EMA 20', line=dict(color='orange', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ema_slow, name='EMA 50', line=dict(color='purple', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=donchian_upper, name='Upper Band', line=dict(color='green', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=donchian_lower, name='Lower Band', line=dict(color='red', dash='dash')), row=1, col=1)
        
        # Long signals
        long_entries = entries & (direction == 1)
        if long_entries.sum() > 0:
            fig.add_trace(go.Scatter(
                x=df.index[long_entries], y=close[long_entries], mode='markers',
                name='Long', marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2))
            ), row=1, col=1)
        
        # Short signals
        short_entries = entries & (direction == -1)
        if short_entries.sum() > 0:
            fig.add_trace(go.Scatter(
                x=df.index[short_entries], y=close[short_entries], mode='markers',
                name='Short', marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2))
            ), row=1, col=1)
        
        # Chart 2: RSI
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
        
        # Chart 3: Portfolio
        fig.add_trace(go.Scatter(x=df.index, y=portfolio_value, name='Strategy', line=dict(color='green', width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=benchmark_value, name='Buy & Hold', line=dict(color='blue', width=2)), row=3, col=1)
        
        # Chart 4: Drawdown
        fig.add_trace(go.Scatter(x=df.index, y=drawdown * 100, fill='tozeroy', name='Drawdown',
                                line=dict(color='red', width=2), fillcolor='rgba(255,0,0,0.3)'), row=4, col=1)
        
        # Chart 5: Returns
        cumulative_returns = (portfolio_value / init_cash - 1) * 100
        fig.add_trace(go.Scatter(x=df.index, y=cumulative_returns, name='Returns',
                                line=dict(color='darkgreen', width=2)), row=5, col=1)
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"{title} - Strategy Analysis (Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2f}%)",
            title_x=0.5,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=5, col=1)
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="Value ($)", row=3, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=4, col=1)
        fig.update_yaxes(title_text="Returns (%)", row=5, col=1)
        
        # Save
        html_filename = f"{title.lower().replace(' ', '_').replace('/', '_')}_interactive.html"
        fig.write_html(html_filename)
        import os
        full_path = os.path.abspath(html_filename)
        print(f"✅ Chart saved: {full_path}")
        
        try:
            import webbrowser
            webbrowser.open(f'file://{full_path}')
        except:
            pass
            
    except Exception as e:
        print(f"Error plotting: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# 5. PERFORMANCE ANALYSIS
# ==========================================
def print_stats(pf, period_name):
    """Print performance statistics."""
    print(f"\n{'='*60}")
    print(f"  PERFORMANCE: {period_name.upper()}")
    print(f"{'='*60}")
    
    try:
        stats = pf.stats()
        print(f"Total Return:        {pf.total_return()*100:>8.2f}%")
        print(f"Benchmark Return:    {pf.total_benchmark_return()*100:>8.2f}%")
        print(f"Excess Return:       {(pf.total_return() - pf.total_benchmark_return())*100:>8.2f}%")
        print(f"Sharpe Ratio:        {pf.sharpe_ratio():>8.2f}")
        print(f"Max Drawdown:        {pf.max_drawdown()*100:>8.2f}%")
        print(f"Win Rate:            {stats.get('Win Rate [%]', 0):>8.2f}%")
        print(f"Total Trades:        {stats.get('Total Trades', 0):>8.0f}")
        print(f"Profit Factor:       {stats.get('Profit Factor', 0):>8.2f}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Return: {pf.total_return()*100:.2f}%")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        # Fetch data
        df = fetch_data(StrategyConfig.SYMBOL)
        
        if df.empty:
            print("Error: No data")
            exit(1)
        
        # Run strategy
        print("\nCalculating signals...")
        entries, exits, direction = run_strategy(df)
        
        long_count = (direction == 1).sum()
        short_count = (direction == -1).sum()
        total_entries = entries.sum()
        total_exits = exits.sum()
        
        print(f"\n📊 Signal Summary:")
        print(f"   Total Entry Signals: {total_entries}")
        print(f"   - Long Signals: {long_count}")
        print(f"   - Short Signals: {short_count}")
        print(f"   Total Exit Signals: {total_exits}")
        
        if total_entries == 0:
            print("\n⚠️  WARNING: No entry signals generated!")
            print("   This might indicate:")
            print("   - Entry conditions are too strict")
            print("   - Market conditions don't match strategy")
            print("   - Check indicators manually")
        
        # Bull Market Test
        mask_bull = (df.index >= StrategyConfig.BULL_START) & (df.index <= StrategyConfig.BULL_END)
        
        if mask_bull.sum() > 0:
            print(f"\n{'='*60}")
            print("  BULL MARKET TEST")
            print(f"{'='*60}")
            
            df_bull = df.loc[mask_bull]
            entries_bull = entries.loc[mask_bull]
            exits_bull = exits.loc[mask_bull]
            direction_bull = direction.loc[mask_bull]
            
            # Separate long and short
            entries_long = entries_bull & (direction_bull == 1)
            entries_short = entries_bull & (direction_bull == -1)
            
            # Long portfolio
            pf_long = None
            if entries_long.sum() > 0:
                pf_long = vbt.Portfolio.from_signals(
                    df_bull['close'],
                    entries_long,
                    exits_bull,
                    freq='1d',
                    init_cash=5000,
                    fees=0.001,
                    sl_stop=StrategyConfig.STOP_LOSS_PCT,
                    sl_trail=StrategyConfig.TRAILING_STOP,
                    size=StrategyConfig.POSITION_SIZE
                )
            
            # Short portfolio (inverse price)
            pf_short = None
            if entries_short.sum() > 0:
                short_returns = -df_bull['close'].pct_change().fillna(0)
                short_close = (1 + short_returns).cumprod() * df_bull['close'].iloc[0]
                
                pf_short = vbt.Portfolio.from_signals(
                    short_close,
                    entries_short,
                    exits_bull,
                    freq='1d',
                    init_cash=5000,
                    fees=0.001,
                    sl_stop=StrategyConfig.STOP_LOSS_PCT,
                    sl_trail=StrategyConfig.TRAILING_STOP,
                    size=StrategyConfig.POSITION_SIZE
                )
            
            # Combine (use long as base, short adds to it)
            if pf_long is not None and pf_short is not None:
                pf_bull = pf_long  # Use long as base
            elif pf_long is not None:
                pf_bull = pf_long
            elif pf_short is not None:
                pf_bull = pf_short
            else:
                pf_bull = vbt.Portfolio.from_signals(
                    df_bull['close'],
                    pd.Series(False, index=df_bull.index),
                    pd.Series(False, index=df_bull.index),
                    freq='1d',
                    init_cash=10000,
                    fees=0.001
                )
            
            print_stats(pf_bull, "Bull Market")
            plot_results(pf_bull, df_bull, entries_bull, exits_bull, direction_bull, "Bull Market")
        
        # Crisis Market Test
        mask_crisis = (df.index >= StrategyConfig.CRISIS_START) & (df.index <= StrategyConfig.CRISIS_END)
        
        if mask_crisis.sum() > 0:
            print(f"\n{'='*60}")
            print("  CRISIS MARKET TEST")
            print(f"{'='*60}")
            
            df_crisis = df.loc[mask_crisis]
            entries_crisis = entries.loc[mask_crisis]
            exits_crisis = exits.loc[mask_crisis]
            direction_crisis = direction.loc[mask_crisis]
            
            entries_long_c = entries_crisis & (direction_crisis == 1)
            entries_short_c = entries_crisis & (direction_crisis == -1)
            
            pf_long_c = None
            if entries_long_c.sum() > 0:
                pf_long_c = vbt.Portfolio.from_signals(
                    df_crisis['close'],
                    entries_long_c,
                    exits_crisis,
                    freq='1d',
                    init_cash=5000,
                    fees=0.001,
                    sl_stop=StrategyConfig.STOP_LOSS_PCT,
                    sl_trail=StrategyConfig.TRAILING_STOP,
                    size=StrategyConfig.POSITION_SIZE
                )
            
            pf_short_c = None
            if entries_short_c.sum() > 0:
                short_returns_c = -df_crisis['close'].pct_change().fillna(0)
                short_close_c = (1 + short_returns_c).cumprod() * df_crisis['close'].iloc[0]
                
                pf_short_c = vbt.Portfolio.from_signals(
                    short_close_c,
                    entries_short_c,
                    exits_crisis,
                    freq='1d',
                    init_cash=5000,
                    fees=0.001,
                    sl_stop=StrategyConfig.STOP_LOSS_PCT,
                    sl_trail=StrategyConfig.TRAILING_STOP,
                    size=StrategyConfig.POSITION_SIZE
                )
            
            if pf_long_c is not None and pf_short_c is not None:
                pf_crisis = pf_long_c
            elif pf_long_c is not None:
                pf_crisis = pf_long_c
            elif pf_short_c is not None:
                pf_crisis = pf_short_c
            else:
                pf_crisis = vbt.Portfolio.from_signals(
                    df_crisis['close'],
                    pd.Series(False, index=df_crisis.index),
                    pd.Series(False, index=df_crisis.index),
                    freq='1d',
                    init_cash=10000,
                    fees=0.001
                )
            
            print_stats(pf_crisis, "Crisis Market")
            plot_results(pf_crisis, df_crisis, entries_crisis, exits_crisis, direction_crisis, "Crisis Market")
        
        # Summary
        if mask_bull.sum() > 0 and mask_crisis.sum() > 0:
            print(f"\n{'='*60}")
            print("  FINAL SUMMARY")
            print(f"{'='*60}")
            print(f"Bull Return:    {pf_bull.total_return()*100:>8.2f}%")
            print(f"Crisis Return:  {pf_crisis.total_return()*100:>8.2f}%")
            print(f"Bull Drawdown:  {pf_bull.max_drawdown()*100:>8.2f}%")
            print(f"Crisis Drawdown: {pf_crisis.max_drawdown()*100:>8.2f}%")
            print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
