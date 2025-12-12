import vectorbt as vbt
import pandas as pd
import ccxt
import numpy as np
from datetime import datetime
import webbrowser  # To open charts automatically
import os          # To find file paths

# --- 1. DATA FETCHING MODULE ---
def get_historical_data(symbol='BTC/USDT', start_date='2021-01-01'):
    """
    Fetches historical OHLCV data from Binance.
    """
    print(f"📡 Fetching data for {symbol} starting from {start_date}...")
    exchange = ccxt.binance()
    limit = 1000
    all_ohlcv = []
    
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=limit)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if pd.to_datetime(ohlcv[-1][0], unit='ms') > datetime.now():
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df.dropna()

# --- 2. STRATEGY ENGINE (OPTIMIZED v3) ---
def run_strategy(df, period_name):
    print(f"\n⚙️ Running Analysis: {period_name}...")
    
    close = df['Close']
    
    # --- INDICATORS ---
    # 1. Trend Filter: EMA 50
    # Faster than EMA 200, allows us to catch trends earlier.
    ema = vbt.MA.run(close, window=50, ewm=True)
    
    # 2. Momentum Trigger: RSI 14
    rsi = vbt.RSI.run(close, window=14)
    
    # --- ENTRY LOGIC (MOMENTUM BURST) ---
    # Rule 1: Trend is UP (Price > EMA 50)
    trend_ok = close > ema.ma
    
    # Rule 2: Momentum is Building (RSI crosses ABOVE 50)
    # We are not buying the dip anymore; we are buying the STRENGTH.
    momentum_signal = rsi.rsi_crossed_above(50)
    
    entries = trend_ok & momentum_signal
    
    # --- EXIT LOGIC ---
    # We rely purely on a 10% Trailing Stop to ride the winners.
    exits = pd.Series(False, index=close.index) 

    # --- SIMULATION ---
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        sl_stop=0.10,       # 10% Stop Loss (Gives market room to breathe)
        sl_trail=True,      # TRAILING MODE: Moves up as price goes up
        init_cash=10000,
        fees=0.001,         # 0.1% Binance Fee
        freq='1h'
    )
    
    # --- RESULTS ---
    print(f"\n{'='*40}")
    print(f"     REPORT: {period_name.upper()}     ")
    print(f"{'='*40}")
    stats = portfolio.stats()
    
    # Print Key Metrics
    print(f"Total Return: [{stats['Total Return [%]']:.2f}%]")
    print(f"Win Rate:     [{stats['Win Rate [%]']:.2f}%]")
    print(f"Max Drawdown: [{stats['Max Drawdown [%]']:.2f}%]")
    print(f"Sharpe Ratio: [{stats['Sharpe Ratio']:.2f}]")
    print(f"{'='*40}")
    
    # --- SAVE & OPEN CHART ---
    filename = f"result_{period_name.lower().replace(' ', '_')}.html"
    portfolio.plot().write_html(filename)
    
    # Get absolute path and open in browser
    full_path = os.path.abspath(filename)
    url = 'file://' + full_path
    print(f"📊 Opening chart in browser: {filename}")
    webbrowser.open(url)
    
    return stats

# --- 3. MAIN EXECUTION ---
full_df = get_historical_data(start_date='2021-01-01')

# Define Periods
# Period 1: The 2021 Bull Run
bull_start = '2021-01-01'
bull_end   = '2021-11-10'

# Period 2: The 2022 Crash
bear_start = '2021-11-11'
bear_end   = '2022-12-31'

df_bull = full_df.loc[bull_start:bull_end]
df_crisis = full_df.loc[bear_start:bear_end]

print("\n--- SCENARIO 1: BULL MARKET (Prosperity) ---")
stats_bull = run_strategy(df_bull, "Bull Market 2021")

print("\n--- SCENARIO 2: ECONOMIC CRISIS (The Bear) ---")
stats_crisis = run_strategy(df_crisis, "Crisis Market 2022")

print("\n\n************************************************")
print("              FINAL COMPARISON                  ")
print("************************************************")
print(f"RETURN DIFF:      Bull [{stats_bull['Total Return [%]']:.2f}%] vs Crisis [{stats_crisis['Total Return [%]']:.2f}%]")
print(f"RISK (DRAWDOWN):  Bull [{stats_bull['Max Drawdown [%]']:.2f}%] vs Crisis [{stats_crisis['Max Drawdown [%]']:.2f}%]")