import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- ADIM 1: VERİ ÇEKME ---
print("Veriler indiriliyor...")
symbol = "AAPL"
# auto_adjust=True varsayılan olduğu için 'Close' artık düzeltilmiş fiyattır.
data = yf.download(symbol, start="2020-01-01", end="2023-12-30", progress=False)

# yfinance bazen çoklu indeks döndürebilir, bunu düzeltelim:
if isinstance(data.columns, pd.MultiIndex):
    # Sadece 'Close' sütununu alıp tek seviyeye indiriyoruz
    data = data.xs(symbol, axis=1, level=1, drop_level=True)

# --- ADIM 2: STRATEJİ KURMA (SMA Crossover) ---
# DİKKAT: Artık 'Adj Close' yerine 'Close' kullanıyoruz
data['SMA_Short'] = data['Close'].rolling(window=20).mean()
data['SMA_Long'] = data['Close'].rolling(window=50).mean()

# Sinyal: Kısa ortalama Uzunu geçerse AL (1), yoksa NAKİT (0)
data['Signal'] = 0
data.loc[data['SMA_Short'] > data['SMA_Long'], 'Signal'] = 1

# --- ADIM 3: BACKTEST ---
# Günlük piyasa getirisi
data['Market_Returns'] = data['Close'].pct_change()

# Strateji getirisi (Sinyali 1 gün kaydırıyoruz)
data['Strategy_Returns'] = data['Signal'].shift(1) * data['Market_Returns']

# Kümülatif (Bileşik) Getiriler
data['Cumulative_Market'] = (1 + data['Market_Returns']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()

# Sonuçları Terminale Yazdır
market_donus = data['Cumulative_Market'].iloc[-1]
strategy_donus = data['Cumulative_Strategy'].iloc[-1]

print("-" * 30)
print(f"Market (Buy & Hold) Getirisi: {market_donus:.2f}x")
print(f"SMA Stratejisi Getirisi:      {strategy_donus:.2f}x")
print("-" * 30)

# --- ADIM 4: GÖRSELLEŞTİRME ---
plt.figure(figsize=(10, 6))
plt.plot(data['Cumulative_Market'], label='Market (Buy & Hold)', color='gray', alpha=0.5)
plt.plot(data['Cumulative_Strategy'], label='SMA Stratejisi', color='green')
plt.title(f'{symbol} - SMA Stratejisi Backtest Sonucu')
plt.legend()
plt.grid(True)
print("Grafik çizdiriliyor... (Pencereyi kontrol et)")
plt.show()