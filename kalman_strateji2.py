import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- AYARLAR ---
SYMBOL = "ETH-USD"
START_DATE = "2022-01-01"  # Ayı piyasasını özellikle seçtik, Short yeteneğini test etmek için
END_DATE = "2024-01-01"
COMMISSION = 0.001  # İşlem başına %0.1 komisyon (Binance/Aracı kurum payı)

print(f"{SYMBOL} verileri indiriliyor ve analiz başlıyor...")

# --- 1. VERİ HAZIRLIĞI ---
data = yf.download(SYMBOL, start=START_DATE, end=END_DATE, progress=False)

# MultiIndex düzeltmesi
if isinstance(data.columns, pd.MultiIndex):
    data = data.xs(SYMBOL, axis=1, level=1, drop_level=True)

# --- 2. MATEMATİKSEL MODELLER ---

# A) Kalman Filtresi (Gürültüyü temizler, gerçek trendi bulur)
def calculate_kalman(prices, Q=1e-5, R=0.01):
    n_iter = len(prices)
    sz = (n_iter,)
    xhat = np.zeros(sz)      # Filtrelenmiş Fiyat
    P = np.zeros(sz)         # Hata varyansı
    xhatminus = np.zeros(sz) 
    Pminus = np.zeros(sz)    
    K = np.zeros(sz)         # Kalman Kazancı
    
    xhat[0] = prices[0]
    P[0] = 1.0
    
    for k in range(1, n_iter):
        # Time Update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        # Measurement Update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (prices[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    return xhat

# B) ATR (Average True Range) - Volatiliteyi Ölçer
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

# Hesaplamaları Uygula
data['Kalman'] = calculate_kalman(data['Close'].values, Q=0.01, R=0.5)
data['ATR'] = calculate_atr(data)

# --- 3. STRATEJİ MANTIĞI (TREND FOLLOWING) ---

# Kalman Eğimini (Slope) Hesapla
# Eğer Kalman bugün dünden yüksekse, trend yukarıdır.
data['Kalman_Slope'] = data['Kalman'].diff()

# Eşik Değeri (Threshold): Sadece güçlü trendlerde gir
# ATR'nin %10'u kadar bir hareket varsa trendi onayla
threshold = data['ATR'] * 0.1

data['Signal'] = 0

# LONG KURALI: Kalman Eğimi Pozitif VE Güçlü
data.loc[data['Kalman_Slope'] > threshold, 'Signal'] = 1 

# SHORT KURALI: Kalman Eğimi Negatif VE Güçlü
data.loc[data['Kalman_Slope'] < -threshold, 'Signal'] = -1

# Pozisyonları doldur (1'den -1'e geçene kadar pozisyonu koru)
data['Position'] = data['Signal'].replace(0, np.nan).ffill().fillna(0)

# --- 4. PROFESYONEL BACKTEST (Short & Komisyon Dahil) ---

# Günlük Market Getirisi
data['Market_Returns'] = data['Close'].pct_change()

# Strateji Getirisi:
# Position * Market_Returns yaparsak:
# Eğer Shorttaysak (-1) ve Market Düşerse (-%5) -> (-1 * -0.05) = +%5 KAZANÇ!
data['Strategy_Raw_Returns'] = data['Position'].shift(1) * data['Market_Returns']

# Komisyon Maliyeti Hesabı
# Pozisyonun değiştiği günleri bul (Al veya Sat yaptıysak komisyon öderiz)
# diff() != 0 olan yerlerde işlem yapılmıştır. abs() ile -1'den 1'e geçişi 2 birim sayarız.
trades = data['Position'].diff().fillna(0).abs()
data['Transaction_Costs'] = trades * COMMISSION

# Net Getiri (Getiri - Komisyon)
data['Strategy_Net_Returns'] = data['Strategy_Raw_Returns'] - data['Transaction_Costs']

# Kümülatif Sonuçlar
data['Cumulative_Market'] = (1 + data['Market_Returns']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Net_Returns']).cumprod()

# --- 5. PERFORMANS METRİKLERİ VE ÇIKTI ---
total_market_return = data['Cumulative_Market'].iloc[-1]
total_strategy_return = data['Cumulative_Strategy'].iloc[-1]

# Sharpe Ratio (Yıllıklandırılmış Risk/Getiri Oranı)
# Sharpe > 1 iyidir, > 2 harikadır.
risk_free_rate = 0.0 # Basitlik için 0 aldık
std_dev = data['Strategy_Net_Returns'].std() * np.sqrt(252) # Yıllık volatilite
annual_return = data['Strategy_Net_Returns'].mean() * 252
sharpe_ratio = (annual_return - risk_free_rate) / std_dev

print("\n" + "="*40)
print(f"STRATEJİ RAPORU ({START_DATE} - {END_DATE})")
print("="*40)
print(f"Market (Buy & Hold) Sonucu:  {total_market_return:.2f}x (Paran bu kata çıktı/indi)")
print(f"Pro Kalman Stratejisi:       {total_strategy_return:.2f}x")
print(f"Sharpe Oranı:                {sharpe_ratio:.2f}")
print(f"İşlem Komisyonu Oranı:       %{COMMISSION*100}")
print("-" * 40)

if total_strategy_return > total_market_return:
    print("✅ BAŞARILI: Strateji piyasayı yendi!")
else:
    print("❌ BAŞARISIZ: Piyasa daha iyi getirdi.")

# --- 6. GÖRSELLEŞTİRME ---
plt.figure(figsize=(12, 6))

# Ana Getiri Grafiği
plt.plot(data['Cumulative_Market'], label='Market (ETH Hold)', color='gray', alpha=0.5, linestyle='--')
plt.plot(data['Cumulative_Strategy'], label='Pro Kalman (Long+Short)', color='blue', linewidth=2)

# Alım Satım Yerlerini İşaretle (Opsiyonel ama şık durur)
# Sadece pozisyon değişimlerini bul
buy_signals = data[data['Position'].diff() == 1].index # -1'den 0'a veya 0'dan 1'e
sell_signals = data[data['Position'].diff() == -1].index # 1'den 0'a veya 0'dan -1'e

# Grafik Ayarları
plt.title(f'Profesyonel Kalman Trend Takipçisi (Komisyonlu Backtest)\nSharpe Ratio: {sharpe_ratio:.2f}')
plt.ylabel('Kümülatif Getiri (x Kat)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log') # Logaritmik ölçek uzun vadede daha doğru gösterir

print("Grafik oluşturuluyor...")
plt.show()

#Test