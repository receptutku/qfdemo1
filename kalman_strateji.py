import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- ADIM 1: VERİ ÇEKME ---
symbol = "ETH-USD" # Bu sefer daha volatil bir şey deneyelim: Ethereum
print(f"{symbol} verisi indiriliyor...")
data = yf.download(symbol, start="2022-01-01", end="2024-01-01", progress=False)

# MultiIndex düzeltmesi
if isinstance(data.columns, pd.MultiIndex):
    data = data.xs(symbol, axis=1, level=1, drop_level=True)

# --- ADIM 2: KALMAN FİLTRESİ MATEMATİĞİ (MANUEL İMPLEMENTASYON) ---
# Burada bir kütüphane kullanmak yerine matematiği kendimiz kuruyoruz.

def apply_kalman_filter(prices, Q=1e-5, R=0.01):
    """
    prices: Hissenin kapanış fiyatları
    Q: Process Noise Covariance (Sürecin gürültüsü - Trendin değişme hızı)
    R: Measurement Noise Covariance (Ölçüm gürültüsü - Piyasa oynaklığı)
    """
    n_iter = len(prices)
    sz = (n_iter,) # Dizi boyutu
    
    # Başlangıç değerleri
    xhat = np.zeros(sz)      # A posteriori estimate (Filtrelenmiş fiyat)
    P = np.zeros(sz)         # A posteriori error estimate (Hata tahmini)
    xhatminus = np.zeros(sz) # A priori estimate
    Pminus = np.zeros(sz)    # A priori error estimate
    K = np.zeros(sz)         # Kalman Gain (Kazanç faktörü)
    
    # İlk değer atamaları
    xhat[0] = prices[0]
    P[0] = 1.0
    
    for k in range(1, n_iter):
        # 1. Tahmin Aşaması (Time Update)
        xhatminus[k] = xhat[k-1] # Bir önceki durumu koru (Random Walk varsayımı)
        Pminus[k] = P[k-1] + Q
        
        # 2. Düzeltme Aşaması (Measurement Update)
        # Kalman Kazancı: K = Pminus / (Pminus + R)
        K[k] = Pminus[k] / (Pminus[k] + R)
        
        # Tahmini Güncelle: x = x_minus + K * (Measurement - x_minus)
        xhat[k] = xhatminus[k] + K[k] * (prices[k] - xhatminus[k])
        
        # Hata Kovaryansını Güncelle: P = (1 - K) * Pminus
        P[k] = (1 - K[k]) * Pminus[k]
        
    return xhat

print("Kalman Filtresi hesaplanıyor...")
# Kapanış fiyatlarını numpy dizisine çevirip fonksiyona sokuyoruz
prices_array = data['Close'].values
kalman_estimates = apply_kalman_filter(prices_array, Q=0.005, R=1.0) # Q ve R ayarlanabilir parametrelerdir

# Sonuçları DataFrame'e ekle
data['Kalman'] = kalman_estimates

# --- ADIM 3: STRATEJİ KURMA (MEAN REVERSION) ---
# Fiyatın Kalman filtresinden ne kadar saptığını bulalım (Residuals)
data['Residual'] = data['Close'] - data['Kalman']

# Bu sapmanın Standart Sapmasını (Z-Score) hesaplayalım
# Son 20 gündeki sapmaların standart sapması
data['Std'] = data['Residual'].rolling(window=30).std()
data['Z_Score'] = data['Residual'] / data['Std']

# SİNYALLER:
# Eğer Fiyat, Kalman'ın 1.5 Standart Sapma altındaysa AL (Ucuz kaldı)
# Eğer Fiyat, Kalman'a geri dönerse veya üzerine çıkarsa POZİSYONU KAPAT
data['Signal'] = 0
data.loc[data['Z_Score'] < -1.5, 'Signal'] = 1 # Al
data.loc[data['Z_Score'] > 0.5, 'Signal'] = 0  # Sat/Kapat (Erken kar alımı)

# Pozisyon takibi (Sürekli al sinyali gelmesin, elimizde yoksa alalım)
# Bu basit backtest için shift kullanıp getiriyi hesaplayacağız
data['Position'] = data['Signal'].shift(1) # Ertesi gün işleme gireriz

# --- ADIM 4: BACKTEST ---
data['Market_Returns'] = data['Close'].pct_change()
data['Strategy_Returns'] = data['Position'] * data['Market_Returns']

# Kümülatif Getiri
data['Cum_Market'] = (1 + data['Market_Returns']).cumprod()
data['Cum_Strategy'] = (1 + data['Strategy_Returns']).cumprod()

# Sonuçlar
print("-" * 40)
print(f"Market Getirisi: {data['Cum_Market'].iloc[-1]:.2f}x")
print(f"Kalman Stratejisi: {data['Cum_Strategy'].iloc[-1]:.2f}x")
print("-" * 40)

# --- ADIM 5: GÖRSELLEŞTİRME (Kompleks Grafik) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Üst Grafik: Fiyat vs Kalman
ax1.plot(data.index, data['Close'], label='Fiyat (Noisy)', color='gray', alpha=0.5)
ax1.plot(data.index, data['Kalman'], label='Kalman Filtresi (True Value)', color='blue', linewidth=1.5)
# Alım yerlerini işaretle
buy_signals = data[data['Signal'] == 1]
ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='AL Sinyali')
ax1.set_title(f'{symbol} - Kalman Filtresi Mean Reversion')
ax1.legend()
ax1.grid(True)

# Alt Grafik: Z-Score (Osilatör)
ax2.plot(data.index, data['Z_Score'], label='Z-Score (Sapma)', color='purple')
ax2.axhline(-1.5, color='green', linestyle='--', label='Alım Eşiği (-1.5 std)')
ax2.axhline(0, color='black', linestyle='-', label='Mean (Denge Noktası)')
ax2.fill_between(data.index, -1.5, -3, color='green', alpha=0.1) # Alım bölgesi
ax2.set_title('Z-Score (Fiyatın Kalman\'dan Sapması)')
ax2.legend()
ax2.grid(True)

print("Gelişmiş grafik çizdiriliyor...")
plt.show()