# Quantitative Finance Trading Strategies

Bu repository, Python kullanılarak geliştirilmiş üç farklı algoritmik trading stratejisi içermektedir. Her strateji, yfinance kütüphanesi ile gerçek piyasa verilerini kullanarak backtest edilmiştir.

## 📊 Stratejiler

### 1. SMA Crossover Stratejisi (`ilk_stratejim.py`)
Basit ve etkili bir trend takip stratejisi. Kısa vadeli (20 günlük) ve uzun vadeli (50 günlük) basit hareketli ortalamaları (SMA) kullanarak alım/satım sinyalleri üretir.

**Özellikler:**
- 20/50 günlük SMA crossover
- Buy & Hold karşılaştırması
- Görselleştirme ile performans analizi

**Kullanım:**
```bash
python ilk_stratejim.py
```

### 2. Kalman Filter Mean Reversion (`kalman_strateji.py`)
Kalman filtresi kullanarak fiyat gürültüsünü temizleyen ve mean reversion (ortalamaya dönüş) stratejisi uygulayan gelişmiş bir yaklaşım.

**Özellikler:**
- Manuel Kalman filtresi implementasyonu
- Z-Score tabanlı sinyal üretimi
- Mean reversion mantığı
- İki panel görselleştirme (fiyat + Z-Score)

**Kullanım:**
```bash
python kalman_strateji.py
```

### 3. Profesyonel Kalman Trend Takipçisi (`kalman_strateji2.py`)
En gelişmiş strateji. Kalman filtresi ve ATR (Average True Range) kullanarak hem long hem short pozisyonlar alabilen, komisyon maliyetlerini de hesaba katan profesyonel bir backtest sistemi.

**Özellikler:**
- Kalman filtresi ile trend tespiti
- ATR ile volatilite ölçümü
- Long ve Short pozisyon desteği
- Komisyon maliyeti hesaplama
- Sharpe Ratio performans metriği
- Logaritmik ölçekli görselleştirme

**Kullanım:**
```bash
python kalman_strateji2.py
```

## 🚀 Kurulum

### Gereksinimler
- Python 3.7+
- pip

### Adımlar

1. Repository'yi klonlayın:
```bash
git clone https://github.com/receptutku/qfdemo1.git
cd qfdemo1
```

2. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

veya manuel olarak:
```bash
pip install yfinance pandas matplotlib numpy
```

## 📦 Bağımlılıklar

- `yfinance>=0.2.66` - Yahoo Finance veri çekme
- `pandas>=1.3.0` - Veri işleme ve analiz
- `matplotlib>=3.0.0` - Görselleştirme
- `numpy` - Sayısal hesaplamalar (pandas ile birlikte gelir)

## 📈 Kullanım Örnekleri

### Strateji Parametrelerini Değiştirme

Her stratejide farklı parametreler test edilebilir:

**SMA Crossover:**
- `window=20` ve `window=50` değerlerini değiştirerek farklı periyotlar deneyebilirsiniz.

**Kalman Filter:**
- `Q` (Process Noise) ve `R` (Measurement Noise) parametrelerini ayarlayarak filtre hassasiyetini değiştirebilirsiniz.
- Z-Score eşik değerlerini (`-1.5`, `0.5`) optimize edebilirsiniz.

**Pro Kalman:**
- `COMMISSION` değerini gerçek komisyon oranınıza göre ayarlayın.
- `Q` ve `R` parametrelerini piyasa koşullarına göre optimize edin.

### Farklı Semboller Test Etme

Her stratejide `symbol` değişkenini değiştirerek farklı hisse senetleri veya kripto paralar test edilebilir:

```python
symbol = "AAPL"  # Apple
symbol = "TSLA"  # Tesla
symbol = "BTC-USD"  # Bitcoin
symbol = "ETH-USD"  # Ethereum
```

## 📊 Performans Metrikleri

Stratejiler şu metrikleri hesaplar:
- **Kümülatif Getiri**: Başlangıç sermayesinin kaç katına çıktığı
- **Sharpe Ratio**: Risk-ayarlı getiri oranı (sadece `kalman_strateji2.py`)
- **Buy & Hold Karşılaştırması**: Pasif yatırım stratejisi ile karşılaştırma

## ⚠️ Uyarılar

- Bu stratejiler eğitim ve araştırma amaçlıdır.
- Geçmiş performans gelecek sonuçları garanti etmez.
- Gerçek trading yapmadan önce kapsamlı test ve risk yönetimi yapın.
- Komisyon, slippage ve likidite gibi gerçek piyasa koşulları backtest'te tam olarak simüle edilemeyebilir.

## 📝 Lisans

Bu proje eğitim amaçlıdır. Kendi sorumluluğunuzda kullanın.

## 🤝 Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır. Büyük değişiklikler için önce bir issue açarak neyi değiştirmek istediğinizi tartışın.

## 📧 İletişim

Sorularınız için issue açabilirsiniz.

---

**Not**: Bu stratejiler finansal tavsiye değildir. Yatırım kararlarınızı kendi araştırmanıza dayanarak alın.

