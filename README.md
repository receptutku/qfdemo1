# BTC Backtesting Research Framework

A clean, reproducible Python framework for BTC-only backtesting research (TRAIN phase) with **multi-provider live data support**.

## Features

- 📊 **Multi-provider data**: OKX, Bybit, Kucoin, Kraken with automatic fallback
- 🔄 **Live data loop**: Continuous polling with health monitoring
- 🧪 **Realistic backtest**: Next-bar execution, fees, slippage, stop-loss
- 📈 **Walk-forward analysis**: Time-series validation across multiple periods
- 💾 **Incremental caching**: Parquet format with safe append

## Project Structure

```
├── src/
│   ├── main.py              # CLI entry point
│   ├── backtest.py          # Backtest engine
│   ├── strategies.py        # Strategy definitions
│   ├── walkforward.py       # Walk-forward runner
│   ├── splitting.py         # Time-series splitter
│   └── data/
│       ├── providers/       # Exchange providers
│       │   ├── base.py      # Provider interface
│       │   ├── ccxt_provider.py
│       │   ├── kraken_provider.py
│       │   └── manager.py   # Fallback manager
│       └── live_fetcher.py  # Update & live loop
├── data/                    # Cached OHLCV (gitignored)
├── reports/                 # Generated outputs (gitignored)
└── configs/                 # Strategy configs
```

## Quick Start

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Download Data (One-shot)

```bash
# Using OKX (default)
python -m src.main update --symbol BTC/USDT --tf 1h --start 2019-01-01 --exchange okx

# With fallback exchanges
python -m src.main update --symbol BTC/USDT --tf 1h --start 2019-01-01 \
    --exchange okx --fallback bybit,kucoin,kraken_direct
```

### 3. Live Data Loop

```bash
# Continuous polling (every 60s)
python -m src.main live --symbol BTC/USDT --tf 1h --exchange okx --poll_seconds 60

# With fallbacks
python -m src.main live --symbol BTC/USDT --tf 1h \
    --exchange okx --fallback bybit,kucoin --poll_seconds 60
```

### 4. Run Backtest

```bash
python -m src.main backtest --strategy sma_atr \
    --symbol BTC/USDT --tf 1h \
    --start 2019-01-01 --end 2023-12-31
```

### 5. Walk-Forward Analysis

```bash
python -m src.main walkforward --strategy sma_atr \
    --symbol BTC/USDT --tf 1h \
    --start 2019-01-01 --end 2023-12-31 \
    --train_days 540 --test_days 90 --step_days 30
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `update` | One-shot fetch/update OHLCV data |
| `live` | Continuous live data polling |
| `backtest` | Run single backtest |
| `walkforward` | Walk-forward analysis |
| `info` | Show cached data info |

### update

```bash
python -m src.main update --symbol BTC/USDT --tf 1h --start 2019-01-01 \
    [--exchange okx] [--fallback bybit,kucoin] [--allow_symbol_fallback]
```

### live

```bash
python -m src.main live --symbol BTC/USDT --tf 1h \
    [--exchange okx] [--fallback bybit,kucoin] [--poll_seconds 60]
```

Output:
```
[14:30:15] Provider: ccxt_okx | Last: 2024-01-15 14:00 UTC | Close: $42,150.00 | Total: 43824 rows
```

## Supported Exchanges

| Exchange | Provider | Status |
|----------|----------|--------|
| OKX | `ccxt_okx` | ✅ Primary |
| Bybit | `ccxt_bybit` | ✅ Fallback |
| Kucoin | `ccxt_kucoin` | ✅ Fallback |
| Kraken | `kraken_direct` | ✅ Non-CCXT fallback |
| Coinbase | `ccxt_coinbase` | ✅ Fallback |

## Strategy: sma_atr

SMA crossover with ATR-based stop-loss:
- Entry: Fast SMA (20) crosses above Slow SMA (50)
- Exit: Crossover reversal OR stop-loss hit
- Stop: Entry price - 2.0 × ATR(14)

Config: `configs/strategy_sma_atr.yaml`

## License

MIT
