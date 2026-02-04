# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

BTC-only backtesting research framework for quantitative trading strategy development. Focuses on realistic backtesting with multi-provider data support, walk-forward validation, and proper cost modeling.

## Commands

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Data Commands
```bash
# Download/update OHLCV data
python3 -m src.main update --symbol BTC/USDT --tf 1h --start 2019-01-01 --exchange okx

# With fallback exchanges
python3 -m src.main update --symbol BTC/USDT --tf 1h --start 2019-01-01 --exchange okx --fallback bybit,kucoin

# Live data polling
python3 -m src.main live --symbol BTC/USDT --tf 1h --exchange okx --poll_seconds 60

# Check cached data info
python3 -m src.main info --symbol BTC/USDT --tf 1h

# Data audit
python3 -m src.main audit --symbol BTC/USDT --tf 1h --start 2024-01-01 --end 2025-12-31 --exchange vectorbt
```

### Backtest Commands
```bash
# Single backtest
python3 -m src.main backtest --strategy sma_atr --symbol BTC/USDT --tf 1h --start 2019-01-01 --end 2023-12-31

# Walk-forward analysis
python3 -m src.main walkforward --strategy sma_atr --symbol BTC/USDT --tf 1h --start 2019-01-01 --end 2023-12-31 --train_days 540 --test_days 90 --step_days 30
```

### Linting
```bash
black src/
ruff check src/
```

## Architecture

### Core Flow
1. **Data Layer** (`src/data/`) → Fetch OHLCV from exchanges with fallback
2. **Strategy Layer** (`src/strategies.py`) → Generate signals with indicators
3. **Backtest Engine** (`src/backtest.py`) → Execute trades with realistic costs
4. **Walk-Forward** (`src/walkforward.py`) → Time-series cross-validation

### Data Provider System
The provider system (`src/data/providers/`) uses a fallback chain pattern:
- `manager.py`: Orchestrates multiple providers, auto-switches on failure
- `base.py`: Abstract interface all providers implement (`OHLCVProvider`)
- Concrete providers: `ccxt_provider.py` (exchanges), `vectorbt_provider.py` (Yahoo), `kraken_provider.py`

Provider priority: Primary → Fallback list → VectorBT (rarely blocked)

Symbol mapping happens per-provider (e.g., `BTC/USDT` → `BTC-USD` for Yahoo).

### Execution Model (Critical)
**No look-ahead bias** is enforced:
- Signals evaluated at bar[i] close
- Trades executed at bar[i+1] open
- Stop-loss: Two modes in `stop_fill_mode` parameter:
  - `intrabar`: If bar low breaches stop, fill at stop price
  - `next_open`: If bar close < stop, fill at next bar open

### Cost Model
Volatility-aware slippage: `slippage_bps = max(min_slippage_bps, slippage_k × (ATR/Close) × 10000)`

Cost profiles defined in `backtest.py`:
- `baseline`: fee=0.06%, slippage_k=0.15
- `high`: fee=0.12%, slippage_k=0.30

### Walk-Forward Validation
`src/splitting.py` generates time-series splits maintaining temporal order:
- Train window → Test window (no overlap)
- Warmup bars prepended for indicator calculation
- Dual summary: All splits vs. Filtered (excludes low-sample)

### Strategy Implementation
Strategies extend `BaseStrategy` in `src/strategies.py`:
- Must implement `generate_signals(df) → DataFrame` with columns: `signal`, `stop_price`, `atr`
- Signal values: 1 (long), 0 (flat)
- Currently only `sma_atr` strategy implemented

Filters in `sma_atr`:
1. ADX filter: Only enter when ADX >= threshold (trend strength)
2. Slope filter: Require SMA_slow slope > 0
3. Daily regime filter: Price > SMA200 on daily timeframe

### Output Directories
- `data/`: Cached OHLCV parquet files (gitignored)
- `reports/`: Backtest results, walk-forward reports, audit JSON (gitignored)
- `configs/`: Strategy YAML configurations

### Key Data Structures
- `BacktestResult`: Complete backtest output with trades, equity curve, metrics
- `WalkForwardResult`: Aggregated walk-forward results with dual summaries
- `Split`: Train/test window definition for walk-forward
- `AuditResult`: Data quality metrics (gaps, duplicates, coverage)

## Important Patterns

### Loading Data
Always use `load_ohlcv()` from `src/data_loader.py` - handles timezone (UTC), filtering, and validation.

### Adding New Strategies
1. Create class extending `BaseStrategy` in `src/strategies.py`
2. Implement `generate_signals()` returning DataFrame with required columns
3. Add to `STRATEGIES` registry dict
4. Create config in `configs/strategy_<name>.yaml`

### Provider Health
Providers track consecutive failures. After 3 failures, provider is marked unhealthy and skipped until recovery.
