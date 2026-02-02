"""
Walk-forward analysis runner.

Runs strategy across multiple time-series splits and produces
per-window metrics plus aggregated stability analysis.
"""

import json
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_ohlcv
from src.splitting import generate_walk_forward_splits, get_split_data, Split
from src.strategies import get_strategy


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass
class SplitMetrics:
    """Metrics for a single walk-forward split."""
    split_index: int
    test_start: str
    test_end: str
    total_return_pct: float  # net_return
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    profit_factor: float
    trades_count: int
    avg_trade_return_pct: float
    final_equity: float
    low_sample_flag: bool  # True if trades < 10
    total_costs: float = 0.0  # fees + slippage for this split
    avg_position_notional: float = 0.0
    avg_risk_used_per_trade: float = 0.0  # fraction of equity


@dataclass
class SplitSummary:
    """Aggregated metrics for a set of splits (all or filtered)."""
    split_count: int
    profitable_splits_pct: float
    pf_above_1_pct: float
    pf_above_1_1_pct: float
    median_return: float
    mean_return: float
    median_sharpe: float
    mean_sharpe: float
    median_win_rate: float
    worst_drawdown: float
    median_trades: float
    stability_score: float
    # Trade-weighted metrics
    weighted_mean_return: float
    weighted_mean_pf: float


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis result with dual summaries."""
    run_id: str
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    train_days: int
    test_days: int
    step_days: int
    min_trades: int
    
    # Per-split results
    split_metrics: List[SplitMetrics]
    
    # Overall counts
    total_splits: int
    low_sample_splits: int
    percent_low_sample_splits: float
    
    # Dual summaries
    all_splits: SplitSummary
    filtered_splits: SplitSummary  # Excludes low_sample
    
    # Stitched equity curve
    stitched_equity: pd.Series



def run_split_backtest(
    df_full: pd.DataFrame,
    split: Split,
    strategy_name: str,
    warmup_bars: int = 200,
    fee_pct: float = 0.0005,
    slippage_bps: float = 5.0,  # Legacy, used if k not set
    daily_df: pd.DataFrame = None,
    sizing_mode: str = "all_in",
    risk_per_trade: float = 0.01,
    max_leverage: float = 1.0,
    slippage_k: float = 0.15,
    min_slippage_bps: float = 1.0,
) -> tuple:
    """
    Run backtest on a single split's test window.
    
    Uses warmup data for indicator calculation but only trades on test period.
    
    Returns:
        (SplitMetrics, trades_list, equity_series)
    """
    from src.backtest import apply_volatility_slippage, Trade
    
    # Get data with warmup
    _, test_df, warmup_df = get_split_data(df_full, split, warmup_bars)
    
    if len(test_df) < 2:
        return None, [], pd.Series()
    
    # Get strategy and prepare regime filter if needed
    strategy = get_strategy(strategy_name)
    
    if daily_df is not None and hasattr(strategy, 'regime_filter_enabled') and strategy.regime_filter_enabled:
        if hasattr(strategy, 'prepare_regime'):
            strategy.prepare_regime(warmup_df, daily_df)
    
    signals_df = strategy.generate_signals(warmup_df)
    
    # Find where test period starts in warmup_df
    test_start_idx = warmup_df.index.get_loc(test_df.index[0])
    
    # Initialize state
    initial_equity = 10000.0
    equity = initial_equity
    position = 0.0
    entry_price = 0.0
    entry_time = None
    stop_price = None
    highest_close = 0.0
    
    # Cost tracking
    total_fees = 0.0
    total_slippage = 0.0
    
    # Position sizing tracking
    position_notionals = []
    risk_amounts_used = [] # as fraction of equity at entry
    
    trades = []
    equity_history = []
    
    # Run simulation only on test period
    for i in range(test_start_idx, len(warmup_df)):
        bar_time = warmup_df.index[i]
        bar_open = warmup_df["open"].iloc[i]
        bar_low = warmup_df["low"].iloc[i]
        bar_close = warmup_df["close"].iloc[i]
        itr_atr = signals_df["atr"].iloc[i]
        
        # Record equity
        current_equity = position * bar_close if position > 0 else equity
        equity_history.append({"timestamp": bar_time, "equity": current_equity})
        
        # Check stop-loss
        if position > 0 and stop_price is not None and bar_low <= stop_price:
            # Volatility slippage
            exit_price, slip_cost = apply_volatility_slippage(
                stop_price, itr_atr, is_buy=False, 
                slippage_k=slippage_k, min_slippage_bps=min_slippage_bps
            )
            exit_price = max(exit_price, bar_low)
            
            total_slippage += slip_cost * position
            
            gross_proceeds = position * exit_price
            fee = gross_proceeds * fee_pct
            total_fees += fee
            net_proceeds = gross_proceeds - fee
            
            pnl = net_proceeds - (position * entry_price)
            return_pct = (exit_price / entry_price - 1) * 100
            duration = (bar_time - entry_time).total_seconds() / 3600
            
            trades.append(Trade(
                entry_time=entry_time, entry_price=entry_price,
                exit_time=bar_time, exit_price=exit_price,
                position_size=position, pnl=pnl,
                return_pct=return_pct, duration_hours=duration,
                exit_type="stop"
            ))
            
            equity = net_proceeds
            position = 0.0
            entry_price = 0.0
            entry_time = None
            stop_price = None
            equity_history[-1]["equity"] = equity
        
        # Update trailing stop
        if position > 0 and strategy.trailing_stop:
            highest_close = max(highest_close, bar_close)
            atr = signals_df["atr"].iloc[i]
            if not pd.isna(atr):
                trailing = highest_close - strategy.atr_multiplier * atr
                stop_price = max(stop_price, trailing) if stop_price else trailing
        
        # Skip last bar
        if i >= len(warmup_df) - 1:
            continue
        
        # Signals
        signal = signals_df["signal"].iloc[i]
        prev_signal = signals_df["signal"].iloc[i-1] if i > test_start_idx else 0
        
        next_open = warmup_df["open"].iloc[i + 1]
        next_time = warmup_df.index[i + 1]
        
        # Signal change
        if signal != prev_signal:
            # EXT & ENTRY LOGIC
            
            # Exit
            if position > 0 and signal == 0:
                exit_price, slip_cost = apply_volatility_slippage(
                    next_open, itr_atr, is_buy=False, 
                    slippage_k=slippage_k, min_slippage_bps=min_slippage_bps
                )
                
                total_slippage += slip_cost * position
                gross_proceeds = position * exit_price
                fee = gross_proceeds * fee_pct
                total_fees += fee
                net_proceeds = gross_proceeds - fee
                
                pnl = net_proceeds - (position * entry_price)
                return_pct = (exit_price / entry_price - 1) * 100
                duration = (next_time - entry_time).total_seconds() / 3600
                
                trades.append(Trade(
                    entry_time=entry_time, entry_price=entry_price,
                    exit_time=next_time, exit_price=exit_price,
                    position_size=position, pnl=pnl,
                    return_pct=return_pct, duration_hours=duration,
                    exit_type="signal"
                ))
                
                equity = net_proceeds
                position = 0.0
                entry_price = 0.0
                entry_time = None
                stop_price = None
            
            # Entry
            if signal == 1 and position == 0:
                exec_price, entry_slip_cost = apply_volatility_slippage(
                    next_open, itr_atr, is_buy=True, 
                    slippage_k=slippage_k, min_slippage_bps=min_slippage_bps
                )
                
                # SIZING LOGIC
                position_size_btc = 0.0
                pos_val = 0.0
                
                if sizing_mode == "risk_per_trade":
                    if not pd.isna(itr_atr) and itr_atr > 0:
                        stop_dist = strategy.atr_multiplier * itr_atr
                        if stop_dist > 0:
                            risk_amt = equity * risk_per_trade
                            max_notional = equity * max_leverage
                            
                            position_size_btc = risk_amt / stop_dist
                            position_val = position_size_btc * exec_price
                            if position_val > max_notional:
                                position_val = max_notional
                                position_size_btc = position_val / exec_price
                            
                            fee = position_val * fee_pct
                            if position_val + fee > equity:
                                cash = equity
                                position_val = cash / (1 + fee_pct)
                                fee = cash - position_val
                                position_size_btc = position_val / exec_price
                                pos_val = position_val
                            else:
                                pos_val = position_val
                else:
                    # All in
                    trade_equity = equity
                    fee = trade_equity * fee_pct
                    available = trade_equity - fee
                    position_size_btc = available / exec_price
                    pos_val = available

                if position_size_btc > 0:
                    position = position_size_btc
                    total_fees += fee
                    total_slippage += entry_slip_cost * position
                    
                    entry_price = exec_price
                    entry_time = next_time
                    highest_close = bar_close
                    
                    # Stop logic
                    stop_sig = signals_df["stop_price"].iloc[i]
                    if not pd.isna(stop_sig):
                         if not pd.isna(itr_atr):
                             stop_price = exec_price - strategy.atr_multiplier * itr_atr
                         else:
                             stop_price = stop_sig
                    else:
                        stop_price = None
                    
                    # Metrics tracking
                    position_notionals.append(pos_val)
                    if stop_price and stop_price > 0:
                        dist = exec_price - stop_price
                        risk_usd = position * dist
                        risk_pct = risk_usd / equity if equity > 0 else 0
                        risk_amounts_used.append(risk_pct)
                    else:
                        risk_amounts_used.append(0.0)
                    
                    equity -= (pos_val + fee)
    
    # Close remaining position
    if position > 0:
        final_close = warmup_df["close"].iloc[-1]
        final_atr = signals_df["atr"].iloc[-1]
        exit_price, slip_cost = apply_volatility_slippage(
            final_close, final_atr, is_buy=False,
            slippage_k=slippage_k, min_slippage_bps=min_slippage_bps
        )
        
        # Track slippage cost
        total_slippage += slip_cost * position
        
        gross_proceeds = position * exit_price
        fee = gross_proceeds * fee_pct
        total_fees += fee
        net_proceeds = gross_proceeds - fee
        
        pnl = net_proceeds - (position * entry_price)
        return_pct = (exit_price / entry_price - 1) * 100
        duration = (warmup_df.index[-1] - entry_time).total_seconds() / 3600
        
        trades.append(Trade(
            entry_time=entry_time, entry_price=entry_price,
            exit_time=warmup_df.index[-1], exit_price=exit_price,
            position_size=position, pnl=pnl,
            return_pct=return_pct, duration_hours=duration,
            exit_type="end"
        ))
        equity = net_proceeds
    
    # Calculate total costs
    total_costs = total_fees + total_slippage
    
    final_equity = equity if position == 0 else position * warmup_df["close"].iloc[-1]
    
    # Build equity series
    if equity_history:
        eq_df = pd.DataFrame(equity_history).set_index("timestamp")
        equity_series = eq_df["equity"]
        equity_series.iloc[-1] = final_equity
    else:
        equity_series = pd.Series()
    
    # Calculate metrics
    total_return = (final_equity / initial_equity - 1) * 100
    
    # CAGR
    if len(equity_series) > 1:
        days = (equity_series.index[-1] - equity_series.index[0]).total_seconds() / 86400
        years = days / 365.25
        cagr = ((final_equity / initial_equity) ** (1/years) - 1) * 100 if years > 0 else 0
    else:
        cagr = 0
    
    # Max Drawdown
    if len(equity_series) > 0:
        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak * 100
        max_dd = dd.min()
    else:
        max_dd = 0
    
    # Sharpe
    if len(equity_series) > 1:
        returns = equity_series.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(8760) if returns.std() > 0 else 0
    else:
        sharpe = 0
    
    # Trade metrics
    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) * 100
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
        avg_ret = np.mean([t.return_pct for t in trades])
    else:
        win_rate = pf = avg_ret = 0
    
    metrics = SplitMetrics(
        split_index=split.index,
        test_start=str(split.test_start.date()),
        test_end=str(split.test_end.date()),
        total_return_pct=round(total_return, 2),
        cagr_pct=round(cagr, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=round(sharpe, 3),
        win_rate_pct=round(win_rate, 1),
        profit_factor=round(pf, 3) if pf < 999 else 999.0,
        trades_count=len(trades),
        avg_trade_return_pct=round(avg_ret, 3),
        final_equity=round(final_equity, 2),
        low_sample_flag=len(trades) < 10,
        total_costs=round(total_costs, 2),
        avg_position_notional=round(sum(position_notionals)/len(position_notionals) if position_notionals else 0.0, 2),
        avg_risk_used_per_trade=round(sum(risk_amounts_used)/len(risk_amounts_used) if risk_amounts_used else 0.0, 4),
    )
    
    return metrics, trades, equity_series


def _calculate_split_summary(metrics_list: List[SplitMetrics]) -> SplitSummary:
    """Calculate aggregated summary for a list of split metrics."""
    if not metrics_list:
        return SplitSummary(
            split_count=0,
            profitable_splits_pct=0.0,
            pf_above_1_pct=0.0,
            pf_above_1_1_pct=0.0,
            median_return=0.0,
            mean_return=0.0,
            median_sharpe=0.0,
            mean_sharpe=0.0,
            median_win_rate=0.0,
            worst_drawdown=0.0,
            median_trades=0.0,
            stability_score=0.0,
            weighted_mean_return=0.0,
            weighted_mean_pf=0.0,
        )
    
    returns = [m.total_return_pct for m in metrics_list]
    sharpes = [m.sharpe_ratio for m in metrics_list]
    pfs = [m.profit_factor for m in metrics_list if m.profit_factor < 999]
    win_rates = [m.win_rate_pct for m in metrics_list]
    drawdowns = [m.max_drawdown_pct for m in metrics_list]
    trades_counts = [m.trades_count for m in metrics_list]
    
    n = len(metrics_list)
    profitable = sum(1 for r in returns if r > 0)
    pf_above_1 = sum(1 for m in metrics_list if m.profit_factor > 1.0)
    pf_above_1_1 = sum(1 for m in metrics_list if m.profit_factor > 1.1)
    
    percent_profitable = (profitable / n * 100) if n > 0 else 0
    median_return = np.median(returns) if returns else 0
    worst_dd = min(drawdowns) if drawdowns else 0
    median_trades = np.median(trades_counts) if trades_counts else 0
    
    # Stability score
    if abs(worst_dd) > 0:
        stability_score = percent_profitable * (median_return / abs(worst_dd))
    else:
        stability_score = percent_profitable * median_return / 100 if median_return > 0 else 0
    
    # Trade-weighted metrics
    total_trades = sum(trades_counts) if trades_counts else 0
    if total_trades > 0:
        weighted_return = sum(m.total_return_pct * m.trades_count for m in metrics_list) / total_trades
        # For weighted PF, only include splits with valid PF
        valid_pf_metrics = [m for m in metrics_list if m.profit_factor < 999 and m.trades_count > 0]
        if valid_pf_metrics:
            weighted_pf = sum(m.profit_factor * m.trades_count for m in valid_pf_metrics) / sum(m.trades_count for m in valid_pf_metrics)
        else:
            weighted_pf = 0
    else:
        weighted_return = 0
        weighted_pf = 0
    
    return SplitSummary(
        split_count=n,
        profitable_splits_pct=round(percent_profitable, 1),
        pf_above_1_pct=round(pf_above_1 / n * 100, 1) if n > 0 else 0,
        pf_above_1_1_pct=round(pf_above_1_1 / n * 100, 1) if n > 0 else 0,
        median_return=round(median_return, 2),
        mean_return=round(np.mean(returns), 2) if returns else 0,
        median_sharpe=round(np.median(sharpes), 3) if sharpes else 0,
        mean_sharpe=round(np.mean(sharpes), 3) if sharpes else 0,
        median_win_rate=round(np.median(win_rates), 1) if win_rates else 0,
        worst_drawdown=round(worst_dd, 2),
        median_trades=round(median_trades, 1),
        stability_score=round(stability_score, 3),
        weighted_mean_return=round(weighted_return, 2),
        weighted_mean_pf=round(weighted_pf, 3),
    )


def run_walk_forward(
    strategy_name: str,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = "2019-01-01",
    end_date: str = "2023-12-31",
    train_days: int = 240,
    test_days: int = 90,
    step_days: int = 30,
    warmup_bars: int = 200,
    fee_pct: float = 0.0005,
    slippage_bps: float = 5.0,
    wf_mode: str = "fixed_params",
    min_trades: int = 10,
    save_reports: bool = True,
    sizing_mode: str = "all_in",
    risk_per_trade: float = 0.01,
    max_leverage: float = 1.0,
    slippage_k: float = 0.15,
    min_slippage_bps: float = 1.0,
) -> WalkForwardResult:
    """
    Run walk-forward analysis across multiple time periods.
    
    wf_mode:
    - "fixed_params": Use config parameters as-is, no tuning.
    - "train_then_test": Tune params on train window, evaluate on test.
    
    min_trades: Minimum trades per split; below this is flagged as low_sample.
    
    Returns aggregated results and saves detailed reports.
    """
    # Generate run ID
    config_str = f"{strategy_name}_{symbol}_{timeframe}_{train_days}_{test_days}_{step_days}_{wf_mode}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config_hash
    
    # Load data
    print(f"[WalkForward] Loading data...")
    df = load_ohlcv(symbol=symbol, timeframe=timeframe, data_dir=DATA_DIR,
                    start_date=start_date, end_date=end_date)
    print(f"[WalkForward] Loaded {len(df)} candles")
    print(f"[WalkForward] Mode: {wf_mode}, Min Trades: {min_trades}")
    
    # Load daily data for regime filter
    daily_df = None
    try:
        daily_df = load_ohlcv(symbol=symbol, timeframe="1d", data_dir=DATA_DIR,
                              start_date=start_date, end_date=end_date)
        print(f"[WalkForward] Loaded {len(daily_df)} daily candles for regime filter")
    except FileNotFoundError:
        print("[WalkForward] Daily data not found, regime filter will be disabled")
    
    # Generate splits
    splits = generate_walk_forward_splits(
        df, train_days=train_days, test_days=test_days, 
        step_days=step_days, warmup_bars=warmup_bars
    )
    print(f"[WalkForward] Generated {len(splits)} splits")
    
    if not splits:
        raise ValueError("No valid splits generated")
    
    # Run backtest on each split
    all_metrics = []
    all_trades = []
    all_equities = []
    selected_params_list = []  # For train_then_test mode
    
    for split in splits:
        if wf_mode == "train_then_test":
            # TRAIN_THEN_TEST: Select best params on train, evaluate on test
            best_params = select_params_on_train(df, split, strategy_name, warmup_bars, fee_pct, slippage_bps)
            selected_params_list.append({"split": split.index, **best_params})
            
            # Run test with selected params
            metrics, trades, equity = run_split_backtest_with_params(
                df, split, strategy_name, best_params, warmup_bars, fee_pct, slippage_bps
            )
        else:
            # FIXED_PARAMS: Use default strategy params
            metrics, trades, equity = run_split_backtest(
                df, split, strategy_name, warmup_bars, fee_pct, slippage_bps, 
                daily_df=daily_df,
                sizing_mode=sizing_mode,
                risk_per_trade=risk_per_trade,
                max_leverage=max_leverage,
                slippage_k=slippage_k,
                min_slippage_bps=min_slippage_bps,
            )
        
        if metrics:
            # Update low_sample_flag based on min_trades parameter
            metrics.low_sample_flag = metrics.trades_count < min_trades
            all_metrics.append(metrics)
            all_trades.append((split.index, trades))
            all_equities.append((split.index, equity))
    
    # Separate splits into all and filtered (non-low-sample)
    filtered_metrics = [m for m in all_metrics if not m.low_sample_flag]
    low_sample_count = sum(1 for m in all_metrics if m.low_sample_flag)
    
    # Calculate dual summaries
    all_splits_summary = _calculate_split_summary(all_metrics)
    filtered_splits_summary = _calculate_split_summary(filtered_metrics)
    
    # Build stitched equity curve (normalized, sequential)
    stitched = []
    current_equity = 1.0
    for idx, equity in all_equities:
        if len(equity) > 0:
            # Normalize this split's equity to start at current_equity
            split_return = equity.iloc[-1] / equity.iloc[0] if equity.iloc[0] > 0 else 1
            for ts, val in equity.items():
                normalized = current_equity * (val / equity.iloc[0]) if equity.iloc[0] > 0 else current_equity
                stitched.append({"timestamp": ts, "equity": normalized})
            current_equity = current_equity * split_return
    
    stitched_equity = pd.DataFrame(stitched).set_index("timestamp")["equity"] if stitched else pd.Series()
    
    result = WalkForwardResult(
        run_id=run_id,
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        min_trades=min_trades,
        split_metrics=all_metrics,
        total_splits=len(all_metrics),
        low_sample_splits=low_sample_count,
        percent_low_sample_splits=round(low_sample_count / len(all_metrics) * 100, 1) if all_metrics else 0,
        all_splits=all_splits_summary,
        filtered_splits=filtered_splits_summary,
        stitched_equity=stitched_equity,
    )
    
    if save_reports:
        save_walk_forward_reports(result, all_trades, all_equities, splits, wf_mode, selected_params_list)
    
    return result


def select_params_on_train(
    df_full: pd.DataFrame,
    split: Split,
    strategy_name: str,
    warmup_bars: int,
    fee_pct: float,
    slippage_bps: float,
) -> dict:
    """
    Select best params on train window using small predefined grid.
    Returns dict of best params.
    """
    from src.strategies import SMACrossATRStrategy
    
    # Small predefined grid for SMA-ATR strategy
    PARAM_GRID = [
        {"fast_period": 20, "slow_period": 50},
        {"fast_period": 30, "slow_period": 80},
        {"fast_period": 40, "slow_period": 100},  # Current default
        {"fast_period": 50, "slow_period": 120},
    ]
    
    # Get train data with warmup
    train_df, _, _ = get_split_data(df_full, split, warmup_bars)
    
    # Prepend warmup bars to train
    train_start_idx = df_full.index.get_loc(split.train_start)
    warmup_start_idx = max(0, train_start_idx - warmup_bars)
    full_train_df = df_full.iloc[warmup_start_idx:df_full.index.get_loc(split.train_end) + 1]
    
    best_sharpe = -999
    best_params = PARAM_GRID[0]
    
    for params in PARAM_GRID:
        # Create temporary strategy with these params
        strategy = SMACrossATRStrategy(**params)
        
        # Run quick backtest on train window (simplified)
        signals = strategy.generate_signals(full_train_df)
        
        # Find train start index in full_train_df
        train_actual_start = full_train_df.index.get_loc(split.train_start) if split.train_start in full_train_df.index else 0
        
        # Very simplified return calculation (just count winning signals)
        signal_changes = signals["signal"].diff().fillna(0)
        entries = signal_changes[train_actual_start:] == 1
        
        # Count crossovers as proxy for activity
        n_entries = entries.sum()
        
        # Calculate simple return based on trend
        close = full_train_df["close"]
        if n_entries > 0:
            # Simple metric: correlation of signal with next-bar return
            next_ret = close.pct_change().shift(-1)
            aligned_ret = next_ret[train_actual_start:]
            aligned_sig = signals["signal"][train_actual_start:]
            
            # Calculate rough Sharpe proxy
            strategy_ret = aligned_ret * aligned_sig
            if strategy_ret.std() > 0:
                sharpe_proxy = strategy_ret.mean() / strategy_ret.std() * np.sqrt(8760)
            else:
                sharpe_proxy = 0
        else:
            sharpe_proxy = 0
        
        if sharpe_proxy > best_sharpe:
            best_sharpe = sharpe_proxy
            best_params = params
    
    return best_params


def run_split_backtest_with_params(
    df_full: pd.DataFrame,
    split: Split,
    strategy_name: str,
    params: dict,
    warmup_bars: int,
    fee_pct: float,
    slippage_bps: float,
) -> tuple:
    """
    Run backtest with specific strategy params.
    Similar to run_split_backtest but uses provided params.
    """
    from src.backtest import apply_slippage, Trade
    from src.strategies import SMACrossATRStrategy
    
    # Get data with warmup
    _, test_df, warmup_df = get_split_data(df_full, split, warmup_bars)
    
    if len(test_df) < 2:
        return None, [], pd.Series()
    
    # Create strategy with selected params
    strategy = SMACrossATRStrategy(**params)
    signals_df = strategy.generate_signals(warmup_df)
    
    # Find where test period starts in warmup_df
    test_start_idx = warmup_df.index.get_loc(test_df.index[0])
    
    # Initialize state
    initial_equity = 10000.0
    equity = initial_equity
    position = 0.0
    entry_price = 0.0
    entry_time = None
    stop_price = None
    highest_close = 0.0
    
    trades = []
    equity_history = []
    
    # Run simulation only on test period (same as run_split_backtest)
    for i in range(test_start_idx, len(warmup_df)):
        bar_time = warmup_df.index[i]
        bar_open = warmup_df["open"].iloc[i]
        bar_low = warmup_df["low"].iloc[i]
        bar_close = warmup_df["close"].iloc[i]
        
        current_equity = position * bar_close if position > 0 else equity
        equity_history.append({"timestamp": bar_time, "equity": current_equity})
        
        # Stop-loss check
        if position > 0 and stop_price is not None and bar_low <= stop_price:
            exit_price = apply_slippage(stop_price, slippage_bps, is_buy=False)
            exit_price = max(exit_price, bar_low)
            
            gross_proceeds = position * exit_price
            fee = gross_proceeds * fee_pct
            net_proceeds = gross_proceeds - fee
            
            pnl = net_proceeds - (position * entry_price)
            return_pct = (exit_price / entry_price - 1) * 100
            duration = (bar_time - entry_time).total_seconds() / 3600
            
            trades.append(Trade(
                entry_time=entry_time, entry_price=entry_price,
                exit_time=bar_time, exit_price=exit_price,
                position_size=position, pnl=pnl,
                return_pct=return_pct, duration_hours=duration,
                exit_type="stop"
            ))
            
            equity = net_proceeds
            position = 0.0
            entry_price = 0.0
            entry_time = None
            stop_price = None
            equity_history[-1]["equity"] = equity
        
        # Trailing stop
        if position > 0 and strategy.trailing_stop:
            highest_close = max(highest_close, bar_close)
            atr = signals_df["atr"].iloc[i]
            if not pd.isna(atr):
                trailing = highest_close - strategy.atr_multiplier * atr
                stop_price = max(stop_price, trailing) if stop_price else trailing
        
        if i >= len(warmup_df) - 1:
            continue
        
        signal = signals_df["signal"].iloc[i]
        prev_signal = signals_df["signal"].iloc[i-1] if i > test_start_idx else 0
        
        next_open = warmup_df["open"].iloc[i + 1]
        next_time = warmup_df.index[i + 1]
        
        if signal != prev_signal:
            if position > 0 and signal == 0:
                exit_price = apply_slippage(next_open, slippage_bps, is_buy=False)
                gross_proceeds = position * exit_price
                fee = gross_proceeds * fee_pct
                net_proceeds = gross_proceeds - fee
                
                pnl = net_proceeds - (position * entry_price)
                return_pct = (exit_price / entry_price - 1) * 100
                duration = (next_time - entry_time).total_seconds() / 3600
                
                trades.append(Trade(
                    entry_time=entry_time, entry_price=entry_price,
                    exit_time=next_time, exit_price=exit_price,
                    position_size=position, pnl=pnl,
                    return_pct=return_pct, duration_hours=duration,
                    exit_type="signal"
                ))
                
                equity = net_proceeds
                position = 0.0
                entry_price = 0.0
                entry_time = None
                stop_price = None
            
            if signal == 1 and position == 0:
                exec_price = apply_slippage(next_open, slippage_bps, is_buy=True)
                fee = equity * fee_pct
                available = equity - fee
                
                position = available / exec_price
                entry_price = exec_price
                entry_time = next_time
                highest_close = bar_close
                
                atr = signals_df["atr"].iloc[i]
                if not pd.isna(atr):
                    stop_price = exec_price - strategy.atr_multiplier * atr
                
                equity = 0.0
    
    # Close remaining position
    if position > 0:
        final_close = warmup_df["close"].iloc[-1]
        exit_price = apply_slippage(final_close, slippage_bps, is_buy=False)
        gross_proceeds = position * exit_price
        fee = gross_proceeds * fee_pct
        net_proceeds = gross_proceeds - fee
        
        pnl = net_proceeds - (position * entry_price)
        return_pct = (exit_price / entry_price - 1) * 100
        duration = (warmup_df.index[-1] - entry_time).total_seconds() / 3600
        
        trades.append(Trade(
            entry_time=entry_time, entry_price=entry_price,
            exit_time=warmup_df.index[-1], exit_price=exit_price,
            position_size=position, pnl=pnl,
            return_pct=return_pct, duration_hours=duration,
            exit_type="end"
        ))
        equity = net_proceeds
    
    final_equity = equity if position == 0 else position * warmup_df["close"].iloc[-1]
    
    # Build equity series
    if equity_history:
        eq_df = pd.DataFrame(equity_history).set_index("timestamp")
        equity_series = eq_df["equity"]
        equity_series.iloc[-1] = final_equity
    else:
        equity_series = pd.Series()
    
    # Calculate metrics (same as run_split_backtest)
    total_return = (final_equity / initial_equity - 1) * 100
    
    if len(equity_series) > 1:
        days = (equity_series.index[-1] - equity_series.index[0]).total_seconds() / 86400
        years = days / 365.25
        cagr = ((final_equity / initial_equity) ** (1/years) - 1) * 100 if years > 0 else 0
    else:
        cagr = 0
    
    if len(equity_series) > 0:
        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak * 100
        max_dd = dd.min()
    else:
        max_dd = 0
    
    if len(equity_series) > 1:
        returns = equity_series.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(8760) if returns.std() > 0 else 0
    else:
        sharpe = 0
    
    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) * 100
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
        avg_ret = np.mean([t.return_pct for t in trades])
    else:
        win_rate = pf = avg_ret = 0
    
    metrics = SplitMetrics(
        split_index=split.index,
        test_start=str(split.test_start.date()),
        test_end=str(split.test_end.date()),
        total_return_pct=round(total_return, 2),
        cagr_pct=round(cagr, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=round(sharpe, 3),
        win_rate_pct=round(win_rate, 1),
        profit_factor=round(pf, 3) if pf < 999 else 999.0,
        trades_count=len(trades),
        avg_trade_return_pct=round(avg_ret, 3),
        final_equity=round(final_equity, 2),
        low_sample_flag=len(trades) < 10,
    )
    
    return metrics, trades, equity_series


def save_walk_forward_reports(
    result: WalkForwardResult,
    all_trades: list,
    all_equities: list,
    splits: List[Split],
    wf_mode: str = "fixed_params",
    selected_params_list: list = None,
) -> Path:
    """Save all walk-forward reports."""
    report_dir = REPORTS_DIR / "walkforward" / result.run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Save selected params if train_then_test mode
    if wf_mode == "train_then_test" and selected_params_list:
        pd.DataFrame(selected_params_list).to_csv(report_dir / "selected_params.csv", index=False)
    
    # Per-split reports
    for metrics in result.split_metrics:
        split_dir = report_dir / f"split_{metrics.split_index:02d}"
        split_dir.mkdir(exist_ok=True)
        
        # metrics.json
        with open(split_dir / "metrics.json", "w") as f:
            json.dump(asdict(metrics), f, indent=2)
        
        # trades.csv
        for idx, trades in all_trades:
            if idx == metrics.split_index and trades:
                trades_data = [{
                    "entry_time": str(t.entry_time),
                    "entry_price": round(t.entry_price, 2),
                    "exit_time": str(t.exit_time),
                    "exit_price": round(t.exit_price, 2),
                    "pnl": round(t.pnl, 2),
                    "return_pct": round(t.return_pct, 3),
                    "exit_type": t.exit_type,
                } for t in trades]
                pd.DataFrame(trades_data).to_csv(split_dir / "trades.csv", index=False)
        
        # equity.csv
        for idx, equity in all_equities:
            if idx == metrics.split_index and len(equity) > 0:
                eq_df = equity.reset_index()
                eq_df.columns = ["timestamp", "equity"]
                eq_df.to_csv(split_dir / "equity.csv", index=False)
    
    # Aggregated reports
    # 1. splits.csv
    splits_data = [{
        "split_index": s.index,
        "train_start": str(s.train_start.date()),
        "train_end": str(s.train_end.date()),
        "test_start": str(s.test_start.date()),
        "test_end": str(s.test_end.date()),
    } for s in splits]
    pd.DataFrame(splits_data).to_csv(report_dir / "splits.csv", index=False)
    
    # 2. metrics_by_split.csv
    metrics_data = [asdict(m) for m in result.split_metrics]
    pd.DataFrame(metrics_data).to_csv(report_dir / "metrics_by_split.csv", index=False)
    
    # 3. summary.json
    summary = {
        "run_id": result.run_id,
        "strategy": result.strategy_name,
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "period": f"{result.start_date} to {result.end_date}",
        "config": {
            "train_days": result.train_days,
            "test_days": result.test_days,
            "step_days": result.step_days,
            "min_trades": result.min_trades,
        },
        "total_splits": result.total_splits,
        "low_sample_splits": result.low_sample_splits,
        "percent_low_sample_splits": result.percent_low_sample_splits,
        "filtered_percent_profitable": result.filtered_splits.profitable_splits_pct,
        "all_splits": asdict(result.all_splits),
        "filtered_splits": asdict(result.filtered_splits),
    }
    with open(report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # 4. stability.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    splits_idx = [m.split_index for m in result.split_metrics]
    
    # Return by split
    ax1 = axes[0, 0]
    returns = [m.total_return_pct for m in result.split_metrics]
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax1.bar(splits_idx, returns, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=result.all_splits.median_return, color='blue', linestyle='--', label=f'Median: {result.all_splits.median_return:.1f}%')
    ax1.set_title("Return by Split")
    ax1.set_xlabel("Split Index")
    ax1.set_ylabel("Return (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sharpe by split
    ax2 = axes[0, 1]
    sharpes = [m.sharpe_ratio for m in result.split_metrics]
    colors = ['green' if s > 0 else 'red' for s in sharpes]
    ax2.bar(splits_idx, sharpes, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=result.all_splits.median_sharpe, color='blue', linestyle='--', label=f'Median: {result.all_splits.median_sharpe:.2f}')
    ax2.set_title("Sharpe Ratio by Split")
    ax2.set_xlabel("Split Index")
    ax2.set_ylabel("Sharpe")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Drawdown by split
    ax3 = axes[1, 0]
    drawdowns = [m.max_drawdown_pct for m in result.split_metrics]
    ax3.bar(splits_idx, drawdowns, color='red', alpha=0.7)
    ax3.set_title("Max Drawdown by Split")
    ax3.set_xlabel("Split Index")
    ax3.set_ylabel("Max Drawdown (%)")
    ax3.grid(True, alpha=0.3)
    
    # Profit Factor by split
    ax4 = axes[1, 1]
    pfs = [min(m.profit_factor, 5) for m in result.split_metrics]  # Cap at 5 for visualization
    colors = ['green' if p > 1 else 'red' for p in pfs]
    ax4.bar(splits_idx, pfs, color=colors, alpha=0.7)
    ax4.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_title("Profit Factor by Split (capped at 5)")
    ax4.set_xlabel("Split Index")
    ax4.set_ylabel("Profit Factor")
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f"Walk-Forward Stability: {result.strategy_name} | {result.symbol}", fontsize=12)
    plt.tight_layout()
    plt.savefig(report_dir / "stability.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. equity_all_splits.png
    if len(result.stitched_equity) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(result.stitched_equity.index, result.stitched_equity.values, 'b-', linewidth=1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f"Stitched Test Equity: {result.strategy_name} | Walk-Forward")
        ax.set_ylabel("Normalized Equity (start=1.0)")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        
        # Add final return annotation
        final_return = (result.stitched_equity.iloc[-1] - 1) * 100
        ax.text(0.02, 0.98, f"Total Return: {final_return:.1f}%", 
                transform=ax.transAxes, va='top', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(report_dir / "equity_all_splits.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"[WalkForward] Reports saved: {report_dir}")
    return report_dir


def print_walk_forward_summary(result: WalkForwardResult):
    """Print formatted walk-forward summary with dual summaries."""
    print("\n" + "=" * 100)
    print(f"WALK-FORWARD RESULTS: {result.strategy_name}")
    print("=" * 100)
    print(f"Symbol: {result.symbol} ({result.timeframe})")
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Config: Train={result.train_days}d, Test={result.test_days}d, Step={result.step_days}d, MinTrades={result.min_trades}")
    print("-" * 100)
    
    # Per-split table
    print(f"{'Split':>5} | {'Test Period':>25} | {'Return':>8} | {'MaxDD':>8} | {'Sharpe':>7} | {'WinRate':>7} | {'PF':>6} | {'Trades':>6} | {'Costs':>8}")
    print("-" * 100)
    
    for m in result.split_metrics:
        flag = "*" if m.low_sample_flag else " "
        print(f"{m.split_index:>5}{flag}| {m.test_start} to {m.test_end} | {m.total_return_pct:>7.1f}% | "
              f"{m.max_drawdown_pct:>7.1f}% | {m.sharpe_ratio:>7.2f} | {m.win_rate_pct:>6.1f}% | "
              f"{m.profit_factor:>6.2f} | {m.trades_count:>6} | ${m.total_costs:>7.2f}")
    
    print("-" * 100)
    print(f"* = low sample (< {result.min_trades} trades)")
    print("-" * 100)
    
    # Summary counts
    print(f"\nTotal Splits: {result.total_splits} | Low Sample: {result.low_sample_splits} ({result.percent_low_sample_splits:.1f}%)")
    
    # Helper to print summary
    def print_summary(name: str, s):
        print(f"\n{'=' * 100}")
        print(f"{name.upper()} (n={s.split_count})")
        print("=" * 100)
        print(f"Profitable Splits:   {s.profitable_splits_pct:.1f}%")
        print(f"PF > 1.0:            {s.pf_above_1_pct:.1f}%")
        print(f"PF > 1.1:            {s.pf_above_1_1_pct:.1f}%")
        print("-" * 100)
        print(f"Median Return:       {s.median_return:.2f}%")
        print(f"Mean Return:         {s.mean_return:.2f}%")
        print(f"Weighted Mean Ret:   {s.weighted_mean_return:.2f}%")
        print("-" * 100)
        print(f"Median Sharpe:       {s.median_sharpe:.3f}")
        print(f"Mean Sharpe:         {s.mean_sharpe:.3f}")
        print(f"Median Win Rate:     {s.median_win_rate:.1f}%")
        print("-" * 100)
        print(f"Worst Drawdown:      {s.worst_drawdown:.2f}%")
        print(f"Median Trades:       {s.median_trades:.0f}")
        print(f"Weighted Mean PF:    {s.weighted_mean_pf:.3f}")
        print("-" * 100)
        print(f"STABILITY SCORE:     {s.stability_score:.3f}")
    
    # Print both summaries
    print_summary("ALL SPLITS", result.all_splits)
    print_summary("FILTERED SPLITS (excludes low sample)", result.filtered_splits)
    
    print("\n" + "=" * 100)
