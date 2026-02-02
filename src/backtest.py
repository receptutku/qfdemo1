"""
Minimal, realistic backtesting engine.

Features:
- Signal evaluation at bar close, execution at next bar open (no look-ahead)
- Long-only support with ATR-based stop-loss
- Fee and slippage model
- Configurable position sizing
- Stop-loss execution: if low breaches stop, exit at stop_price + slippage
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_ohlcv


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Cost profiles
COST_PROFILES = {
    "baseline": {"fee_pct": 0.0006, "slippage_k": 0.15, "min_slippage_bps": 1.0},
    "high": {"fee_pct": 0.0012, "slippage_k": 0.30, "min_slippage_bps": 2.0},
}


@dataclass
class Trade:
    """Single trade record."""
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    position_size: float  # BTC quantity
    pnl: float  # USD profit/loss
    return_pct: float  # Percentage return
    duration_hours: float
    exit_type: str  # 'signal' or 'stop'


@dataclass
class BacktestResult:
    """Complete backtest result."""
    # Metadata
    run_id: str
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    
    # Parameters
    initial_capital: float
    fee_pct: float
    slippage_bps: float
    position_size_pct: float
    
    # Results
    final_equity: float
    trades: List[Trade]
    equity_curve: pd.Series
    
    # Metrics
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    avg_trade_return_pct: float
    avg_trade_duration_hours: float
    
    # Cost tracking
    total_fees_paid: float = 0.0
    total_slippage_cost: float = 0.0
    cost_as_pct_of_gross_pnl: float = 0.0


def apply_slippage(price: float, slippage_bps: float, is_buy: bool) -> float:
    """Apply constant slippage to price (legacy)."""
    slippage_pct = slippage_bps / 10000
    if is_buy:
        return price * (1 + slippage_pct)
    else:
        return price * (1 - slippage_pct)


def calculate_volatility_slippage(
    price: float, 
    atr: float, 
    slippage_k: float = 0.15,
    min_slippage_bps: float = 1.0,
) -> float:
    """
    Calculate volatility-aware slippage in bps.
    
    Formula: slippage_bps = max(min_slippage_bps, slippage_k * (ATR / Close) * 10_000)
    
    Args:
        price: Current price
        atr: ATR value at current bar
        slippage_k: Slippage constant (default 0.15)
        min_slippage_bps: Minimum slippage in bps (default 1.0)
    
    Returns:
        Slippage in basis points
    """
    if pd.isna(atr) or atr <= 0 or price <= 0:
        return min_slippage_bps
    
    vol_slippage_bps = slippage_k * (atr / price) * 10000
    return max(min_slippage_bps, vol_slippage_bps)


def apply_volatility_slippage(
    price: float, 
    atr: float, 
    is_buy: bool,
    slippage_k: float = 0.15,
    min_slippage_bps: float = 1.0,
) -> tuple:
    """
    Apply volatility-aware slippage to price.
    
    Returns:
        (adjusted_price, slippage_cost_per_unit)
    """
    slippage_bps = calculate_volatility_slippage(price, atr, slippage_k, min_slippage_bps)
    slippage_pct = slippage_bps / 10000
    
    if is_buy:
        adjusted_price = price * (1 + slippage_pct)
        slippage_cost = price * slippage_pct  # Cost per unit
    else:
        adjusted_price = price * (1 - slippage_pct)
        slippage_cost = price * slippage_pct  # Cost per unit
    
    return adjusted_price, slippage_cost


def calculate_metrics(
    equity_curve: pd.Series,
    trades: List[Trade],
    initial_capital: float,
    timeframe: str,
) -> Dict[str, float]:
    """Calculate performance metrics."""
    if len(equity_curve) < 2:
        return {
            "cagr_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0,
            "win_rate_pct": 0.0, "profit_factor": 0.0, "total_trades": 0,
            "avg_trade_return_pct": 0.0, "avg_trade_duration_hours": 0.0,
        }
    
    final_equity = equity_curve.iloc[-1]
    
    # CAGR
    start_time = equity_curve.index[0]
    end_time = equity_curve.index[-1]
    years = (end_time - start_time).total_seconds() / (365.25 * 24 * 3600)
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0
    
    # Max Drawdown
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio
    returns = equity_curve.pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        tf_hours = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "1h": 1, "4h": 4, "1d": 24}
        hours_per_bar = tf_hours.get(timeframe, 1)
        bars_per_year = 8760 / hours_per_bar
        sharpe = returns.mean() / returns.std() * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0
    
    # Trade metrics
    if trades:
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning) / len(trades) * 100
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
        avg_return = np.mean([t.return_pct for t in trades])
        avg_duration = np.mean([t.duration_hours for t in trades])
    else:
        win_rate = profit_factor = avg_return = avg_duration = 0.0
    
    return {
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3),
        "total_trades": len(trades),
        "avg_trade_return_pct": round(avg_return, 3),
        "avg_trade_duration_hours": round(avg_duration, 1),
    }


def run_backtest(
    strategy_name: str,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = "2019-01-01",
    end_date: str = "2023-12-31",
    initial_capital: float = 10000.0,
    fee_pct: float = None,
    slippage_bps: float = None,  # Legacy, ignored if cost_profile set
    position_size_pct: float = 1.0,
    stop_fill_mode: str = "intrabar",
    sizing_mode: str = "all_in",
    risk_per_trade: float = 0.01,
    max_leverage: float = 1.0,
    cost_profile: str = "baseline",
    slippage_k: float = None,
    min_slippage_bps: float = None,
    save_reports: bool = True,
) -> BacktestResult:
    """
    Run backtest for a given strategy.
    
    Execution Model:
    - Signals evaluated at bar[i] close
    - Trades executed at bar[i+1] open
    
    Stop-Loss Fill Modes:
    - "intrabar" (default): If bar[i] low <= stop_price, fill at stop_price + slippage.
    - "next_open": If bar[i] close < stop_price, fill at bar[i+1] open + slippage.
    
    Position Sizing Modes:
    - "all_in": Use position_size_pct of equity (e.g., 1.0 = 100%)
    - "risk_per_trade": Size based on risk budget and stop distance.
    
    Cost Model:
    - fee_pct: Applied on notional for entry and exit
    - Slippage: Volatility-aware = max(min_slippage_bps, slippage_k * (ATR/Close) * 10000)
    - cost_profile: "baseline" or "high" (presets for fee/slippage params)
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    
    # Apply cost profile if params not explicitly set
    profile = COST_PROFILES.get(cost_profile, COST_PROFILES["baseline"])
    if fee_pct is None:
        fee_pct = profile["fee_pct"]
    if slippage_k is None:
        slippage_k = profile["slippage_k"]
    if min_slippage_bps is None:
        min_slippage_bps = profile["min_slippage_bps"]
    
    # Load data
    print(f"[Backtest] Loading data...")
    df = load_ohlcv(symbol=symbol, timeframe=timeframe, data_dir=DATA_DIR,
                    start_date=start_date, end_date=end_date)
    print(f"[Backtest] Loaded {len(df)} candles ({df.index.min()} to {df.index.max()})")
    
    if len(df) < 2:
        raise ValueError("Not enough data")
    
    # Get strategy signals
    from src.strategies import get_strategy
    strategy = get_strategy(strategy_name)
    
    # Prepare regime filter if strategy supports it
    if hasattr(strategy, 'regime_filter_enabled') and strategy.regime_filter_enabled:
        try:
            daily_df = load_ohlcv(symbol=symbol, timeframe="1d", data_dir=DATA_DIR,
                                  start_date=start_date, end_date=end_date)
            print(f"[Backtest] Loaded {len(daily_df)} daily candles for regime filter")
            if hasattr(strategy, 'prepare_regime'):
                strategy.prepare_regime(df, daily_df)
        except FileNotFoundError:
            print("[Backtest] WARNING: Daily data not found, regime filter disabled")
            strategy.regime_filter_enabled = False
    
    signals_df = strategy.generate_signals(df)
    
    print(f"[Backtest] Strategy: {strategy_name}")
    print(f"[Backtest] Fee: {fee_pct*100:.3f}%, Slippage_k: {slippage_k}, Min: {min_slippage_bps}bps")
    
    # State
    equity = initial_capital
    position = 0.0
    entry_price = 0.0
    entry_time = None
    stop_price = None
    highest_close = 0.0  # For trailing stop
    
    # Cost tracking
    total_fees_paid = 0.0
    total_slippage_cost = 0.0
    
    trades: List[Trade] = []
    equity_history = []
    
    # Main loop
    for i in range(len(df)):
        bar_time = df.index[i]
        bar_open = df["open"].iloc[i]
        bar_high = df["high"].iloc[i]
        bar_low = df["low"].iloc[i]
        bar_close = df["close"].iloc[i]
        
        # Record equity (position value + uninvested cash)
        current_equity = (position * bar_close) + equity if position > 0 else equity
        equity_history.append({"timestamp": bar_time, "equity": current_equity})
        
        # Check stop-loss based on stop_fill_mode
        if position > 0 and stop_price is not None:
            stop_triggered = False
            exit_price = 0.0
            slip_cost = 0.0
            atr = signals_df["atr"].iloc[i]
            
            if stop_fill_mode == "intrabar":
                # INTRABAR: If bar low breaches stop, fill at stop_price (within bar)
                if bar_low <= stop_price:
                    stop_triggered = True
                    exit_price, slip_cost = apply_volatility_slippage(
                        stop_price, atr, is_buy=False, 
                        slippage_k=slippage_k, min_slippage_bps=min_slippage_bps
                    )
                    # Ensure exit price is within bar range
                    exit_price = max(exit_price, bar_low)
            
            elif stop_fill_mode == "next_open":
                # NEXT_OPEN: If bar close < stop, fill at next bar open
                if i > 0:
                    prev_close = df["close"].iloc[i - 1]
                    prev_stop = signals_df["stop_price"].iloc[i - 1] if not pd.isna(signals_df["stop_price"].iloc[i - 1]) else stop_price
                    if prev_close < prev_stop:
                        stop_triggered = True
                        exit_price, slip_cost = apply_volatility_slippage(
                            bar_open, atr, is_buy=False,
                            slippage_k=slippage_k, min_slippage_bps=min_slippage_bps
                        )
            
            if stop_triggered:
                gross_proceeds = position * exit_price
                fee = gross_proceeds * fee_pct
                net_proceeds = gross_proceeds - fee
                
                # Track costs
                total_fees_paid += fee
                total_slippage_cost += slip_cost * position
                
                pnl = net_proceeds - (position * entry_price)
                return_pct = (exit_price / entry_price - 1) * 100
                duration = (bar_time - entry_time).total_seconds() / 3600
                
                trades.append(Trade(
                    entry_time=entry_time, entry_price=entry_price,
                    exit_time=bar_time, exit_price=exit_price,
                    position_size=position, pnl=pnl,
                    return_pct=return_pct, duration_hours=duration,
                    exit_type=f"stop_{stop_fill_mode}"
                ))
                
                # Add proceeds to remaining equity
                equity = equity + net_proceeds
                position = 0.0
                entry_price = 0.0
                entry_time = None
                stop_price = None
                highest_close = 0.0
                
                # Update equity after stop
                equity_history[-1]["equity"] = equity
        
        # Update trailing stop if enabled and in position
        if position > 0 and strategy.trailing_stop:
            highest_close = max(highest_close, bar_close)
            atr = signals_df["atr"].iloc[i]
            if not pd.isna(atr):
                trailing_stop = highest_close - strategy.atr_multiplier * atr
                stop_price = max(stop_price, trailing_stop) if stop_price else trailing_stop
        
        # Skip last bar for signal processing
        if i >= len(df) - 1:
            continue
        
        # Get signal (evaluated at this bar's close)
        signal = signals_df["signal"].iloc[i]
        prev_signal = signals_df["signal"].iloc[i-1] if i > 0 else 0
        
        # Next bar data (execution)
        next_open = df["open"].iloc[i + 1]
        next_time = df.index[i + 1]
        
        # Signal change
        if signal != prev_signal:
            # EXIT on signal (only if still in position - stop might have triggered)
            if position > 0 and signal == 0:
                atr = signals_df["atr"].iloc[i]
                exit_price, slip_cost = apply_volatility_slippage(
                    next_open, atr, is_buy=False,
                    slippage_k=slippage_k, min_slippage_bps=min_slippage_bps
                )
                
                gross_proceeds = position * exit_price
                fee = gross_proceeds * fee_pct
                net_proceeds = gross_proceeds - fee
                
                # Track costs
                total_fees_paid += fee
                total_slippage_cost += slip_cost * position
                
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
                
                # Add proceeds to remaining equity
                equity = equity + net_proceeds
                position = 0.0
                entry_price = 0.0
                entry_time = None
                stop_price = None
                highest_close = 0.0
            
            # ENTRY
            if signal == 1 and position == 0:
                atr = signals_df["atr"].iloc[i]
                exec_price, entry_slip_cost = apply_volatility_slippage(
                    next_open, atr, is_buy=True,
                    slippage_k=slippage_k, min_slippage_bps=min_slippage_bps
                )
                
                # Calculate position size based on sizing_mode
                if sizing_mode == "risk_per_trade":
                    # Risk-based sizing: position_size = risk_amount / stop_distance
                    if pd.isna(atr) or atr <= 0:
                        # Skip trade if ATR is invalid
                        continue
                    
                    stop_distance = strategy.atr_multiplier * atr
                    if stop_distance <= 0:
                        continue
                    
                    risk_amount = equity * risk_per_trade
                    max_notional = equity * max_leverage
                    
                    # Position size in base currency (BTC)
                    position_size_btc = risk_amount / stop_distance
                    position_value = position_size_btc * exec_price
                    
                    # Cap by max notional
                    if position_value > max_notional:
                        position_value = max_notional
                        position_size_btc = position_value / exec_price
                    
                    # Apply fee - fee comes from the position value
                    fee = position_value * fee_pct
                    cash_spent = position_value + fee
                    
                    # Check if we have enough equity
                    if cash_spent > equity:
                        cash_spent = equity
                        position_value = cash_spent / (1 + fee_pct)
                        fee = cash_spent - position_value
                        position_size_btc = position_value / exec_price
                    
                    position = position_size_btc
                    entry_cost = cash_spent
                    
                else:  # sizing_mode == "all_in"
                    trade_equity = equity * position_size_pct
                    fee = trade_equity * fee_pct
                    available = trade_equity - fee
                    position = available / exec_price
                    entry_cost = trade_equity
                
                # Track costs
                total_fees_paid += fee
                total_slippage_cost += entry_slip_cost * position
                
                entry_price = exec_price
                entry_time = next_time
                highest_close = bar_close
                
                # Set initial stop-loss
                stop_signal = signals_df["stop_price"].iloc[i]
                if not pd.isna(stop_signal):
                    if not pd.isna(atr):
                        stop_price = exec_price - strategy.atr_multiplier * atr
                    else:
                        stop_price = stop_signal
                else:
                    stop_price = None
                
                # Deduct cash spent from equity
                equity = equity - entry_cost
    
    # Close remaining position
    if position > 0:
        final_close = df["close"].iloc[-1]
        final_atr = signals_df["atr"].iloc[-1]
        exit_price, slip_cost = apply_volatility_slippage(
            final_close, final_atr, is_buy=False,
            slippage_k=slippage_k, min_slippage_bps=min_slippage_bps
        )
        
        gross_proceeds = position * exit_price
        fee = gross_proceeds * fee_pct
        net_proceeds = gross_proceeds - fee
        
        # Track costs
        total_fees_paid += fee
        total_slippage_cost += slip_cost * position
        
        pnl = net_proceeds - (position * entry_price)
        return_pct = (exit_price / entry_price - 1) * 100
        duration = (df.index[-1] - entry_time).total_seconds() / 3600
        
        trades.append(Trade(
            entry_time=entry_time, entry_price=entry_price,
            exit_time=df.index[-1], exit_price=exit_price,
            position_size=position, pnl=pnl,
            return_pct=return_pct, duration_hours=duration,
            exit_type="end"
        ))
        
        # Add proceeds to remaining equity
        equity = equity + net_proceeds
    
    # Build equity curve
    equity_df = pd.DataFrame(equity_history).set_index("timestamp")
    equity_curve = equity_df["equity"]
    
    final_equity = equity
    
    # Metrics
    metrics = calculate_metrics(equity_curve, trades, initial_capital, timeframe)
    
    # Calculate cost as % of gross PnL
    gross_pnl = sum(t.pnl for t in trades if t.pnl > 0)  # Only winning trades
    total_costs = total_fees_paid + total_slippage_cost
    cost_as_pct = (total_costs / gross_pnl * 100) if gross_pnl > 0 else 0.0
    
    result = BacktestResult(
        run_id=run_id, strategy_name=strategy_name, symbol=symbol,
        timeframe=timeframe, start_date=start_date, end_date=end_date,
        initial_capital=initial_capital, fee_pct=fee_pct,
        slippage_bps=0.0,  # Legacy field, actual slippage is volatility-aware
        position_size_pct=position_size_pct,
        final_equity=round(final_equity, 2), trades=trades,
        equity_curve=equity_curve,
        total_fees_paid=round(total_fees_paid, 2),
        total_slippage_cost=round(total_slippage_cost, 2),
        cost_as_pct_of_gross_pnl=round(cost_as_pct, 2),
        **metrics,
    )
    
    if save_reports:
        save_backtest_reports(result)
    
    return result


def save_backtest_reports(result: BacktestResult) -> Path:
    """Save backtest results to files."""
    report_dir = REPORTS_DIR / result.run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. metrics.json
    metrics = {
        "run_id": result.run_id,
        "strategy_name": result.strategy_name,
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": result.initial_capital,
        "final_equity": result.final_equity,
        "fee_pct": result.fee_pct,
        "slippage_bps": result.slippage_bps,
        "cagr_pct": result.cagr_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "win_rate_pct": result.win_rate_pct,
        "profit_factor": result.profit_factor,
        "total_trades": result.total_trades,
        "avg_trade_return_pct": result.avg_trade_return_pct,
        "avg_trade_duration_hours": result.avg_trade_duration_hours,
    }
    with open(report_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # 2. trades.csv
    if result.trades:
        trades_data = [{
            "entry_time": str(t.entry_time),
            "entry_price": round(t.entry_price, 2),
            "exit_time": str(t.exit_time),
            "exit_price": round(t.exit_price, 2),
            "position_size": round(t.position_size, 8),
            "pnl": round(t.pnl, 2),
            "return_pct": round(t.return_pct, 3),
            "duration_hours": round(t.duration_hours, 1),
            "exit_type": t.exit_type,
        } for t in result.trades]
        pd.DataFrame(trades_data).to_csv(report_dir / "trades.csv", index=False)
    
    # 3. equity.csv
    equity_df = result.equity_curve.reset_index()
    equity_df.columns = ["timestamp", "equity"]
    equity_df.to_csv(report_dir / "equity.csv", index=False)
    
    # 4. equity.png
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1 = axes[0]
    ax1.plot(result.equity_curve.index, result.equity_curve.values, 'b-', linewidth=1)
    ax1.axhline(y=result.initial_capital, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title(f"{result.strategy_name} | {result.symbol} {result.timeframe} | {result.start_date} to {result.end_date}")
    ax1.set_ylabel("Equity (USD)")
    ax1.grid(True, alpha=0.3)
    
    # Metrics annotation
    txt = (f"CAGR: {result.cagr_pct:.1f}%  |  MaxDD: {result.max_drawdown_pct:.1f}%  |  "
           f"Sharpe: {result.sharpe_ratio:.2f}  |  WinRate: {result.win_rate_pct:.1f}%  |  "
           f"Trades: {result.total_trades}  |  PF: {result.profit_factor:.2f}")
    ax1.text(0.5, 1.02, txt, transform=ax1.transAxes, ha='center', fontsize=9)
    
    # Drawdown
    ax2 = axes[1]
    peak = result.equity_curve.expanding().max()
    dd = (result.equity_curve - peak) / peak * 100
    ax2.fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(report_dir / "equity.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Backtest] Reports saved: {report_dir}")
    return report_dir


def print_backtest_summary(result: BacktestResult):
    """Print formatted backtest summary."""
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS: {result.strategy_name}")
    print("=" * 60)
    print(f"Run ID:         {result.run_id}")
    print(f"Symbol:         {result.symbol} ({result.timeframe})")
    print(f"Period:         {result.start_date} to {result.end_date}")
    print("-" * 60)
    print(f"Initial:        ${result.initial_capital:,.2f}")
    print(f"Final:          ${result.final_equity:,.2f}")
    print(f"P&L:            ${result.final_equity - result.initial_capital:,.2f}")
    print("-" * 60)
    print(f"CAGR:           {result.cagr_pct:.2f}%")
    print(f"Max Drawdown:   {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio:   {result.sharpe_ratio:.3f}")
    print("-" * 60)
    print(f"Total Trades:   {result.total_trades}")
    print(f"Win Rate:       {result.win_rate_pct:.1f}%")
    print(f"Profit Factor:  {result.profit_factor:.2f}")
    print(f"Avg Return:     {result.avg_trade_return_pct:.2f}%")
    print(f"Avg Duration:   {result.avg_trade_duration_hours:.1f}h")
    print("-" * 60)
    print(f"COST BREAKDOWN:")
    print(f"  Fees Paid:     ${result.total_fees_paid:,.2f}")
    print(f"  Slippage Cost: ${result.total_slippage_cost:,.2f}")
    print(f"  Costs/Gross:   {result.cost_as_pct_of_gross_pnl:.1f}%")
    print("=" * 60)

