"""
Data audit module for verifying data correctness.

Checks:
- Timestamps are UTC and monotonic
- Gap detection (missing candles)
- Duplicate detection
- Missing % calculation for 24/7 crypto
- Weekend candle verification
"""

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


# Timeframe to expected interval in seconds
TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
    "6h": 21600, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800,
}


@dataclass
class Gap:
    """Single gap in time series."""
    start: str  # ISO format
    end: str
    expected_candles: int
    duration_seconds: int


@dataclass
class AuditResult:
    """Complete data audit result."""
    run_id: str
    audit_timestamp: str
    
    # Data info
    symbol: str
    canonical_symbol: Optional[str]
    timeframe: str
    data_source: str
    data_file: str
    
    # Row counts
    rows: int
    first_ts: str
    last_ts: str
    
    # Time checks
    is_utc: bool
    is_monotonic: bool
    
    # Gap analysis
    gap_count: int
    gaps: List[Dict[str, Any]]
    
    # Duplicate analysis
    duplicate_count: int
    duplicate_timestamps: List[str]
    
    # Coverage analysis
    expected_candles: int
    actual_candles: int
    missing_pct: float
    
    # Crypto 24/7 verification
    weekend_candles: int
    weekend_expected: int
    weekend_coverage_pct: float
    
    # Execution config (for transparency)
    execution_config: Dict[str, Any]


def detect_gaps(df: pd.DataFrame, timeframe: str) -> Tuple[List[Gap], int]:
    """
    Detect gaps (missing candles) in time series.
    
    Returns:
        Tuple of (list of Gap objects, total gap count)
    """
    if len(df) < 2:
        return [], 0
    
    expected_interval = timedelta(seconds=TIMEFRAME_SECONDS.get(timeframe, 3600))
    
    gaps = []
    total_missing = 0
    
    for i in range(1, len(df)):
        prev_time = df.index[i - 1]
        curr_time = df.index[i]
        
        actual_interval = curr_time - prev_time
        
        if actual_interval > expected_interval * 1.5:  # Allow some tolerance
            expected_candles = int(actual_interval / expected_interval) - 1
            total_missing += expected_candles
            
            gaps.append(Gap(
                start=prev_time.isoformat(),
                end=curr_time.isoformat(),
                expected_candles=expected_candles,
                duration_seconds=int(actual_interval.total_seconds()),
            ))
    
    return gaps, total_missing


def detect_duplicates(df: pd.DataFrame) -> Tuple[int, List[str]]:
    """
    Detect duplicate timestamps.
    
    Returns:
        Tuple of (duplicate count, list of duplicate timestamps)
    """
    duplicates = df.index.duplicated(keep=False)
    dup_count = duplicates.sum()
    
    if dup_count > 0:
        dup_times = df.index[duplicates].unique()
        return dup_count, [t.isoformat() for t in dup_times[:10]]  # Limit to 10
    
    return 0, []


def count_weekend_candles(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Count weekend candles for 24/7 crypto verification.
    
    Returns:
        Tuple of (actual weekend candles, expected weekend candles)
    """
    # Weekend = Saturday (5) or Sunday (6)
    weekend_mask = df.index.dayofweek.isin([5, 6])
    actual = weekend_mask.sum()
    
    # Calculate expected: approximately 2/7 of total
    total_days = (df.index[-1] - df.index[0]).days
    weekend_days = (total_days * 2) // 7
    
    # Expected candles per day depends on timeframe
    # For simplicity, use same ratio as actual
    expected = int(len(df) * (2 / 7))
    
    return actual, expected


def audit_data(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    canonical_symbol: Optional[str] = None,
    data_source: str = "unknown",
    data_file: str = "",
    stop_fill_mode: str = "intrabar",
    fee_pct: float = 0.0005,
    slippage_bps: float = 5.0,
) -> AuditResult:
    """
    Run comprehensive data audit.
    
    Args:
        symbol: Original symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe (e.g., '1h')
        df: DataFrame with DatetimeIndex and OHLCV columns
        canonical_symbol: Actual symbol used by data source (e.g., 'BTC-USD')
        data_source: Data provider name
        data_file: Path to data file
        stop_fill_mode: 'intrabar' or 'next_open'
        fee_pct: Trading fee percentage
        slippage_bps: Slippage in basis points
    
    Returns:
        AuditResult with comprehensive audit data
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    
    if len(df) == 0:
        return AuditResult(
            run_id=run_id,
            audit_timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            canonical_symbol=canonical_symbol,
            timeframe=timeframe,
            data_source=data_source,
            data_file=data_file,
            rows=0,
            first_ts="",
            last_ts="",
            is_utc=False,
            is_monotonic=False,
            gap_count=0,
            gaps=[],
            duplicate_count=0,
            duplicate_timestamps=[],
            expected_candles=0,
            actual_candles=0,
            missing_pct=100.0,
            weekend_candles=0,
            weekend_expected=0,
            weekend_coverage_pct=0.0,
            execution_config={
                "stop_fill_mode": stop_fill_mode,
                "fee_pct": fee_pct,
                "slippage_bps": slippage_bps,
            },
        )
    
    # Time checks
    is_utc = df.index.tz is not None and str(df.index.tz) == "UTC"
    is_monotonic = df.index.is_monotonic_increasing
    
    # Gap detection
    gaps, gap_count = detect_gaps(df, timeframe)
    
    # Duplicate detection
    dup_count, dup_times = detect_duplicates(df)
    
    # Coverage calculation
    interval_seconds = TIMEFRAME_SECONDS.get(timeframe, 3600)
    total_seconds = (df.index[-1] - df.index[0]).total_seconds()
    expected_candles = int(total_seconds / interval_seconds) + 1
    actual_candles = len(df)
    missing_pct = ((expected_candles - actual_candles) / expected_candles * 100) if expected_candles > 0 else 0.0
    
    # Weekend verification (24/7 crypto)
    weekend_actual, weekend_expected = count_weekend_candles(df)
    weekend_coverage = (weekend_actual / weekend_expected * 100) if weekend_expected > 0 else 0.0
    
    return AuditResult(
        run_id=run_id,
        audit_timestamp=datetime.now(timezone.utc).isoformat(),
        symbol=symbol,
        canonical_symbol=canonical_symbol,
        timeframe=timeframe,
        data_source=data_source,
        data_file=data_file,
        rows=len(df),
        first_ts=df.index[0].isoformat(),
        last_ts=df.index[-1].isoformat(),
        is_utc=is_utc,
        is_monotonic=is_monotonic,
        gap_count=gap_count,
        gaps=[asdict(g) for g in gaps[:20]],  # Limit to 20
        duplicate_count=dup_count,
        duplicate_timestamps=dup_times,
        expected_candles=expected_candles,
        actual_candles=actual_candles,
        missing_pct=round(missing_pct, 2),
        weekend_candles=weekend_actual,
        weekend_expected=weekend_expected,
        weekend_coverage_pct=round(weekend_coverage, 2),
        execution_config={
            "stop_fill_mode": stop_fill_mode,
            "fee_pct": fee_pct,
            "slippage_bps": slippage_bps,
        },
    )


def save_audit_report(result: AuditResult) -> Path:
    """Save audit result to JSON file."""
    report_dir = REPORTS_DIR / "audit" / result.run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / "audit.json"
    
    # Convert to dict and handle numpy types
    data = asdict(result)
    
    def convert_numpy(obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    data = convert_numpy(data)
    
    with open(report_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return report_path


def print_audit_summary(result: AuditResult):
    """Print formatted audit summary."""
    print("\n" + "=" * 70)
    print("DATA AUDIT REPORT")
    print("=" * 70)
    print(f"Run ID:           {result.run_id}")
    print(f"Timestamp:        {result.audit_timestamp}")
    print("-" * 70)
    print(f"Symbol:           {result.symbol}")
    if result.canonical_symbol:
        print(f"Canonical Symbol: {result.canonical_symbol}")
    print(f"Timeframe:        {result.timeframe}")
    print(f"Data Source:      {result.data_source}")
    print(f"Data File:        {result.data_file}")
    print("-" * 70)
    print(f"Rows:             {result.rows:,}")
    print(f"First Timestamp:  {result.first_ts}")
    print(f"Last Timestamp:   {result.last_ts}")
    print("-" * 70)
    
    # Time checks
    utc_status = "✓" if result.is_utc else "✗"
    mono_status = "✓" if result.is_monotonic else "✗"
    print(f"UTC Timestamps:   {utc_status}")
    print(f"Monotonic:        {mono_status}")
    print("-" * 70)
    
    # Gaps
    gap_status = "✓ None" if result.gap_count == 0 else f"⚠ {result.gap_count} gaps"
    print(f"Gap Count:        {gap_status}")
    if result.gaps and len(result.gaps) <= 5:
        for g in result.gaps:
            print(f"  - {g['start']} → {g['end']} ({g['expected_candles']} missing)")
    elif result.gaps:
        print(f"  (First 5 of {len(result.gaps)} gaps shown)")
        for g in result.gaps[:5]:
            print(f"  - {g['start']} → {g['end']} ({g['expected_candles']} missing)")
    
    # Duplicates
    dup_status = "✓ None" if result.duplicate_count == 0 else f"✗ {result.duplicate_count}"
    print(f"Duplicates:       {dup_status}")
    
    # Coverage
    print("-" * 70)
    print(f"Expected Candles: {result.expected_candles:,}")
    print(f"Actual Candles:   {result.actual_candles:,}")
    missing_color = "✓" if result.missing_pct < 5 else "⚠"
    print(f"Missing:          {missing_color} {result.missing_pct:.2f}%")
    
    # Weekend (24/7)
    print("-" * 70)
    print(f"Weekend Candles:  {result.weekend_candles:,} (expected ~{result.weekend_expected:,})")
    weekend_status = "✓" if result.weekend_coverage_pct > 90 else "⚠"
    print(f"Weekend Coverage: {weekend_status} {result.weekend_coverage_pct:.1f}%")
    
    # Execution config
    print("-" * 70)
    print("EXECUTION CONFIG:")
    print(f"  Stop Fill Mode: {result.execution_config['stop_fill_mode']}")
    print(f"  Fee:            {result.execution_config['fee_pct']*100:.3f}%")
    print(f"  Slippage:       {result.execution_config['slippage_bps']} bps")
    print("=" * 70)
