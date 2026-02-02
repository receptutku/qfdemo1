"""
Time-series splitting for walk-forward analysis.

Implements rolling window splitting that maintains temporal order
and prevents data leakage between train and test periods.
"""

import pandas as pd
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Generator


@dataclass
class Split:
    """A single train/test split."""
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    def __repr__(self) -> str:
        return (f"Split {self.index}: "
                f"Train [{self.train_start.date()} to {self.train_end.date()}] | "
                f"Test [{self.test_start.date()} to {self.test_end.date()}]")


def generate_walk_forward_splits(
    df: pd.DataFrame,
    train_days: int = 540,
    test_days: int = 90,
    step_days: int = 30,
    warmup_bars: int = 200,
) -> List[Split]:
    """
    Generate walk-forward splits from OHLCV data.
    
    Time-series rules:
    - Never shuffle data
    - Test window is strictly AFTER train window
    - No overlap between train and test
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        train_days: Training window length in days
        test_days: Test window length in days
        step_days: How many days to shift between splits
        warmup_bars: Minimum bars before first tradeable signal (for indicator warmup)
    
    Returns:
        List of Split objects
    """
    if df.empty:
        return []
    
    # Get data range
    data_start = df.index.min()
    data_end = df.index.max()
    
    # Convert days to timedelta
    train_delta = timedelta(days=train_days)
    test_delta = timedelta(days=test_days)
    step_delta = timedelta(days=step_days)
    
    splits = []
    split_idx = 0
    
    # First split starts at data_start
    current_train_start = data_start
    
    while True:
        # Calculate window boundaries
        train_end = current_train_start + train_delta
        test_start = train_end  # Test starts immediately after train
        test_end = test_start + test_delta
        
        # Check if we have enough data
        if test_end > data_end:
            # Try to fit a partial test window
            if test_start < data_end:
                test_end = data_end
            else:
                break
        
        # Verify we have actual data in these ranges
        train_mask = (df.index >= current_train_start) & (df.index < train_end)
        test_mask = (df.index >= test_start) & (df.index <= test_end)
        
        train_bars = train_mask.sum()
        test_bars = test_mask.sum()
        
        # Skip if not enough data (need at least warmup_bars in train)
        if train_bars < warmup_bars or test_bars < 10:
            current_train_start += step_delta
            continue
        
        split = Split(
            index=split_idx,
            train_start=current_train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        splits.append(split)
        
        split_idx += 1
        current_train_start += step_delta
        
        # Safety: prevent infinite loop
        if split_idx > 1000:
            break
    
    return splits


def get_split_data(
    df: pd.DataFrame,
    split: Split,
    warmup_bars: int = 200,
) -> tuple:
    """
    Get train and test data for a split.
    
    For test period, we include warmup bars from the end of training
    to allow indicators to calculate, but we only trade on test period.
    
    Args:
        df: Full OHLCV DataFrame
        split: Split definition
        warmup_bars: Bars to include before test_start for indicator warmup
    
    Returns:
        (train_df, test_df, warmup_df)
        - train_df: Training period data
        - test_df: Test period data (for trading)
        - warmup_df: Data including warmup period (for indicator calculation)
    """
    # Training data
    train_df = df[(df.index >= split.train_start) & (df.index < split.train_end)]
    
    # Test data (actual trading period)
    test_df = df[(df.index >= split.test_start) & (df.index <= split.test_end)]
    
    # Warmup: get bars before test_start
    pre_test_data = df[df.index < split.test_start]
    if len(pre_test_data) >= warmup_bars:
        warmup_start_idx = len(pre_test_data) - warmup_bars
        warmup_start = pre_test_data.index[warmup_start_idx]
    else:
        warmup_start = pre_test_data.index[0] if len(pre_test_data) > 0 else split.test_start
    
    # Combined data for indicator calculation (warmup + test)
    warmup_df = df[(df.index >= warmup_start) & (df.index <= split.test_end)]
    
    return train_df, test_df, warmup_df


def print_splits_summary(splits: List[Split]):
    """Print a summary table of all splits."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD SPLITS")
    print("=" * 80)
    print(f"{'Split':>5} | {'Train Start':>12} | {'Train End':>12} | {'Test Start':>12} | {'Test End':>12}")
    print("-" * 80)
    
    for s in splits:
        print(f"{s.index:>5} | {str(s.train_start.date()):>12} | {str(s.train_end.date()):>12} | "
              f"{str(s.test_start.date()):>12} | {str(s.test_end.date()):>12}")
    
    print("=" * 80)
    print(f"Total splits: {len(splits)}")
