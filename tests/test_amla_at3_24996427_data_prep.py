import pytest
import pandas as pd
import numpy as np
import os
from io import StringIO
from pathlib import Path

# Import your module
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # project root

from amla_at1_24996427.data.DataPrep import standardize_colname, coerce_numeric, load_and_clean_csv


# ---------------------------------------------------------------------
# 1. TEST: standardize_colname
# ---------------------------------------------------------------------
@pytest.mark.parametrize("input_col,expected", [
    (" Close ", "close"),
    ("MarketCap", "market_cap"),
    ("Market-Cap", "market_cap"),
    ("Time*Open", "timeopen"),
    ("Rolling Mean 7", "rolling_mean_7"),
])
def test_standardize_colname(input_col, expected):
    assert standardize_colname(input_col) == expected


# ---------------------------------------------------------------------
# 2. TEST: coerce_numeric
# ---------------------------------------------------------------------
@pytest.mark.parametrize("input_val,expected", [
    (123, 123.0),
    (123.45, 123.45),
    ("1,234.56", 1234.56),
    ("$9.87", 9.87),
    ("  7.5  ", 7.5),
    ("invalid", np.nan),
    (None, np.nan),
])
def test_coerce_numeric(input_val, expected):
    result = coerce_numeric(input_val)
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------
# 3. TEST: load_and_clean_csv
# ---------------------------------------------------------------------
@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file for testing."""
    content = """timestamp;open;high;low;close;volume;marketcap;name
2024-01-01;1,000;1,100;900;1,050;5,000;$10,000,000;XRP
2024-01-02;1050;1100;950;1000;5500;$10,500,000;XRP
"""
    path = tmp_path / "sample.csv"
    path.write_text(content)
    return path


def test_load_and_clean_csv_basic(sample_csv):
    df = load_and_clean_csv(sample_csv)

    # Structure checks
    assert isinstance(df, pd.DataFrame)
    assert "date" in df.columns
    assert df["date"].dtype == "datetime64[ns]"
    assert "source_file" in df.columns
    assert df["source_file"].iloc[0] == os.path.basename(sample_csv)

    # Numeric coercion checks
    assert df["open"].iloc[0] == 1000.0
    assert df["market_cap"].iloc[0] == 10000000.0
    assert not df.isna().any().any()

    # Sorted by date
    assert df["date"].is_monotonic_increasing


def test_load_and_clean_csv_with_timeopen(tmp_path):
    """Test fallback if 'timeopen' column is used instead of timestamp."""
    content = """timeOpen;open;close;market_cap
2024-01-01;100;110;1000
2024-01-02;105;115;1200
"""
    path = tmp_path / "alt_time.csv"
    path.write_text(content)
    df = load_and_clean_csv(path)
    assert "date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_and_clean_csv_no_date_column(tmp_path):
    """Should raise ValueError if no recognizable date column."""
    content = """price;open;close
1;2;3
"""
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match="No recognizable date column"):
        load_and_clean_csv(path)


def test_load_and_clean_csv_drops_name_column(tmp_path):
    content = """timestamp;open;close;name
2024-01-01;1;2;ABC
"""
    path = tmp_path / "drop_name.csv"
    path.write_text(content)
    df = load_and_clean_csv(path)
    assert "name" not in df.columns


def test_load_and_clean_csv_handles_missing_values(tmp_path):
    content = """timestamp;open;close;market_cap
2024-01-01;1,000;;$2,000
2024-01-02;;1,100;$2,200
"""
    path = tmp_path / "missing.csv"
    path.write_text(content)
    df = load_and_clean_csv(path)
    # Ensure partial NaNs are preserved but date column is clean
    assert not df["date"].isna().any()
    assert df["open"].isna().any() or df["close"].isna().any()
