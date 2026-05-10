"""
tests/test_data_loader.py

Tests for research/data/loader.py
"""
from __future__ import annotations

import pandas as pd
import pytest

from research.data import load_ohlcv, load_ohlcv_from_csv


_TZ = "Asia/Ho_Chi_Minh"


# ---------------------------------------------------------------------------
# load_ohlcv with DataFrame source
# ---------------------------------------------------------------------------


class TestLoadOhlcvFromDataFrame:
    def test_returns_copy_not_same_object(self, raw_ohlcv_df):
        result = load_ohlcv(raw_ohlcv_df)
        assert result is not raw_ohlcv_df

    def test_index_is_datetimeindex(self, raw_ohlcv_df):
        result = load_ohlcv(raw_ohlcv_df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_index_is_timezone_aware(self, raw_ohlcv_df):
        result = load_ohlcv(raw_ohlcv_df)
        assert result.index.tzinfo is not None
        assert str(result.index.tz) == _TZ

    def test_columns_are_lowercase(self, raw_ohlcv_df):
        result = load_ohlcv(raw_ohlcv_df)
        assert all(c == c.lower() for c in result.columns)

    def test_datetime_not_in_columns_after_normalise(self, raw_ohlcv_df):
        result = load_ohlcv(raw_ohlcv_df)
        assert "datetime" not in result.columns

    def test_required_columns_present(self, raw_ohlcv_df):
        result = load_ohlcv(raw_ohlcv_df)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in result.columns

    def test_index_is_sorted(self, raw_ohlcv_df):
        # Shuffle first to ensure sorting is applied
        shuffled = raw_ohlcv_df.sample(frac=1, random_state=0).reset_index(drop=True)
        result = load_ohlcv(shuffled)
        assert result.index.is_monotonic_increasing

    def test_missing_required_column_raises_value_error(self, raw_ohlcv_df):
        incomplete = raw_ohlcv_df.drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing required columns"):
            load_ohlcv(incomplete)

    def test_already_tz_aware_datetime_is_converted(self, raw_ohlcv_df):
        """If input datetime column is already tz-aware, it must be converted."""
        df = raw_ohlcv_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")
        result = load_ohlcv(df)
        assert str(result.index.tz) == _TZ

    def test_extra_columns_are_preserved(self, raw_ohlcv_df):
        df = raw_ohlcv_df.copy()
        df["open_interest"] = 999
        result = load_ohlcv(df)
        assert "open_interest" in result.columns

    def test_row_count_preserved(self, raw_ohlcv_df):
        result = load_ohlcv(raw_ohlcv_df)
        assert len(result) == len(raw_ohlcv_df)

    def test_column_names_are_stripped_and_lowercased(self, raw_ohlcv_df):
        """Columns are lowercased and stripped during normalisation."""
        result = load_ohlcv(raw_ohlcv_df)
        # After normalisation every column name must be lower-stripped
        for col in result.columns:
            assert col == col.lower().strip()


# ---------------------------------------------------------------------------
# load_ohlcv_from_csv
# ---------------------------------------------------------------------------


class TestLoadOhlcvFromCsv:
    def test_loads_valid_csv(self, tmp_path, raw_ohlcv_df):
        csv_path = tmp_path / "ohlcv.csv"
        raw_ohlcv_df.to_csv(csv_path, index=False)
        result = load_ohlcv_from_csv(csv_path)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_ohlcv_from_csv(tmp_path / "nonexistent.csv")

    def test_csv_missing_column_raises(self, tmp_path, raw_ohlcv_df):
        bad = raw_ohlcv_df.drop(columns=["close"])
        csv_path = tmp_path / "bad.csv"
        bad.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_ohlcv_from_csv(csv_path)

    def test_accepts_string_path(self, tmp_path, raw_ohlcv_df):
        csv_path = tmp_path / "ohlcv.csv"
        raw_ohlcv_df.to_csv(csv_path, index=False)
        result = load_ohlcv_from_csv(str(csv_path))
        assert len(result) == len(raw_ohlcv_df)

    def test_load_ohlcv_delegates_path_to_csv_loader(self, tmp_path, raw_ohlcv_df):
        csv_path = tmp_path / "ohlcv.csv"
        raw_ohlcv_df.to_csv(csv_path, index=False)
        result = load_ohlcv(csv_path)
        assert len(result) == len(raw_ohlcv_df)
