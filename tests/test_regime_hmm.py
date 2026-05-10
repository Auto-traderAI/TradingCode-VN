"""
tests/test_regime_hmm.py

Tests for research/regime/hmm_regime.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.regime import HMMRegimeDetector


@pytest.fixture
def hmm_features() -> pd.DataFrame:
    """Noise-free feature DataFrame suitable for HMM fitting."""
    n = 300
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Ho_Chi_Minh")
    return pd.DataFrame(
        {
            "ret": rng.normal(0, 0.01, n),
            "vol": np.abs(rng.normal(0.15, 0.05, n)),
        },
        index=idx,
    )


class TestHMMRegimeDetectorErrors:
    def test_predict_before_fit_raises_runtime_error(self, hmm_features):
        detector = HMMRegimeDetector()
        with pytest.raises(RuntimeError, match="Call fit"):
            detector.predict(hmm_features)

    def test_transition_matrix_before_fit_raises_runtime_error(self):
        detector = HMMRegimeDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = detector.transition_matrix


class TestHMMRegimeDetectorFit:
    def test_fit_returns_self(self, hmm_features):
        detector = HMMRegimeDetector(n_regimes=2, n_iter=10)
        result = detector.fit(hmm_features)
        assert result is detector

    def test_fit_sets_model(self, hmm_features):
        detector = HMMRegimeDetector(n_regimes=2, n_iter=10)
        detector.fit(hmm_features)
        assert detector._model is not None


class TestHMMRegimeDetectorPredict:
    def test_output_is_series(self, hmm_features):
        detector = HMMRegimeDetector(n_regimes=2, n_iter=10)
        labels = detector.fit_predict(hmm_features)
        assert isinstance(labels, pd.Series)

    def test_output_length_matches_input(self, hmm_features):
        detector = HMMRegimeDetector(n_regimes=3, n_iter=10)
        labels = detector.fit_predict(hmm_features)
        assert len(labels) == len(hmm_features)

    def test_index_matches_input(self, hmm_features):
        detector = HMMRegimeDetector(n_regimes=2, n_iter=10)
        labels = detector.fit_predict(hmm_features)
        assert labels.index.equals(hmm_features.index)

    def test_regime_values_within_range(self, hmm_features):
        n_regimes = 3
        detector = HMMRegimeDetector(n_regimes=n_regimes, n_iter=10)
        labels = detector.fit_predict(hmm_features)
        assert labels.min() >= 0
        assert labels.max() < n_regimes

    def test_two_regime_model(self, hmm_features):
        detector = HMMRegimeDetector(n_regimes=2, n_iter=10)
        labels = detector.fit_predict(hmm_features)
        assert labels.max() < 2

    def test_predict_on_separate_test_set(self, hmm_features):
        train = hmm_features.iloc[:200]
        test = hmm_features.iloc[200:]
        detector = HMMRegimeDetector(n_regimes=2, n_iter=10)
        detector.fit(train)
        labels = detector.predict(test)
        assert len(labels) == len(test)


class TestHMMTransitionMatrix:
    def test_returns_dataframe(self, hmm_features):
        detector = HMMRegimeDetector(n_regimes=2, n_iter=10)
        detector.fit(hmm_features)
        tm = detector.transition_matrix
        assert isinstance(tm, pd.DataFrame)

    def test_shape_matches_n_regimes(self, hmm_features):
        n = 3
        detector = HMMRegimeDetector(n_regimes=n, n_iter=10)
        detector.fit(hmm_features)
        tm = detector.transition_matrix
        assert tm.shape == (n, n)

    def test_rows_sum_to_one(self, hmm_features):
        detector = HMMRegimeDetector(n_regimes=2, n_iter=10)
        detector.fit(hmm_features)
        tm = detector.transition_matrix
        np.testing.assert_allclose(tm.sum(axis=1).values, 1.0, atol=1e-6)
