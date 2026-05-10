"""
research/regime/hmm_regime.py

Hidden Markov Model (HMM) regime detector for VN30F.

Theory
------
A Gaussian HMM is fit to a feature vector (typically log-returns +
realised volatility) using the Baum-Welch (EM) algorithm.  The Viterbi
algorithm decodes the most likely hidden state sequence — each state
corresponds to a latent market *regime* (e.g. trending, mean-reverting,
high-vol crash).

Reference
---------
Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
Nonstationary Time Series and the Business Cycle." Econometrica, 57(2).

This is *research prototype* code — not production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class HMMRegimeDetector:
    """Fit a Gaussian HMM and decode market regimes.

    Parameters
    ----------
    n_regimes:
        Number of latent states (default 3 — bull / bear / neutral).
    n_iter:
        Maximum EM iterations for Baum-Welch.
    covariance_type:
        HMM covariance structure. ``"full"`` recommended for financial data.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        n_iter: int = 200,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model: GaussianHMM | None = None

    # ------------------------------------------------------------------
    def fit(self, features: pd.DataFrame) -> "HMMRegimeDetector":
        """Fit the HMM on ``features`` (rows = observations, cols = features).

        The feature matrix is z-score normalised internally.

        Parameters
        ----------
        features:
            DataFrame of shape (T, F) — no NaNs allowed.

        Returns
        -------
        self
        """
        X = self._scaler.fit_transform(features.values.astype(float))
        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._model.fit(X)
        return self

    # ------------------------------------------------------------------
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Decode the most likely regime sequence via Viterbi.

        Parameters
        ----------
        features:
            DataFrame of shape (T, F) — same columns as fit().

        Returns
        -------
        pd.Series
            Integer regime labels (0 … n_regimes-1), indexed like ``features``.
        """
        if self._model is None:
            raise RuntimeError("[HMMRegimeDetector] Call fit() before predict().")
        X = self._scaler.transform(features.values.astype(float))
        labels = self._model.predict(X)
        return pd.Series(labels, index=features.index, name="regime")

    # ------------------------------------------------------------------
    def fit_predict(self, features: pd.DataFrame) -> pd.Series:
        """Convenience: fit then predict on the same data."""
        return self.fit(features).predict(features)

    # ------------------------------------------------------------------
    @property
    def transition_matrix(self) -> pd.DataFrame:
        """Return the estimated state-transition probability matrix."""
        if self._model is None:
            raise RuntimeError("[HMMRegimeDetector] Model not fitted yet.")
        tm = pd.DataFrame(
            self._model.transmat_,
            index=[f"from_{i}" for i in range(self.n_regimes)],
            columns=[f"to_{i}" for i in range(self.n_regimes)],
        )
        return tm
