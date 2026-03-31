"""Feature engineering for the linear attention asset pricing model.

Builds the firm-characteristic matrix X_t used in:

    A_t = X_t W X_t^T      (Kelly et al. 2025)

Characteristics align with those listed in ``config/default.yaml``
under ``data.characteristics``:
    log_mcap, bm, mom_12_2, prof, inv, turnover, vol, ep

The final matrix is normalised cross-sectionally each month before
being fed to the attention model.
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata


_CHARACTERISTICS = [
    "log_mcap",
    "bm",
    "mom_12_2",
    "prof",
    "inv",
    "turnover",
    "vol",
    "ep",
]


def compute_firm_characteristics(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Build the cross-sectional firm-characteristic matrix X_t.

    Derives each of the eight characteristics used in the thesis from the
    merged CRSP–Compustat panel.  Missing values are forward-filled within
    each firm and then set to cross-sectional median.

    Args:
        merged_df: Merged CRSP–Compustat panel with at minimum columns
            ``[permno, date, mcap, bm, ret, roa, at, capx, turnover]``.

    Returns:
        DataFrame with columns ``[permno, date] + _CHARACTERISTICS`` where
        each characteristic is available for all ``(permno, date)`` pairs
        that survive the filters.
    """
    df = merged_df.copy()

    # log_mcap: natural log of market capitalisation (size)
    df["log_mcap"] = np.log(df["mcap"].clip(lower=1e-6))

    # bm: book-to-market (value signal) — expected from Compustat cleaning
    # Already computed in CompustatLoader.clean()

    # mom_12_2: 12-2 month price momentum
    df = df.sort_values(["permno", "date"])
    df["cum_ret"] = df.groupby("permno")["ret"].transform(
        lambda s: s.rolling(12).apply(lambda x: (1 + x).prod() - 1, raw=True)
    )
    # Exclude the most recent month (skip-1 convention)
    df["mom_12_2"] = df.groupby("permno")["cum_ret"].shift(1)

    # prof: profitability (ROA from Compustat)
    df["prof"] = df.get("roa", pd.Series(np.nan, index=df.index))

    # inv: asset growth (investment)
    df["inv"] = df.groupby("permno")["at"].pct_change()

    # turnover: average monthly share turnover — expected from CRSP
    # (volume / shares outstanding); if not present, fill with NaN
    if "turnover" not in df.columns:
        df["turnover"] = np.nan

    # vol: idiosyncratic volatility (36-month rolling std of returns)
    df["vol"] = df.groupby("permno")["ret"].transform(
        lambda s: s.rolling(36, min_periods=12).std()
    )

    # ep: earnings-to-price ratio
    if "ni" in df.columns and "mcap" in df.columns:
        df["ep"] = df["ni"] / df["mcap"].clip(lower=1e-6)
    else:
        df["ep"] = np.nan

    # Forward-fill within firm, then cross-sectional median imputation
    for col in _CHARACTERISTICS:
        df[col] = df.groupby("permno")[col].ffill()
        df[col] = df[col].fillna(df.groupby("date")[col].transform("median"))

    return df[["permno", "date"] + _CHARACTERISTICS].reset_index(drop=True)


def normalize_features(
    features_df: pd.DataFrame,
    method: Literal["zscore", "rank"] = "rank",
) -> pd.DataFrame:
    """Normalise firm characteristics cross-sectionally each month.

    Args:
        features_df: DataFrame from :func:`compute_firm_characteristics`
            with columns ``[permno, date] + characteristics``.
        method: Normalisation method:
            - ``'zscore'``: subtract cross-sectional mean, divide by std.
            - ``'rank'``: cross-sectional rank normalised to [-0.5, 0.5].

    Returns:
        DataFrame with the same shape as ``features_df`` but with
        normalised characteristic values.
    """
    df = features_df.copy()
    chars = [c for c in _CHARACTERISTICS if c in df.columns]

    if method == "zscore":
        # Cross-sectional z-score each month
        def _zscore(x: pd.Series) -> pd.Series:
            mu, sigma = x.mean(), x.std()
            return (x - mu) / (sigma + 1e-8)

        df[chars] = df.groupby("date")[chars].transform(_zscore)

    elif method == "rank":
        # Cross-sectional rank normalised to [-0.5, 0.5] (Kelly et al. 2025)
        def _rank(x: pd.Series) -> pd.Series:
            r = rankdata(x.fillna(x.median()), method="average")
            return (r - 1) / (len(r) - 1 + 1e-8) - 0.5

        df[chars] = df.groupby("date")[chars].transform(_rank)

    else:
        raise ValueError(f"Unknown normalisation method: '{method}'.  Choose 'zscore' or 'rank'.")

    return df


def construct_pairs(
    features_df: pd.DataFrame,
    max_pairs_per_date: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build firm-pair feature table for attention analysis.

    For each cross-sectional month, constructs all (i, j) firm pairs and
    computes pair-level features used in the H1 and H2 regressions:
    - ``size_diff``: absolute log-mcap difference
    - ``style_diff``: Euclidean distance in BM / momentum / profitability

    Args:
        features_df: Normalised features DataFrame with columns
            ``[permno, date, log_mcap, bm, mom_12_2, prof, inv, turnover, vol, ep]``.
        max_pairs_per_date: If provided, randomly sub-samples this many
            pairs per date to limit memory usage for large cross-sections
            (e.g. set to ``config.hypothesis_tests.h1.pair_sample_size``).
        seed: Random seed for reproducible sub-sampling.

    Returns:
        Long-format DataFrame with one row per (date, permno_i, permno_j)
        pair containing the pair-level features.

    Note:
        Without sub-sampling, memory usage grows as O(N²) per date.  For
        N=3000 stocks this is ~9M pairs per month.  Always set
        ``max_pairs_per_date`` in production runs.
    """
    rng = np.random.default_rng(seed)
    records = []
    style_cols = ["bm", "mom_12_2", "prof", "inv"]

    for date, group in features_df.groupby("date"):
        permnos = group["permno"].values
        n = len(permnos)
        if n < 2:
            continue

        vals = group.set_index("permno")
        # Generate all (i, j) index pairs with i < j
        idx_i, idx_j = np.triu_indices(n, k=1)

        if max_pairs_per_date is not None and len(idx_i) > max_pairs_per_date:
            sampled = rng.choice(len(idx_i), size=max_pairs_per_date, replace=False)
            idx_i, idx_j = idx_i[sampled], idx_j[sampled]

        for ii, jj in zip(idx_i, idx_j):
            pi, pj = permnos[ii], permnos[jj]
            xi, xj = vals.loc[pi], vals.loc[pj]

            size_diff = abs(xi["log_mcap"] - xj["log_mcap"])

            # Style distance across BM, momentum, profitability, investment
            style_i = np.array([xi.get(c, 0.0) for c in style_cols], dtype=float)
            style_j = np.array([xj.get(c, 0.0) for c in style_cols], dtype=float)
            style_diff = float(np.linalg.norm(style_i - style_j))

            records.append(
                {
                    "date": date,
                    "permno_i": pi,
                    "permno_j": pj,
                    "size_diff": size_diff,
                    "style_diff": style_diff,
                }
            )

    pairs_df = pd.DataFrame(records)
    return pairs_df
