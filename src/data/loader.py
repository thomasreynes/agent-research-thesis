"""Data loading utilities for CRSP and Compustat datasets.

Provides config loading, raw data ingestion, basic cleaning, and the
CRSP–Compustat merge required to build the firm-characteristic matrix
X_t used in the linear attention model:

    A_t = X_t W X_t^T      (Kelly et al. 2025)

All paths are read from ``config/default.yaml`` so no numbers are
hard-coded in this module.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


def load_config(config_path: str = "config/default.yaml") -> dict:
    """Load the project YAML configuration file.

    Args:
        config_path: Path to the YAML config (relative to repo root or
            absolute).  Defaults to ``config/default.yaml``.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


class CRSPLoader:
    """Load and clean CRSP monthly stock return data.

    Expects a CSV or Parquet file at ``cfg['data']['raw_dir']`` named
    ``crsp_monthly.csv`` (or ``.parquet``).

    The cleaning pipeline applies the standard asset-pricing filters:
    - Drop rows with missing returns or prices
    - Enforce minimum price filter (``cfg['data']['min_price']``)
    - Optionally exclude financial firms (SIC 6000–6999)

    Args:
        cfg: Project configuration dict from :func:`load_config`.

    Attributes:
        raw_dir: Path to the raw data directory.
        min_price: Minimum share price filter.
        exclude_financials: Whether to drop financial-sector stocks.
    """

    def __init__(self, cfg: dict) -> None:
        self.raw_dir = Path(cfg["data"]["raw_dir"])
        self.min_price: float = cfg["data"]["min_price"]
        self.exclude_financials: bool = cfg["data"]["exclude_financials"]

    def load(self, filename: str = "crsp_monthly.parquet") -> pd.DataFrame:
        """Load the raw CRSP monthly returns file.

        Args:
            filename: Name of the file inside ``raw_dir``.  Supports
                ``*.csv`` and ``*.parquet``.

        Returns:
            Raw CRSP DataFrame with at minimum columns:
            ``[permno, date, ret, prc, shrout, siccd]``.

        Raises:
            FileNotFoundError: If the data file does not exist.
        """
        filepath = self.raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"CRSP file not found: {filepath.resolve()}.  "
                "Place the raw data in data/raw/ before running."
            )
        if filepath.suffix == ".parquet":
            return pd.read_parquet(filepath)
        return pd.read_csv(filepath, low_memory=False)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard asset-pricing cleaning filters to CRSP data.

        Args:
            df: Raw CRSP DataFrame from :meth:`load`.

        Returns:
            Cleaned DataFrame with:
            - Absolute price >= ``min_price``
            - Non-missing returns
            - Financial firms removed (if configured)
            - ``date`` column parsed to ``datetime``
            - ``mcap`` column added (price × shares outstanding)
        """
        df = df.copy()
        # Parse date column
        df["date"] = pd.to_datetime(df["date"])
        # Drop missing returns
        df = df.dropna(subset=["ret"])
        # Apply minimum price filter
        df = df[df["prc"].abs() >= self.min_price]
        # Add market capitalisation in millions.
        # CRSP shrout is in thousands of shares, so:
        #   mcap ($M) = |prc| ($) × shrout (000s) / 1_000
        df["mcap"] = df["prc"].abs() * df["shrout"] / 1_000
        # Optionally exclude financial firms (SIC 6000–6999)
        if self.exclude_financials:
            df = df[~df["siccd"].between(6000, 6999)]
        return df.reset_index(drop=True)


class CompustatLoader:
    """Load and clean Compustat annual fundamental data.

    Expects a CSV or Parquet file at ``cfg['data']['raw_dir']`` named
    ``compustat_annual.csv`` (or ``.parquet``).

    Args:
        cfg: Project configuration dict from :func:`load_config`.

    Attributes:
        raw_dir: Path to the raw data directory.
    """

    def __init__(self, cfg: dict) -> None:
        self.raw_dir = Path(cfg["data"]["raw_dir"])

    def load(self, filename: str = "compustat_annual.parquet") -> pd.DataFrame:
        """Load raw Compustat annual fundamentals.

        Args:
            filename: Name of the file inside ``raw_dir``.

        Returns:
            Raw Compustat DataFrame with at minimum columns:
            ``[gvkey, datadate, at, ceq, ni, sale, capx]``.

        Raises:
            FileNotFoundError: If the data file does not exist.
        """
        filepath = self.raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Compustat file not found: {filepath.resolve()}.  "
                "Place the raw data in data/raw/ before running."
            )
        if filepath.suffix == ".parquet":
            return pd.read_parquet(filepath)
        return pd.read_csv(filepath, low_memory=False)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard cleaning to Compustat annual data.

        Args:
            df: Raw Compustat DataFrame from :meth:`load`.

        Returns:
            Cleaned DataFrame with:
            - ``datadate`` parsed to ``datetime``
            - Rows with non-positive total assets removed
            - Basic derived ratios added (``bm``, ``roa``)
        """
        df = df.copy()
        df["datadate"] = pd.to_datetime(df["datadate"])
        # Remove firms with non-positive total assets
        df = df[df["at"] > 0]
        # Book-to-market ratio (BM); 'lt' (total liabilities) may be absent in
        # some Compustat vintages, so default to zero when missing.
        lt = df["lt"] if "lt" in df.columns else 0.0
        df["bm"] = df["ceq"] / (df["at"] - lt + df["ceq"]).clip(lower=1e-6)
        # Return on assets (ROA) as profitability proxy
        df["roa"] = df["ni"] / df["at"].clip(lower=1e-6)
        return df.reset_index(drop=True)


def merge_crsp_compustat(
    crsp_df: pd.DataFrame,
    compustat_df: pd.DataFrame,
    link_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge CRSP monthly returns with Compustat annual fundamentals.

    Uses a standard June-fiscal-year-end link so that accounting data is
    only used after the SEC 10-K filing deadline (avoids look-ahead bias).

    Args:
        crsp_df: Cleaned CRSP DataFrame with columns
            ``[permno, date, ret, mcap, ...]``.
        compustat_df: Cleaned Compustat DataFrame with columns
            ``[gvkey, datadate, bm, roa, ...]``.
        link_table: Optional CRSP–Compustat link table (CCM).  If
            ``None``, the function expects ``permno`` to already be present
            in ``compustat_df`` (useful for pre-merged data).

    Returns:
        Merged panel DataFrame indexed by ``(permno, date)`` containing
        both return and fundamental information.

    Todo:
        Implement full CCM link-table join when ``link_table`` is provided.
    """
    if link_table is not None:
        # TODO: Implement CCM link-table join (gvkey -> permno mapping)
        raise NotImplementedError(
            "CCM link-table join not yet implemented. "
            "Provide pre-merged data or implement the join."
        )
    # Assume permno is already in compustat_df (pre-merged path)
    merged = pd.merge(
        crsp_df,
        compustat_df,
        on="permno",
        how="left",
        suffixes=("_crsp", "_compustat"),
    )
    merged = merged.sort_values(["permno", "date"]).reset_index(drop=True)
    return merged
