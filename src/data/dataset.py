"""Dataset utilities for monthly cross-sectional asset pricing tensors.

This module builds month-by-month tensors from the processed panel data.
Each item corresponds to a single month t with:
- X_t: characteristic matrix (N_t, D)
- r_next_t: next-month excess returns vector (N_t,)

These tensors are the direct inputs for the linear attention equation
A_t = X_t W X_t^T (Kelly et al. 2025, linear case).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MonthlyBatch:
    """Container for one monthly cross-section.

    Attributes:
        date: Month-end timestamp for the batch.
        permnos: Asset identifiers in the same order as X and r_next.
        X: Characteristic matrix with shape (N_t, D).
        r_next: Next-month excess return vector with shape (N_t,).
    """

    date: pd.Timestamp
    permnos: list[int]
    X: torch.Tensor
    r_next: torch.Tensor


class MonthlyCrossSectionDataset(Dataset[MonthlyBatch]):
    """PyTorch dataset that yields one cross-section per month.

    This dataset preserves varying cross-sectional sizes across months, which
    is required by the thesis panel setting. The output is suitable for
    linear-attention training and decomposition scripts.

    Args:
        panel_df: Processed panel DataFrame containing at least date/entity,
            feature columns, and return target.
        feature_cols: Ordered list of characteristic column names for X_t.
        date_col: Name of the date column in panel_df.
        entity_col: Name of the entity identifier column in panel_df.
        target_col: Name of the next-month excess return target column.
        min_assets_per_month: Minimum number of assets required to keep a
            month in the dataset.

    Raises:
        ValueError: If required columns are missing or no valid months remain.
    """

    def __init__(
        self,
        panel_df: pd.DataFrame,
        feature_cols: list[str],
        date_col: str = "date",
        entity_col: str = "permno",
        target_col: str = "ret_exc_lead1m",
        min_assets_per_month: int = 20,
    ) -> None:
        self._feature_cols = feature_cols
        self._date_col = date_col
        self._entity_col = entity_col
        self._target_col = target_col

        required = [date_col, entity_col, target_col] + feature_cols
        missing = [col for col in required if col not in panel_df.columns]
        if missing:
            raise ValueError(f"Missing required columns for dataset: {missing}")

        df = panel_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([date_col, entity_col]).reset_index(drop=True)

        monthly_batches: list[MonthlyBatch] = []
        for month, group in df.groupby(date_col, sort=True):
            clean = group.dropna(subset=feature_cols + [target_col])
            if len(clean) < min_assets_per_month:
                continue

            permnos = clean[entity_col].astype(int).tolist()
            x_values = clean[feature_cols].to_numpy(dtype="float32")
            r_values = clean[target_col].to_numpy(dtype="float32")

            monthly_batches.append(
                MonthlyBatch(
                    date=pd.Timestamp(month),
                    permnos=permnos,
                    X=torch.from_numpy(x_values),
                    r_next=torch.from_numpy(r_values),
                )
            )

        if not monthly_batches:
            raise ValueError(
                "No valid monthly batches found. Check input data and filters."
            )

        self._batches = monthly_batches

    def __len__(self) -> int:
        """Return the number of monthly cross-sections."""
        return len(self._batches)

    def __getitem__(self, index: int) -> MonthlyBatch:
        """Return one monthly batch by index.

        Args:
            index: Position of the month in chronological order.

        Returns:
            MonthlyBatch containing date, permnos, X, and r_next.
        """
        return self._batches[index]

    def dates(self) -> list[pd.Timestamp]:
        """Return all batch dates in chronological order."""
        return [batch.date for batch in self._batches]

    def as_metadata_frame(self) -> pd.DataFrame:
        """Summarize dataset composition for logging and audit trails.

        Returns:
            DataFrame with one row per month and the number of assets.
        """
        rows: list[dict[str, Any]] = []
        for batch in self._batches:
            rows.append({"date": batch.date, "n_assets": len(batch.permnos)})
        return pd.DataFrame(rows)
