"""Data subpackage for the AIPM Linear Attention thesis.

Exposes data loading, merging, and feature engineering utilities.
"""

from src.data.loader import load_config, CRSPLoader, CompustatLoader, merge_crsp_compustat
from src.data.features import compute_firm_characteristics, normalize_features, construct_pairs
from src.data.dataset import MonthlyBatch, MonthlyCrossSectionDataset

__all__ = [
    "load_config",
    "CRSPLoader",
    "CompustatLoader",
    "merge_crsp_compustat",
    "compute_firm_characteristics",
    "normalize_features",
    "construct_pairs",
    "MonthlyBatch",
    "MonthlyCrossSectionDataset",
]
