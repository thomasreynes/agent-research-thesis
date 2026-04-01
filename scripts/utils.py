"""Shared utilities for the linear-attention asset pricing thesis.

Provides config loading, seed setting, logging setup, and data loading
helpers used across all entry-point scripts.
"""

import logging
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Filesystem path to the YAML config file.

    Returns:
        Parsed config as a nested dictionary.
    """
    with open(path, "r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    return cfg


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for the built-in ``random`` module, NumPy, and PyTorch
    (both CPU and CUDA). Call this once at the start of every script.

    Args:
        seed: Integer seed value (loaded from config, never hardcoded).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a timestamped console handler.

    Args:
        level: Logging level string, e.g. ``"INFO"``, ``"DEBUG"``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_data(config: dict[str, Any]) -> pd.DataFrame:
    """Load the processed panel dataset specified in *config*.

    Reads the Parquet file at ``config['data']['processed_path']``, parses
    the date column, and returns a tidy long-format panel DataFrame with
    columns for entity ID, date, characteristics, and the return column.

    Args:
        config: Config dict containing at minimum::

            data:
              processed_path: "data/processed/panel.parquet"
              date_col: "date"
              entity_col: "permno"

    Returns:
        Panel DataFrame sorted by ``(date_col, entity_col)``.

    Raises:
        FileNotFoundError: If the processed Parquet file does not exist.
    """
    data_cfg = config["data"]
    path: str = data_cfg["processed_path"]
    date_col: str = data_cfg["date_col"]
    entity_col: str = data_cfg["entity_col"]

    logger = logging.getLogger(__name__)
    logger.info("Loading panel data from %s", path)

    df = pd.read_parquet(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col, entity_col]).reset_index(drop=True)

    logger.info(
        "Loaded %d rows | %d unique dates | %d unique entities",
        len(df),
        df[date_col].nunique(),
        df[entity_col].nunique(),
    )
    return df
