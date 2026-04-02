"""Build processed panel data for linear-attention thesis experiments.

Usage:
    python scripts/build_panel.py --config config/default.yaml

Pipeline:
1) Load raw CRSP and Compustat data.
2) Clean each source with configured filters.
3) Merge into a panel keyed by (permno, date).
4) Engineer and normalise thesis characteristics.
5) Save processed panel to data/processed/panel.parquet.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data.features import compute_firm_characteristics, normalize_features
from src.data.loader import CRSPLoader, CompustatLoader, load_config, merge_crsp_compustat


LOGGER = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure script logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _save_panel(panel: pd.DataFrame, output_path: Path) -> None:
    """Save processed panel and ensure destination directory exists.

    Args:
        panel: Processed long-format panel.
        output_path: Output parquet path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)


def build_panel(config_path: str) -> Path:
    """Run panel-building pipeline from YAML config.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Path to the saved processed panel parquet file.
    """
    cfg = load_config(config_path)
    data_cfg = cfg["data"]

    crsp_loader = CRSPLoader(cfg)
    comp_loader = CompustatLoader(cfg)

    LOGGER.info("Loading CRSP and Compustat raw datasets")
    crsp_raw = crsp_loader.load()
    comp_raw = comp_loader.load()

    LOGGER.info("Cleaning CRSP and Compustat datasets")
    crsp_clean = crsp_loader.clean(crsp_raw)
    comp_clean = comp_loader.clean(comp_raw)

    LOGGER.info("Merging cleaned datasets into panel")
    merged = merge_crsp_compustat(crsp_clean, comp_clean)

    LOGGER.info("Computing and normalizing firm characteristics")
    features = compute_firm_characteristics(merged)
    normalised = normalize_features(features, method="rank")

    if "ret_exc_lead1m" in merged.columns:
        target = merged[["permno", "date", "ret_exc_lead1m"]].copy()
        panel = normalised.merge(target, on=["permno", "date"], how="left")
    else:
        panel = normalised.copy()

    panel = panel.sort_values(["date", "permno"]).reset_index(drop=True)

    output_dir = Path(data_cfg["processed_dir"])
    output_path = output_dir / "panel.parquet"
    _save_panel(panel=panel, output_path=output_path)

    LOGGER.info("Saved processed panel: %s", output_path)
    LOGGER.info(
        "Panel shape: rows=%d | dates=%d | assets=%d",
        len(panel),
        panel["date"].nunique(),
        panel["permno"].nunique(),
    )

    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build processed panel dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for panel construction."""
    _setup_logging()
    args = parse_args()
    build_panel(config_path=args.config)


if __name__ == "__main__":
    main()
