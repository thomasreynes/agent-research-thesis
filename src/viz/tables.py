"""LaTeX table generation for the AIPM Linear Attention thesis.

All tables default to saving in ``outputs/tables/`` (configurable via
``config/default.yaml → visualization.table_dir``).

Generated files can be directly ``\\input{}`` into the thesis LaTeX
document.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


_TABLE_DIR = Path("outputs/tables")


def regression_table(
    results_list: "List[src.analysis.regressions.RegressionResults]",
    save_path: Optional[str] = None,
    caption: str = "Panel regression results",
    label: str = "tab:regressions",
) -> str:
    """Generate a publication-quality regression table in LaTeX.

    Produces a multi-column table where each column corresponds to one
    :class:`~src.analysis.regressions.RegressionResults` object.
    Coefficients are shown with t-statistics in parentheses; stars
    indicate significance at 10%/5%/1% levels.

    Args:
        results_list: List of regression result objects, one per column.
        save_path: Path to save the ``.tex`` file.  Defaults to
            ``outputs/tables/regression_table.tex``.
        caption: LaTeX table caption.
        label: LaTeX ``\\label{}`` identifier.

    Returns:
        LaTeX table string (also written to ``save_path``).
    """
    save_path = save_path or str(_TABLE_DIR / "regression_table.tex")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Collect all variable names across specifications
    all_vars: List[str] = []
    for res in results_list:
        for k in res.coefficients:
            if k != "const" and k not in all_vars:
                all_vars.append(k)

    n_cols = len(results_list)
    col_headers = " & ".join(
        [f"({i + 1}) {res.hypothesis}" for i, res in enumerate(results_list)]
    )

    def _stars(pval: float) -> str:
        if pval < 0.01:
            return "***"
        if pval < 0.05:
            return "**"
        if pval < 0.10:
            return "*"
        return ""

    rows: List[str] = []
    for var in all_vars:
        coef_cells = []
        tstat_cells = []
        for res in results_list:
            coef = res.coefficients.get(var)
            tstat = res.t_stats.get(var)
            pval = res.p_values.get(var, 1.0)
            if coef is None:
                coef_cells.append("")
                tstat_cells.append("")
            else:
                coef_cells.append(f"{coef:.4f}{_stars(pval)}")
                tstat_cells.append(f"({tstat:.2f})")
        rows.append(f"\\textit{{{var}}} & " + " & ".join(coef_cells) + " \\\\")
        rows.append("& " + " & ".join(tstat_cells) + " \\\\")

    # Footer rows
    r2_cells = " & ".join([f"{res.r_squared:.3f}" for res in results_list])
    n_cells = " & ".join([f"{res.n_obs:,}" for res in results_list])

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{l" + "r" * n_cols + "}\n"
        "\\hline\\hline\n"
        f" & {col_headers} \\\\\n"
        "\\hline\n"
        + "\n".join(rows) + "\n"
        "\\hline\n"
        f"$R^2$ & {r2_cells} \\\\\n"
        f"$N$ & {n_cells} \\\\\n"
        "\\hline\\hline\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\small\n"
        "\\item \\textit{Notes:} t-statistics in parentheses. "
        "*, **, *** denote significance at 10\\%, 5\\%, 1\\% levels. "
        "Standard errors are clustered as indicated in the text.\n"
        "\\end{tablenotes}\n"
        "\\end{table}\n"
    )

    Path(save_path).write_text(latex, encoding="utf-8")
    return latex


def summary_statistics_table(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    caption: str = "Summary statistics",
    label: str = "tab:summary",
) -> str:
    """Generate a LaTeX summary statistics table.

    Reports mean, standard deviation, 25th percentile, median, 75th
    percentile, and number of observations for each column.

    Args:
        df: DataFrame containing the variables to summarise.
        columns: Subset of columns to include.  Defaults to all numeric
            columns.
        save_path: Path to save the ``.tex`` file.  Defaults to
            ``outputs/tables/summary_statistics.tex``.
        caption: LaTeX table caption.
        label: LaTeX ``\\label{}`` identifier.

    Returns:
        LaTeX table string (also written to ``save_path``).
    """
    save_path = save_path or str(_TABLE_DIR / "summary_statistics.tex")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if columns is None:
        columns = list(df.select_dtypes(include="number").columns)

    stats = df[columns].describe(percentiles=[0.25, 0.5, 0.75]).T[
        ["count", "mean", "std", "25%", "50%", "75%"]
    ]
    stats.columns = ["N", "Mean", "Std", "P25", "Median", "P75"]

    latex_body = stats.to_latex(
        float_format="{:.4f}".format,
        caption=caption,
        label=label,
        column_format="l" + "r" * len(stats.columns),
        escape=False,
    )

    Path(save_path).write_text(latex_body, encoding="utf-8")
    return latex_body
