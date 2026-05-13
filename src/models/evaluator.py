"""
src/models/evaluator.py

Reliability and validity metrics for the trained models.

Includes
--------
- Pearson r between predicted and ground-truth scores
- Mean Absolute Error per trait
- Test-retest reliability (ICC) across multiple calls of the same CEO/user
- Cronbach's alpha for scale internal consistency
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


# ── Prediction accuracy ───────────────────────────────────────────────────────

def evaluate_predictions(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    traits: List[str] = None,
) -> pd.DataFrame:
    """
    Compute MAE and Pearson r for each trait.

    Parameters
    ----------
    y_true, y_pred : pd.DataFrame  — columns = trait abbreviations (O,C,E,A,N)

    Returns
    -------
    pd.DataFrame  — index = traits, columns = [mae, pearson_r, pearson_p]
    """
    if traits is None:
        traits = list(y_true.columns)

    rows = []
    for trait in traits:
        yt = y_true[trait].values
        yp = y_pred[trait].values

        mae          = mean_absolute_error(yt, yp)
        r, p         = stats.pearsonr(yt, yp)
        rows.append({"trait": trait, "mae": mae, "pearson_r": r, "pearson_p": p})

    df = pd.DataFrame(rows).set_index("trait")
    logger.info("Evaluation complete:\n%s", df.to_string())
    return df


# ── Test-retest reliability ───────────────────────────────────────────────────

def intraclass_correlation(
    scores_per_session: List[np.ndarray],
) -> float:
    """
    Two-way mixed-effects ICC(3,1) — appropriate for test-retest reliability.

    Parameters
    ----------
    scores_per_session : list of 1-D arrays
        Each array is the trait scores from one session (same user, different
        comment samples). All arrays must be the same length.

    Returns
    -------
    float  — ICC estimate in [0, 1]; > 0.75 is considered good.
    """
    k = len(scores_per_session)
    if k < 2:
        raise ValueError("Need at least 2 sessions for ICC")

    n = len(scores_per_session[0])
    data = np.column_stack(scores_per_session)   # (n_subjects, k_raters)

    grand_mean   = data.mean()
    row_means    = data.mean(axis=1)
    col_means    = data.mean(axis=0)

    SSw = sum(((data[i] - row_means[i]) ** 2).sum() for i in range(n))
    SSb = k * ((row_means - grand_mean) ** 2).sum()
    SSc = n * ((col_means - grand_mean) ** 2).sum()
    SSe = SSw - SSc

    MSb = SSb / (n - 1)
    MSe = SSe / ((n - 1) * (k - 1))

    icc = (MSb - MSe) / (MSb + (k - 1) * MSe)
    return float(icc)


# ── Cronbach's alpha ──────────────────────────────────────────────────────────

def cronbach_alpha(data: np.ndarray) -> float:
    """
    Cronbach's alpha for internal consistency of a scale.

    Parameters
    ----------
    data : np.ndarray  — shape (n_subjects, n_items)

    Returns
    -------
    float  — alpha; > 0.70 generally acceptable
    """
    n_items = data.shape[1]
    item_variances = data.var(axis=0, ddof=1)
    total_variance = data.sum(axis=1).var(ddof=1)
    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return float(alpha)


# ── Summary report ────────────────────────────────────────────────────────────

def reliability_report(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> str:
    """
    Human-readable reliability report.
    """
    metrics = evaluate_predictions(y_true, y_pred)
    lines   = ["─" * 55, "  Reliability Report", "─" * 55]
    lines.append(f"{'Trait':<20} {'MAE':>8} {'Pearson r':>12} {'p-value':>10}")
    lines.append("─" * 55)
    for trait, row in metrics.iterrows():
        sig = "***" if row["pearson_p"] < 0.001 else ("**" if row["pearson_p"] < 0.01 else "*")
        lines.append(
            f"  {trait:<18} {row['mae']:>8.3f} {row['pearson_r']:>12.3f}{sig:>6}"
        )
    lines.append("─" * 55)
    return "\n".join(lines)
