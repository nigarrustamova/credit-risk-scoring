"""Turning model scores into business numbers.

- bad rate at a fixed approval rate, and approval rate at a fixed bad rate (model A vs model B)
- a loss PROXY among approved loans = sum of the original loan amount of approved applicants with payment
  difficulties x LGD (no amortisation, cures or recoveries; the target is early delinquency, not default)
- expected credit loss of the approved book, ECL = PD x LGD x EAD

LGD (loss given default) is an ASSUMPTION (default 45%, a common placeholder for unsecured lending; it is not a
regulatory value for retail). The percentage changes reported do not depend on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LGD = 0.45


def approve_mask(pd_scores, approval_rate: float) -> np.ndarray:
    """Approve the approval_rate share of applicants with the lowest PD."""
    pd_scores = np.asarray(pd_scores, float)
    cut = np.quantile(pd_scores, approval_rate)
    return pd_scores <= cut


def at_approval(y, pd_scores, amount, approval_rate: float, lgd: float = LGD) -> dict:
    y, amount = np.asarray(y), np.asarray(amount, float)
    m = approve_mask(pd_scores, approval_rate)
    defaulted = (y == 1) & m
    return {"approval_rate": approval_rate, "approved": int(m.sum()), "bad_rate": float(y[m].mean()),
            "approved_amount": float(amount[m].sum()), "realised_loss": float(amount[defaulted].sum() * lgd),
            "loss_rate": float(amount[defaulted].sum() * lgd / max(amount[m].sum(), 1)),
            "ecl": float(np.sum(np.asarray(pd_scores)[m] * lgd * amount[m])),
            "pd_cutoff": float(np.quantile(pd_scores, approval_rate))}


def approval_at_bad_rate(y, pd_scores, target_bad_rate: float) -> float:
    """Highest approval rate whose bad rate stays at or below target_bad_rate."""
    y = np.asarray(y)[np.argsort(pd_scores)]
    cum_bad = np.cumsum(y) / np.arange(1, len(y) + 1)
    ok = np.where(cum_bad <= target_bad_rate)[0]
    return float((ok.max() + 1) / len(y)) if len(ok) else 0.0


def compare(y, pd_base, pd_new, amount, rates=(0.5, 0.6, 0.7, 0.8, 0.9), lgd: float = LGD) -> pd.DataFrame:
    rows = []
    for r in rates:
        a, b = at_approval(y, pd_base, amount, r, lgd), at_approval(y, pd_new, amount, r, lgd)
        rows.append({"approval_rate": r, "bad_rate_base": a["bad_rate"], "bad_rate_new": b["bad_rate"],
                     "bad_rate_change": b["bad_rate"] / a["bad_rate"] - 1,
                     "loss_rate_base": a["loss_rate"], "loss_rate_new": b["loss_rate"],
                     "loss_change": b["realised_loss"] / a["realised_loss"] - 1 if a["realised_loss"] else np.nan,
                     "extra_approvals_same_bad_rate": approval_at_bad_rate(y, pd_new, a["bad_rate"]) - r})
    return pd.DataFrame(rows)


def illustrate(portfolio_azn: float, loss_rate_base: float, loss_rate_new: float) -> dict:
    """Scale the measured loss-rate change to a portfolio size (an illustration, not a forecast)."""
    return {"portfolio_azn": portfolio_azn, "loss_base_azn": portfolio_azn * loss_rate_base,
            "loss_new_azn": portfolio_azn * loss_rate_new, "saving_azn": portfolio_azn * (loss_rate_base - loss_rate_new)}
