"""Decision layer on top of the model: Central Bank of Azerbaijan (CBA) debt-burden rules, risk grades and a
counter-offer when the requested loan does not fit.

This is a rule engine, not a model, on purpose: legal limits must be applied exactly and explained, not learned.

Rules encoded:
- CBA "Regulations on governing credit risks on a single borrower or a group of related borrowers", para 4.6:
  consumer loans are issued to borrowers whose debt-to-income (DTI) ratio is not over 70% (exceptions apply for
  loans secured by deposits or precious metals)  -> hard decline above 70%
- 45%: an internal "high burden" threshold (an earlier CBA DTI guideline used 45%); here it sends the application
  to review and is the target of the counter-offer. It is a lender policy choice, not a CBA limit.
- CBA Decision 39/3 (15 Sep 2026, as reported): where income can be documented, a behaviour-model income estimate
  may not be used; a guarantor's DTI may not exceed 70%. This model never estimates income: it uses declared or
  documented income only, and payment behaviour enters solely as risk features.
- Any missing or non-finite input fails closed: the application goes to manual review, never to auto-approval.
"""

from __future__ import annotations

import math

from dataclasses import asdict, dataclass, field

DTI_HARD_LIMIT = 0.70
DTI_HIGH_BURDEN = 0.45

# PD bands for risk grades (calibrated PD). The payment-difficulty rate in the data is ~8%.
GRADES = [("A", 0.025), ("B", 0.05), ("C", 0.08), ("D", 0.12), ("E", 1.01)]


def grade(pd_: float) -> str:
    for g, upper in GRADES:
        if pd_ < upper:
            return g
    return "E"


def monthly_payment(amount: float, months: int, annual_rate: float = 0.0) -> float:
    if months <= 0:
        return float("inf")
    r = annual_rate / 12
    if r == 0:
        return amount / months
    return amount * r / (1 - (1 + r) ** -months)


@dataclass
class Decision:
    decision: str                       # APPROVE | CONDITIONAL | DECLINE
    pd: float
    grade: str
    dti_total: float
    flags: list[str] = field(default_factory=list)
    reasons_en: list[str] = field(default_factory=list)
    reasons_az: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _finite(*xs) -> bool:
    return all(x is not None and isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def decide(pd_: float, new_payment: float, existing_payments: float, income: float,
           pd_approve: float, pd_decline: float) -> Decision:
    """pd_approve / pd_decline: cut-offs chosen from the portfolio analysis (see reports/REPORT.md)."""
    existing_payments = 0.0 if existing_payments is None or not _finite(existing_payments) else max(existing_payments, 0.0)
    if not _finite(pd_, new_payment, income) or income <= 0 or new_payment < 0:
        d = Decision("CONDITIONAL", float(pd_) if _finite(pd_) else float("nan"), "?", float("nan"), ["MISSING_INPUT"])
        d.reasons_en.append("A required input (payment, income or score) is missing or invalid: manual review.")
        d.reasons_az.append("Tələb olunan məlumat (ödəniş, gəlir və ya skor) yoxdur və ya səhvdir: əl ilə yoxlama.")
        return d
    dti = (new_payment + existing_payments) / income
    d = Decision("APPROVE", round(float(pd_), 4), grade(pd_), round(float(dti), 3))
    if dti > DTI_HARD_LIMIT:
        d.decision = "DECLINE"
        d.flags.append("CBA_DTI_OVER_70")
        d.reasons_en.append(f"Debt service would be {dti:.0%} of income; the CBA limit for consumer loans is 70%.")
        d.reasons_az.append(f"Borc ödənişi gəlirin {dti:.0%}-i olardı; istehlak kreditləri üçün AMB limiti 70%-dir.")
    elif dti > DTI_HIGH_BURDEN:
        d.flags.append("CBA_HIGH_BURDEN_45")
        d.reasons_en.append(f"Debt service is {dti:.0%} of income (above the 45% high-burden threshold): review.")
        d.reasons_az.append(f"Borc ödənişi gəlirin {dti:.0%}-idir (45%-lik yüksək yük həddindən yuxarı): yoxlama.")
    if pd_ > pd_decline:
        d.decision = "DECLINE"
        d.reasons_en.append(f"Probability of default {pd_:.1%} is above the decline cut-off {pd_decline:.1%}.")
        d.reasons_az.append(f"Ödəməmə ehtimalı {pd_:.1%} imtina həddindən ({pd_decline:.1%}) yuxarıdır.")
    elif d.decision != "DECLINE" and (pd_ > pd_approve or d.flags):
        d.decision = "CONDITIONAL"
        if pd_ > pd_approve:
            d.reasons_en.append(f"Probability of default {pd_:.1%} is in the review band.")
            d.reasons_az.append(f"Ödəməmə ehtimalı {pd_:.1%} yoxlama zonasındadır.")
    return d


def counter_offers(amount: float, income: float, existing_payments: float, terms=(6, 12, 18, 24, 36),
                   annual_rate: float = 0.0, target_dti: float = DTI_HIGH_BURDEN) -> list[dict]:
    """For each term: the largest amount whose total debt service stays within target_dti, capped at the request."""
    room = target_dti * income - max(existing_payments, 0.0)
    out = []
    for t in terms:
        if room <= 0:
            out.append({"months": t, "max_amount": 0.0, "monthly": 0.0})
            continue
        per_unit = monthly_payment(1.0, t, annual_rate)
        max_amt = min(amount, room / per_unit)
        out.append({"months": t, "max_amount": round(max_amt, 2), "monthly": round(monthly_payment(max_amt, t, annual_rate), 2)})
    return out
