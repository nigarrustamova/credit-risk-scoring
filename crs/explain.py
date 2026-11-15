"""Per-applicant explanations from LightGBM's built-in SHAP values (pred_contrib=True): the same TreeSHAP values the
`shap` package computes, without the extra dependency at serving time.

Each feature is mapped to a plain reason in English and Azerbaijani, so a loan officer (and the customer) can read
why the score went up or down. Features of one family (e.g. all late-payment aggregates) are merged into one reason.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# (regex on feature name, family key, English, Azerbaijani). First match wins; order matters.
# Families in CUSTOMER_FACING may be given to the applicant as reasons; the others are for internal review only
# (age, family, region, education and similar are sensitive or not something the applicant can act on).
FAMILIES = [
    (r"^EXT_", "ext", "Credit bureau scores", "Kredit bürosu skorları"),
    (r"^(INST_.*(LATE|DPD|UNDERPAID|SHORTFALL)|POS_.*DPD|CC_.*DPD)", "late", "Late or partial payments on past loans",
     "Keçmiş kreditlər üzrə gecikmiş və ya natamam ödənişlər"),
    (r"^INST_", "inst", "Repayment history on past loans", "Keçmiş kreditlərin ödəniş tarixçəsi"),
    (r"^BURO_.*(OVERDUE|BB_LATE|BB_WORST|DAY_OVERDUE)", "buro_late", "Overdue loans at other lenders (credit bureau)",
     "Digər kreditorlarda gecikmiş kreditlər (kredit bürosu)"),
    (r"^(BURO_.*(DEBT|ACTIVE)|BURO_DEBT_INCOME)", "buro_debt", "Current debt at other lenders",
     "Digər kreditorlarda mövcud borc"),
    (r"^BURO_", "buro", "Credit history length and activity (bureau)", "Kredit tarixçəsinin uzunluğu və aktivliyi (büro)"),
    (r"^PREV_.*REFUSED", "refused", "Previously refused applications", "Əvvəllər rədd edilmiş müraciətlər"),
    (r"^PREV_POS", "pos_prev", "Past point-of-sale installment loans", "Keçmiş mağaza taksit kreditləri"),
    (r"^PREV_", "prev", "Past applications with this lender", "Bu kreditorla keçmiş müraciətlər"),
    (r"^POS_", "pos", "Point-of-sale and cash loan history", "Mağaza və nağd kredit tarixçəsi"),
    (r"^CC_", "cc", "Credit card use", "Kredit kartından istifadə"),
    (r"(ANNUITY_INCOME|TOTAL_DEBT_SERVICE|CREDIT_INCOME|INCOME_PER_PERSON|AMT_INCOME)", "burden",
     "Payment burden relative to income", "Ödəniş yükünün gəlirə nisbəti"),
    (r"(CREDIT_TERM|PAYMENT_RATE|AMT_ANNUITY|AMT_CREDIT|GOODS|CREDIT_MINUS_GOODS)", "loan",
     "Loan amount and term", "Kreditin məbləği və müddəti"),
    (r"(DAYS_EMPLOYED|EMPLOYED_TO_AGE|INCOME_TYPE|OCCUPATION|ORGANIZATION)", "job", "Employment", "İş yeri və iş stajı"),
    (r"(DAYS_BIRTH|AGE_YEARS)", "age", "Age", "Yaş"),
    (r"(DAYS_ID_PUBLISH|ID_PUBLISH)", "id", "How recently the ID document was issued", "Şəxsiyyət vəsiqəsinin nə vaxt verildiyi"),
    (r"DAYS_LAST_PHONE_CHANGE|PHONE_CHANGE", "phone", "Recent change of phone number", "Telefon nömrəsinin yaxında dəyişməsi"),
    (r"^FLAG_(MOBIL|EMP_PHONE|WORK_PHONE|CONT_MOBILE|PHONE|EMAIL)$", "contact", "Contact details provided",
     "Təqdim olunmuş əlaqə məlumatları"),
    (r"(REGION|CITY)", "region", "Region", "Region"),
    (r"(EDUCATION)", "edu", "Education", "Təhsil"),
    (r"(FAMILY|CHILDREN|FAM_MEMBERS)", "family", "Family situation", "Ailə vəziyyəti"),
    (r"(OWN_CAR|OWN_REALTY|HOUSING)", "assets", "Owned car or property", "Mülkiyyətdə avtomobil və ya əmlak"),
    (r"(_AVG|_MODE|_MEDI|TOTALAREA)$", "home", "Details of the applicant's home building", "Yaşadığı binanın göstəriciləri"),
    (r"SOCIAL_CIRCLE", "social", "Defaults among the applicant's contacts", "Tanışları arasında ödəməmə halları"),
    (r"DAYS_REGISTRATION", "registration", "How recently the registration changed", "Qeydiyyatın nə vaxt dəyişdiyi"),
    (r"DOCUMENT", "docs", "Documents provided", "Təqdim olunmuş sənədlər"),
    (r"APPR_PROCESS", "timing", "Day and hour of the application", "Müraciətin günü və saatı"),
]
_COMPILED = [(re.compile(p), k, en, az) for p, k, en, az in FAMILIES]
CUSTOMER_FACING = {"ext", "late", "inst", "buro_late", "buro_debt", "buro", "refused", "pos_prev", "prev", "pos", "cc",
                   "burden", "loan", "job"}
OTHER = ("other", "Other application details", "Müraciətin digər məlumatları")


def family(feature: str) -> tuple[str, str, str]:
    for rx, k, en, az in _COMPILED:
        if rx.search(feature):
            return k, en, az
    return OTHER


def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP values in log-odds, one column per feature (the last column, the base value, is dropped)."""
    booster = model.booster_ if hasattr(model, "booster_") else model
    vals = booster.predict(X, pred_contrib=True)
    return pd.DataFrame(vals[:, :-1], columns=X.columns, index=X.index)


def reasons(contrib_row: pd.Series, top: int = 3) -> dict:
    """Group SHAP values by family; return the strongest risk-increasing and risk-decreasing reasons.
    `increase_risk` / `decrease_risk` hold only reasons that may be given to the customer;
    `internal` holds every family, for the analyst."""
    agg: dict[str, list] = {}
    for feat, v in contrib_row.items():
        if not np.isfinite(v) or v == 0:
            continue
        k, en, az = family(str(feat))
        a = agg.setdefault(k, [0.0, en, az, k])
        a[0] += float(v)
    items = sorted(agg.values(), key=lambda a: a[0])
    fmt = lambda lst: [{"en": a[1], "az": a[2], "impact": round(a[0], 3)} for a in lst]  # noqa: E731
    cust = [a for a in items if a[3] in CUSTOMER_FACING]
    return {"increase_risk": fmt([a for a in reversed(cust) if a[0] > 0][:top]),
            "decrease_risk": fmt([a for a in cust if a[0] < 0][:top]),
            "internal": fmt(sorted(items, key=lambda a: -abs(a[0]))[:8])}


def global_importance(contrib: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    imp = contrib.abs().mean().sort_values(ascending=False)
    fam = pd.Series({f: family(f)[1] for f in imp.index})
    by_family = imp.groupby(fam).sum().sort_values(ascending=False)
    return pd.DataFrame({"feature": imp.index[:top], "mean_abs_shap": imp.values[:top].round(4)}), by_family
