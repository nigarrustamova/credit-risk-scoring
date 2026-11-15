"""Feature engineering: one row per applicant (SK_ID_CURR) from the application and 6 history tables.

Two kinds of features:
- application features (`app_features`): computed from the current application only. The demo recomputes them when
  the operator changes the loan amount, term or income, so they must stay a pure function of one row.
- history features (`history_features`): aggregates of the credit bureau, previous applications, installment
  payments, POS/cash loans and credit cards. These describe past behaviour and do not change with the new loan.

All functions only use the columns that exist, so the same code runs on the real data and on the small
synthetic tables used in tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_ANOMALY = 365243            # Home Credit's placeholder for "not applicable" in DAYS_* columns
KEY = "SK_ID_CURR"

# Fields excluded from the model by policy: protected (gender), not about the applicant's own credit behaviour
# (contacts' defaults, building statistics, who came with the applicant), or arbitrary (day/hour of the application).
EXCLUDED_EXACT = {"CODE_GENDER", "NAME_TYPE_SUITE", "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START",
                  "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE", "TOTALAREA_MODE"}


def is_excluded(col: str) -> bool:
    return (col in EXCLUDED_EXACT or "SOCIAL_CIRCLE" in col or col.endswith(("_AVG", "_MODE", "_MEDI")))


def _div(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def _has(df: pd.DataFrame, *cols: str) -> bool:
    return all(c in df.columns for c in cols)


# ---------------------------------------------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------------------------------------------
def app_features(app: pd.DataFrame) -> pd.DataFrame:
    """Application-level features. Works on any number of rows (the API scores one row at a time)."""
    df = app.copy()
    if "DAYS_EMPLOYED" in df:
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_ANOMALY).astype("int8")
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] != DAYS_ANOMALY)
    df = df.drop(columns=[c for c in df.columns if is_excluded(c)])
    if _has(df, "AMT_CREDIT", "AMT_INCOME_TOTAL"):
        df["CREDIT_INCOME_RATIO"] = _div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    if _has(df, "AMT_ANNUITY", "AMT_INCOME_TOTAL"):
        # payment-to-income of the NEW loan. Home Credit does not state the income period; annuity / income is the
        # standard ratio used with this data and is read as a debt-service ratio.
        df["ANNUITY_INCOME_RATIO"] = _div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    if _has(df, "AMT_CREDIT", "AMT_ANNUITY"):
        df["CREDIT_TERM"] = _div(df["AMT_CREDIT"], df["AMT_ANNUITY"])          # ~number of payments
        df["PAYMENT_RATE"] = _div(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    if _has(df, "AMT_GOODS_PRICE", "AMT_CREDIT"):
        df["GOODS_CREDIT_RATIO"] = _div(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
        df["CREDIT_MINUS_GOODS"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
    if _has(df, "DAYS_BIRTH"):
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    if _has(df, "DAYS_EMPLOYED", "DAYS_BIRTH"):
        df["EMPLOYED_TO_AGE"] = _div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    if _has(df, "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"):
        df["INCOME_PER_PERSON"] = _div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    if _has(df, "DAYS_ID_PUBLISH", "DAYS_BIRTH"):
        df["ID_PUBLISH_TO_AGE"] = _div(df["DAYS_ID_PUBLISH"], df["DAYS_BIRTH"])
    if _has(df, "DAYS_LAST_PHONE_CHANGE", "DAYS_BIRTH"):
        df["PHONE_CHANGE_TO_AGE"] = _div(df["DAYS_LAST_PHONE_CHANGE"], df["DAYS_BIRTH"])
    ext = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in df]
    if ext:
        e = df[ext].astype("float64")
        df["EXT_MEAN"] = e.mean(axis=1)
        df["EXT_MIN"] = e.min(axis=1)
        df["EXT_MAX"] = e.max(axis=1)
        df["EXT_STD"] = e.std(axis=1)
        df["EXT_PROD"] = e.prod(axis=1, min_count=len(ext))
        df["EXT_NAN_COUNT"] = e.isna().sum(axis=1)
    docs = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if docs:
        df["DOCUMENT_COUNT"] = df[docs].sum(axis=1)
        df = df.drop(columns=docs)
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("category")
    return df


# ---------------------------------------------------------------------------------------------------------------
# history tables
# ---------------------------------------------------------------------------------------------------------------
def _agg(df: pd.DataFrame, by: str, prefix: str, spec: dict[str, list[str]]) -> pd.DataFrame:
    spec = {c: a for c, a in spec.items() if c in df.columns}
    if not spec or df.empty:
        return pd.DataFrame(index=pd.Index([], name=by))
    g = df.groupby(by, observed=True).agg(spec)
    g.columns = [f"{prefix}_{c}_{a.upper()}" for c, a in g.columns]
    return g.astype("float32")


def _numeric(df: pd.DataFrame, exclude: tuple[str, ...]) -> list[str]:
    return [c for c in df.select_dtypes("number").columns if c not in exclude and not c.startswith("SK_ID")]


CARD_MIN_PAYMENT = 0.05       # monthly payment on a revolving line: 5% of the balance
MIN_TERM_MONTHS = 12          # never spread an instalment loan's remaining debt over fewer months than this
DEFAULT_TERM_MONTHS = 24      # remaining term when the end date is missing or already passed


def estimate_monthly_payment(b: pd.DataFrame) -> pd.Series:
    """Estimated monthly payment of each ACTIVE bureau loan (0 for closed loans).

    1. the reported annuity when there is one;
    2. credit cards / credit lines: 5% of the outstanding balance;
    3. instalment loans: outstanding debt / remaining months, with at least 12 months, and 24 months when the end
       date is missing or already passed (otherwise the whole debt would count as one month's payment).
    The CBA defines debt burden from contractual monthly repayments; a lender would take those from the credit
    registry. This estimate only stands in for them in the public data."""
    idx = b.index
    active = b["CREDIT_ACTIVE"].astype(str).eq("Active") if "CREDIT_ACTIVE" in b else pd.Series(False, index=idx)
    debt = b["AMT_CREDIT_SUM_DEBT"].astype("float64").clip(lower=0).fillna(0) if "AMT_CREDIT_SUM_DEBT" in b \
        else pd.Series(0.0, index=idx)
    revolving = b["CREDIT_TYPE"].astype(str).str.contains("card|line", case=False) if "CREDIT_TYPE" in b \
        else pd.Series(False, index=idx)
    if "DAYS_CREDIT_ENDDATE" in b:
        months_left = b["DAYS_CREDIT_ENDDATE"].astype("float64") / 30.4
        months_left = months_left.where(months_left > 1, DEFAULT_TERM_MONTHS).fillna(DEFAULT_TERM_MONTHS)
    else:
        months_left = pd.Series(float(DEFAULT_TERM_MONTHS), index=idx)
    est = pd.Series(np.where(revolving, debt * CARD_MIN_PAYMENT, debt / months_left.clip(lower=MIN_TERM_MONTHS)), index=idx)
    if "AMT_ANNUITY" in b:
        ann = b["AMT_ANNUITY"].astype("float64")
        est = ann.where(ann > 0, est)
    return est.where(active, 0.0).fillna(0.0)


def bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame | None) -> pd.DataFrame:
    b = bureau.copy()
    for c in ("DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE"):
        if c in b:
            b[c] = b[c].where(b[c].abs() < 50000)
    for c in ("DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE"):    # facts/updates cannot be after the application date
        if c in b:
            b[c] = b[c].where(b[c] <= 0)
    if bureau_balance is not None and not bureau_balance.empty and "STATUS" in bureau_balance:
        bb = bureau_balance.copy()
        st = bb["STATUS"].astype(str)
        bb["LATE"] = st.isin(["1", "2", "3", "4", "5"]).astype("int8")          # 30+ days past due in that month
        bb["LATE_DEPTH"] = pd.to_numeric(st.where(st.str.isdigit()), errors="coerce")
        g = bb.groupby("SK_ID_BUREAU").agg(BB_MONTHS=("MONTHS_BALANCE", "size"), BB_LATE_SHARE=("LATE", "mean"),
                                           BB_WORST=("LATE_DEPTH", "max"))
        b = b.merge(g, left_on="SK_ID_BUREAU", right_index=True, how="left")
    if _has(b, "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM"):
        b["DEBT_CREDIT_RATIO"] = _div(b["AMT_CREDIT_SUM_DEBT"], b["AMT_CREDIT_SUM"])
    if _has(b, "AMT_CREDIT_SUM_OVERDUE", "AMT_CREDIT_SUM_DEBT"):
        b["OVERDUE_DEBT_RATIO"] = _div(b["AMT_CREDIT_SUM_OVERDUE"], b["AMT_CREDIT_SUM_DEBT"])
    active = b["CREDIT_ACTIVE"].astype(str).eq("Active") if "CREDIT_ACTIVE" in b else pd.Series(False, index=b.index)
    b["EST_MONTHLY_PAYMENT"] = estimate_monthly_payment(b)

    num = _numeric(b, exclude=())
    spec = {c: ["mean", "max"] for c in num}
    for c in ("AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE", "CNT_CREDIT_PROLONG",
              "EST_MONTHLY_PAYMENT"):
        if c in spec:
            spec[c] = ["mean", "max", "sum"]
    if "DAYS_CREDIT" in spec:
        spec["DAYS_CREDIT"] = ["min", "max", "mean", "size"]
    out = _agg(b, KEY, "BURO", spec)
    if "CREDIT_ACTIVE" in b:
        out = out.join(_agg(b[active], KEY, "BURO_ACTIVE", {c: ["mean", "sum"] for c in
                       ("AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DAYS_CREDIT", "EST_MONTHLY_PAYMENT", "BB_LATE_SHARE")}),
                       how="left")
        out["BURO_ACTIVE_SHARE"] = b.assign(a=active.astype("float32")).groupby(KEY)["a"].mean().astype("float32")
    if "CREDIT_TYPE" in b:
        b["IS_CONSUMER"] = b["CREDIT_TYPE"].astype(str).eq("Consumer credit").astype("float32")
        out["BURO_CONSUMER_SHARE"] = b.groupby(KEY)["IS_CONSUMER"].mean().astype("float32")
    return out


def previous_features(prev: pd.DataFrame) -> pd.DataFrame:
    p = prev.drop(columns=[c for c in ("WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START", "NAME_TYPE_SUITE")
                           if c in prev.columns])
    for c in [c for c in p.columns if c.startswith("DAYS_")]:
        p[c] = p[c].where(p[c] != DAYS_ANOMALY)
    if _has(p, "AMT_APPLICATION", "AMT_CREDIT"):
        p["APP_CREDIT_RATIO"] = _div(p["AMT_APPLICATION"], p["AMT_CREDIT"])
    if _has(p, "AMT_CREDIT", "AMT_ANNUITY"):
        p["CREDIT_TERM"] = _div(p["AMT_CREDIT"], p["AMT_ANNUITY"])
    status = p["NAME_CONTRACT_STATUS"].astype(str) if "NAME_CONTRACT_STATUS" in p else pd.Series("", index=p.index)
    ctype = p["NAME_CONTRACT_TYPE"].astype(str) if "NAME_CONTRACT_TYPE" in p else pd.Series("", index=p.index)
    p["REFUSED"] = status.eq("Refused").astype("float32")
    p["APPROVED"] = status.eq("Approved").astype("float32")
    p["IS_CONSUMER_LOAN"] = ctype.eq("Consumer loans").astype("float32")      # point-of-sale installment loans

    num = _numeric(p, exclude=("SELLERPLACE_AREA",))
    spec = {c: ["mean", "max"] for c in num}
    for c in ("REFUSED", "APPROVED", "IS_CONSUMER_LOAN"):
        if c in spec:
            spec[c] = ["mean", "sum"]
    if "DAYS_DECISION" in spec:
        spec["DAYS_DECISION"] = ["min", "max", "mean", "size"]
    out = _agg(p, KEY, "PREV", spec)
    small = {c: ["mean", "max"] for c in ("AMT_CREDIT", "AMT_ANNUITY", "APP_CREDIT_RATIO", "CNT_PAYMENT",
                                           "DAYS_DECISION", "RATE_DOWN_PAYMENT", "CREDIT_TERM")}
    out = out.join(_agg(p[status.eq("Approved")], KEY, "PREV_APPROVED", small), how="left")
    out = out.join(_agg(p[status.eq("Refused")], KEY, "PREV_REFUSED", small), how="left")
    out = out.join(_agg(p[ctype.eq("Consumer loans")], KEY, "PREV_POS", small), how="left")
    return out


def installments_features(ins: pd.DataFrame) -> pd.DataFrame:
    i = ins.copy()
    if _has(i, "DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"):
        diff = i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]
        i["DPD"] = diff.clip(lower=0)                     # days paid late
        i["DBD"] = (-diff).clip(lower=0)                  # days paid early
        i["LATE"] = (diff > 0).astype("float32")
        i["LATE_30"] = (diff > 30).astype("float32")
    if _has(i, "AMT_PAYMENT", "AMT_INSTALMENT"):
        i["PAYMENT_RATIO"] = _div(i["AMT_PAYMENT"], i["AMT_INSTALMENT"])
        i["PAYMENT_SHORTFALL"] = (i["AMT_INSTALMENT"] - i["AMT_PAYMENT"]).clip(lower=0)
        i["UNDERPAID"] = (i["AMT_PAYMENT"] < i["AMT_INSTALMENT"] - 0.01).astype("float32")
    spec = {"NUM_INSTALMENT_VERSION": ["nunique"], "DPD": ["max", "mean", "sum"], "DBD": ["max", "mean"],
            "LATE": ["mean", "sum"], "LATE_30": ["mean", "sum"], "PAYMENT_RATIO": ["mean", "min"],
            "PAYMENT_SHORTFALL": ["max", "mean", "sum"], "UNDERPAID": ["mean"],
            "AMT_INSTALMENT": ["max", "mean", "sum"], "AMT_PAYMENT": ["min", "max", "mean", "sum"],
            "DAYS_ENTRY_PAYMENT": ["max", "mean"], "SK_ID_PREV": ["nunique"]}
    out = _agg(i, KEY, "INST", spec)
    if "DAYS_INSTALMENT" in i:
        out["INST_COUNT"] = i.groupby(KEY).size().astype("float32")
        recent = i[i["DAYS_INSTALMENT"] >= -365]          # the last 12 months say most about current behaviour
        out = out.join(_agg(recent, KEY, "INST_12M", {"DPD": ["max", "mean"], "LATE": ["mean"], "LATE_30": ["mean"],
                                                       "PAYMENT_RATIO": ["mean"], "UNDERPAID": ["mean"]}), how="left")
    return out


def pos_features(pos: pd.DataFrame) -> pd.DataFrame:
    p = pos.copy()
    if "NAME_CONTRACT_STATUS" in p:
        p["COMPLETED"] = p["NAME_CONTRACT_STATUS"].astype(str).eq("Completed").astype("float32")
    if "SK_DPD_DEF" in p:
        p["DPD_DEF_POS"] = (p["SK_DPD_DEF"] > 0).astype("float32")
    spec = {"MONTHS_BALANCE": ["min", "max", "size"], "SK_DPD": ["max", "mean"], "SK_DPD_DEF": ["max", "mean"],
            "DPD_DEF_POS": ["mean"], "CNT_INSTALMENT": ["mean", "max"], "CNT_INSTALMENT_FUTURE": ["mean", "max"],
            "COMPLETED": ["mean"]}
    out = _agg(p, KEY, "POS", spec)
    if "MONTHS_BALANCE" in p:
        out = out.join(_agg(p[p["MONTHS_BALANCE"] >= -12], KEY, "POS_12M",
                            {"SK_DPD": ["max", "mean"], "SK_DPD_DEF": ["max"]}), how="left")
    return out


def credit_card_features(cc: pd.DataFrame) -> pd.DataFrame:
    c = cc.copy()
    if _has(c, "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"):
        c["UTILIZATION"] = _div(c["AMT_BALANCE"], c["AMT_CREDIT_LIMIT_ACTUAL"])
    if _has(c, "AMT_PAYMENT_CURRENT", "AMT_INST_MIN_REGULARITY"):
        c["PAID_OVER_MIN"] = _div(c["AMT_PAYMENT_CURRENT"], c["AMT_INST_MIN_REGULARITY"])
    if "SK_DPD" in c:
        c["DPD_POS"] = (c["SK_DPD"] > 0).astype("float32")
    num = _numeric(c, exclude=())
    spec = {k: ["mean", "max"] for k in num}
    if "MONTHS_BALANCE" in spec:
        spec["MONTHS_BALANCE"] = ["min", "size"]
    out = _agg(c, KEY, "CC", spec)
    if "MONTHS_BALANCE" in c:
        out = out.join(_agg(c[c["MONTHS_BALANCE"] >= -6], KEY, "CC_6M",
                            {"UTILIZATION": ["mean", "max"], "DPD_POS": ["mean"], "PAID_OVER_MIN": ["mean"]}),
                       how="left")
    return out


def history_features(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    if "bureau" in t:
        parts.append(bureau_features(t["bureau"], t.get("bureau_balance")))
    if "previous_application" in t:
        parts.append(previous_features(t["previous_application"]))
    if "installments_payments" in t:
        parts.append(installments_features(t["installments_payments"]))
    if "POS_CASH_balance" in t:
        parts.append(pos_features(t["POS_CASH_balance"]))
    if "credit_card_balance" in t:
        parts.append(credit_card_features(t["credit_card_balance"]))
    out = pd.concat(parts, axis=1) if parts else pd.DataFrame()
    out.index.name = KEY
    return out


# ---------------------------------------------------------------------------------------------------------------
def combine(app_feats: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Join application and history features and add the cross-table ratios (total debt service, etc.)."""
    df = app_feats.join(hist, on=KEY, how="left").copy()
    existing = df.get("BURO_ACTIVE_EST_MONTHLY_PAYMENT_SUM")
    if existing is not None and _has(df, "AMT_ANNUITY", "AMT_INCOME_TOTAL"):
        df["TOTAL_DEBT_SERVICE_RATIO"] = _div(df["AMT_ANNUITY"] + existing.fillna(0), df["AMT_INCOME_TOTAL"])
    if _has(df, "BURO_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM", "AMT_INCOME_TOTAL"):
        df["BURO_DEBT_INCOME_RATIO"] = _div(df["BURO_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], df["AMT_INCOME_TOTAL"])
    return df


def build(t: dict[str, pd.DataFrame], app_table: str = "application_train") -> pd.DataFrame:
    return combine(app_features(t[app_table]), history_features(t))
