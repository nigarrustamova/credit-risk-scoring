"""Small synthetic Home Credit-like tables with a planted risk signal. Used by the tests and by CI, where the real
Kaggle data is not available. Column names and value conventions follow the real tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make(n: int = 3000, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(100001, 100001 + n)
    ext2, ext3 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    ext1 = np.where(rng.uniform(size=n) < 0.5, np.nan, rng.uniform(0, 1, n))
    income = rng.lognormal(12, 0.5, n).round(-2)
    credit = (income * rng.uniform(0.5, 5, n)).round(-2)
    term = rng.choice([6, 12, 18, 24, 36, 48], n)
    annuity = (credit / term * 1.1).round(1)
    age_days = -rng.integers(21 * 365, 68 * 365, n)
    emp = -rng.integers(30, 30 * 365, n)
    emp = np.where(rng.uniform(size=n) < 0.15, 365243, emp)
    late_history = rng.beta(1, 12, n)                         # each applicant's tendency to pay late
    dti = annuity / income
    logit = (-2.6 - 2.2 * (ext2 - 0.5) - 2.0 * (ext3 - 0.5) + 6 * late_history + 1.2 * (dti > 0.45)
             + 0.6 * (age_days > -30 * 365))
    target = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype("int8")
    app = pd.DataFrame({
        "SK_ID_CURR": ids, "TARGET": target,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "CODE_GENDER": rng.choice(["F", "M"], n), "FLAG_OWN_CAR": rng.choice(["Y", "N"], n),
        "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n), "CNT_CHILDREN": rng.integers(0, 4, n),
        "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": (credit * rng.uniform(0.8, 1.0, n)).round(-2),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "Commercial associate", "State servant"], n),
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education"], n),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Civil marriage"], n),
        "DAYS_BIRTH": age_days, "DAYS_EMPLOYED": emp, "DAYS_ID_PUBLISH": -rng.integers(0, 6000, n),
        "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 3000, n).astype(float),
        "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
        "REGION_RATING_CLIENT": rng.integers(1, 4, n),
        "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
        "FLAG_DOCUMENT_3": rng.integers(0, 2, n), "FLAG_DOCUMENT_6": rng.integers(0, 2, n),
    })
    test = app.drop(columns="TARGET").iloc[: n // 5].copy()
    test["SK_ID_CURR"] = test["SK_ID_CURR"] + 900000

    # bureau: 0-5 loans per applicant
    nb = rng.integers(0, 6, n)
    b_ids = np.repeat(ids, nb)
    m = len(b_ids)
    b_late = np.repeat(late_history, nb)
    bureau = pd.DataFrame({
        "SK_ID_CURR": b_ids, "SK_ID_BUREAU": np.arange(5000000, 5000000 + m),
        "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], m, p=[0.4, 0.6]),
        "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan"], m, p=[0.7, 0.25, 0.05]),
        "DAYS_CREDIT": -rng.integers(30, 3000, m), "CREDIT_DAY_OVERDUE": (rng.uniform(size=m) < b_late) * rng.integers(1, 90, m),
        "DAYS_CREDIT_ENDDATE": rng.integers(-2000, 1500, m).astype(float),
        "AMT_CREDIT_SUM": rng.lognormal(12, 0.8, m).round(-2), "AMT_CREDIT_SUM_DEBT": rng.lognormal(10, 1.2, m).round(-2),
        "AMT_CREDIT_SUM_OVERDUE": np.where(rng.uniform(size=m) < b_late, rng.lognormal(8, 1, m), 0.0),
        "CNT_CREDIT_PROLONG": rng.integers(0, 2, m), "AMT_ANNUITY": np.where(rng.uniform(size=m) < 0.3, rng.lognormal(9, 0.6, m), np.nan),
    })
    reps = rng.integers(1, 8, m)
    bb_ids = np.repeat(bureau["SK_ID_BUREAU"].values, reps)
    bb_late = np.repeat(b_late, reps)
    bureau_balance = pd.DataFrame({
        "SK_ID_BUREAU": bb_ids, "MONTHS_BALANCE": -np.concatenate([np.arange(r) for r in reps]),
        "STATUS": np.where(rng.uniform(size=len(bb_ids)) < bb_late, "1", rng.choice(["0", "C", "X"], len(bb_ids))),
    })

    # previous Home Credit applications
    npv = rng.integers(0, 5, n)
    p_ids = np.repeat(ids, npv)
    k = len(p_ids)
    prev = pd.DataFrame({
        "SK_ID_PREV": np.arange(2000000, 2000000 + k), "SK_ID_CURR": p_ids,
        "NAME_CONTRACT_TYPE": rng.choice(["Consumer loans", "Cash loans", "Revolving loans"], k, p=[0.5, 0.4, 0.1]),
        "AMT_ANNUITY": rng.lognormal(9, 0.6, k), "AMT_APPLICATION": rng.lognormal(11, 0.8, k).round(-2),
        "AMT_CREDIT": rng.lognormal(11, 0.8, k).round(-2), "CNT_PAYMENT": rng.choice([6, 10, 12, 24], k).astype(float),
        "NAME_CONTRACT_STATUS": np.where(rng.uniform(size=k) < np.repeat(late_history, npv) * 2, "Refused",
                                         rng.choice(["Approved", "Canceled"], k, p=[0.85, 0.15])),
        "DAYS_DECISION": -rng.integers(10, 2900, k), "RATE_DOWN_PAYMENT": rng.uniform(0, 0.3, k),
        "DAYS_TERMINATION": np.where(rng.uniform(size=k) < 0.3, 365243, -rng.integers(1, 2000, k)).astype(float),
    })
    # installment payments of previous loans
    ni = rng.integers(1, 12, k)
    i_prev = np.repeat(prev["SK_ID_PREV"].values, ni)
    i_cur = np.repeat(prev["SK_ID_CURR"].values, ni)
    i_late = np.repeat(np.repeat(late_history, npv), ni)
    q = len(i_prev)
    days_inst = -rng.integers(1, 2500, q).astype(float)
    late = rng.uniform(size=q) < i_late
    inst_amt = rng.lognormal(9, 0.6, q)
    installments = pd.DataFrame({
        "SK_ID_PREV": i_prev, "SK_ID_CURR": i_cur, "NUM_INSTALMENT_VERSION": rng.integers(0, 2, q).astype(float),
        "NUM_INSTALMENT_NUMBER": np.concatenate([np.arange(1, r + 1) for r in ni]),
        "DAYS_INSTALMENT": days_inst,
        "DAYS_ENTRY_PAYMENT": days_inst + np.where(late, rng.integers(1, 60, q), -rng.integers(0, 15, q)),
        "AMT_INSTALMENT": inst_amt, "AMT_PAYMENT": np.where(late & (rng.uniform(size=q) < 0.5), inst_amt * 0.5, inst_amt),
    })
    pos = pd.DataFrame({
        "SK_ID_PREV": i_prev, "SK_ID_CURR": i_cur, "MONTHS_BALANCE": -rng.integers(1, 80, q),
        "CNT_INSTALMENT": rng.choice([6.0, 12.0, 24.0], q), "CNT_INSTALMENT_FUTURE": rng.integers(0, 12, q).astype(float),
        "NAME_CONTRACT_STATUS": rng.choice(["Active", "Completed"], q, p=[0.8, 0.2]),
        "SK_DPD": np.where(late, rng.integers(1, 40, q), 0), "SK_DPD_DEF": np.where(late & (rng.uniform(size=q) < 0.3), 5, 0),
    })
    cc_mask = prev["NAME_CONTRACT_TYPE"].eq("Revolving loans").values
    cc_prev, cc_cur = prev["SK_ID_PREV"].values[cc_mask], prev["SK_ID_CURR"].values[cc_mask]
    nc = 6
    cc_late = np.repeat(np.repeat(late_history, npv)[cc_mask], nc)
    r = len(cc_prev) * nc
    limit = rng.choice([45000.0, 90000.0, 135000.0], r)
    cc = pd.DataFrame({
        "SK_ID_PREV": np.repeat(cc_prev, nc), "SK_ID_CURR": np.repeat(cc_cur, nc), "MONTHS_BALANCE": -np.tile(np.arange(1, nc + 1), len(cc_prev)),
        "AMT_BALANCE": limit * np.clip(rng.beta(2, 3, r) + cc_late, 0, 1.2), "AMT_CREDIT_LIMIT_ACTUAL": limit,
        "AMT_INST_MIN_REGULARITY": limit * 0.05, "AMT_PAYMENT_CURRENT": limit * 0.05 * rng.uniform(0.3, 3, r),
        "SK_DPD": np.where(rng.uniform(size=r) < cc_late, rng.integers(1, 30, r), 0),
    })
    return {"application_train": app, "application_test": test, "bureau": bureau, "bureau_balance": bureau_balance,
            "previous_application": prev, "installments_payments": installments, "POS_CASH_balance": pos,
            "credit_card_balance": cc}
