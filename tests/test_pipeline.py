"""Tests on synthetic data: they run anywhere (CI included) without the Kaggle files."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crs import business, explain, features, metrics, policy, synthetic  # noqa: E402
from crs.calibration import Calibrator  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    return synthetic.make(1500, seed=3)


def test_features_one_row_per_applicant(tables):
    df = features.build(tables)
    assert len(df) == len(tables["application_train"])
    assert df["SK_ID_CURR"].is_unique
    for c in ("ANNUITY_INCOME_RATIO", "EXT_MEAN", "INST_LATE_MEAN", "BURO_ACTIVE_EST_MONTHLY_PAYMENT_SUM",
              "PREV_REFUSED_MEAN", "TOTAL_DEBT_SERVICE_RATIO"):
        assert c in df, c


def test_gender_is_not_a_feature(tables):
    assert "CODE_GENDER" not in features.app_features(tables["application_train"])


def test_days_employed_anomaly_becomes_missing():
    a = pd.DataFrame({"SK_ID_CURR": [1, 2], "DAYS_EMPLOYED": [365243, -1000], "DAYS_BIRTH": [-12000, -12000]})
    f = features.app_features(a)
    assert np.isnan(f.loc[0, "DAYS_EMPLOYED"]) and f.loc[0, "DAYS_EMPLOYED_ANOM"] == 1
    assert f.loc[1, "DAYS_EMPLOYED"] == -1000


def test_app_features_are_row_independent(tables):
    """The demo rescores one row: the result must not depend on the other rows."""
    a = tables["application_train"]
    full = features.app_features(a).iloc[[5]].reset_index(drop=True)
    one = features.app_features(a.iloc[[5]]).reset_index(drop=True)
    num = full.select_dtypes("number").columns
    pd.testing.assert_frame_equal(full[num], one[num], check_dtype=False)


def test_metrics():
    y = np.array([0, 0, 1, 1])
    r = metrics.report(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert r["auc"] == 1.0 and r["gini"] == 1.0 and r["ks"] == 1.0
    assert metrics.psi(np.random.default_rng(0).random(5000), np.random.default_rng(1).random(5000)) < 0.01


def test_calibrator_is_monotone_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    s = rng.random(5000)
    y = (rng.random(5000) < s ** 2).astype(int)
    c = Calibrator.fit(s, y)
    out = c(np.linspace(0, 1, 50))
    assert np.all(np.diff(out) >= -1e-12)
    c.save(tmp_path / "c.json")
    assert np.allclose(Calibrator.load(tmp_path / "c.json")(s[:100]), c(s[:100]))


def test_policy_cba_limits():
    d = policy.decide(0.02, new_payment=800, existing_payments=0, income=1000, pd_approve=0.05, pd_decline=0.10)
    assert d.decision == "DECLINE" and "CBA_DTI_OVER_70" in d.flags
    d = policy.decide(0.02, new_payment=500, existing_payments=0, income=1000, pd_approve=0.05, pd_decline=0.10)
    assert d.decision == "CONDITIONAL" and "CBA_HIGH_BURDEN_45" in d.flags
    d = policy.decide(0.02, new_payment=200, existing_payments=100, income=1000, pd_approve=0.05, pd_decline=0.10)
    assert d.decision == "APPROVE" and not d.flags
    d = policy.decide(0.20, new_payment=100, existing_payments=0, income=1000, pd_approve=0.05, pd_decline=0.10)
    assert d.decision == "DECLINE" and d.reasons_az and d.reasons_en


def test_policy_fails_closed_on_missing_inputs():
    nan = float("nan")
    for args in [(0.05, nan, 0, 1000), (nan, 100, 0, 1000), (0.05, 100, 0, nan), (0.05, 100, 0, 0),
                 (0.05, float("inf"), 0, 1000), (0.05, None, 0, 1000)]:
        d = policy.decide(*args, pd_approve=0.08, pd_decline=0.15)
        assert d.decision != "APPROVE" and "MISSING_INPUT" in d.flags, args
    d = policy.decide(0.05, 100, nan, 1000, pd_approve=0.08, pd_decline=0.15)   # unknown existing debt -> 0
    assert d.decision == "APPROVE"


def test_existing_payment_estimate_is_bounded():
    b = pd.DataFrame({"SK_ID_CURR": [1, 1, 1, 1], "CREDIT_ACTIVE": ["Active", "Active", "Active", "Closed"],
                      "CREDIT_TYPE": ["Consumer credit", "Credit card", "Consumer credit", "Consumer credit"],
                      "AMT_CREDIT_SUM_DEBT": [24000.0, 10000.0, 12000.0, 5000.0],
                      "DAYS_CREDIT_ENDDATE": [-100.0, np.nan, 3000.0, 100.0], "AMT_ANNUITY": [np.nan, np.nan, np.nan, 900.0]})
    est = features.estimate_monthly_payment(b).tolist()
    assert est[0] == 24000 / 24            # end date passed: spread over 24 months, not one month
    assert est[1] == 10000 * 0.05          # card: 5% of the balance
    assert abs(est[2] - 12000 / (3000 / 30.4)) < 1e-6
    assert est[3] == 0                     # closed loan


def test_policy_excluded_fields_are_not_features(tables):
    a = tables["application_train"].assign(DEF_30_CNT_SOCIAL_CIRCLE=1.0, APARTMENTS_AVG=0.1,
                                           HOUR_APPR_PROCESS_START=10, NAME_TYPE_SUITE="Family")
    cols = features.app_features(a).columns
    for c in ("CODE_GENDER", "DEF_30_CNT_SOCIAL_CIRCLE", "APARTMENTS_AVG", "HOUR_APPR_PROCESS_START", "NAME_TYPE_SUITE"):
        assert c not in cols


def test_counter_offers_respect_45_percent():
    offers = policy.counter_offers(amount=20000, income=1000, existing_payments=200)
    for o in offers:
        assert o["monthly"] + 200 <= 0.45 * 1000 + 0.01
        assert o["max_amount"] <= 20000
    assert offers[-1]["max_amount"] >= offers[0]["max_amount"]        # longer term -> larger amount


def test_explanation_families():
    assert explain.family("INST_12M_LATE_MEAN")[0] == "late"
    assert explain.family("EXT_SOURCE_2")[0] == "ext"
    assert explain.family("PREV_REFUSED_AMT_CREDIT_MEAN")[0] == "refused"
    r = explain.reasons(pd.Series({"EXT_SOURCE_2": -0.5, "INST_LATE_MEAN": 0.4, "INST_DPD_MAX": 0.2, "AGE_YEARS": 0.05}))
    assert r["increase_risk"][0]["en"].startswith("Late") and abs(r["increase_risk"][0]["impact"] - 0.6) < 1e-9
    assert r["decrease_risk"][0]["az"] == "Kredit bürosu skorları"
    assert explain.family("FLAG_EMP_PHONE")[0] == "contact"
    assert explain.family("DAYS_LAST_PHONE_CHANGE")[0] == "phone"
    r = explain.reasons(pd.Series({"AGE_YEARS": 0.9, "INST_LATE_MEAN": 0.1}))
    assert [x["en"] for x in r["increase_risk"]] == ["Late or partial payments on past loans"]   # age: internal only
    assert r["internal"][0]["en"] == "Age"


def test_business_better_model_has_lower_bad_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.08).astype(int)
    good = y * 0.6 + rng.random(20000)
    weak = y * 0.2 + rng.random(20000)
    amount = np.full(20000, 1000.0)
    comp = business.compare(y, weak, good, amount, rates=(0.8,))
    assert comp["bad_rate_new"].iloc[0] < comp["bad_rate_base"].iloc[0]
    assert comp["extra_approvals_same_bad_rate"].iloc[0] > 0


@pytest.mark.slow
def test_train_end_to_end_and_api(tmp_path):
    """Full pipeline on synthetic data into a temp folder, then the API on its artifacts."""
    env = {**os.environ, "CRS_OUT": str(tmp_path)}
    r = subprocess.run([sys.executable, "train.py", "--synthetic", "--folds", "3", "--max-trees", "200"], cwd=ROOT,
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    res = json.loads((tmp_path / "reports" / "results.json").read_text())
    assert res["ladder_holdout"]["M2 LightGBM, all tables"]["auc"] > 0.6
    assert (tmp_path / "reports" / "REPORT.md").exists()

    from crs.scorer import Scorer
    s = Scorer(tmp_path / "artifacts")
    sk = int(s.profiles["SK_ID_CURR"].iloc[0])
    app, hist = s.profile(sk)
    out = s.assess(app, hist)
    assert 0 < out["pd"] < 1 and out["decision"] in ("APPROVE", "CONDITIONAL", "DECLINE")
    assert out["explanation"]["increase_risk"] or out["explanation"]["decrease_risk"]
    big = s.assess({**app, "AMT_ANNUITY": app["AMT_INCOME_TOTAL"] * 0.9}, hist)
    assert big["decision"] == "DECLINE" and "CBA_DTI_OVER_70" in big["flags"]
    for o in big.get("counter_offers", []):            # every offer was re-scored and is not a decline
        assert o["decision"] in ("APPROVE", "CONDITIONAL") and 0 < o["pd"] < 1
    miss = s.assess({**app, "AMT_ANNUITY": float("nan")}, hist)
    assert miss["decision"] == "CONDITIONAL" and "MISSING_INPUT" in miss["flags"]
    cat_col = next(iter(s.categories))
    unk = s.assess({**app, cat_col: "never-seen-value"}, hist)
    assert "UNKNOWN_CATEGORY" in unk["flags"] and unk["decision"] != "APPROVE"

    # API on the same artifacts
    import importlib
    os.environ["CRS_ARTIFACTS"] = str(tmp_path / "artifacts")
    import api.main as api_main
    importlib.reload(api_main)
    from fastapi.testclient import TestClient
    c = TestClient(api_main.app)
    assert c.get("/health").json()["status"] == "ok"
    same = c.post("/score", json={"sk_id_curr": sk}).json()
    assert abs(same["pd"] - float(s.profiles.loc[s.profiles.SK_ID_CURR == sk, "PD_AT_REQUEST"].iloc[0])) < 1e-4
    orig = api_main.original_term(app)
    unchanged = api_main.apply_request(app, api_main.LoanRequest(sk_id_curr=sk, amount=app["AMT_CREDIT"], months=orig))
    assert unchanged["AMT_ANNUITY"] == app["AMT_ANNUITY"] and unchanged.get("AMT_GOODS_PRICE") == app.get("AMT_GOODS_PRICE")
    assert c.post("/score", json={"sk_id_curr": sk, "income": "Infinity"}).status_code == 422
    assert c.post("/score", json={"sk_id_curr": sk, "amount": -5}).status_code == 422
    assert c.post("/score", json={"sk_id_curr": 1}).status_code == 404
