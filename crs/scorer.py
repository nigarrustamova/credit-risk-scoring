"""Scoring service used by the API and the demo: load the artifacts once, then score an applicant for any loan
amount / term / income. History features stay fixed (they describe the past); application features are recomputed."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import explain, features, policy
from .calibration import Calibrator

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


class Scorer:
    def __init__(self, artifacts: Path = ARTIFACTS):
        a = Path(artifacts)
        self.booster = lgb.Booster(model_file=str(a / "m2.txt"))
        self.cal = Calibrator.load(a / "calibrator.json")
        meta = json.loads((a / "meta.json").read_text())
        self.cols: list[str] = meta["features"]
        self.categories: dict[str, list] = meta["categories"]
        self.cutoffs: dict = meta["cutoffs"]
        self.meta = meta
        self.profiles = pd.read_parquet(a / "profiles.parquet") if (a / "profiles.parquet").exists() else None

    # -----------------------------------------------------------------------------------------------------------
    def _frame(self, app_row: pd.DataFrame, hist_row: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        X = features.combine(features.app_features(app_row), hist_row)
        for c in self.cols:
            if c not in X:
                X[c] = np.nan
        X = X[self.cols].copy()
        unknown = []
        for c in self.cols:                       # one row with a missing value must not turn a number into text
            if c not in self.categories:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
        for c, levels in self.categories.items():
            v = X[c].astype(object).where(X[c].notna(), None)
            unknown += [f"{c}={x}" for x in v if x is not None and str(x) not in levels]
            X[c] = pd.Categorical(v.astype(str).where(v.notna(), None), categories=levels)
        return X, unknown

    def score_frame(self, X: pd.DataFrame) -> np.ndarray:
        return self.cal(self.booster.predict(X))

    def _pd(self, app: dict, hist: dict):
        app_row = pd.DataFrame([app])
        hist_row = pd.DataFrame([hist], index=pd.Index([app.get("SK_ID_CURR", 0)], name="SK_ID_CURR"))
        X, unknown = self._frame(app_row, hist_row)
        raw = float(self.booster.predict(X)[0])
        return X, raw, float(self.cal([raw])[0]), unknown

    def assess(self, app: dict, hist: dict, explain_top: int = 3, annual_rate: float = 0.0,
               offers: bool = True) -> dict:
        """app: raw application fields (Home Credit names); hist: precomputed history features for this applicant.
        Counter-offers are re-scored by the model and kept only if they would not be declined."""
        X, raw, pd_, unknown = self._pd(app, hist)
        contrib = explain.contributions(self.booster, X).iloc[0]
        existing = hist.get("BURO_ACTIVE_EST_MONTHLY_PAYMENT_SUM")
        existing = float(existing) if existing is not None and np.isfinite(existing) else 0.0
        d = policy.decide(pd_, _num(app.get("AMT_ANNUITY")), existing, _num(app.get("AMT_INCOME_TOTAL")),
                          self.cutoffs["approve"], self.cutoffs["decline"])
        if unknown:
            d.flags.append("UNKNOWN_CATEGORY")
            d.reasons_en.append(f"Values not seen in training were treated as missing: {', '.join(unknown[:3])}.")
            d.reasons_az.append(f"Təlimdə olmayan dəyərlər boş kimi qəbul edildi: {', '.join(unknown[:3])}.")
            if d.decision == "APPROVE":
                d.decision = "CONDITIONAL"
        out = d.to_dict()
        out.update({"raw_score": round(raw, 5), "existing_monthly_payments": round(existing, 2),
                    "explanation": explain.reasons(contrib, explain_top)})
        if offers and d.decision != "APPROVE" and "MISSING_INPUT" not in d.flags:
            out["counter_offers"] = self.counter_offers(app, hist, existing, annual_rate)
        return out

    def counter_offers(self, app: dict, hist: dict, existing: float, annual_rate: float) -> list[dict]:
        """Largest amount per term with debt service <= 45% of income, then re-scored; keep only offers the full
        decision layer would not decline."""
        out = []
        amount, income = _num(app.get("AMT_CREDIT")), _num(app.get("AMT_INCOME_TOTAL"))
        goods_ratio = _num(app.get("AMT_GOODS_PRICE")) / amount if amount else 1.0
        for o in policy.counter_offers(amount, income, existing, annual_rate=annual_rate):
            if o["max_amount"] <= 0:
                continue
            a = {**app, "AMT_CREDIT": o["max_amount"], "AMT_ANNUITY": o["monthly"],
                 "AMT_GOODS_PRICE": o["max_amount"] * (goods_ratio if np.isfinite(goods_ratio) else 1.0)}
            r = self.assess(a, hist, annual_rate=annual_rate, offers=False)
            if r["decision"] != "DECLINE":
                out.append({**o, "pd": r["pd"], "decision": r["decision"]})
        return out

    def profile(self, sk_id: int) -> tuple[dict, dict]:
        """A stored applicant from the hold-out set: (raw application fields, history features)."""
        row = self.profiles.loc[self.profiles["SK_ID_CURR"] == sk_id].iloc[0]
        app_cols = self.meta["profile_app_columns"]
        app = {c: (np.nan if pd.isna(row[c]) else row[c].item() if hasattr(row[c], "item") else row[c]) for c in app_cols}
        hist = {c: (np.nan if pd.isna(row[c]) else float(row[c])) for c in self.profiles.columns
                if c not in app_cols and c not in ("TARGET",)}
        return app, hist
