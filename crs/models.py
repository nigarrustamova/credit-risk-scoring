"""The model ladder. Each step is measured on the same folds, so the gain of every step is visible.

B0  logistic regression on a handful of application fields  ("a simple bank rule")
B1  WoE-binned logistic scorecard, 15 fields from a hand-picked list  (a compact classic scorecard)
B2  WoE-binned logistic scorecard, best 50 of ALL features by information value  (a rich scorecard: the fair
    "same features, simpler method" benchmark)
M1  LightGBM on application features only
M2  LightGBM on application + all history features  (the main model)
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET, KEY = "TARGET", "SK_ID_CURR"

B0_FEATURES = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AGE_YEARS", "DAYS_EMPLOYED",
               "ANNUITY_INCOME_RATIO", "CREDIT_INCOME_RATIO", "CNT_CHILDREN"]

# candidates for the scorecard: fields a risk team can read and defend; binning keeps the strongest
B1_CANDIDATES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_MEAN", "AGE_YEARS", "DAYS_EMPLOYED",
                 "EMPLOYED_TO_AGE", "ANNUITY_INCOME_RATIO", "CREDIT_INCOME_RATIO", "PAYMENT_RATE", "GOODS_CREDIT_RATIO",
                 "AMT_INCOME_TOTAL", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE", "REGION_RATING_CLIENT",
                 "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "FLAG_OWN_CAR",
                 "INST_LATE_MEAN", "INST_DPD_MAX", "INST_12M_LATE_MEAN", "INST_PAYMENT_RATIO_MEAN",
                 "PREV_REFUSED_MEAN", "PREV_APPROVED_SUM", "PREV_DAYS_DECISION_MAX",
                 "BURO_DAYS_CREDIT_MAX", "BURO_ACTIVE_SHARE", "BURO_DEBT_CREDIT_RATIO_MEAN", "BURO_BB_LATE_SHARE_MEAN",
                 "POS_SK_DPD_DEF_MAX", "CC_UTILIZATION_MEAN", "TOTAL_DEBT_SERVICE_RATIO"]

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=80,
                  subsample=0.85, subsample_freq=1, colsample_bytree=0.35, reg_alpha=0.05, reg_lambda=5.0,
                  min_split_gain=0.0, n_estimators=6000, n_jobs=-1, verbose=-1)


def folds(y, n_splits: int = 5, seed: int = 42):
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (TARGET, KEY)]


def app_columns(df: pd.DataFrame) -> list[str]:
    hist = ("BURO_", "PREV_", "INST_", "POS_", "CC_")
    extra = ("TOTAL_DEBT_SERVICE_RATIO", "BURO_DEBT_INCOME_RATIO")
    return [c for c in feature_columns(df) if not c.startswith(hist) and c not in extra]


# ---------------------------------------------------------------------------------------------------------------
class B0:
    name = "B0 logistic (8 fields)"

    def __init__(self):
        self.cols = None
        self.pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500))

    def fit(self, X, y):
        self.cols = [c for c in B0_FEATURES if c in X]
        self.pipe.fit(X[self.cols].astype(float), y)
        return self

    def predict(self, X):
        return self.pipe.predict_proba(X[self.cols].astype(float))[:, 1]


class B1:
    """Classic scorecard: optimal binning -> weight of evidence -> logistic regression -> points (PDO scaling)."""
    name = "B1 WoE scorecard"

    def __init__(self, max_features: int = 15, candidates: list[str] | None = None, name: str | None = None):
        self.max_features = max_features
        self.candidates = candidates
        if name:
            self.name = name

    def fit(self, X, y):
        from optbinning import BinningProcess, Scorecard
        cols = [c for c in (self.candidates or B1_CANDIDATES) if c in X]
        cat = [c for c in cols if str(X[c].dtype) == "category"]
        Xs = X[cols].copy()
        for c in cat:
            Xs[c] = Xs[c].astype(str)
        bp = BinningProcess(cols, categorical_variables=cat, max_n_prebins=20, min_prebin_size=0.02, n_jobs=2,
                            selection_criteria={"iv": {"min": 0.02, "strategy": "highest", "top": self.max_features}})
        self.sc = Scorecard(binning_process=bp, estimator=LogisticRegression(max_iter=1000),
                            scaling_method="pdo_odds", scaling_method_params={"pdo": 20, "odds": 50, "scorecard_points": 600})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sc.fit(Xs, np.asarray(y))
        self.cols, self.cat = cols, cat
        return self

    def _prep(self, X):
        Xs = X[self.cols].copy()
        for c in self.cat:
            Xs[c] = Xs[c].astype(str)
        return Xs

    def predict(self, X):
        return self.sc.predict_proba(self._prep(X))[:, 1]

    def points(self, X):
        return self.sc.score(self._prep(X))

    def table(self) -> pd.DataFrame:
        return self.sc.table(style="summary")

    def selected(self) -> list[str]:
        return list(self.sc.binning_process_.get_support(names=True))


class LGBM:
    def __init__(self, cols: list[str] | None = None, name: str = "LightGBM", params: dict | None = None,
                 early_stopping: int = 200):
        self.cols, self.name = cols, name
        self.params = {**LGB_PARAMS, **(params or {})}
        self.early_stopping = early_stopping
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        self.cols = self.cols or feature_columns(X)
        self.model = lgb.LGBMClassifier(**self.params)
        kw = {}
        if X_val is not None:
            kw = dict(eval_set=[(X_val[self.cols], y_val)], eval_metric="auc",
                      callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
        self.model.fit(X[self.cols], y, **kw)
        return self

    def predict(self, X):
        return self.model.predict_proba(X[self.cols])[:, 1]

    @property
    def best_iteration(self) -> int:
        return int(self.model.best_iteration_ or self.model.n_estimators)


def inner_split(y_tr, frac: float = 0.1, seed: int = 0):
    """Stratified split of a training fold into fit / early-stopping parts, so the scored fold stays untouched."""
    rng = np.random.default_rng(seed)
    y_tr = np.asarray(y_tr)
    stop = np.zeros(len(y_tr), bool)
    for cls in (0, 1):
        idx = np.where(y_tr == cls)[0]
        stop[rng.choice(idx, int(round(frac * len(idx))), replace=False)] = True
    return np.where(~stop)[0], np.where(stop)[0]


def cross_validate(make_model, X: pd.DataFrame, y: pd.Series, fold_idx, early_stop: bool = False, log=print):
    """Out-of-fold predictions. For LightGBM, early stopping uses 10% of the TRAINING fold, so the scored fold's
    labels are never used for any choice."""
    oof = np.zeros(len(y))
    models = []
    for k, (tr, va) in enumerate(fold_idx):
        m = make_model()
        if early_stop:
            fit_i, stop_i = inner_split(y.iloc[tr], seed=k)
            tr_fit, tr_stop = tr[fit_i], tr[stop_i]
            m.fit(X.iloc[tr_fit], y.iloc[tr_fit], X.iloc[tr_stop], y.iloc[tr_stop])
        else:
            m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
        models.append(m)
        from sklearn.metrics import roc_auc_score
        log(f"  {getattr(m, 'name', '')} fold {k + 1}: AUC {roc_auc_score(y.iloc[va], oof[va]):.4f}"
            + (f" ({m.best_iteration} trees)" if early_stop else ""))
    return oof, models
