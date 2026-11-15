"""Rebuild the B2 (50-feature WoE scorecard) hold-out scores with the exact split used in train.py,
and save them next to the existing hold-out scores for the swap-set analysis."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train  # noqa: E402
from crs import features, metrics, models  # noqa: E402

t = train.load_tables(False)
hist = features.history_features(t)
df = features.combine(features.app_features(t["application_train"]), hist)
y_all = df["TARGET"].astype(int)
rng = np.random.default_rng(7)
hold = np.zeros(len(df), bool)
for cls in (0, 1):
    idx = np.where(y_all == cls)[0]
    hold[rng.choice(idx, int(round(0.2 * len(idx))), replace=False)] = True
dev, ho = df[~hold].reset_index(drop=True), df[hold].reset_index(drop=True)
y, y_ho = dev["TARGET"].astype(int), ho["TARGET"].astype(int)

b2 = models.B1(50, models.feature_columns(df), "B2").fit(dev, y)
s = b2.predict(ho)
print("B2 hold-out", metrics.report(y_ho, s))

keep = ["SK_ID_CURR", "AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED"]
extra = ho[[c for c in keep if c in ho.columns]].copy()
extra["y"] = y_ho.values
extra["pd_b2"] = s
extra.to_parquet(ROOT / "artifacts" / "holdout_b2.parquet", index=False)
print("saved", len(extra))
