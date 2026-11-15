"""Isotonic calibration: maps the model's score to a probability of default that matches observed default rates.
Fitted on out-of-fold predictions, so the calibrator never sees a score the model made on its own training rows.
Stored as two arrays, so the API needs only numpy to apply it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self, x=None, y=None):
        self.x = np.asarray(x if x is not None else [0.0, 1.0], float)
        self.y = np.asarray(y if y is not None else [0.0, 1.0], float)

    PD_FLOOR = 0.0005      # Basel III retail PD floor (0.05%)
    PD_CAP = 0.95

    @classmethod
    def fit(cls, scores, outcomes) -> "Calibrator":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=cls.PD_FLOOR, y_max=cls.PD_CAP).fit(scores, outcomes)
        return cls(iso.X_thresholds_, iso.y_thresholds_)

    def __call__(self, scores):
        return np.interp(np.asarray(scores, float), self.x, self.y)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"x": self.x.tolist(), "y": self.y.tolist()}))

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        d = json.loads(Path(path).read_text())
        return cls(d["x"], d["y"])
