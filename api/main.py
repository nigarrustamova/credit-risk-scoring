"""Scoring API.

    uvicorn api.main:app --reload            # http://127.0.0.1:8000/docs

POST /score with a stored applicant id and the requested loan returns the decision, PD, risk grade, debt-service
ratio, the reasons (English and Azerbaijani) and counter-offers when the request does not fit.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from crs.policy import monthly_payment
from crs.scorer import Scorer

app = FastAPI(title="Installment credit risk scoring", version="1.0")


@lru_cache(maxsize=1)
def scorer() -> Scorer:
    art = os.getenv("CRS_ARTIFACTS")
    return Scorer(Path(art)) if art else Scorer()


class LoanRequest(BaseModel):
    """Leave a field empty to keep the applicant's original value. The original monthly payment is kept unless the
    amount or the term changes; then it is recomputed (amount change at the same term: scaled; new term: annuity
    at `annual_rate`)."""
    model_config = {"allow_inf_nan": False}
    sk_id_curr: int = Field(..., description="Applicant id from /applicants")
    amount: float | None = Field(None, gt=0, le=1e8, description="Loan amount (Home Credit currency units)")
    months: int | None = Field(None, ge=3, le=120, description="Term in months")
    annual_rate: float = Field(0.0, ge=0, le=1, description="Annual interest rate used when the term changes")
    income: float | None = Field(None, gt=0, le=1e9, description="Documented income; default: the declared income")


@app.get("/health")
def health() -> dict:
    s = scorer()
    return {"status": "ok", "model_trees": s.meta["trees"], "features": len(s.cols), "trained_on": s.meta["trained_on"]}


@app.get("/applicants")
def applicants(limit: int = 50) -> list[dict]:
    p = scorer().profiles
    cols = ["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "PD_AT_REQUEST"]
    return p[cols].head(limit).to_dict("records")


def original_term(app_row: dict) -> int:
    return max(int(round(app_row["AMT_CREDIT"] / app_row["AMT_ANNUITY"])), 1)


def apply_request(app_row: dict, req: LoanRequest) -> dict:
    """Change only what the request changes, so an unchanged request scores exactly the original application."""
    a = dict(app_row)
    term_changed = req.months is not None and req.months != original_term(app_row)
    if req.amount is not None and abs(req.amount - a["AMT_CREDIT"]) > 0.5:
        k = req.amount / a["AMT_CREDIT"]
        a["AMT_CREDIT"] = req.amount
        if a.get("AMT_GOODS_PRICE") is not None and a["AMT_GOODS_PRICE"] == a["AMT_GOODS_PRICE"]:
            a["AMT_GOODS_PRICE"] = a["AMT_GOODS_PRICE"] * k       # keep the goods-to-credit ratio
        if not term_changed:
            a["AMT_ANNUITY"] = a["AMT_ANNUITY"] * k               # same term and implied rate, scaled amount
    if term_changed:
        a["AMT_ANNUITY"] = round(monthly_payment(a["AMT_CREDIT"], req.months, req.annual_rate), 2)
    if req.income is not None:
        a["AMT_INCOME_TOTAL"] = req.income
    return a


@app.post("/score")
def score(req: LoanRequest) -> dict:
    s = scorer()
    try:
        app_row, hist = s.profile(req.sk_id_curr)
    except (IndexError, KeyError):
        raise HTTPException(404, f"unknown applicant {req.sk_id_curr}")
    a = apply_request(app_row, req)
    out = s.assess(a, hist, annual_rate=req.annual_rate)
    out["request"] = {"amount": a["AMT_CREDIT"], "monthly_payment": a["AMT_ANNUITY"], "income": a["AMT_INCOME_TOTAL"]}
    return out
