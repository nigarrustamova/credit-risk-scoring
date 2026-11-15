"""reports/REPORT.md from reports/results.json (written by train.py)."""

from __future__ import annotations

from pathlib import Path

REP = Path(__file__).resolve().parents[1] / "reports"
B1, B2, M2 = "B1 WoE scorecard", "B2 WoE scorecard, best 50 of all features", "M2 LightGBM, all tables"
M2X = "M2x LightGBM, all tables, no external scores"


def _pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def _biz_table(rows, base_label):
    L = [f"| Approval rate | Bad rate, {base_label} | Bad rate, M2 | Change | Loss-proxy change* | Extra approvals at the same bad rate |",
         "|---|---|---|---|---|---|"]
    for x in rows:
        L.append(f"| {_pct(x['approval_rate'], 0)} | {_pct(x['bad_rate_base'], 2)} | {_pct(x['bad_rate_new'], 2)} | "
                 f"{x['bad_rate_change'] * 100:+.1f}% | {x['loss_change'] * 100:+.1f}% | "
                 f"{x['extra_approvals_same_bad_rate'] * 100:+.1f} pp |")
    return L


def write(r: dict, rep: Path = REP) -> None:
    lad, cv = r["ladder_holdout"], r["ladder_cv"]
    meta = r["meta"]
    b1, m2 = lad[B1], lad[M2]
    b2, m2x = lad.get(B2), lad.get(M2X)
    at = lambda rows: next(x for x in rows if abs(x["approval_rate"] - 0.8) < 1e-9)  # noqa: E731
    a1, a2 = at(r["business"]), at(r["business_vs_b2"]) if r.get("business_vs_b2") else None
    cal = r["calibration"]
    L = [
        "# Evaluation report", "",
        f"Data: {meta['trained_on']} · development {meta['n_dev']:,} applicants (5-fold CV) · hold-out "
        f"{meta['n_holdout']:,} applicants (never used for training, tuning, calibration or cut-offs) · "
        f"{meta['features']} features in M2.", "",
        f"**Target.** Home Credit's `TARGET` = the client had *payment difficulties* (late more than X days on at least "
        f"one of the first Y instalments). It is an early-delinquency flag, not a 90+ days-past-due or IFRS 9 default. "
        f"Rate in the data: {_pct(m2['default_rate'])}. \"PD\" below means the probability of this event; a lender "
        "would re-calibrate to its own default definition.", "",
        "## Model ladder (hold-out)", "", "![Model ladder](model_ladder.png)", "",
        "| Model | AUC | Gini | KS | PR-AUC | CV AUC (dev) |", "|---|---|---|---|---|---|",
    ]
    for n, x in lad.items():
        L.append(f"| {n} | {x['auc']:.4f} | {x['gini']:.4f} | {x['ks']:.4f} | {x['pr_auc']:.4f} | {cv[n]['auc']:.4f} |")
    L += ["", f"- M2 vs the compact 15-field scorecard B1: **Gini {b1['gini']:.3f} → {m2['gini']:.3f}**."]
    if b2:
        L.append(f"- M2 vs the 50-field scorecard B2 built from the **same features**: **Gini {b2['gini']:.3f} → "
                 f"{m2['gini']:.3f}**. This is the fair \"same data, better method\" gain.")
    if m2x:
        L.append(f"- Without Home Credit's anonymous external scores (`EXT_SOURCE_1-3`), which an Azerbaijani lender would "
                 f"replace with its own bureau score, M2x reaches AUC {m2x['auc']:.3f}.")
    L += ["", "## Business effect (hold-out)", "", "![Approval vs bad rate](approval_vs_bad_rate.png)", "",
          "Applicants are approved in order of score; each model approves the same share.", "",
          "**M2 vs the compact scorecard B1**", ""] + _biz_table(r["business"], "B1")
    if a2:
        L += ["", "**M2 vs the 50-field scorecard B2**", ""] + _biz_table(r["business_vs_b2"], "B2")
    L += ["", f"At 80% approval, M2 approves {abs(a1['bad_rate_change']) * 100:.0f}% fewer bad payers than B1"
          + (f" and {abs(a2['bad_rate_change']) * 100:.0f}% fewer than B2" if a2 else "") + ".", "",
          "*Loss proxy = original loan amount of approved applicants with payment difficulties × an assumed LGD. "
          "It ignores amortisation, cures and recoveries and uses an early-delinquency flag, so treat it as a "
          "direction, not a loss forecast. The percentage change does not depend on the LGD value.", "",
          "## The full decision policy (hold-out)", "",
          f"Cut-offs from development scores only: approve PD ≤ {_pct(r['cutoffs']['approve'], 2)}, decline PD > "
          f"{_pct(r['cutoffs']['decline'], 2)}, review in between; plus the CBA 70% DTI decline and the 45% high-burden "
          "review threshold, applied exactly as the scorer applies them.", "",
          "| Decision | Applicants | Share | Bad rate |", "|---|---|---|---|"]
    for b in r.get("policy", []):
        L.append(f"| {b['decision']} | {b['size']:,} | {_pct(b['share'])} | {_pct(b['mean'], 2)} |")
    L += ["", "Debt-service ratio (new payment + estimated payments on open bureau loans) / declared income:", "",
          "| DTI band | Applicants | Bad rate |", "|---|---|---|"]
    for b in r["dti"]:
        L.append(f"| {b['band']} | {b['size']:,} | {_pct(b['mean'], 2) if b['size'] else '—'} |")
    L += ["", "The DTI here is a stand-in: Home Credit does not state the income period, and payments on open bureau "
          "loans are estimated (reported annuity; else 5% of a card balance; else debt over the remaining term, at "
          "least 12 months). A lender would use contractual instalments from the credit registry. The rule engine "
          "itself is exact; its inputs here are approximate.", "",
          "## Calibration (hold-out)", "", "![Calibration](calibration.png)", "",
          "| Grade | Applicants | Predicted PD | Observed bad rate |", "|---|---|---|---|"]
    for g in r["grades"]:
        L.append(f"| {g['grade']} | {g['n']:,} | {_pct(g['predicted'], 2)} | {_pct(g['observed'], 2)} |")
    L += ["", f"Brier {cal['m2_calibrated']['brier']:.4f}, ECE {cal['m2_calibrated']['ece']:.4f}. An ECE this small is "
          "within the sampling noise of a perfectly calibrated model, so the grade table is the better evidence. "
          "Isotonic calibration fitted on out-of-fold scores; PD floored at 0.05% (Basel III retail floor) and "
          "capped at 95%.", "",
          "## Stability", "",
          f"PSI development vs hold-out {r['stability']['psi_dev_vs_holdout']:.4f}; vs Kaggle's unlabeled applicants "
          f"{r['stability']['psi_dev_vs_new_applicants']:.4f}. Both samples come from the same period, so a low PSI is "
          "expected and is **not** evidence of stability over time. The data has no dates: out-of-time stability "
          "cannot be tested and belongs in a pilot.", "",
          "## Fairness and excluded fields", "",
          "Excluded from the model by policy: gender, defaults among the applicant's contacts (`*_SOCIAL_CIRCLE`), "
          "statistics of the applicant's building (`*_AVG/_MODE/_MEDI`), who accompanied the applicant, and the day "
          "and hour of the application. Age, family status, region and education stay in the model but are never "
          "given to the customer as reasons.", ""]
    if r.get("gender_proxy_auc"):
        L.append(f"Excluding gender does not remove it: the model's inputs predict gender with AUC "
                 f"{r['gender_proxy_auc']:.2f}.")
    L += ["", "| Group | Applicants | Approval at 80% policy | Bad rate |", "|---|---|---|---|"]
    for gname, groups in r["fairness"].items():
        for k, v in groups.items():
            L.append(f"| {gname}: {k} | {v['n']:,} | {_pct(v['approval'])} | {_pct(v['default_rate'], 2)} |")
    age = r["fairness"].get("age", {})
    if "<25" in age and "50+" in age:
        ratio = age["<25"]["approval"] / age["50+"]["approval"]
        L += ["", f"Approval ratio under-25 vs 50+: **{ratio:.2f}** (the \"four-fifths\" screening rule flags values "
              "below 0.80). The gaps track observed bad rates, but this is a finding for a compliance review, not a "
              "clean bill of health."]
    L += ["", "## What drives the score", "", "![Drivers](drivers.png)", "",
          "## Limitations", "",
          "- Home Credit's public Kaggle data (2018), not Azerbaijani data. Method, pipeline and decision layer "
          "transfer; every number must be re-measured on a lender's own data.",
          "- No application dates: random split, no out-of-time test; real performance on future applicants is usually lower.",
          "- No reject inference: outcomes exist only for loans that were granted.",
          "- The target is early payment difficulty, not default; the loss figures are a proxy.",
          "- Independent validation findings and how they were addressed: [VALIDATION.md](VALIDATION.md).",
          ]
    (Path(rep) / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
