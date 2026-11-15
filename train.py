"""End-to-end training and evaluation.

    python train.py                 # real Kaggle data (run `python -m crs.data` first)
    python train.py --synthetic     # small synthetic data: checks the whole pipeline in ~1 minute

Steps
1. features for application_train / application_test (+ 6 history tables)
2. stratified split: 80% development, 20% hold-out (never used for any choice)
3. 5-fold CV on development for B0, B1, M1, M2 (out-of-fold scores)
4. final models refit on all development data; M2 calibrated on its out-of-fold scores
5. hold-out evaluation, stability (PSI), business analysis, fairness check, explanations
6. artifacts/ for the API and demo, reports/ for the README and slides, submission.csv for Kaggle
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from crs import business, data, explain, features, metrics, models, policy  # noqa: E402
from crs.calibration import Calibrator  # noqa: E402

import os  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = Path(os.getenv("CRS_OUT", ROOT))                 # tests point this at a temp folder
ART, REP = OUT / "artifacts", OUT / "reports"
COLORS = {"B0": "#8a8985", "B1": "#eb6834", "B2": "#c2410c", "M1": "#7fb3ea", "M2": "#2a78d6", "M2x": "#9cc3ef"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
T0 = time.time()


def _versions() -> dict:
    import platform
    from importlib.metadata import version
    out = {"python": platform.python_version()}
    for p in ("lightgbm", "scikit-learn", "pandas", "numpy", "optbinning"):
        try:
            out[p] = version(p)
        except Exception:
            out[p] = None
    return out


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def load_tables(synthetic: bool) -> dict[str, pd.DataFrame]:
    if synthetic:
        from crs import synthetic as syn
        return syn.make(6000, seed=1)
    return data.load_all()


# ---------------------------------------------------------------------------------------------------------------
def charts(ladder: dict, curves: pd.DataFrame, rel: pd.DataFrame, fam: pd.Series) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style(ax, grid="x"):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        (ax.xaxis if grid == "x" else ax.yaxis).grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    chart_ladder(ladder)
    _rest_charts(curves, rel, fam, plt, style)


def _style(ax, grid="x"):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    (ax.xaxis if grid == "x" else ax.yaxis).grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_ladder(ladder: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = _style
    names = list(ladder)
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(names) + 1.2), dpi=160)
    vals = [ladder[n]["auc"] for n in names]
    ax.barh(range(len(names)), [v - 0.5 for v in vals], left=0.5, color=[COLORS[n.split()[0]] for n in names], height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"AUC {v:.3f} · Gini {2 * v - 1:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(names)), names, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0.5, max(vals) + 0.135)
    ax.set_xlabel("ROC-AUC on the hold-out set (axis starts at 0.5 = random guessing)", color=MUTED, fontsize=9)
    style(ax)
    ax.set_title("From a simple rule to the full model", loc="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REP / "model_ladder.png")
    plt.close(fig)


def _rest_charts(curves, rel, fam, plt, style) -> None:

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    for col, name, color in (("bad_rate_b1", "B1 scorecard, 15 fields", COLORS["B1"]),
                             ("bad_rate_b2", "B2 scorecard, 50 fields", COLORS["B2"]), ("bad_rate_m2", "M2 LightGBM", COLORS["M2"])):
        ax.plot(curves["approval_rate"] * 100, curves[col] * 100, color=color, linewidth=2, label=name)
    ax.set_xlabel("Approval rate, %", color=MUTED, fontsize=9)
    ax.set_ylabel("Payment-difficulty rate among approved, %", color=MUTED, fontsize=9)
    style(ax, "y")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("At the same approval rate, M2 approves fewer future bad payers", loc="left", fontsize=12,
                 color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REP / "approval_vs_bad_rate.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    lim = max(rel[["pred_raw", "pred_cal", "observed"]].max()) * 1.08
    ax.plot([0, lim * 100], [0, lim * 100], color=GRID, linewidth=1.5, zorder=0)
    ax.plot(rel["pred_raw"] * 100, rel["observed"] * 100, "o-", color="#8a8985", markersize=5, linewidth=1.5, label="before calibration")
    ax.plot(rel["pred_cal"] * 100, rel["observed"] * 100, "o-", color=COLORS["M2"], markersize=5, linewidth=2, label="after calibration")
    ax.set_xlabel("Predicted PD, % (deciles)", color=MUTED, fontsize=9)
    ax.set_ylabel("Observed default rate, %", color=MUTED, fontsize=9)
    style(ax, "y")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Calibration on the hold-out set", loc="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REP / "calibration.png")
    plt.close(fig)

    fam = fam.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(fam) + 1.1), dpi=160)
    ax.barh(range(len(fam)), fam.values, color=COLORS["M2"], height=0.6)
    ax.set_yticks(range(len(fam)), fam.index, color=INK, fontsize=9)
    ax.set_xlabel("Mean |SHAP| (log-odds), summed per family", color=MUTED, fontsize=9)
    style(ax)
    ax.set_title("What drives the risk score", loc="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REP / "drivers.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-trees", type=int, default=6000)
    args = ap.parse_args()
    ART.mkdir(exist_ok=True)
    REP.mkdir(exist_ok=True)

    t = load_tables(args.synthetic)
    log("tables:", {k: len(v) for k, v in t.items()})
    hist = features.history_features(t)
    raw_train, raw_test = t["application_train"], t["application_test"]
    df = features.combine(features.app_features(raw_train), hist)
    df_test = features.combine(features.app_features(raw_test), hist)
    for k in [k for k in t if k not in ("application_train", "application_test")]:
        del t[k]
    log(f"features: {df.shape[1] - 2} for {len(df):,} applicants; default rate {df['TARGET'].mean():.2%}")

    y_all = df["TARGET"].astype(int)
    rng = np.random.default_rng(7)
    hold = np.zeros(len(df), bool)
    for cls in (0, 1):                                    # stratified 20% hold-out
        idx = np.where(y_all == cls)[0]
        hold[rng.choice(idx, int(round(0.2 * len(idx))), replace=False)] = True
    dev, ho = df[~hold].reset_index(drop=True), df[hold].reset_index(drop=True)
    y, y_ho = dev["TARGET"].astype(int), ho["TARGET"].astype(int)
    fold_idx = models.folds(y, args.folds)
    app_cols = models.app_columns(df)
    params = {"n_estimators": args.max_trees}

    ladder_dev, oofs, cv_models = {}, {}, {}
    all_cols = models.feature_columns(df)
    no_ext = [c for c in all_cols if not c.startswith("EXT_")]
    makers = {"B0 logistic (8 fields)": (models.B0, False), "B1 WoE scorecard": (models.B1, False),
              "B2 WoE scorecard, best 50 of all features": (lambda: models.B1(50, all_cols, "B2"), False),
              "M1 LightGBM, application only": (lambda: models.LGBM(app_cols, "M1", params), True),
              "M2x LightGBM, all tables, no external scores": (lambda: models.LGBM(no_ext, "M2x", params), True),
              "M2 LightGBM, all tables": (lambda: models.LGBM(None, "M2", params), True)}
    for name, (mk, es) in makers.items():
        log("CV", name)
        oofs[name], cv_models[name] = models.cross_validate(mk, dev, y, fold_idx, early_stop=es, log=log)
        ladder_dev[name] = metrics.report(y, oofs[name])
        log("  OOF", ladder_dev[name])

    # ---- final models on all development data -------------------------------------------------------------
    finals = {}
    for name, (mk, es) in makers.items():
        m = mk()
        if es:
            trees = int(np.mean([cm.best_iteration for cm in cv_models[name]]) * 1.1)
            m.params["n_estimators"] = max(trees, 50)
        finals[name] = m.fit(dev, y)
        log("final", name, f"({m.params['n_estimators']} trees)" if es else "")
    m2_name = "M2 LightGBM, all tables"
    m2 = finals[m2_name]
    cal = Calibrator.fit(oofs[m2_name], y)

    # ---- hold-out ---------------------------------------------------------------------------------------------
    raw_ho = {n: m.predict(ho) for n, m in finals.items()}
    ladder = {n: metrics.report(y_ho, s) for n, s in raw_ho.items()}
    pd_m2 = cal(raw_ho[m2_name])
    pd_b1 = raw_ho["B1 WoE scorecard"]
    cal_report = {"m2_raw": metrics.report(y_ho, raw_ho[m2_name]), "m2_calibrated": metrics.report(y_ho, pd_m2)}
    for n, r in ladder.items():
        log("HOLD-OUT", n, r)

    dec = pd.qcut(pd_m2, 10, labels=False, duplicates="drop")
    rel = pd.DataFrame({"pred_raw": raw_ho[m2_name], "pred_cal": pd_m2, "observed": y_ho, "dec": dec}).groupby("dec").mean()

    # stability: development scores vs hold-out and vs Kaggle's unlabeled applicants
    pd_dev_oof = cal(oofs[m2_name])
    pd_kaggle = cal(m2.predict(df_test))
    stability = {"psi_dev_vs_holdout": round(metrics.psi(pd_dev_oof, pd_m2), 4),
                 "psi_dev_vs_new_applicants": round(metrics.psi(pd_dev_oof, pd_kaggle), 4)}
    log("stability", stability)

    # business: B1 scorecard vs M2 at the same approval rates
    amount = ho["AMT_CREDIT"].astype(float).values
    rates = np.round(np.arange(0.50, 0.951, 0.05), 2)
    # rank by the raw M2 score: same order as the calibrated PD, but without the ties isotonic calibration creates
    rank_m2 = raw_ho[m2_name]
    comp = business.compare(y_ho, pd_b1, rank_m2, amount, rates=rates)
    pd_b2 = raw_ho["B2 WoE scorecard, best 50 of all features"]
    comp_b2 = business.compare(y_ho, pd_b2, rank_m2, amount, rates=rates)
    curves = pd.DataFrame({"approval_rate": np.round(np.arange(0.30, 1.0001, 0.01), 2)})
    curves["bad_rate_b1"] = [business.at_approval(y_ho, pd_b1, amount, r)["bad_rate"] for r in curves["approval_rate"]]
    curves["bad_rate_b2"] = [business.at_approval(y_ho, pd_b2, amount, r)["bad_rate"] for r in curves["approval_rate"]]
    curves["bad_rate_m2"] = [business.at_approval(y_ho, rank_m2, amount, r)["bad_rate"] for r in curves["approval_rate"]]

    # decision cut-offs: approve the lowest-risk 70% outright, decline the riskiest 15%, review the rest
    cutoffs = {"approve": float(np.quantile(pd_dev_oof, 0.70)), "decline": float(np.quantile(pd_dev_oof, 0.85))}
    bands = pd.cut(pd_m2, [-1, cutoffs["approve"], cutoffs["decline"], 2], labels=["APPROVE", "REVIEW", "DECLINE"])
    band_table = pd.DataFrame({"band": bands, "y": y_ho}).groupby("band", observed=False)["y"].agg(["size", "mean"])
    grades = pd.Series([policy.grade(p) for p in pd_m2])
    grade_table = pd.DataFrame({"grade": grades, "y": y_ho, "pd": pd_m2}).groupby("grade").agg(
        n=("y", "size"), observed=("y", "mean"), predicted=("pd", "mean"))

    # CBA debt-burden rules on the hold-out applicants
    dti = ho.get("TOTAL_DEBT_SERVICE_RATIO", ho.get("ANNUITY_INCOME_RATIO")).astype(float)
    dti_table = pd.DataFrame({"band": pd.cut(dti, [-1, 0.45, 0.70, np.inf], labels=["<=45%", "45-70%", ">70%"]),
                              "y": y_ho}).groupby("band", observed=False)["y"].agg(["size", "mean"])

    # the full decision policy end to end (PD cut-offs + CBA debt-service rules), as the scorer applies it
    decisions = np.array([policy.decide(p_, d_ * 1.0, 0.0, 1.0, cutoffs["approve"], cutoffs["decline"]).decision
                          for p_, d_ in zip(pd_m2, dti.fillna(np.inf).values)])
    policy_table = pd.DataFrame({"decision": decisions, "y": y_ho.values}).groupby("decision")["y"].agg(["size", "mean"])
    policy_table["share"] = policy_table["size"] / len(decisions)

    # fairness: approval at the 80% approval policy by group (gender is NOT a model input)
    approved = business.approve_mask(rank_m2, 0.80)
    ho_raw = raw_train.set_index("SK_ID_CURR").loc[ho["SK_ID_CURR"]].reset_index()
    groups = {}
    if "CODE_GENDER" in ho_raw:
        g = ho_raw["CODE_GENDER"].astype(str).values
        groups["gender"] = {k: {"approval": float(approved[g == k].mean()), "default_rate": float(y_ho[g == k].mean()),
                                "n": int((g == k).sum())} for k in np.unique(g) if (g == k).sum() > 100}
    age = ho["AGE_YEARS"].values
    ab = pd.cut(age, [0, 25, 35, 50, 120], labels=["<25", "25-35", "35-50", "50+"]).astype(str)
    groups["age"] = {k: {"approval": float(approved[ab == k].mean()), "default_rate": float(y_ho[ab == k].mean()),
                         "n": int((ab == k).sum())} for k in ["<25", "25-35", "35-50", "50+"] if (ab == k).sum() > 0}
    # how well the model's inputs reveal gender, although gender itself is excluded (a proxy check)
    proxy_auc = None
    if "CODE_GENDER" in ho_raw:
        from sklearn.metrics import roc_auc_score
        g_ok = ho_raw["CODE_GENDER"].astype(str).isin(["F", "M"]).values
        gy = (ho_raw["CODE_GENDER"].astype(str).values[g_ok] == "M").astype(int)
        gx = ho[m2.cols][g_ok]
        half = len(gy) // 2
        import lightgbm as lgb
        gm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, verbose=-1).fit(gx.iloc[:half], gy[:half])
        proxy_auc = float(roc_auc_score(gy[half:], gm.predict_proba(gx.iloc[half:])[:, 1]))

    # explanations
    sample = ho.sample(min(5000, len(ho)), random_state=0)
    contrib = explain.contributions(m2.model, sample[m2.cols])
    top_feats, fam = explain.global_importance(contrib)

    charts(ladder, curves, rel, fam)

    # ---- artifacts ---------------------------------------------------------------------------------------------
    m2.model.booster_.save_model(str(ART / "m2.txt"))
    cal.save(ART / "calibrator.json")
    with open(ART / "b1_scorecard.pkl", "wb") as f:
        pickle.dump(finals["B1 WoE scorecard"], f)
    cats = {c: [str(x) for x in df[c].cat.categories] for c in m2.cols if str(df[c].dtype) == "category"}
    app_raw_cols = [c for c in raw_train.columns if c != "TARGET"]
    # demo profiles: hold-out applicants (never seen in training), both defaulters and non-defaulters
    pick = pd.concat([ho[y_ho == 1].sample(min(60, int(y_ho.sum())), random_state=1),
                      ho[y_ho == 0].sample(min(240, int((y_ho == 0).sum())), random_state=1)])
    prof = raw_train.set_index("SK_ID_CURR").loc[pick["SK_ID_CURR"]].reset_index()
    prof = prof.merge(hist.reset_index(), on="SK_ID_CURR", how="left")
    prof["PD_AT_REQUEST"] = cal(m2.predict(pick)).round(4)
    prof.to_parquet(ART / "profiles.parquet", index=False)
    meta = {"features": m2.cols, "categories": cats, "cutoffs": cutoffs, "profile_app_columns": app_raw_cols,
            "trees": m2.params["n_estimators"], "trained_on": "synthetic" if args.synthetic else "home-credit-default-risk",
            "n_dev": int(len(dev)), "n_holdout": int(len(ho)), "n_features": len(m2.cols),
            "profile_mix": {"payment_difficulty": int((pick["TARGET"] == 1).sum()), "no_difficulty": int((pick["TARGET"] == 0).sum())},
            "versions": _versions()}
    (ART / "meta.json").write_text(json.dumps(meta, indent=1))

    # hold-out scores for the demo's portfolio view
    pd.DataFrame({"y": y_ho.values, "pd_m2": pd_m2, "score_m2": rank_m2, "pd_b1": pd_b1, "amount": amount,
                  "annuity": ho["AMT_ANNUITY"].astype(float).values, "income": ho["AMT_INCOME_TOTAL"].astype(float).values,
                  "dti": dti.values}).to_parquet(ART / "holdout_scores.parquet", index=False)

    sub = pd.DataFrame({"SK_ID_CURR": df_test["SK_ID_CURR"].values,
                        "TARGET": np.mean([cm.predict(df_test) for cm in cv_models[m2_name]], axis=0)})
    sub.to_csv(REP / "submission.csv", index=False)

    results = {"ladder_holdout": ladder, "ladder_cv": ladder_dev, "calibration": cal_report, "stability": stability,
               "cutoffs": cutoffs, "bands": band_table.reset_index().to_dict("records"),
               "grades": grade_table.reset_index().to_dict("records"),
               "dti": dti_table.reset_index().to_dict("records"), "business": comp.to_dict("records"),
               "business_vs_b2": comp_b2.to_dict("records"), "policy": policy_table.reset_index().to_dict("records"),
               "gender_proxy_auc": proxy_auc,
               "fairness": groups, "top_features": top_feats.to_dict("records"),
               "driver_families": fam.round(4).to_dict(), "meta": meta | {"features": len(m2.cols)},
               "minutes": round((time.time() - T0) / 60, 1)}
    (REP / "results.json").write_text(json.dumps(results, indent=1, default=float))
    from crs import report
    report.write(results, REP)
    log("done ->", ART, REP)


if __name__ == "__main__":
    main()
