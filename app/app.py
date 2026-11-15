"""Demo: an installment-credit decision with explanation, CBA rules and counter-offers.

    python -m streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from api.main import LoanRequest, apply_request, original_term  # noqa: E402
from crs import business  # noqa: E402
from crs.policy import DTI_HARD_LIMIT, DTI_HIGH_BURDEN  # noqa: E402
from crs.scorer import Scorer  # noqa: E402

st.set_page_config(page_title="Installment Credit Risk", page_icon="💳", layout="wide")
ART, REP = ROOT / "artifacts", ROOT / "reports"
COLOR = {"APPROVE": "#16703c", "CONDITIONAL": "#a35a00", "DECLINE": "#a11"}
LABEL = {"APPROVE": "✅ Approve", "CONDITIONAL": "🟠 Approve with conditions / review", "DECLINE": "⛔ Decline"}


@st.cache_resource
def load() -> Scorer:
    if not (ART / "m2.txt").exists():
        st.error("No trained model in artifacts/. Run: python train.py  (or python train.py --synthetic)")
        st.stop()
    return Scorer(ART)


s = load()
prof = s.profiles

with st.sidebar:
    st.title("💳 Installment Credit Risk")
    st.caption("Probability of default for consumer and point-of-sale installment loans, with Central Bank of "
               "Azerbaijan debt-burden rules and an explanation for every decision.")
    st.markdown(f"**Model:** LightGBM, {s.meta['n_features']} features, {s.meta['trees']} trees")
    st.markdown(f"**Trained on:** {s.meta['trained_on']} ({s.meta['n_dev']:,} applicants)")
    st.caption("Target = Home Credit's *payment difficulties* flag (early late payments), not a 90-day default. "
               "Amounts are in the data's own currency units.")
    lang = st.radio("Explanation language", ["Azərbaycan", "English"], horizontal=True)
    show_truth = st.toggle("Show actual outcome (demo only)", value=False,
                           help="Applicants come from the hold-out set: the model never saw their outcome.")

tab_dec, tab_port, tab_rep = st.tabs(["🧾 Loan decision", "📊 Portfolio impact", "📈 Model report"])

with tab_dec:
    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("Application")
        ids = prof["SK_ID_CURR"].tolist()

        def label(i):
            r = prof.loc[prof["SK_ID_CURR"] == i].iloc[0]
            return f"#{i} · income {r['AMT_INCOME_TOTAL']:,.0f} · requested {r['AMT_CREDIT']:,.0f}"

        mix = s.meta.get("profile_mix", {})
        sk = st.selectbox("Applicant (from the hold-out set)", ids, format_func=label,
                          help=f"{len(ids)} hold-out applicants the model never saw: {mix.get('payment_difficulty', '?')} "
                               f"with payment difficulties (over-sampled for the demo) and {mix.get('no_difficulty', '?')} without.")
        app_row, hist = s.profile(sk)
        base_term = original_term(app_row)
        amount = st.number_input("Loan amount", min_value=1000.0, value=float(app_row["AMT_CREDIT"]), step=5000.0,
                                 help="Changing the amount at the same term scales the original monthly payment.")
        months = st.select_slider("Term, months", options=sorted({6, 12, 18, 24, 36, 48, 60, base_term}),
                                  value=base_term, help="The applicant's original term is pre-selected.")
        income = st.number_input("Documented income", min_value=1000.0, value=float(app_row["AMT_INCOME_TOTAL"]),
                                 step=5000.0, help="Declared or documented income only. The model never estimates "
                                                   "income (CBA Decision 39/3).")
        rate = st.slider("Annual interest rate, % (used when the term changes)", 0.0, 40.0, 0.0, 0.5,
                         help="0 = interest-free installments. Also applied to counter-offers.") / 100

        st.markdown("**Credit history (fixed, from the bureau and past loans)**")
        h = pd.Series(hist)

        def fmt(v, kind="num"):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            return f"{v:.0%}" if kind == "pct" else f"{v:,.0f}"
        facts = {
            "Past installments paid late": fmt(h.get("INST_LATE_MEAN"), "pct"),
            "Late in the last 12 months": fmt(h.get("INST_12M_LATE_MEAN"), "pct"),
            "Previous applications refused": fmt(h.get("PREV_REFUSED_MEAN"), "pct"),
            "Past point-of-sale installment loans": fmt(h.get("PREV_IS_CONSUMER_LOAN_SUM")),
            "Loans at other lenders (bureau)": fmt(h.get("BURO_DAYS_CREDIT_SIZE")),
            "Est. monthly payments on open loans": fmt(h.get("BURO_ACTIVE_EST_MONTHLY_PAYMENT_SUM")),
        }
        st.markdown("\n".join(f"- {k}: **{v}**" for k, v in facts.items()))

    req = LoanRequest(sk_id_curr=int(sk), amount=amount, months=months, annual_rate=rate, income=income)
    a = apply_request(app_row, req)
    out = s.assess(a, hist, annual_rate=rate)
    with right:
        st.subheader("Decision")
        c = COLOR[out["decision"]]
        st.markdown(f"<div style='border:2px solid {c};border-radius:12px;padding:14px 18px'>"
                    f"<div style='font-size:1.5rem;font-weight:700;color:{c}'>{LABEL[out['decision']]}</div>"
                    f"<div style='margin-top:6px'>Monthly payment <b>{a['AMT_ANNUITY']:,.0f}</b> · term {months} months"
                    f"</div></div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("PD (payment difficulty)", f"{out['pd']:.1%}")
        m2.metric("Risk grade", out["grade"])
        m3.metric("Debt service / income", f"{out['dti_total']:.0%}",
                  help=f"New payment + estimated payments on open loans, divided by income. CBA: >{DTI_HIGH_BURDEN:.0%} "
                       f"high burden, >{DTI_HARD_LIMIT:.0%} not allowed.")
        key = "reasons_az" if lang == "Azərbaycan" else "reasons_en"
        for r in out[key]:
            st.warning(r)
        ex = out["explanation"]
        lk = "az" if lang == "Azərbaycan" else "en"
        up_t, down_t = (("Riski artıran", "Riski azaldan") if lk == "az" else ("Raises the risk", "Lowers the risk"))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{up_t}**")
            for e in ex["increase_risk"]:
                st.markdown(f"🔺 {e[lk]}")
        with c2:
            st.markdown(f"**{down_t}**")
            for e in ex["decrease_risk"]:
                st.markdown(f"🔻 {e[lk]}")
        with st.expander("Internal analyst view (not shown to the customer)"):
            st.caption("All driver families, including ones that are never given to a customer as reasons "
                       "(age, family, region, education...).")
            for e in ex.get("internal", []):
                st.markdown(f"{'🔺' if e['impact'] > 0 else '🔻'} {e['en']} ({e['impact']:+.2f} log-odds)")
        if out["decision"] != "APPROVE" and "counter_offers" in out:
            st.markdown("**Counter-offer:** largest amount with debt service ≤ 45% of income, re-scored by the model")
            if out["counter_offers"]:
                co = pd.DataFrame(out["counter_offers"])
                co["max_amount"] = co["max_amount"].map(lambda v: f"{v:,.0f}")
                co["monthly"] = co["monthly"].map(lambda v: f"{v:,.0f}")
                co["pd"] = co["pd"].map(lambda v: f"{v:.1%}")
                co = co.rename(columns={
                    "months": "Term, months", "max_amount": "Max amount", "monthly": "Monthly payment",
                    "pd": "PD", "decision": "Decision"})
                st.dataframe(co, hide_index=True, width="stretch")
            else:
                st.caption("No smaller loan passes: the risk comes from the applicant's history, not the amount.")
        if show_truth:
            y = int(prof.loc[prof["SK_ID_CURR"] == sk, "TARGET"].iloc[0])
            st.info(f"Actual outcome of this applicant's original loan: "
                    f"{'had payment difficulties' if y else 'no payment difficulties'}.")
        with st.expander("API response (JSON)"):
            st.code(json.dumps(out, indent=1, ensure_ascii=False, default=float), language="json")

with tab_port:
    st.subheader("Same approval rate, fewer bad payers")
    hp = ART / "holdout_scores.parquet"
    if not hp.exists():
        st.info("Run train.py to create the hold-out scores.")
    else:
        h = pd.read_parquet(hp)
        st.caption(f"{len(h):,} hold-out applicants the models never saw. Each model approves the same share, lowest "
                   "risk first. Benchmark: the compact 15-field scorecard B1 (against the 50-field scorecard B2 the "
                   "gain is smaller; see the Model report).")
        rate_a = st.slider("Approval rate, %", 50, 95, 80, 5) / 100
        lgd = st.slider("Loss given default (assumption), %", 20, 80, 45, 5) / 100
        a1 = business.at_approval(h["y"], h["pd_b1"], h["amount"], rate_a, lgd)
        a2 = business.at_approval(h["y"], h.get("score_m2", h["pd_m2"]), h["amount"], rate_a, lgd)
        c1, c2, c3 = st.columns(3)
        c1.metric("Bad rate · B1 scorecard", f"{a1['bad_rate']:.2%}")
        c2.metric("Bad rate · M2 model", f"{a2['bad_rate']:.2%}", f"{(a2['bad_rate'] / a1['bad_rate'] - 1):+.1%}",
                  delta_color="inverse")
        c3.metric("Loss-proxy change", f"{(a2['realised_loss'] / a1['realised_loss'] - 1):+.1%}",
                  help="Original loan amount of approved applicants with payment difficulties × LGD", delta_color="off")
        st.markdown("**Scale illustration (not a forecast)**")
        st.warning("This multiplies the loss-proxy gap measured on Home Credit data by a volume you type in. It uses an "
                   "early-delinquency flag, the full loan amount and an assumed LGD, on non-Azerbaijani data. "
                   "It shows the order of magnitude a pilot would test, not an expected saving.")
        port = st.number_input("Installment volume, AZN", value=100_000_000, step=10_000_000)
        ill = business.illustrate(port, a1["loss_rate"], a2["loss_rate"])
        st.metric("Illustrative loss-proxy difference", f"{ill['saving_azn']:,.0f} AZN")
        st.caption(f"Loss-proxy rate per approved amount: B1 {a1['loss_rate']:.2%} → M2 {a2['loss_rate']:.2%} "
                   f"(LGD {lgd:.0%}, assumption).")

with tab_rep:
    rp = REP / "REPORT.md"
    if rp.exists():
        text = rp.read_text(encoding="utf-8")
        for block in text.split("\n\n"):
            if block.startswith("!["):
                img = block[block.index("(") + 1: block.index(")")]
                st.image(str(REP / img))
            else:
                st.markdown(block)
    else:
        st.info("Run train.py to create the report.")
