"""DataSphere ML predictor — new models, 3CLpro-style (not a mirror of index.html).

Three models trained by train_models.py on one synthetic polyglot dataset:
  1. High-value customer classifier (top-20% spenders)
  2. Next order amount regressor
  3. Review rating classifier (1-5 stars)

Features combine all three stores: MySQL order aggregates,
Mongo-style catalogue/review descriptors, Neo4j graph centrality.
Every metric shown comes from models/metrics.json — nothing is made up.

Run: /opt/anaconda3/bin/streamlit run streamlit_app/ml_app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="DataSphere ML — polyglot predictors", layout="wide")
HERE = Path(__file__).parent
MODELS = HERE / "models"

CITIES = ["London", "Bengaluru", "Pune", "Berlin", "Austin", "Lagos"]
CATS = ["Shoes", "Electronics", "Home", "Books", "Beauty", "Sports"]

# feature -> store badge
STORE = {"city": "MySQL", "tenure_days": "MySQL", "n_orders": "MySQL",
         "avg_amount": "MySQL", "recency_days": "MySQL", "past_avg": "MySQL",
         "month": "MySQL", "day_of_week": "MySQL", "total_spent": "MySQL",
         "price": "MongoDB", "category": "MongoDB", "fav_category": "MongoDB",
         "reviewer_n_reviews": "MongoDB", "reviewer_avg_stars": "MongoDB",
         "reviewer_recency": "MongoDB",
         "n_reviews": "MongoDB", "n_categories": "MongoDB",
         "degree": "Neo4j", "pagerank": "Neo4j", "reviewer_pagerank": "Neo4j"}


@st.cache_resource
def load_all():
    metrics = json.loads((MODELS / "metrics.json").read_text())
    extra = {}
    for name in ("calibration", "shap", "distributions"):
        p = MODELS / f"{name}.json"
        extra[name] = json.loads(p.read_text()) if p.exists() else None
    return (metrics,
            joblib.load(MODELS / "model_customer.pkl"),
            joblib.load(MODELS / "model_amount.pkl"),
            joblib.load(MODELS / "model_rating.pkl"),
            extra)


def importances(pipe, top_n=8):
    """Feature importances mapped back through the OneHot encoder."""
    try:
        enc = pipe.named_steps["enc"]
        names = list(enc.get_feature_names_out())
    except Exception:
        return None
    m = pipe.named_steps["m"]
    imp = getattr(m, "feature_importances_", None)
    if imp is None:
        return None
    df = pd.DataFrame({"feature": names, "importance": np.asarray(imp, float)})
    return df.sort_values("importance", ascending=False).head(top_n)


try:
    metrics, m_cust, m_amt, m_rate, extra = load_all()
except Exception as e:
    st.error(f"Models not found — run `python train_models.py` first. ({e})")
    st.stop()

st.title("DataSphere ML — predictions across three databases")
st.caption(f"Trained {metrics['generated']} · seed {metrics['seed']} · "
           f"{metrics['n_customers']} customers / {metrics['n_orders']} orders / "
           f"{metrics['n_reviews']} reviews · best: customer={metrics['customer']['best']}, "
           f"amount={metrics['amount']['best']}, rating={metrics['rating']['best']}")
st.write("Each prediction fuses **MySQL** order history, **MongoDB** catalogue/review "
         "shape and **Neo4j** graph centrality — the point of the polyglot design. "
         "Feature badges show where every input comes from.")

tab1, tab2, tab3, tab_cal, tab_shap, tab_data = st.tabs(
    ["High-value customer", "Order amount", "Review rating",
     "Calibration", "Why? (SHAP)", "Data quality"])

with tab1:
    st.header("Will this customer land in the top 20% of spenders?")
    st.caption("Classifier · RF / XGBoost / LightGBM compared, best kept. Test metrics below are real.")
    c1, c2, c3 = st.columns(3)
    city = c1.selectbox("City (MySQL)", CITIES)
    fav = c1.selectbox("Favourite category (MongoDB)", CATS)
    tenure = c1.slider("Tenure, days (MySQL)", 30, 900, 300)
    n_orders = c2.slider("Orders placed (MySQL)", 1, 120, 12)
    avg_amt = c2.slider("Average order, £ (MySQL)", 5.0, 300.0, 65.0)
    recency = c2.slider("Days since last order (MySQL)", 0, 400, 25)
    degree = c3.slider("Friend links (Neo4j degree)", 1, 60, 6)
    pr = c3.slider("Influence score (Neo4j PageRank×n)", 0.1, 5.0, 1.0)
    n_cat = c3.slider("Distinct categories (MongoDB)", 1, 6, 3)
    n_rev = c3.slider("Reviews written (MongoDB)", 0, 30, 3)
    if st.button("Predict customer value", type="primary"):
        X = pd.DataFrame([{**{"city": city, "tenure_days": tenure, "degree": degree,
                               "pagerank": pr, "n_orders": n_orders, "avg_amount": avg_amt,
                               "recency_days": recency, "n_categories": n_cat,
                               "n_reviews": n_rev, "fav_category": fav}}])
        st.session_state["last_customer_X"] = X
        p = float(m_cust.predict_proba(X)[0, 1])
        st.metric("P(high-value)", f"{p:.0%}", "likely top-20%" if p >= 0.5 else "likely not")
        st.progress(min(max(p, 0.0), 1.0))
    dfm = pd.DataFrame(metrics["customer"]["models"]).T
    st.subheader("Held-out test metrics (all three trained)")
    st.dataframe(dfm)
    imp = importances(m_cust)
    if imp is not None:
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                               title="Best model: feature importances"), use_container_width=True)

with tab2:
    st.header("How much will the next order be?")
    c1, c2, c3 = st.columns(3)
    tenure2 = c1.slider("Customer tenure, days", 30, 900, 300, key="t2")
    past = c1.slider("Customer past average, £", 5.0, 300.0, 65.0, key="p2")
    rec2 = c1.slider("Recency, days", 0, 400, 25, key="r2")
    deg2 = c2.slider("Customer links (Neo4j)", 1, 60, 6, key="d2")
    pr2 = c2.slider("Customer influence", 0.1, 5.0, 1.0, key="pr2")
    cat2 = c2.selectbox("Order category", CATS, key="c2")
    month = c3.selectbox("Month", list(range(1, 13)), index=11)
    dow = c3.selectbox("Day of week (0=Mon)", list(range(7)), index=4)
    if st.button("Predict order amount", type="primary"):
        X = pd.DataFrame([{**{"tenure_days": tenure2, "past_avg": past, "degree": deg2,
                               "pagerank": pr2, "recency_days": rec2, "month": month,
                               "day_of_week": dow, "category": cat2}}])
        pred = float(m_amt.predict(X)[0])
        s = metrics["amount"]["resid_std"]
        st.metric("Predicted amount", f"£{pred:.2f}", f"± ~£{1.96*s:.0f} (95% from test residuals)")
    st.subheader("Held-out test metrics")
    st.dataframe(pd.DataFrame(metrics["amount"]["models"]).T)
    imp = importances(m_amt)
    if imp is not None:
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                               title="Best model: feature importances"), use_container_width=True)

with tab3:
    st.header("How many stars will this review give?")
    c1, c2 = st.columns(2)
    price = c1.slider("Product price, £", 10.0, 300.0, 89.0)
    cat3 = c1.selectbox("Product category", CATS, key="c3")
    rn = c2.slider("Reviewer's past review count", 0, 30, 4)
    ravg = c2.slider("Reviewer's average stars so far", 1.0, 5.0, 3.8, step=0.1)
    rpr = c2.slider("Reviewer influence (Neo4j)", 0.1, 5.0, 1.0, key="rpr")
    rrec = c2.slider("Reviewer recency, days", 0, 400, 30, key="rrec")
    if st.button("Predict rating", type="primary"):
        X = pd.DataFrame([{**{"price": price, "category": cat3, "reviewer_n_reviews": rn,
                               "reviewer_avg_stars": ravg,
                               "reviewer_pagerank": rpr, "reviewer_recency": rrec}}])
        proba = m_rate.predict_proba(X)[0]
        stars = int(proba.argmax()) + 1
        st.metric("Predicted rating", f"{stars} ★", f"confidence {proba.max():.0%}")
        st.plotly_chart(px.bar(x=[f"{i+1} ★" for i in range(5)], y=proba,
                               labels={"x": "Stars", "y": "Probability"}), use_container_width=True)
    st.subheader("Held-out test metrics (5-class, imbalanced like real shops)")
    st.dataframe(pd.DataFrame(metrics["rating"]["models"]).T)
    st.caption("Most reviews are 4★, 1★ is rare — so accuracy beats the 48% majority baseline "
               "but 1★/2★ recall stays low. That imbalance is the finding, not a bug.")

with tab_cal:
    st.header("Calibration: does 70% mean 70%?")
    st.write("Held-out reliability — predicted probability vs actual high-value rate. "
             "The diagonal is perfect calibration.")
    calib = extra.get("calibration")
    if not calib:
        st.warning("Run train_models.py to generate calibration.json.")
    else:
        import plotly.graph_objects as go
        fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                    line=dict(dash="dash", color="grey"), name="ideal")])
        df = pd.DataFrame(calib["overall"])
        fig.add_trace(go.Scatter(x=df["mean_pred"], y=df["observed"], mode="lines+markers",
                                 name="overall (sized by n)",
                                 marker=dict(size=np.sqrt(df["n"]) * 2)))
        cat = st.selectbox("Per favourite-category curve", ["(all)"] + CATS)
        if cat != "(all)":
            d2 = pd.DataFrame(calib["by_fav_category"][cat])
            if len(d2):
                fig.add_trace(go.Scatter(x=d2["mean_pred"], y=d2["observed"],
                                         mode="lines+markers", name=cat))
        fig.update_layout(xaxis_title="Mean predicted P", yaxis_title="Observed rate",
                          height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df)

with tab_shap:
    st.header("Why did it predict that? (SHAP, like 3CLpro attribution)")
    st.write("Global bars are mean |SHAP| on the held-out test set (precomputed in training). "
             "The local chart explains one customer live.")
    shap_pre = extra.get("shap")
    if shap_pre and shap_pre.get("customer"):
        s = shap_pre["customer"]
        st.plotly_chart(px.bar(x=s["mean_abs"][:8], y=s["features"][:8], orientation="h",
                               labels={"x": "mean |SHAP|", "y": ""},
                               title="Customer model: global attributions"), use_container_width=True)
    if shap_pre and shap_pre.get("amount"):
        s = shap_pre["amount"]
        st.plotly_chart(px.bar(x=s["mean_abs"][:8], y=s["features"][:8], orientation="h",
                               labels={"x": "mean |SHAP| (£)", "y": ""},
                               title="Amount model: global attributions"), use_container_width=True)
    st.subheader("Explain one customer")
    X0 = st.session_state.get("last_customer_X")
    if X0 is None:
        st.info("Predict on the first tab first — or explain these defaults.")
        X0 = pd.DataFrame([{"city": "London", "tenure_days": 300, "degree": 6,
                             "pagerank": 1.0, "n_orders": 12, "avg_amount": 65.0,
                             "recency_days": 25, "n_categories": 3, "n_reviews": 3,
                             "fav_category": "Shoes"}])
    if st.button("Explain this customer"):
        try:
            import shap as _shap
            enc = m_cust.named_steps["enc"]
            Xt = enc.transform(X0)
            names = list(enc.get_feature_names_out())
            if hasattr(Xt, "toarray"):
                Xt = Xt.toarray()
            sv = np.asarray(_shap.TreeExplainer(m_cust.named_steps["m"]).shap_values(Xt))[0]
            if sv.ndim == 2:
                sv = sv[:, -1]
            d = pd.DataFrame({"feature": names, "shap": sv}).sort_values("shap")
            st.plotly_chart(px.bar(d.tail(8), x="shap", y="feature", orientation="h",
                                   labels={"shap": "SHAP (→ pushes high-value)"},
                                   title="Local forces for this customer"), use_container_width=True)
            p = float(m_cust.predict_proba(X0)[0, 1])
            st.caption(f"Model output for this row: P(high-value) = {p:.1%}. "
                       "Positive bars pushed it up, negative bars pushed it down.")
        except Exception as e:
            st.warning(f"Live SHAP unavailable ({e}); global bars above still hold.")

with tab_data:
    st.header("Data quality — what the models actually ate")
    st.write("Drawn from the real seeded training data (distributions.json), not mock-ups.")
    dd = extra.get("distributions")
    if not dd:
        st.warning("Run train_models.py to generate distributions.json.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(x=list(dd["stars"]), y=list(dd["stars"].values()),
                                   labels={"x": "Stars", "y": "Reviews"},
                                   title="Review stars (J-shape, like real shops)"),
                            use_container_width=True)
            st.plotly_chart(px.bar(x=dd["amount_hist"]["centers"], y=dd["amount_hist"]["counts"],
                                   labels={"x": "Order £", "y": "Orders"},
                                   title="Order amounts (1st–99th pct)"), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(x=dd["total_spent_hist"]["centers"],
                                   y=dd["total_spent_hist"]["counts"],
                                   labels={"x": "Customer lifetime £", "y": "Customers"},
                                   title="Lifetime spend (heavy tail)"), use_container_width=True)
            st.plotly_chart(px.bar(x=list(dd["city_mix"]), y=list(dd["city_mix"].values()),
                                   labels={"x": "City", "y": "Customers"},
                                   title="City mix"), use_container_width=True)
        st.write(f"Graph: mean degree {dd['degree']['mean']}, p90 {dd['degree']['p90']}, "
                 f"max {dd['degree']['max']} — hub-and-spoke, as designed. "
                 f"Order categories: {dd['order_category_mix']}")
