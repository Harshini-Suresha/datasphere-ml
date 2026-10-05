"""Train 3 polyglot models for DataSphere (like export_results.py in 3clpro).

Generates one synthetic e-commerce dataset (seeded), engineers features
from all three paradigms (MySQL orders, Mongo-style catalogue/reviews,
Neo4j graph centrality), trains RF/XGBoost/LightGBM per task, keeps the
best, and writes artifacts for the Streamlit predictor.

Usage:
    /opt/anaconda3/bin/python train_models.py           # full
    /opt/anaconda3/bin/python train_models.py --fast    # fewer rows/trees
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except Exception:
    HAS_LGB = False

SEED = 42
CITIES = ["London", "Bengaluru", "Pune", "Berlin", "Austin", "Lagos"]
CITY_P = [0.28, 0.22, 0.15, 0.14, 0.12, 0.09]          # realistic home-base mix
CITY_PREM = {"London": 14, "Austin": 10, "Berlin": 8,   # price-level per city
             "Bengaluru": -2, "Pune": -4, "Lagos": -6}
CATS = ["Shoes", "Electronics", "Home", "Books", "Beauty", "Sports"]
OUT = Path(__file__).parent / "streamlit_app" / "models"
OUT.mkdir(parents=True, exist_ok=True)


def gen_customers(n=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    # graph: Barabasi-Albert-like popularity (hubs spend more) without networkx
    # so train script has zero extra deps beyond sklearn/xgb/lgb.
    degree = np.ones(n, dtype=float)
    for i in range(1, n):
        w = degree[:i] + 1.0
        p = w / w.sum()
        for _ in range(2 if i > 3 else 1):
            j = rng.choice(i, p=p)
            degree[i] += 1
            degree[j] += 1
    pagerank = degree / degree.sum() * n  # normalised influence score
    city = rng.choice(CITIES, size=n, p=CITY_P)
    prem = np.array([CITY_PREM[c] for c in city])
    tenure = np.clip(rng.gamma(5.0, 90.0, n).astype(int), 30, 900)
    # latent taste: high-propensity buyers exist in every city/graph position,
    # so the label is NOT a deterministic function of the features.
    propensity = rng.normal(0, 1, n)
    activity = rng.gamma(2.0, 1.0, n) * (0.5 + pagerank) + 0.4 * propensity
    n_orders = np.clip((rng.power(1.6, n) * 34 * (0.4 + activity / np.mean(activity))
                        + (propensity > 1).astype(int) * 4).astype(int), 1, 120)
    avg_amount = np.round(22 + prem + rng.gamma(2.2, 16, n)
                          + (pagerank - 1) * 5 + propensity * 7, 2)
    avg_amount = np.clip(avg_amount, 5, 400)
    total = np.round(n_orders * avg_amount * rng.lognormal(0, 0.12, n), 2)
    recency = np.clip((rng.exponential(45, n) / (0.5 + activity / 3)).astype(int), 0, 400)
    n_cat = np.clip(rng.integers(1, 6, n) + (activity > np.mean(activity)).astype(int), 1, 6)
    n_rev = (rng.poisson(2.5, n) + (activity > 2).astype(int) * 2).clip(0, 30)
    # favourite category correlates with city (local demand), not pure noise
    fav_idx = (rng.integers(0, 6, n)
               + (city == "London") * 1 + (city == "Bengaluru") * 3) % 6
    leniency = np.round(rng.normal(0, 0.6, n), 3)  # reviewer generosity (for reviews table)
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "city": city,
        "tenure_days": tenure, "degree": degree.astype(int),
        "pagerank": np.round(pagerank, 4),
        "n_orders": n_orders, "avg_amount": avg_amount,
        "recency_days": recency, "n_categories": n_cat,
        "n_reviews": n_rev, "fav_category": [CATS[i] for i in fav_idx],
        "leniency": leniency,
        "total_spent": total,
    })
    # noisy label: rank by log-spend plus judgement noise, so boundary
    # customers overlap and AUC lands in a believable ~0.9, not 0.997.
    score = np.log(total) + rng.normal(0, 0.30, n)
    df["high_value"] = (score >= np.quantile(score, 0.8)).astype(int)
    return df


def gen_orders(cust, n_orders=50_000, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    cid = rng.choice(cust["customer_id"].values, size=n_orders,
                     p=(cust["n_orders"] / cust["n_orders"].sum()).values)
    c = cust.set_index("customer_id").loc[cid].reset_index(drop=True)
    month = rng.integers(1, 13, n_orders)
    dow = rng.integers(0, 7, n_orders)
    cat = [CATS[i] for i in rng.integers(0, 6, n_orders)]
    cat_eff = {"Shoes": 8, "Electronics": 55, "Home": 12, "Books": -8, "Beauty": 2, "Sports": 15}
    weekend = (dow >= 5).astype(int)
    wknd_eff = np.array([8 if x in ("Shoes", "Beauty", "Sports") else
                         (-5 if x == "Electronics" else 0) for x in cat]) * weekend
    seas_eff = (month == 12) * 25 + (month == 11) * 10 - (month == 1) * 8
    base = (c["past_avg"].values if "past_avg" in c else c["avg_amount"].values) * 0.50 \
        + np.minimum(c["tenure_days"].values, 600) * 0.015 \
        + np.array([cat_eff[x] for x in cat]) + wknd_eff + seas_eff
    amount = base * rng.lognormal(0, 0.28, n_orders)  # multiplicative noise: rich orders vary more
    return pd.DataFrame({
        "tenure_days": c["tenure_days"].values, "past_avg": c["avg_amount"].values,
        "degree": c["degree"].values, "pagerank": c["pagerank"].values,
        "recency_days": c["recency_days"].values, "month": month,
        "day_of_week": dow, "category": cat,
        "amount": np.round(np.clip(amount, 5, 900), 2)})


def gen_reviews(cust, n_rev=30_000, seed=SEED + 2):
    rng = np.random.default_rng(seed)
    w = (cust["n_reviews"] + 0.5).values
    cid = rng.choice(cust["customer_id"].values, size=n_rev, p=w / w.sum())
    c = cust.set_index("customer_id").loc[cid].reset_index(drop=True)
    price = np.round(10 + rng.power(2, n_rev) * 290, 2)
    cat = [CATS[i] for i in rng.integers(0, 6, n_rev)]
    # reviewer generosity persists per customer -> legitimate behavioural signal
    reviewer_avg = np.clip(3.8 + c["leniency"].values * 0.7 + rng.normal(0, 0.15, n_rev), 1, 5)
    quality = (np.array([0.3 if x == "Books" else (-0.2 if x == "Electronics" else 0.0) for x in cat])
               - (price > 200) * 0.5 + rng.normal(0, 0.5, n_rev))
    latent = 4.0 + quality * 0.7 + (reviewer_avg - 3.8) * 0.9
    stars = np.clip(np.round(latent + rng.normal(0, 0.5, n_rev)), 1, 5).astype(int)
    return pd.DataFrame({"price": price, "category": cat,
                         "reviewer_n_reviews": c["n_reviews"].values,
                         "reviewer_avg_stars": np.round(reviewer_avg, 2),
                         "reviewer_pagerank": c["pagerank"].values,
                         "reviewer_recency": c["recency_days"].values,
                         "stars": stars})


def enc(df, cat_cols):
    return ColumnTransformer(
        [(  # noqa
            "cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough")


def fit_predict(task, X_train, y_train, X_test, y_test, cat_cols, kind, fast):
    n_est = 60 if fast else 200
    cands = {}
    if kind == "clf":
        cands["rf"] = RandomForestClassifier(n_estimators=n_est, n_jobs=-1, random_state=SEED)
        if HAS_XGB:
            cands["xgb"] = XGBClassifier(n_estimators=n_est, max_depth=6, learning_rate=0.08,
                                        subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=SEED)
        if HAS_LGB:
            from lightgbm import LGBMClassifier as L
            cands["lgbm"] = L(n_estimators=n_est, num_leaves=63, n_jobs=-1, random_state=SEED, verbose=-1)
    elif kind == "reg":
        cands["rf"] = RandomForestRegressor(n_estimators=n_est, n_jobs=-1, random_state=SEED)
        if HAS_XGB:
            cands["xgb"] = XGBRegressor(n_estimators=n_est, max_depth=6, learning_rate=0.08,
                                       subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=SEED)
        if HAS_LGB:
            from lightgbm import LGBMRegressor as L
            cands["lgbm"] = L(n_estimators=n_est, num_leaves=63, n_jobs=-1, random_state=SEED, verbose=-1)
    else:  # multiclass stars
        cands["rf"] = RandomForestClassifier(n_estimators=n_est, n_jobs=-1, random_state=SEED)
        if HAS_XGB:
            cands["xgb"] = XGBClassifier(n_estimators=n_est, max_depth=6, learning_rate=0.08,
                                        n_jobs=-1, random_state=SEED)
        if HAS_LGB:
            from lightgbm import LGBMClassifier as L
            cands["lgbm"] = L(n_estimators=n_est, num_leaves=63, n_jobs=-1, random_state=SEED, verbose=-1)

    results, best = {}, None
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    for name, est in cands.items():
        pipe = Pipeline([("enc", enc(X_train, cat_cols)), ("m", est)])
        t = time.perf_counter()
        if kind == "clf":
            cvv = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        else:
            cvv = cross_val_score(pipe, X_train, y_train, cv=3, scoring="r2" if kind == "reg" else "accuracy")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        dt = time.perf_counter() - t
        if kind == "clf":
            proba = pipe.predict_proba(X_test)[:, 1]
            m = {"cv_mean": round(float(cvv.mean()), 4), "cv_std": round(float(cvv.std()), 4),
                 "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
                 "f1": round(float(f1_score(y_test, pred)), 4),
                 "accuracy": round(float(accuracy_score(y_test, pred)), 4), "fit_s": round(dt, 1)}
            key = m["roc_auc"]
        elif kind == "reg":
            m = {"cv_mean": round(float(cvv.mean()), 4),
                 "rmse": round(float(np.sqrt(np.mean((y_test - pred) ** 2))), 2),
                 "mae": round(float(mean_absolute_error(y_test, pred)), 2),
                 "r2": round(float(r2_score(y_test, pred)), 4), "fit_s": round(dt, 1)}
            key = m["r2"]
        else:
            m = {"cv_mean": round(float(cvv.mean()), 4),
                 "accuracy": round(float(accuracy_score(y_test, pred)), 4),
                 "f1_macro": round(float(f1_score(y_test, pred, average="macro")), 4),
                 "fit_s": round(dt, 1)}
            key = m["accuracy"]
        results[name] = m
        if best is None or key > best[0]:
            best = (key, name, pipe)
    return results, best


def reliability(y_true, proba, n_bins=10):
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (proba > lo) & (proba <= hi) if lo > 0 else (proba >= lo) & (proba <= hi)
        if m.sum() > 0:
            out.append({"bin_mid": round(float((lo + hi) / 2), 3), "n": int(m.sum()),
                        "mean_pred": round(float(proba[m].mean()), 4),
                        "observed": round(float(y_true[m].mean()), 4)})
    return out


def shap_summary(pipe, X_sample, max_rows=800):
    """Mean |SHAP| per (encoded) feature + one example. Tree models only."""
    import shap
    Xs = X_sample.iloc[:max_rows]
    Xt = pipe.named_steps["enc"].transform(Xs)
    try:
        names = list(pipe.named_steps["enc"].get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(Xt.shape[1])]
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    model = pipe.named_steps["m"]
    try:
        sv = shap.TreeExplainer(model).shap_values(Xt)
    except Exception:
        return None
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, f, classes) -> positive class
        sv = sv[:, :, -1]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:12]
    return {"features": [names[i] for i in order],
            "mean_abs": [round(float(mean_abs[i]), 5) for i in order],
            "example": {"features": names,
                        "values": [round(float(v), 5) for v in sv[0]]}}


def dists(cust, ords, revs, n_bins=24):
    def hist(s, bins=24, lo=None, hi=None):
        v = np.asarray(s, float)
        lo = float(np.quantile(v, 0.01)) if lo is None else lo
        hi = float(np.quantile(v, 0.99)) if hi is None else hi
        c, e = np.histogram(v, bins=bins, range=(lo, hi))
        return {"lo": round(lo, 2), "hi": round(hi, 2),
                "centers": [round(float(x), 2) for x in (e[:-1] + e[1:]) / 2],
                "counts": [int(x) for x in c]}
    return {
        "stars": {str(k): int(v) for k, v in revs["stars"].value_counts().sort_index().items()},
        "amount_hist": hist(ords["amount"]),
        "total_spent_hist": hist(cust["total_spent"]),
        "city_mix": {k: int(v) for k, v in cust["city"].value_counts().items()},
        "fav_category_mix": {k: int(v) for k, v in cust["fav_category"].value_counts().items()},
        "order_category_mix": {k: int(v) for k, v in ords["category"].value_counts().items()},
        "review_category_mix": {k: int(v) for k, v in revs["category"].value_counts().items()},
        "degree": {"mean": round(float(cust["degree"].mean()), 2),
                   "p90": int(cust["degree"].quantile(0.9)),
                   "max": int(cust["degree"].max())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    import joblib

    cust = gen_customers()
    feats_c = ["city", "tenure_days", "degree", "pagerank", "n_orders",
               "avg_amount", "recency_days", "n_categories", "n_reviews", "fav_category"]
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(cust[feats_c], cust["high_value"],
                                                  test_size=0.2, stratify=cust["high_value"],
                                                  random_state=SEED)
    res_c, (key_c, name_c, pipe_c) = fit_predict("customer", Xc_tr, yc_tr, Xc_te, yc_te,
                                                ["city", "fav_category"], "clf", a.fast)
    joblib.dump(pipe_c, OUT / "model_customer.pkl")

    ords = gen_orders(cust, 20_000 if a.fast else 50_000)
    feats_o = ["tenure_days", "past_avg", "degree", "pagerank", "recency_days",
               "month", "day_of_week", "category"]
    Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(ords[feats_o], ords["amount"],
                                                  test_size=0.2, random_state=SEED)
    res_o, (key_o, name_o, pipe_o) = fit_predict("amount", Xo_tr, yo_tr, Xo_te, yo_te,
                                                ["category"], "reg", a.fast)
    joblib.dump(pipe_o, OUT / "model_amount.pkl")
    resid_std = float((yo_te - pipe_o.predict(Xo_te)).std())

    revs = gen_reviews(cust, 12_000 if a.fast else 30_000)
    feats_r = ["price", "category", "reviewer_n_reviews", "reviewer_avg_stars",
               "reviewer_pagerank", "reviewer_recency"]
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(revs[feats_r], revs["stars"] - 1, test_size=0.2,
                                                  stratify=revs["stars"], random_state=SEED)
    res_r, (key_r, name_r, pipe_r) = fit_predict("rating", Xr_tr, yr_tr, Xr_te, yr_te,
                                                ["category"], "multi", a.fast)
    joblib.dump(pipe_r, OUT / "model_rating.pkl")

    metrics = {
        "generated": time.strftime("%Y-%m-%d"), "seed": SEED, "fast": a.fast,
        "n_customers": len(cust), "n_orders": len(ords), "n_reviews": len(revs),
        "pos_rate": round(float(cust["high_value"].mean()), 3),
        "customer": {"models": res_c, "best": name_c},
        "amount": {"models": res_o, "best": name_o, "resid_std": round(resid_std, 2)},
        "rating": {"models": res_r, "best": name_r},
        "features": {"customer": feats_c, "amount": feats_o, "rating": feats_r},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))

    # --- calibration: customer model, overall + per favourite-category ---
    proba_c = pipe_c.predict_proba(Xc_te)[:, 1]
    calib = {"overall": reliability(yc_te, proba_c),
             "by_fav_category": {cat: reliability(yc_te[Xc_te["fav_category"] == cat],
                                                  proba_c[Xc_te["fav_category"] == cat], n_bins=5)
                                 for cat in CATS}}
    (OUT / "calibration.json").write_text(json.dumps(calib, indent=1))

    # --- SHAP summaries (global mean|SHAP| + one example row) ---
    shap_out = {"customer": shap_summary(pipe_c, Xc_te),
                "amount": shap_summary(pipe_o, Xo_te)}
    (OUT / "shap.json").write_text(json.dumps(shap_out, indent=1))

    # --- data-quality distributions (drawn from the real generated data) ---
    (OUT / "distributions.json").write_text(json.dumps(dists(cust, ords, revs), indent=1))
    print(json.dumps(metrics, indent=1))
    print(f"saved to {OUT} in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
