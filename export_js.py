"""Distill small linear models for the in-page calculators (zero hosting).

Fits LogisticRegression / Ridge on the SAME generator as train_models.py,
exports weights + transforms as window.ML_MODELS in ml_models.js,
with honest held-out test metrics printed + embedded.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_models import (CATS, CITIES, SEED, gen_customers, gen_orders,
                          gen_reviews)

NUM_C = ["tenure_days", "degree", "pagerank", "n_orders", "avg_amount",
         "recency_days", "n_categories", "n_reviews"]
NUM_O = ["tenure_days", "past_avg", "degree", "pagerank", "recency_days",
         "month", "day_of_week"]
NUM_R = ["price", "reviewer_n_reviews", "reviewer_avg_stars",
         "reviewer_pagerank", "reviewer_recency"]


def featurize_fit(df, num, cat):
    sc = StandardScaler().fit(df[num].astype(float))
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[cat])
    Xn = sc.transform(df[num].astype(float))
    Xc = enc.transform(df[cat])
    return np.hstack([Xn, Xc]), sc, enc


def featurize_apply(df, num, cat, sc, enc):
    return np.hstack([sc.transform(df[num].astype(float)), enc.transform(df[cat])])


def export_linear(name, num, cat, coef, intercept, sc, enc, metrics):
    return {
        "num": num, "cat": cat,
        "num_mean": [float(v) for v in sc.mean_],
        "num_std": [float(v) for v in sc.scale_],
        "cat_levels": [list(map(str, lv)) for lv in enc.categories_],
        "coef": [float(v) for v in coef], "intercept": float(intercept),
        "metrics": metrics,
        "defaults": {},  # filled below
    }


def main():
    cust = gen_customers()
    ords = gen_orders(cust)
    revs = gen_reviews(cust)

    out = {}
    # 1. customer: P(high-value)
    X = cust[NUM_C + ["city", "fav_category"]]
    tr, te, ytr, yte = train_test_split(X, cust["high_value"], test_size=0.2,
                                        stratify=cust["high_value"], random_state=SEED)
    Mtr, sc, enc = featurize_fit(tr, NUM_C, ["city", "fav_category"])
    Mte = featurize_apply(te, NUM_C, ["city", "fav_category"], sc, enc)
    clf = LogisticRegression(max_iter=2000).fit(Mtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Mte)[:, 1])
    out["customer"] = export_linear("customer", NUM_C, ["city", "fav_category"],
                                    clf.coef_[0], clf.intercept_[0], sc, enc,
                                    {"test_auc": round(float(auc), 3), "kind": "probability"})
    # 2. amount: Ridge £
    X = ords[NUM_O + ["category"]]
    tr, te, ytr, yte = train_test_split(X, ords["amount"], test_size=0.2, random_state=SEED)
    Mtr, sc, enc = featurize_fit(tr, NUM_O, ["category"])
    Mte = featurize_apply(te, NUM_O, ["category"], sc, enc)
    reg = Ridge().fit(Mtr, ytr)
    pred = reg.predict(Mte)
    out["amount"] = export_linear("amount", NUM_O, ["category"],
                                  reg.coef_, reg.intercept_, sc, enc,
                                  {"test_r2": round(float(r2_score(yte, pred)), 3),
                                   "test_rmse": round(float(np.sqrt(np.mean((yte - pred) ** 2))), 1),
                                   "kind": "pounds"})
    # 3. rating: Ridge -> round to 1..5
    X = revs[NUM_R + ["category"]]
    tr, te, ytr, yte = train_test_split(X, revs["stars"], test_size=0.2,
                                        stratify=revs["stars"], random_state=SEED)
    Mtr, sc, enc = featurize_fit(tr, NUM_R, ["category"])
    Mte = featurize_apply(te, NUM_R, ["category"], sc, enc)
    rr = Ridge().fit(Mtr, ytr)
    pr = np.clip(np.round(rr.predict(Mte)), 1, 5)
    out["rating"] = export_linear("rating", NUM_R, ["category"],
                                  rr.coef_, rr.intercept_, sc, enc,
                                  {"test_accuracy": round(float(accuracy_score(yte, pr)), 3),
                                   "test_mae": round(float(mean_absolute_error(yte, pr)), 3),
                                   "kind": "stars"})
    # typical-value defaults (medians/modes) so the page can offer one-click fills
    out["customer"]["defaults"] = {c: float(cust[c].median()) for c in NUM_C} | \
        {"city": cust["city"].mode()[0], "fav_category": cust["fav_category"].mode()[0]}
    out["amount"]["defaults"] = {c: float(ords[c].median()) for c in NUM_O} | \
        {"category": ords["category"].mode()[0]}
    out["rating"]["defaults"] = {c: float(revs[c].median()) for c in NUM_R} | \
        {"category": revs["category"].mode()[0]}

    with open("ml_models.js", "w") as f:
        f.write("window.ML_MODELS=" + json.dumps(out) + ";")
    print(json.dumps({k: v["metrics"] for k, v in out.items()}, indent=1))
    print("wrote ml_models.js")


if __name__ == "__main__":
    main()
