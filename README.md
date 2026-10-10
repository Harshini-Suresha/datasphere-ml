---
title: DataSphere ML
emoji: 🔮
colorFrom: blue
colorTo: pink
sdk: streamlit
app_file: streamlit_app/ml_app.py
pinned: false
---

# DataSphere ML — polyglot predictors

[![Website](https://img.shields.io/badge/website-live-brightgreen)](https://harshini-suresha.github.io/datasphere-ml/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harshini-Suresha/datasphere-ml/blob/main/notebooks/datasphere_ml_demo.ipynb)
[![Streamlit App](https://img.shields.io/badge/streamlit-app-red)](https://datasphere-ml-i9nv6eofsxkupnbfsdnqrx.streamlit.app/)

**Live links:** [Website (github.io)](https://harshini-suresha.github.io/datasphere-ml/) · [Colab notebook](https://colab.research.google.com/github/Harshini-Suresha/datasphere-ml/blob/main/notebooks/datasphere_ml_demo.ipynb) · [Download .ipynb](https://raw.githubusercontent.com/Harshini-Suresha/datasphere-ml/main/notebooks/datasphere_ml_demo.ipynb) · [Streamlit app](https://datasphere-ml-i9nv6eofsxkupnbfsdnqrx.streamlit.app/) · [GitHub repo](https://github.com/Harshini-Suresha/datasphere-ml)

Three models trained on one synthetic e-commerce dataset spanning MySQL-style
orders, MongoDB-style catalogue/reviews and Neo4j-style graph centrality:

1. **High-value customer** classifier (top-20% spenders)
2. **Order amount** regressor
3. **Review rating** classifier (1–5 stars)

Plus calibration curves, SHAP attributions and data-quality plots.
All metrics/plots come from the committed `streamlit_app/models/*.json` —
nothing is made up.

## Run locally

```bash
/opt/anaconda3/bin/streamlit run streamlit_app/ml_app.py
```

## Retrain

```bash
/opt/anaconda3/bin/python train_models.py        # full
/opt/anaconda3/bin/python train_models.py --fast # quick
```

## Deploy

- **Hugging Face Spaces**: create a Streamlit Space and push this repo to it
  (entry point is already set via the frontmatter above).
- **Streamlit Cloud**: main file `streamlit_app/ml_app.py`, Python 3.11.
