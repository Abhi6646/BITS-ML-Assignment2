import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from eval_utils import compute_metrics

MODEL_DIR = Path("model")
meta = json.loads((MODEL_DIR / "meta.json").read_text())
TARGET, CLASSES, FEATURES = meta["target_col"], meta["classes"], meta["feature_cols"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

st.set_page_config(page_title="Classifier Playground", layout="wide")
st.title("Classification Model Playground")
st.caption("Upload test data, choose a pre-trained model, inspect how it performs. "
           "Models are loaded from disk - nothing is retrained.")


@st.cache_resource
def load_model(fname):
    return joblib.load(MODEL_DIR / fname)


with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    model_name = st.selectbox("Select model", list(meta["models"].keys()))
    st.markdown(f"**Target column:** `{TARGET}`")
    st.markdown(f"**Classes:** {', '.join(CLASSES)}")
    with open("test_data.csv", "rb") as f:
        st.download_button("Download sample test_data.csv", f, "test_data.csv")

if uploaded is None:
    st.info("Upload a CSV in the sidebar (use the sample test_data.csv to try it out).")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Data preview")
st.write(f"{df.shape[0]} rows x {df.shape[1]} columns")
st.dataframe(df.head())

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing required columns: {missing}")
    st.stop()

unknown = set(df[TARGET].astype(str)) - set(CLASSES)
if unknown:
    st.error(f"Target column has labels not seen in training: {sorted(unknown)}")
    st.stop()

X = df[FEATURES]
y_true = df[TARGET].astype(str).map(CLASS_TO_ID).values
pipe = load_model(meta["models"][model_name])
metrics, y_pred = compute_metrics(pipe, X, y_true, len(CLASSES))

st.subheader(f"Evaluation metrics - {model_name}")
cols = st.columns(len(metrics))
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    rep = classification_report(y_true, y_pred, target_names=CLASSES,
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3))

st.subheader("All models on this uploaded data")
all_rows = {n: compute_metrics(load_model(f), X, y_true, len(CLASSES))[0]
            for n, f in meta["models"].items()}
st.dataframe(pd.DataFrame(all_rows).T.round(4))