"""Train 5 classifiers on one dataset and save pipelines, test_data.csv and metrics.

Edit the CONFIG block, then run:  python train_models.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from eval_utils import compute_metrics


DATA_PATH = "/home/s2026aj05276/workspace/ML Assignment 2/data/ObesityDataSet.csv"
TARGET_COL = "NObeyesdad"
DROP_COLS = []
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_TRAIN_ROWS = 20000

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH).drop(columns=DROP_COLS)
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
print("Shape:", df.shape)
print(df[TARGET_COL].value_counts())

X = df.drop(columns=[TARGET_COL])
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COL].astype(str))
n_classes = len(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

if len(X_train) > MAX_TRAIN_ROWS:
    keep = X_train.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE).index
    y_train = pd.Series(y_train, index=X_train.index).loc[keep].values
    X_train = X_train.loc[keep]


test_df = X_test.copy()
test_df[TARGET_COL] = le.inverse_transform(y_test)
test_df.to_csv("test_data.csv", index=False)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat_cols),
    ])


models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True,
                                  random_state=RANDOM_STATE),
}

rows = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    metrics, _ = compute_metrics(pipe, X_test, y_test, n_classes)
    rows[name] = metrics
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, MODEL_DIR / fname)
    print(f"{name:25s}", {k: round(v, 4) for k, v in metrics.items()})

results = pd.DataFrame(rows).T.round(4)
results.index.name = "ML Model Name"
results.to_csv("model/metrics_summary.csv")
print("\n", results.to_markdown())

meta = {
    "target_col": TARGET_COL,
    "classes": [str(c) for c in le.classes_],
    "feature_cols": X.columns.tolist(),
    "models": {n: n.lower().replace(" ", "_") + ".joblib" for n in models},
}
Path("model/meta.json").write_text(json.dumps(meta, indent=2))
print("\nSaved models, model/meta.json and test_data.csv")