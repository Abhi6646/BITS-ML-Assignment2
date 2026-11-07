import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)


def compute_metrics(pipe, X, y_true, n_classes):
    """y_true must be integer-encoded labels (0..n_classes-1)."""
    y_pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    if n_classes == 2:
        avg = "binary"
        auc = roc_auc_score(y_true, proba[:, 1])
    else:
        avg = "weighted"
        auc = roc_auc_score(y_true, proba, multi_class="ovr", average="weighted",
                            labels=np.arange(n_classes))
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }, y_pred