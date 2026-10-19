"""Training, evaluation, saving and loading of the activity classifier."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data import generate_dataset

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def candidate_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "k-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3, random_state=42)),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    }


def train_and_evaluate(df, out_dir=MODEL_DIR, data_source="sample (synthetic)", target="activity"):
    """Compare all candidate models, keep the best (by 5-fold CV accuracy) and save everything."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = [c for c in df.columns if c != target]
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted = [], {}
    for name, model in candidate_models().items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "name": name,
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_f1_weighted": float(f1_score(y_test, pred, average="weighted")),
        })
        fitted[name] = model

    best = max(rows, key=lambda r: r["cv_mean"])["name"]
    best_model = fitted[best]
    pred = best_model.predict(X_test)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)

    metrics = {
        "data_source": data_source,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": features,
        "labels": labels,
        "models": rows,
        "best_model": best,
        "classification_report": classification_report(y_test, pred, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }
    joblib.dump(best_model, out_dir / "best_model.joblib", compress=3)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_confusion_matrix_plot(cm, labels, out_dir / "confusion_matrix.png", best)
    return metrics


def save_confusion_matrix_plot(cm, labels, path, title=""):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    ax.set_title(f"Confusion matrix: {title}" if title else "Confusion matrix")
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_model(model_dir=MODEL_DIR):
    return joblib.load(Path(model_dir) / "best_model.joblib")


def load_metrics(model_dir=MODEL_DIR):
    return json.loads((Path(model_dir) / "metrics.json").read_text())


def ensure_model(model_dir=MODEL_DIR):
    """Train on the synthetic sample data if no saved model exists yet."""
    model_dir = Path(model_dir)
    if not (model_dir / "best_model.joblib").exists():
        train_and_evaluate(generate_dataset(), out_dir=model_dir)
    return load_model(model_dir)


def predict_activities(model, df, features):
    """Predict an activity for every row; raises a clear error if columns are missing."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"The file is missing {len(missing)} required column(s): {missing[:8]}")
    return model.predict(df[features])
