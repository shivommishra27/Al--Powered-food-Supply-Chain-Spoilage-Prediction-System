"""Train spoilage classifier + remaining-shelf-life regressor."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (CATEGORICAL, CLF_PATH, DATA_PATH, FEATURES, METRICS_PATH,
                     MODEL_DIR, NUMERIC, RANDOM_STATE, REG_PATH, TARGET_CLF,
                     TARGET_REG)
from .generate_data import generate


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])


def build_classifier() -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor()),
        ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])


def build_regressor() -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor()),
        ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                        random_state=RANDOM_STATE)),
    ])


def train(df: pd.DataFrame, model_dir: Path = MODEL_DIR) -> dict:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X = df[FEATURES]
    y_clf, y_reg = df[TARGET_CLF], df[TARGET_REG]
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf)

    clf = build_classifier().fit(X_tr, yc_tr)
    reg = build_regressor().fit(X_tr, yr_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    reg_pred = reg.predict(X_te)

    metrics = {
        "classifier": {
            "accuracy": round(accuracy_score(yc_te, pred), 4),
            "f1": round(f1_score(yc_te, pred), 4),
            "roc_auc": round(roc_auc_score(yc_te, proba), 4),
        },
        "regressor": {
            "mae_days": round(mean_absolute_error(yr_te, reg_pred), 4),
            "r2": round(r2_score(yr_te, reg_pred), 4),
        },
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }

    joblib.dump(clf, model_dir / CLF_PATH.name)
    joblib.dump(reg, model_dir / REG_PATH.name)
    (model_dir / METRICS_PATH.name).write_text(json.dumps(metrics, indent=2))
    _plot_importance(reg, model_dir / "feature_importance.png")
    return metrics


def _plot_importance(reg: Pipeline, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = reg.named_steps["prep"].get_feature_names_out()
    imp = reg.named_steps["model"].feature_importances_
    idx = np.argsort(imp)[-12:]
    plt.figure(figsize=(8, 5))
    plt.barh([names[i].split("__")[-1] for i in idx], imp[idx], color="#2e7d32")
    plt.xlabel("Importance")
    plt.title("Top features - shelf-life regressor")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main() -> None:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        print("Dataset not found, generating synthetic data...")
        df = generate()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    metrics = train(df)
    print(json.dumps(metrics, indent=2))
    print(f"Models saved in {MODEL_DIR}")


if __name__ == "__main__":
    main()
