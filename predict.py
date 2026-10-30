"""Inference helpers + CLI."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import CLF_PATH, FEATURES, MODEL_DIR, REG_PATH

_cache = {}


def load_models(model_dir=None):
    d = Path(model_dir) if model_dir else MODEL_DIR
    key = str(d)
    if key not in _cache:
        clf_path, reg_path = d / CLF_PATH.name, d / REG_PATH.name
        if not clf_path.exists() or not reg_path.exists():
            raise FileNotFoundError("Models not found. Run: python -m src.train")
        _cache[key] = (joblib.load(clf_path), joblib.load(reg_path))
    return _cache[key]


def risk_level(prob: float) -> str:
    if prob < 0.30:
        return "LOW"
    if prob < 0.65:
        return "MEDIUM"
    return "HIGH"


def recommendation(level: str, remaining: float) -> str:
    if level == "HIGH":
        return "Urgent: divert to nearest outlet / discount sale or discard. Check cold chain."
    if level == "MEDIUM":
        return f"Prioritise dispatch (FEFO). About {remaining:.1f} days left - fix temperature drift."
    return "Safe: normal handling. Continue monitoring."


def predict_batch(df: pd.DataFrame, model_dir=None) -> pd.DataFrame:
    clf, reg = load_models(model_dir)
    X = df[FEATURES]
    out = df.copy()
    out["spoilage_probability"] = clf.predict_proba(X)[:, 1].round(4)
    out["remaining_shelf_life_days"] = reg.predict(X).clip(min=0).round(2)
    out["risk_level"] = out["spoilage_probability"].apply(risk_level)
    return out


def predict_one(record: dict, model_dir=None) -> dict:
    res = predict_batch(pd.DataFrame([record]), model_dir).iloc[0]
    level = res["risk_level"]
    return {
        "spoilage_probability": float(res["spoilage_probability"]),
        "remaining_shelf_life_days": float(res["remaining_shelf_life_days"]),
        "risk_level": level,
        "recommendation": recommendation(level, res["remaining_shelf_life_days"]),
    }


SAMPLE = {
    "product": "Milk", "packaging": "Standard", "avg_temp_c": 9.0,
    "temp_std_c": 2.0, "humidity_pct": 85, "transit_hours": 36,
    "days_since_harvest": 2, "co2_ppm": 700, "ethylene_ppm": 5,
    "door_open_events": 4, "vibration_g": 0.5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict food spoilage")
    parser.add_argument("--json", type=str, help="JSON string of one record")
    parser.add_argument("--csv", type=str, help="CSV file for batch prediction")
    parser.add_argument("--out", type=str, default="predictions.csv")
    args = parser.parse_args()

    if args.csv:
        res = predict_batch(pd.read_csv(args.csv))
        res.to_csv(args.out, index=False)
        print(f"Saved predictions -> {args.out}")
    else:
        record = json.loads(args.json) if args.json else SAMPLE
        print(json.dumps(predict_one(record), indent=2))


if __name__ == "__main__":
    main()
