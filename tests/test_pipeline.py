from src.generate_data import generate
from src.predict import predict_one
from src.train import train

BAD = {
    "product": "Fish", "packaging": "Standard", "avg_temp_c": 15.0, "temp_std_c": 3.0,
    "humidity_pct": 90, "transit_hours": 60, "days_since_harvest": 2,
    "co2_ppm": 900, "ethylene_ppm": 10, "door_open_events": 6, "vibration_g": 0.8,
}
GOOD = {**BAD, "avg_temp_c": 1.0, "temp_std_c": 0.3, "transit_hours": 6,
        "days_since_harvest": 0.2, "packaging": "Vacuum", "door_open_events": 0}


def test_data_generation():
    df = generate(500)
    assert len(df) == 500
    assert 0.02 < df["spoiled"].mean() < 0.9
    assert df.isna().sum().sum() == 0


def test_train_and_predict(tmp_path):
    metrics = train(generate(2000), tmp_path)
    assert metrics["classifier"]["roc_auc"] > 0.85
    bad = predict_one(BAD, tmp_path)
    good = predict_one(GOOD, tmp_path)
    assert bad["spoilage_probability"] > good["spoilage_probability"]
    assert bad["remaining_shelf_life_days"] < good["remaining_shelf_life_days"]
    assert bad["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
