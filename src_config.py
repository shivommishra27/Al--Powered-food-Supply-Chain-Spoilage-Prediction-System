"""Shared configuration for the spoilage prediction system."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "supply_chain_data.csv"
MODEL_DIR = ROOT / "models"
CLF_PATH = MODEL_DIR / "spoilage_classifier.joblib"
REG_PATH = MODEL_DIR / "shelf_life_regressor.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# product: (base shelf life in days, ideal storage temp in C, temperature sensitivity)
PRODUCTS = {
    "Milk": (7, 4, 1.6),
    "Fish": (3, 1, 2.0),
    "Chicken": (5, 2, 1.8),
    "Lettuce": (10, 3, 1.3),
    "Tomato": (14, 12, 1.0),
    "Banana": (7, 14, 1.1),
    "Apple": (60, 2, 0.6),
    "Bread": (6, 22, 0.8),
    "Yogurt": (21, 4, 1.2),
}

PACKAGING = {"Standard": 1.0, "Crate": 1.1, "Vacuum": 0.7, "Modified Atmosphere": 0.6}

CATEGORICAL = ["product", "packaging"]
NUMERIC = [
    "avg_temp_c", "temp_std_c", "humidity_pct", "transit_hours",
    "days_since_harvest", "co2_ppm", "ethylene_ppm",
    "door_open_events", "vibration_g",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET_CLF = "spoiled"
TARGET_REG = "remaining_shelf_life_days"
RANDOM_STATE = 42
